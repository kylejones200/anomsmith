"""Tests that reinforce layer boundaries (no new behavior assertions beyond contracts)."""

import numpy as np
import pandas as pd
import pytest

from anomsmith.constants import (
    DEFAULT_ASSET_HEALTH_ANOMALY_WEIGHT,
    DEFAULT_ASSET_HEALTH_CLASSIFICATION_WEIGHT,
)
from anomsmith.primitives.detectors.pca import PCADetector
from anomsmith.primitives.scorers.robust_zscore import RobustZScoreScorer
from anomsmith.primitives.thresholding import ThresholdRule
from anomsmith.tasks.detect import run_scoring
from anomsmith.workflows.asset_health import assess_asset_health
from anomsmith.workflows.batch_inference import _score_batch
from anomsmith.workflows.detect import detect_anomalies, sweep_thresholds
from anomsmith.workflows.eval.metrics import (
    compute_f1,
    compute_precision,
    compute_recall,
)
from anomsmith.workflows.pca_pm import track_mahalanobis_distance
from anomsmith.workflows.survival import predict_rul_from_survival


class TestPCADelegatesScoringToPrimitive:
    def test_track_mahalanobis_matches_detector_score(self) -> None:
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((40, 3))
        X_new = rng.standard_normal((15, 3))
        det = PCADetector(n_components=2, score_method="mahalanobis")
        det.fit(X_train)

        expected = det.score(X_new).scores
        series = track_mahalanobis_distance(X_new, det)
        np.testing.assert_allclose(series.values, expected)

    def test_track_requires_mahalanobis_method(self) -> None:
        det = PCADetector(n_components=2, score_method="reconstruction")
        det.fit(np.random.default_rng(1).standard_normal((30, 2)))
        with pytest.raises(ValueError, match="score_method='mahalanobis'"):
            track_mahalanobis_distance(
                np.random.default_rng(1).standard_normal((5, 2)), det
            )


class TestSweepThresholdsUsesMetricsModule:
    def test_sweep_matches_per_threshold_metrics(self) -> None:
        y = pd.Series(np.linspace(-3, 3, 120), index=pd.RangeIndex(120))
        labels = pd.Series((np.random.default_rng(2).random(120) > 0.85).astype(int))
        scorer = RobustZScoreScorer()
        scorer.fit(y.values)
        score_view = run_scoring(y, scorer)
        scores = score_view.scores
        thresholds = np.array([0.5, 1.0, 1.5])
        aligned = labels.reindex(score_view.index, fill_value=0).values

        result = sweep_thresholds(y, scorer, thresholds, labels=labels)

        for i, t in enumerate(thresholds):
            y_pred = (scores >= t).astype(int)
            assert result["precision"].iloc[i] == pytest.approx(
                float(compute_precision(aligned, y_pred))
            )
            assert result["recall"].iloc[i] == pytest.approx(
                float(compute_recall(aligned, y_pred))
            )
            assert result["f1"].iloc[i] == pytest.approx(
                float(compute_f1(aligned, y_pred))
            )


class TestBatchScoringUsesTasksForSeries:
    def test_score_batch_uses_task_path_for_series(self) -> None:
        y = pd.Series(np.random.default_rng(3).standard_normal(50))
        scorer = RobustZScoreScorer()
        scorer.fit(y.values)
        direct = run_scoring(y, scorer)
        via = _score_batch(y, scorer)
        np.testing.assert_array_equal(via.scores, direct.scores)
        assert via.index.equals(direct.index)


class TestSurvivalRulIndexContract:
    def test_predict_rul_defaults_index_from_dataframe(self) -> None:
        class _StubModel:
            def predict_time_to_failure(self, X, threshold=0.5):
                n = len(X)
                return np.arange(n, dtype=float)

        X = pd.DataFrame(
            {"a": [1.0, 2.0]},
            index=pd.Index(["u1", "u2"], name="asset"),
        )
        rul = predict_rul_from_survival(_StubModel(), X)  # type: ignore[arg-type]
        assert rul.index.equals(X.index)


class TestAssessAssetHealthExplicitPolicyParams:
    def test_default_fusion_matches_constants(self) -> None:
        rng = np.random.default_rng(5)
        X = pd.DataFrame(rng.standard_normal((16, 2)), columns=["a", "b"])
        labels = (rng.random(16) > 0.5).astype(int)
        fast = dict(
            failure_labels=labels,
            random_state=0,
            n_estimators=5,
            isolation_n_estimators=10,
        )
        r1 = assess_asset_health(X, **fast)
        r2 = assess_asset_health(
            X,
            **fast,
            classification_weight=DEFAULT_ASSET_HEALTH_CLASSIFICATION_WEIGHT,
            anomaly_weight=DEFAULT_ASSET_HEALTH_ANOMALY_WEIGHT,
        )
        pd.testing.assert_frame_equal(r1, r2)

    def test_fusion_weights_must_sum_to_one(self) -> None:
        X = pd.DataFrame(np.zeros((5, 2)))
        labels = np.zeros(5, dtype=int)
        with pytest.raises(ValueError, match="sum to 1"):
            assess_asset_health(
                X,
                failure_labels=labels,
                use_anomaly_detection=False,
                classification_weight=0.5,
                anomaly_weight=0.4,
            )

    def test_probability_thresholds_ordered(self) -> None:
        X = pd.DataFrame(np.zeros((5, 2)))
        labels = np.zeros(5, dtype=int)
        with pytest.raises(ValueError, match="risk_proba_warning_threshold"):
            assess_asset_health(
                X,
                failure_labels=labels,
                use_anomaly_detection=False,
                risk_proba_warning_threshold=0.9,
                risk_proba_distress_threshold=0.5,
            )


class TestDetectWorkflowThinWrapper:
    def test_detect_delegates_scoring_to_tasks(self) -> None:
        y = pd.Series(np.random.default_rng(4).standard_normal(40))
        scorer = RobustZScoreScorer()
        scorer.fit(y.values)
        rule = ThresholdRule(method="absolute", value=0.0)
        out = detect_anomalies(y, scorer, rule)
        task_scores = run_scoring(y, scorer).scores
        np.testing.assert_array_equal(out["score"].values, task_scores)
