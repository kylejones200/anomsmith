"""Tests for model persistence."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anomsmith.primitives.detectors.ml import IsolationForestDetector
from anomsmith.primitives.model_persistence import (
    get_model_metadata,
    load_model,
    save_model,
)
from anomsmith.primitives.scorers.robust_zscore import RobustZScoreScorer


class TestSaveModel:
    """Tests for save_model."""

    def test_save_raises_if_not_fitted(self) -> None:
        """Test save_model raises ValueError if model not fitted."""
        scorer = RobustZScoreScorer()
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="must be fitted"):
                save_model(scorer, Path(tmp) / "model")

    def test_save_succeeds_when_fitted(self) -> None:
        """Test save_model creates files when model is fitted."""
        scorer = RobustZScoreScorer()
        y = np.random.randn(50)
        scorer.fit(y)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            save_model(scorer, path)
            assert (path / "model.pkl").exists()
            assert (path / "metadata.json").exists()


class TestLoadModel:
    """Tests for load_model."""

    def test_load_raises_file_not_found_for_missing_path(self) -> None:
        """Test load_model raises FileNotFoundError for non-existent path."""
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nonexistent"
            with pytest.raises(FileNotFoundError, match="does not exist"):
                load_model(missing)

    def test_load_raises_file_not_found_for_missing_pkl(self) -> None:
        """Test load_model raises FileNotFoundError when model.pkl missing."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            path.mkdir()
            (path / "metadata.json").write_text("{}")
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                load_model(path)


class TestRoundTrip:
    """Tests for save/load round-trip."""

    def test_round_trip_produces_same_scores(self) -> None:
        """Test save → load → score produces same results within tolerance."""
        scorer = RobustZScoreScorer()
        y_train = np.random.randn(100)
        y_test = np.random.randn(30)
        scorer.fit(y_train)
        scores_before = scorer.score(pd.Series(y_test)).scores

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            save_model(scorer, path)
            loaded = load_model(path)
            scores_after = loaded.score(pd.Series(y_test)).scores

        np.testing.assert_array_almost_equal(scores_before, scores_after)

    def test_round_trip_detector(self) -> None:
        """Test round-trip with IsolationForest detector."""
        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        X = np.random.randn(80, 3)
        detector.fit(X)
        X_test = np.random.randn(20, 3)
        labels_before = detector.predict(X_test).labels

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detector"
            save_model(detector, path)
            loaded = load_model(path)
            labels_after = loaded.predict(X_test).labels

        np.testing.assert_array_equal(labels_before, labels_after)


class TestGetModelMetadata:
    """Tests for get_model_metadata."""

    def test_get_metadata_without_loading(self) -> None:
        """Test get_model_metadata returns metadata without loading model."""
        scorer = RobustZScoreScorer()
        scorer.fit(np.random.randn(50))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            save_model(scorer, path, metadata={"version": "1.0"})
            meta = get_model_metadata(path)
            assert meta["model_class"] == "RobustZScoreScorer"
            assert meta["version"] == "1.0"
            assert meta["fitted"] is True
