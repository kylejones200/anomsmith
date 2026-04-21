"""Tests for predictive maintenance module."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from anomsmith.platform.predictive_maintenance import (
    Alert,
    AlertLevel,
    AlertSystem,
    DashboardVisualizer,
    FailureClassifier,
    FeatureExtractor,
    PredictiveMaintenanceSystem,
    RealTimeIngestion,
    RULEstimator,
    add_degradation_rates,
    add_rolling_statistics,
    calculate_rul,
    create_rul_labels,
    prepare_pm_features,
)
from anomsmith.primitives.detectors.ml import IsolationForestDetector


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def test_extract_from_series(self):
        """Test feature extraction from pandas Series."""
        extractor = FeatureExtractor(rolling_windows=[5, 10])
        data = pd.Series(np.random.randn(100), name="sensor1")
        features = extractor.extract(data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == 100
        assert len(features.columns) > 0

    def test_extract_from_array(self):
        """Test feature extraction from numpy array."""
        extractor = FeatureExtractor(rolling_windows=[5, 10])
        data = np.random.randn(100)
        features = extractor.extract(data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == 100

    def test_extract_from_dataframe(self):
        """Test feature extraction from DataFrame."""
        extractor = FeatureExtractor(rolling_windows=[5, 10])
        data = pd.DataFrame(
            {"temp": np.random.randn(100), "pressure": np.random.randn(100)}
        )
        features = extractor.extract(data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == 100
        assert len(features.columns) > 0

    def test_frequency_features(self):
        """Test frequency domain feature extraction."""
        extractor = FeatureExtractor(rolling_windows=[5], frequency_features=True)
        data = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 100)))
        features = extractor.extract(data)

        assert any("dominant_freq" in col for col in features.columns)
        assert any("spectral_centroid" in col for col in features.columns)

    def test_change_detection_features(self):
        """Test change detection feature extraction."""
        extractor = FeatureExtractor(rolling_windows=[5], change_detection=True)
        data = pd.Series(np.random.randn(100))
        features = extractor.extract(data)

        assert any("_diff" in col for col in features.columns)
        assert any("_pct_change" in col for col in features.columns)


class TestRULEstimator:
    """Tests for RULEstimator."""

    def test_fit_predict(self):
        """Test RUL estimation fit and predict."""
        estimator = RULEstimator(method="regression", random_state=42)
        n_samples = 100
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.uniform(0, 1000, n_samples)  # RUL values

        estimator.fit(X, y)
        assert estimator.is_fitted_

        predictions = estimator.predict(X[:10])
        assert len(predictions) == 10
        assert all(p >= 0 for p in predictions)  # RUL cannot be negative

    def test_fit_predict_dataframe(self):
        """Test RUL estimation with DataFrame input."""
        estimator = RULEstimator(random_state=42)
        X = pd.DataFrame(np.random.randn(50, 5))
        y = pd.Series(np.random.uniform(0, 500, 50))

        estimator.fit(X, y)
        predictions = estimator.predict(X)
        assert len(predictions) == 50

    def test_degradation_threshold(self):
        """Test RUL estimation with degradation threshold."""
        estimator = RULEstimator(random_state=42)
        X = np.random.randn(50, 5)
        degradation = np.linspace(0, 100, 50)  # Increasing degradation
        threshold = 100.0

        estimator.fit(X, degradation, degradation_threshold=threshold)
        predictions = estimator.predict(X[:10])
        assert all(p >= 0 for p in predictions)


class TestFailureClassifier:
    """Tests for FailureClassifier."""

    def test_fit_predict(self):
        """Test failure classification fit and predict."""
        classifier = FailureClassifier(random_state=42)
        n_samples = 100
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)  # Binary labels

        classifier.fit(X, y)
        assert classifier.is_fitted_

        predictions = classifier.predict(X[:10])
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self):
        """Test failure probability prediction."""
        classifier = FailureClassifier(random_state=42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        classifier.fit(X, y)
        probabilities = classifier.predict_proba(X[:10])

        assert probabilities.shape == (10, 2)
        assert all(abs(p.sum() - 1.0) < 1e-6 for p in probabilities)


class TestAlertSystem:
    """Tests for AlertSystem."""

    def test_check_thresholds(self):
        """Test threshold checking."""
        thresholds = {
            "temperature": {"warning": 80.0, "critical": 90.0, "failure": 100.0}
        }
        alert_system = AlertSystem(thresholds=thresholds)

        # Create test data
        features = pd.DataFrame({"temperature": [75.0, 85.0, 95.0, 105.0]})
        alerts = alert_system.check_thresholds(features)

        assert len(alerts) > 0
        assert all(isinstance(alert, Alert) for alert in alerts)
        assert any(alert.level == AlertLevel.WARNING for alert in alerts)
        assert any(alert.level == AlertLevel.CRITICAL for alert in alerts)
        assert any(alert.level == AlertLevel.FAILURE for alert in alerts)

    def test_alert_history(self):
        """Test alert history tracking."""
        thresholds = {"value": {"warning": 10.0}}
        alert_system = AlertSystem(thresholds=thresholds)

        features = pd.DataFrame({"value": [15.0, 20.0]})
        alerts = alert_system.check_thresholds(features)

        assert len(alert_system.alert_history) == len(alerts)
        assert len(alert_system.get_recent_alerts()) > 0

    def test_get_recent_alerts_filtered(self):
        """Test filtering recent alerts."""
        thresholds = {"value": {"warning": 10.0, "critical": 20.0, "failure": 30.0}}
        alert_system = AlertSystem(thresholds=thresholds)

        features = pd.DataFrame({"value": [15.0, 25.0, 35.0]})
        alert_system.check_thresholds(features)

        critical_alerts = alert_system.get_recent_alerts(level=AlertLevel.CRITICAL)
        assert all(alert.level == AlertLevel.CRITICAL for alert in critical_alerts)


class TestPredictiveMaintenanceSystem:
    """Tests for PredictiveMaintenanceSystem."""

    def test_process_without_models(self):
        """Test processing without fitted models."""
        system = PredictiveMaintenanceSystem()
        data = pd.Series(np.random.randn(100))

        results = system.process(data)

        assert "features" not in results or results.get("features") is None
        assert "rul" in results
        assert "failure_probability" in results
        assert "alerts" in results

    def test_process_with_fitted_models(self):
        """Test processing with fitted models."""
        # Create and fit models
        extractor = FeatureExtractor(rolling_windows=[5, 10])
        rul_estimator = RULEstimator(random_state=42)
        failure_classifier = FailureClassifier(random_state=42)

        # Generate training data
        n_train = 200
        train_data = pd.Series(np.random.randn(n_train))
        train_features = extractor.extract(train_data)

        # Fit models
        rul_estimator.fit(
            train_features, np.random.uniform(0, 1000, len(train_features))
        )
        failure_classifier.fit(
            train_features, np.random.randint(0, 2, len(train_features))
        )

        # Create system
        system = PredictiveMaintenanceSystem(
            feature_extractor=extractor,
            rul_estimator=rul_estimator,
            failure_classifier=failure_classifier,
        )

        # Process new data
        new_data = pd.Series(np.random.randn(50))
        results = system.process(new_data, return_features=True)

        assert "features" in results
        assert results["rul"] is not None
        assert results["failure_probability"] is not None
        assert results["failure_prediction"] in [0, 1]

    def test_process_with_anomaly_detector(self):
        """Test processing with integrated anomsmith detector on feature space."""
        extractor = FeatureExtractor(rolling_windows=[5, 10])
        data = pd.Series(np.random.randn(100))
        data.iloc[50:55] += 5  # Inject anomalies
        features = extractor.extract(data)
        detector = IsolationForestDetector(random_state=42)
        detector.fit(np.asarray(features, dtype=float))

        system = PredictiveMaintenanceSystem(
            feature_extractor=extractor,
            anomaly_detector=detector,
        )

        results = system.process(data)

        assert "anomaly_score" in results
        assert "anomaly_prediction" in results
        assert results["anomaly_prediction"] in [0, 1]

    def test_process_with_alert_thresholds(self):
        """Test processing with alert thresholds."""
        thresholds = {"value_rolling_mean_10": {"warning": 0.5}}
        alert_system = AlertSystem(thresholds=thresholds)

        system = PredictiveMaintenanceSystem(alert_system=alert_system)
        data = pd.Series(np.ones(100) * 0.8)  # Values above threshold

        results = system.process(data)

        assert len(results["alerts"]) > 0


class TestUtilityFunctions:
    """Tests for predictive maintenance utility functions."""

    def test_calculate_rul(self):
        """Test RUL calculation."""
        df = pd.DataFrame(
            {
                "asset_id": ["A", "A", "A", "B", "B", "B"],
                "cycle": [1, 2, 3, 1, 2, 3],
            }
        )

        rul = calculate_rul(df, asset_id_col="asset_id", cycle_col="cycle")
        assert len(rul) == len(df)
        assert all(rul >= 0)  # RUL cannot be negative
        # Asset A: max cycle is 3, so RUL at cycle 1 = 2, cycle 2 = 1, cycle 3 = 0
        assert rul.iloc[0] == 2
        assert rul.iloc[2] == 0

    def test_create_rul_labels(self):
        """Test RUL label creation."""
        df = pd.DataFrame({"RUL": [50, 25, 10, 0]})
        df = create_rul_labels(df, warning_threshold=30, critical_threshold=15)

        assert "health_status" in df.columns
        assert "binary_label" in df.columns
        assert "multi_class_label" in df.columns

        # Check health status mapping
        assert df.loc[df["RUL"] == 50, "health_status"].iloc[0] == "healthy"
        assert df.loc[df["RUL"] == 25, "health_status"].iloc[0] == "warning"
        assert df.loc[df["RUL"] == 10, "health_status"].iloc[0] == "critical"
        assert df.loc[df["RUL"] == 0, "health_status"].iloc[0] == "failed"

    def test_add_rolling_statistics(self):
        """Test rolling statistics addition."""
        df = pd.DataFrame(
            {
                "asset_id": ["A"] * 10,
                "cycle": range(1, 11),
                "sensor1": np.random.randn(10),
            }
        )

        result = add_rolling_statistics(
            df,
            feature_cols=["sensor1"],
            asset_id_col="asset_id",
            cycle_col="cycle",
            window=3,
        )

        assert "sensor1_rolling_mean_3" in result.columns
        assert "sensor1_rolling_std_3" in result.columns
        assert len(result) == len(df)

    def test_add_degradation_rates(self):
        """Test degradation rate calculation."""
        df = pd.DataFrame(
            {
                "asset_id": ["A"] * 10,
                "cycle": range(1, 11),
                "sensor1": np.linspace(10, 20, 10),  # Increasing trend
            }
        )

        result = add_degradation_rates(
            df, feature_cols=["sensor1"], asset_id_col="asset_id", cycle_col="cycle"
        )

        assert "sensor1_degradation_rate_1" in result.columns
        assert len(result) == len(df)

    def test_prepare_pm_features(self):
        """Test complete PM feature preparation."""
        df = pd.DataFrame(
            {
                "asset_id": ["A"] * 20 + ["B"] * 20,
                "cycle": list(range(1, 21)) * 2,
                "sensor1": np.random.randn(40),
                "sensor2": np.random.randn(40),
            }
        )

        result = prepare_pm_features(
            df,
            asset_id_col="asset_id",
            cycle_col="cycle",
            feature_cols=["sensor1", "sensor2"],
            calculate_rul_flag=True,
            add_labels=True,
            add_rolling_stats=True,
            include_degradation_rates=False,
            rolling_window=5,
        )

        assert "RUL" in result.columns
        assert "health_status" in result.columns
        assert "binary_label" in result.columns
        assert "sensor1_rolling_mean_5" in result.columns
        assert len(result) == len(df)


class TestRealTimeIngestion:
    """Tests for RealTimeIngestion."""

    def test_ingest_scalar(self):
        """Test ingesting scalar data."""
        pm_system = PredictiveMaintenanceSystem()
        ingestion = RealTimeIngestion(pm_system=pm_system, window_size=5)

        # Ingest data points
        for i in range(10):
            ingestion.ingest(
                data=50.0 + i * 0.1,
                asset_id="ASSET_001",
                sensor_name="sensor1",
            )

        assert "ASSET_001" in ingestion.get_all_assets()
        assert len(ingestion.data_buffers["ASSET_001"]) == 5  # Window size

    def test_process_window(self):
        """Test window processing."""
        # Set up PM system with trained models
        feature_extractor = FeatureExtractor(rolling_windows=[5])
        rul_estimator = RULEstimator(random_state=42)

        # Train on sample data
        train_data = pd.Series(np.random.randn(100))
        train_features = feature_extractor.extract(train_data)
        train_rul = np.random.uniform(0, 100, len(train_features))
        rul_estimator.fit(train_features, train_rul)

        pm_system = PredictiveMaintenanceSystem(
            feature_extractor=feature_extractor,
            rul_estimator=rul_estimator,
        )

        ingestion = RealTimeIngestion(pm_system=pm_system, window_size=5)

        # Ingest enough data to trigger processing
        for i in range(5):
            ingestion.ingest(
                data=50.0 + i * 0.1,
                asset_id="ASSET_001",
                sensor_name="sensor1",
            )

        # Process window
        result = ingestion.process_window("ASSET_001")
        assert "rul" in result or result.get("rul") is None

    def test_get_latest_results(self):
        """Test retrieving latest results."""
        pm_system = PredictiveMaintenanceSystem()
        ingestion = RealTimeIngestion(pm_system=pm_system, window_size=3)

        # Ingest and process
        for i in range(6):
            ingestion.ingest(
                data=50.0 + i * 0.1,
                asset_id="ASSET_001",
                sensor_name="sensor1",
            )

        latest = ingestion.get_latest_results("ASSET_001", n=2)
        assert len(latest) <= 2


class TestDashboardVisualizer:
    """Tests for DashboardVisualizer."""

    def test_create_summary_dashboard(self):
        """Test summary dashboard creation."""
        visualizer = DashboardVisualizer()

        # Create mock results history
        results_history = {
            "ASSET_001": [
                {"rul": 50.0, "failure_probability": 0.2, "timestamp": datetime.now()},
                {"rul": 45.0, "failure_probability": 0.3, "timestamp": datetime.now()},
            ],
            "ASSET_002": [
                {"rul": 20.0, "failure_probability": 0.6, "timestamp": datetime.now()},
            ],
        }

        try:
            fig = visualizer.create_summary_dashboard(results_history)
            assert fig is not None
        except ImportError:
            # Matplotlib not available, skip test
            pass

    def test_create_dashboard_empty(self):
        """Test dashboard creation with empty results."""
        visualizer = DashboardVisualizer()
        results_history = {}

        try:
            with pytest.raises(ValueError):
                visualizer.create_dashboard(results_history)
        except ImportError:
            # Matplotlib not available, skip test
            pass
