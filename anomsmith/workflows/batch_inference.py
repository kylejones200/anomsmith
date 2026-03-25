"""Batch inference workflows for processing large datasets.

Designed for cloud data pipelines (AWS Kinesis, S3, etc.) where data
arrives in batches and needs to be processed efficiently.
"""

import logging
from typing import TYPE_CHECKING, Iterator, Optional, Union

import numpy as np
import pandas as pd

from anomsmith.objects.views import LabelView, ScoreView
from anomsmith.primitives.base import BaseDetector, BaseScorer
from anomsmith.tasks.detect import run_detection, run_scoring

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


def _score_batch(batch_data: Union[np.ndarray, pd.Series, pd.DataFrame], scorer: BaseScorer) -> ScoreView:
    """Score one batch via tasks when input is series-like; else delegate to scorer."""
    if isinstance(batch_data, pd.DataFrame):
        if batch_data.shape[1] != 1:
            return scorer.score(batch_data)
        batch_data = batch_data.iloc[:, 0]
    return run_scoring(batch_data, scorer)


def _predict_batch(
    batch_data: Union[np.ndarray, pd.Series, pd.DataFrame], detector: BaseDetector
) -> tuple[LabelView, ScoreView]:
    if isinstance(batch_data, pd.DataFrame):
        if batch_data.shape[1] != 1:
            label_view = detector.predict(batch_data)
            score_view = detector.score(batch_data)
            return label_view, score_view
        batch_data = batch_data.iloc[:, 0]
    return run_detection(batch_data, detector)


def _series_from_s3_csv_body(body) -> pd.Series:
    """Parse S3 object body as CSV and return the first column as a Series."""
    df = pd.read_csv(body)
    y = df.iloc[:, 0]
    return y if isinstance(y, pd.Series) else pd.Series(y)


def batch_score(
    data_iterator: Iterator[Union[np.ndarray, pd.Series, pd.DataFrame]],
    scorer: BaseScorer,
) -> Iterator[ScoreView]:
    """Score anomalies in batches for efficient processing of large datasets.

    Designed for stream processing (e.g., AWS Kinesis, S3 batch jobs)
    where data arrives in chunks.

    Args:
        data_iterator: Iterator yielding batches of time series data
        scorer: Fitted BaseScorer instance

    Yields:
        ScoreView for each batch

    Examples:
        >>> def data_stream():
        ...     for i in range(0, 10000, 1000):
        ...         yield pd.Series(np.random.randn(1000), index=pd.date_range(start=f"2024-01-01", periods=1000, freq="H") + pd.Timedelta(hours=i))
        >>> scorer = RobustZScoreScorer()
        >>> scorer.fit(y_train)
        >>> for batch_scores in batch_score(data_stream(), scorer):
        ...     process_scores(batch_scores)
    """
    if not scorer.is_fitted:
        raise ValueError("Scorer must be fitted before batch scoring.")

    for batch_idx, batch_data in enumerate(data_iterator):
        score_view = _score_batch(batch_data, scorer)
        logger.debug(f"Scored batch {batch_idx}: {len(score_view.scores)} samples")
        yield score_view


def batch_predict(
    data_iterator: Iterator[Union[np.ndarray, pd.Series, pd.DataFrame]],
    detector: BaseDetector,
) -> Iterator[tuple[LabelView, ScoreView]]:
    """Predict anomalies in batches for efficient processing.

    Args:
        data_iterator: Iterator yielding batches of time series data
        detector: Fitted BaseDetector instance

    Yields:
        Tuple of (LabelView, ScoreView) for each batch

    Examples:
        >>> detector = IsolationForestDetector(contamination=0.05)
        >>> detector.fit(X_train)
        >>> for labels, scores in batch_predict(data_stream(), detector):
        ...     process_predictions(labels, scores)
    """
    if not detector.is_fitted:
        raise ValueError("Detector must be fitted before batch prediction.")

    for batch_idx, batch_data in enumerate(data_iterator):
        label_view, score_view = _predict_batch(batch_data, detector)
        logger.debug(f"Predicted batch {batch_idx}: {len(label_view.labels)} samples")
        yield (label_view, score_view)


def process_s3_batch(
    s3_keys: list[str],
    model: Union[BaseScorer, BaseDetector],
    bucket: str,
    s3_client=None,
) -> pd.DataFrame:
    """Process a batch of S3 files with anomaly detection.

    Designed for AWS Lambda or SageMaker batch jobs that process
    S3 data in batches.

    Args:
        s3_keys: List of S3 object keys to process
        model: Fitted model (BaseScorer or BaseDetector)
        bucket: S3 bucket name (required)
        s3_client: Optional boto3 S3 client (will create if not provided)

    Returns:
        DataFrame with results for all processed files

    Raises:
        ImportError: If boto3 not available
        ValueError: If model not fitted or bucket not specified

    Examples:
        >>> s3_keys = ["data/2024/01/01/file1.csv", "data/2024/01/01/file2.csv"]
        >>> results = process_s3_batch(s3_keys, scorer, bucket="my-data-bucket")
    """
    if not s3_keys:
        raise ValueError("s3_keys must be non-empty.")

    if not model.is_fitted:
        raise ValueError("Model must be fitted before processing.")

    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 batch processing. Install with: pip install boto3"
        )

    if s3_client is None:
        s3_client = boto3.client("s3")

    all_results = []

    for s3_key in s3_keys:
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
            y = _series_from_s3_csv_body(obj["Body"])

            if isinstance(model, BaseScorer):
                score_view = run_scoring(y, model)
                result_df = pd.DataFrame(
                    {"score": score_view.scores, "s3_key": s3_key},
                    index=score_view.index,
                )
            else:
                label_view, score_view = run_detection(y, model)
                result_df = pd.DataFrame(
                    {
                        "label": label_view.labels,
                        "score": score_view.scores,
                        "s3_key": s3_key,
                    },
                    index=label_view.index,
                )

            all_results.append(result_df)
            logger.info(f"Processed {s3_key}: {len(result_df)} samples")

        except Exception as e:
            logger.exception("Error processing %s: %s", s3_key, e)
            continue

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=False)
