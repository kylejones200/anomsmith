"""Batch inference workflows for processing large datasets.

Designed for cloud data pipelines (AWS Kinesis, S3, etc.) where data
arrives in batches and needs to be processed efficiently.
"""

import logging
from typing import TYPE_CHECKING, Iterator, Optional, Union

import numpy as np
import pandas as pd

from anomsmith.primitives.base import BaseDetector, BaseScorer
from anomsmith.objects.views import LabelView, ScoreView

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


def batch_score(
    data_iterator: Iterator[Union[np.ndarray, pd.Series, pd.DataFrame]],
    scorer: BaseScorer,
    batch_size: int = 1000,
    return_index: bool = True,
) -> Iterator[ScoreView]:
    """Score anomalies in batches for efficient processing of large datasets.

    Designed for stream processing (e.g., AWS Kinesis, S3 batch jobs)
    where data arrives in chunks.

    Args:
        data_iterator: Iterator yielding batches of time series data
        scorer: Fitted BaseScorer instance
        batch_size: Number of samples per batch (default 1000)
        return_index: Whether to preserve index in results (default True)

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
        try:
            score_view = scorer.score(batch_data)
            logger.debug(f"Scored batch {batch_idx}: {len(score_view.scores)} samples")
            yield score_view
        except Exception as e:
            logger.error(f"Error scoring batch {batch_idx}: {e}")
            # Yield empty scores on error
            if isinstance(batch_data, pd.Series):
                index = batch_data.index
            else:
                index = pd.RangeIndex(start=0, stop=len(batch_data))
            yield ScoreView(index=index, scores=np.zeros(len(index)))


def batch_predict(
    data_iterator: Iterator[Union[np.ndarray, pd.Series, pd.DataFrame]],
    detector: BaseDetector,
    batch_size: int = 1000,
    return_index: bool = True,
) -> Iterator[tuple[LabelView, ScoreView]]:
    """Predict anomalies in batches for efficient processing.

    Args:
        data_iterator: Iterator yielding batches of time series data
        detector: Fitted BaseDetector instance
        batch_size: Number of samples per batch (default 1000)
        return_index: Whether to preserve index in results (default True)

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
        try:
            label_view = detector.predict(batch_data)
            score_view = detector.score(batch_data)
            logger.debug(f"Predicted batch {batch_idx}: {len(label_view.labels)} samples")
            yield (label_view, score_view)
        except Exception as e:
            logger.error(f"Error predicting batch {batch_idx}: {e}")
            # Yield empty results on error
            if isinstance(batch_data, pd.Series):
                index = batch_data.index
            else:
                index = pd.RangeIndex(start=0, stop=len(batch_data))
            empty_labels = LabelView(index=index, labels=np.zeros(len(index), dtype=int))
            empty_scores = ScoreView(index=index, scores=np.zeros(len(index)))
            yield (empty_labels, empty_scores)


def process_s3_batch(
    s3_keys: list[str],
    model: Union[BaseScorer, BaseDetector],
    s3_client=None,
    bucket: Optional[str] = None,
) -> pd.DataFrame:
    """Process a batch of S3 files with anomaly detection.

    Designed for AWS Lambda or SageMaker batch jobs that process
    S3 data in batches.

    Args:
        s3_keys: List of S3 object keys to process
        model: Fitted model (BaseScorer or BaseDetector)
        s3_client: Optional boto3 S3 client (will create if not provided)
        bucket: S3 bucket name (required if s3_client not provided)

    Returns:
        DataFrame with results for all processed files

    Raises:
        ImportError: If boto3 not available
        ValueError: If model not fitted or bucket not specified

    Examples:
        >>> s3_keys = ["data/2024/01/01/file1.csv", "data/2024/01/01/file2.csv"]
        >>> results = process_s3_batch(s3_keys, scorer, bucket="my-data-bucket")
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 batch processing. Install with: pip install boto3"
        )

    if not model.is_fitted:
        raise ValueError("Model must be fitted before processing.")

    if s3_client is None:
        if bucket is None:
            raise ValueError("Either s3_client or bucket must be provided.")
        s3_client = boto3.client("s3")

    if bucket is None:
        # Extract bucket from first key if possible
        if "/" in s3_keys[0]:
            bucket = s3_keys[0].split("/")[0]
        else:
            raise ValueError("Bucket must be specified.")

    all_results = []

    for s3_key in s3_keys:
        try:
            # Download data from S3
            obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
            df = pd.read_csv(obj["Body"])

            # Assume first column is time series values
            y = df.iloc[:, 0]

            # Score or predict
            if isinstance(model, BaseScorer):
                score_view = model.score(y)
                result_df = pd.DataFrame(
                    {"score": score_view.scores, "s3_key": s3_key}, index=score_view.index
                )
            else:
                label_view = model.predict(y)
                score_view = model.score(y)
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
            logger.error(f"Error processing {s3_key}: {e}")
            continue

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=False)

