"""Model persistence utilities for integration with cloud ML systems.

Provides serialization/deserialization of anomsmith models for deployment
to cloud platforms like AWS SageMaker, Azure ML, or GCP Vertex AI.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from anomsmith.primitives.base import BaseDetector, BaseEstimator, BaseScorer

logger = logging.getLogger(__name__)


def save_model(
    model: BaseEstimator,
    path: Union[str, Path],
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Save an anomsmith model to disk for deployment.

    Saves the model's state, parameters, and metadata in a format suitable
    for cloud deployment (e.g., AWS SageMaker, containerized endpoints).

    Args:
        model: An anomsmith estimator (BaseScorer, BaseDetector, etc.)
        path: Directory path where model will be saved
        metadata: Optional metadata dict (model version, training date, etc.)

    Raises:
        ValueError: If model is not fitted
        OSError: If path cannot be created

    Examples:
        >>> from anomsmith.primitives.scorers.robust_zscore import RobustZScoreScorer
        >>> scorer = RobustZScoreScorer()
        >>> scorer.fit(y_train)
        >>> save_model(scorer, "models/robust_zscore_v1", metadata={"version": "1.0"})
    """
    if not model.is_fitted:
        raise ValueError(f"Model {model.__class__.__name__} must be fitted before saving.")

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save model state
    model_path = path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save metadata
    metadata = metadata or {}
    metadata.update(
        {
            "model_class": model.__class__.__name__,
            "model_module": model.__class__.__module__,
            "parameters": model.get_params(deep=False),
            "fitted": model.is_fitted,
        }
    )

    metadata_path = path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved model to {path}")


def load_model(path: Union[str, Path]) -> BaseEstimator:
    """Load an anomsmith model from disk.

    Args:
        path: Directory path where model was saved

    Returns:
        Loaded model instance

    Raises:
        FileNotFoundError: If model files not found
        ValueError: If model cannot be loaded

    Examples:
        >>> model = load_model("models/robust_zscore_v1")
        >>> scores = model.score(y_test)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")

    model_path = path / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Loaded model from {path}")
    return model


def get_model_metadata(path: Union[str, Path]) -> dict[str, Any]:
    """Get metadata for a saved model without loading it.

    Args:
        path: Directory path where model was saved

    Returns:
        Metadata dictionary

    Raises:
        FileNotFoundError: If metadata file not found
    """
    path = Path(path)
    metadata_path = path / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata


def export_model_for_sagemaker(
    model: BaseEstimator,
    s3_path: str,
    metadata: Optional[dict[str, Any]] = None,
    local_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Export model in format ready for AWS SageMaker deployment.

    Creates a model package that can be uploaded to S3 and deployed
    as a SageMaker endpoint. The model is saved locally first, then
    S3 upload instructions are returned.

    Args:
        model: An anomsmith estimator to export
        s3_path: S3 path where model will be uploaded (e.g., "s3://bucket/models/v1/")
        metadata: Optional metadata for deployment
        local_path: Local path to save model (default: temp directory)

    Returns:
        Dictionary with export information including:
        - local_path: Local path where model was saved
        - s3_path: S3 path for upload
        - upload_command: AWS CLI command to upload
        - inference_code_template: Template for SageMaker inference script

    Examples:
        >>> export_info = export_model_for_sagemaker(
        ...     model, "s3://my-bucket/models/anomaly-detector/v1.0"
        ... )
        >>> print(export_info["upload_command"])
    """
    import tempfile

    if local_path is None:
        local_path = Path(tempfile.mkdtemp()) / "model"
    else:
        local_path = Path(local_path)

    # Save model
    save_model(model, local_path, metadata=metadata)

    # Generate SageMaker-compatible inference code template
    inference_code = _generate_sagemaker_inference_code(model)

    inference_code_path = local_path / "inference.py"
    with open(inference_code_path, "w") as f:
        f.write(inference_code)

    # Generate requirements.txt if needed
    requirements = _generate_requirements(model)
    requirements_path = local_path / "requirements.txt"
    with open(requirements_path, "w") as f:
        f.write("\n".join(requirements))

    # Generate upload command
    upload_command = f"aws s3 sync {local_path} {s3_path} --exclude '*.pyc' --exclude '__pycache__'"

    return {
        "local_path": str(local_path),
        "s3_path": s3_path,
        "upload_command": upload_command,
        "inference_code_template": inference_code,
        "metadata": get_model_metadata(local_path),
    }


def _generate_sagemaker_inference_code(model: BaseEstimator) -> str:
    """Generate SageMaker inference code template for the model."""
    model_class = model.__class__.__name__
    model_module = model.__class__.__module__

    code = f'''"""
SageMaker inference script for {model_class}.

This script is auto-generated for AWS SageMaker deployment.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd


def model_fn(model_dir: str):
    """Load model from disk.
    
    SageMaker calls this function to load the model when the endpoint starts.
    """
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def input_fn(request_body: str, request_content_type: str):
    """Parse input data.
    
    SageMaker calls this function to parse incoming requests.
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        
        # Expect format: {{"instances": [[...], [...]]}}
        if "instances" in data:
            return np.array(data["instances"])
        elif "data" in data:
            return np.array(data["data"])
        else:
            return np.array([data])
    elif request_content_type == "text/csv":
        # CSV input
        from io import StringIO
        df = pd.read_csv(StringIO(request_body), header=None)
        return df.values
    else:
        raise ValueError(f"Unsupported content type: {{request_content_type}}")


def predict_fn(input_data, model):
    """Generate predictions.
    
    SageMaker calls this function to make predictions.
    """
    # Convert to pandas Series if needed for SeriesLike compatibility
    if isinstance(input_data, np.ndarray):
        if input_data.ndim == 1:
            input_series = pd.Series(input_data)
        else:
            # For batch predictions, process each row
            results = []
            for row in input_data:
                input_series = pd.Series(row)
                score_view = model.score(input_series)
                results.append(score_view.scores[0])  # Get first score
            return np.array(results)
    else:
        input_series = input_data
    
    # Score the input
    if hasattr(model, "score"):
        score_view = model.score(input_series)
        scores = score_view.scores
    elif hasattr(model, "predict"):
        label_view = model.predict(input_series)
        scores = label_view.labels.astype(float)
    else:
        raise ValueError(f"Model {{model.__class__.__name__}} has no score or predict method")
    
    return scores


def output_fn(prediction, content_type: str):
    """Format output.
    
    SageMaker calls this function to format the response.
    """
    if content_type == "application/json":
        response = {{"predictions": prediction.tolist()}}
        return json.dumps(response)
    elif content_type == "text/csv":
        return ",".join(str(p) for p in prediction)
    else:
        raise ValueError(f"Unsupported content type: {{content_type}}")
'''
    return code


def _generate_requirements(model: BaseEstimator) -> list[str]:
    """Generate requirements.txt for model deployment."""
    requirements = [
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ]

    # Add model-specific requirements
    model_module = model.__class__.__module__
    if "wavelet" in model_module.lower():
        requirements.append("PyWavelets>=1.3.0")
    if "lstm" in model_module.lower() or "attention" in model_module.lower():
        requirements.append("tensorflow>=2.8.0")
    if "drift" in model_module.lower() or "arima" in model_module.lower():
        requirements.append("statsmodels>=0.13.0")

    # Check for timesmith dependency
    try:
        import timesmith
        requirements.append("timesmith>=0.1.0,<1.0.0")
    except ImportError:
        pass

    return sorted(set(requirements))

