"""Visualization utilities for anomaly detection comparison and analysis."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from anomsmith.platform.evaluation import _anomaly_scores, _prediction_labels
from anomsmith.primitives.detectors.pca import PCADetector

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import signalplot

    SIGNALPLOT_AVAILABLE = True
except ImportError:
    SIGNALPLOT_AVAILABLE = False


def _apply_plot_style():
    """Apply signalplot style if available, otherwise use minimal matplotlib defaults."""
    if SIGNALPLOT_AVAILABLE:
        signalplot.apply()
    elif MATPLOTLIB_AVAILABLE:
        # Apply minimal clean defaults similar to signalplot philosophy
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.linewidth": 0.8,
                "axes.grid": True,
                "axes.grid.axis": "y",
                "grid.alpha": 0.3,
                "grid.linewidth": 0.5,
                "font.family": "sans-serif",
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "xtick.direction": "out",
                "ytick.direction": "out",
            }
        )


def _get_colors():
    """Get color palette following signalplot conventions."""
    if SIGNALPLOT_AVAILABLE:
        # Use signalplot's accent color for anomalies
        try:
            accent = signalplot.ACCENT
        except AttributeError:
            accent = "#d62728"  # Default red accent
        return {
            "primary": "black",
            "secondary": "#666666",
            "accent": accent,
            "normal": "#333333",
            "anomaly": accent,
            "true_anomaly": "#2ca02c",  # Green for true labels
        }
    else:
        # Fallback colors (minimalist)
        return {
            "primary": "black",
            "secondary": "#666666",
            "accent": "#d62728",
            "normal": "#333333",
            "anomaly": "#d62728",
            "true_anomaly": "#2ca02c",
        }


def _save_figure(fig, save_path: str):
    """Save figure with signalplot defaults (300 DPI, white background, tight bbox)."""
    if save_path:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        logger.info("Figure saved to: %s", save_path)


def plot_pca_boundary(
    detector: PCADetector,
    X: np.ndarray | pd.DataFrame,
    y_true: np.ndarray | None = None,
    n_components_plot: int = 2,
    save_path: str | None = None,
):
    """Visualize PCA boundary in 2D projection (anomsmith :class:`~anomsmith.primitives.detectors.pca.PCADetector`)."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    if detector.pca_ is None:
        raise ValueError("Detector must be fitted first.")

    if n_components_plot < 2:
        raise ValueError("n_components_plot must be at least 2")

    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    X_scaled = detector.scaler_.transform(arr)
    X_transformed = detector.pca_.transform(X_scaled)  # type: ignore[union-attr]
    n_actual_components = X_transformed.shape[1]

    if n_components_plot > n_actual_components:
        n_components_plot = n_actual_components

    X_2d = X_transformed[:, :n_components_plot]

    predictions = _prediction_labels(detector, X)

    _apply_plot_style()
    colors = _get_colors()

    fig, ax = plt.subplots(figsize=(10, 8))

    normal_mask = predictions == 0
    anomaly_mask = predictions == 1

    if np.any(normal_mask):
        ax.scatter(
            X_2d[normal_mask, 0],
            X_2d[normal_mask, 1],
            c=colors["normal"],
            alpha=0.6,
            label="Normal",
            s=50,
            edgecolors="none",
        )

    if np.any(anomaly_mask):
        ax.scatter(
            X_2d[anomaly_mask, 0],
            X_2d[anomaly_mask, 1],
            c=colors["anomaly"],
            alpha=0.7,
            label="Anomaly",
            s=60,
            marker="x",
            linewidths=1.5,
        )

    if y_true is not None:
        true_anomaly_mask = (y_true == 1) | (y_true == -1)
        if np.any(true_anomaly_mask):
            ax.scatter(
                X_2d[true_anomaly_mask, 0],
                X_2d[true_anomaly_mask, 1],
                c=colors["true_anomaly"],
                alpha=0.4,
                label="True Anomaly",
                s=40,
                marker="o",
                edgecolors=colors["primary"],
                linewidths=0.5,
            )

    evr = detector.pca_.explained_variance_ratio_  # type: ignore[union-attr]
    if detector.mean_ is not None and detector.mean_.shape[0] >= n_components_plot:
        ax.scatter(
            float(detector.mean_[0]),
            float(detector.mean_[1]),
            c=colors["primary"],
            marker="*",
            s=200,
            label="PC mean (score space)",
            zorder=5,
            edgecolors="none",
        )

    ax.set_xlabel(f"Principal Component 1 ({evr[0]:.2%} variance)")
    ax.set_ylabel(f"Principal Component 2 ({evr[1]:.2%} variance)")
    ax.set_title("PCA Anomaly Detection - 2D Projection")
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_reconstruction_error(
    detector,
    X: np.ndarray | pd.DataFrame,
    y_true: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    save_path: str | None = None,
):
    """
    Plot reconstruction error over time for LSTM or PCA detector.

    Parameters
    ----------
    detector : BaseDetector
        Fitted detector (PCA or LSTM).
    X : array-like
        Data to plot.
    y_true : ndarray, optional
        True labels for marking actual anomalies.
    timestamps : ndarray, optional
        Timestamps for x-axis.
    save_path : str, optional
        Path to save the figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    scores = _anomaly_scores(detector, X)
    predictions = _prediction_labels(detector, X)

    if timestamps is None:
        timestamps = np.arange(len(scores))

    _apply_plot_style()
    colors = _get_colors()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot reconstruction error
    ax.plot(
        timestamps,
        scores,
        color=colors["primary"],
        linewidth=1.5,
        label="Reconstruction Error",
        alpha=0.8,
    )

    # Mark predicted anomalies (anomsmith: 1 = anomaly)
    anomaly_mask = predictions == 1
    if np.any(anomaly_mask):
        ax.scatter(
            timestamps[anomaly_mask],
            scores[anomaly_mask],
            c=colors["anomaly"],
            s=100,
            marker="x",
            label="Detected Anomalies",
            zorder=5,
            linewidths=1.5,
        )

    # Mark true anomalies if provided
    if y_true is not None:
        true_anomaly_mask = (y_true == 1) | (y_true == -1)
        if np.any(true_anomaly_mask):
            ax.scatter(
                timestamps[true_anomaly_mask],
                scores[true_anomaly_mask],
                c=colors["true_anomaly"],
                s=50,
                marker="o",
                alpha=0.6,
                label="True Anomalies",
                edgecolors=colors["primary"],
                linewidths=0.5,
                zorder=4,
            )

    # Plot threshold if available
    if hasattr(detector, "threshold_") and detector.threshold_ is not None:
        ax.axhline(
            y=detector.threshold_,
            color=colors["secondary"],
            linestyle="--",
            linewidth=1,
            label="Threshold",
            alpha=0.7,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Reconstruction Error / Anomaly Score")
    ax.set_title("Reconstruction Error Over Time")
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_comparison_metrics(
    comparison_df: pd.DataFrame,
    metrics: list[str] | None = None,
    save_path: str | None = None,
):
    """
    Create comparison chart for multiple detectors.

    Parameters
    ----------
    comparison_df : DataFrame
        DataFrame from compare_detectors().
    metrics : list of str, optional
        Metrics to plot. Default: ['precision', 'recall', 'f1'].
    save_path : str, optional
        Path to save the figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    if metrics is None:
        metrics = ["precision", "recall", "f1"]

    available_metrics = [m for m in metrics if m in comparison_df.columns]
    if len(available_metrics) == 0:
        raise ValueError("No valid metrics found in comparison DataFrame")

    _apply_plot_style()
    colors = _get_colors()

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    detectors = comparison_df["detector"].values

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        values = comparison_df[metric].values

        bars = ax.bar(
            detectors,
            values,
            color=colors["primary"],
            alpha=0.8,
            edgecolor=colors["primary"],
            linewidth=0.5,
        )
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Comparison")
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=colors["primary"],
            )

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_sensor_drift(
    sensor_data: np.ndarray | pd.Series,
    predictions: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    save_path: str | None = None,
):
    """
    Visualize sensor drift with anomaly flags.

    Parameters
    ----------
    sensor_data : array-like
        Sensor readings over time.
    predictions : ndarray, optional
        Anomaly predictions (``1`` for anomaly, ``0`` for normal).
    timestamps : ndarray, optional
        Timestamps for x-axis.
    save_path : str, optional
        Path to save the figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    sensor_data = np.asarray(sensor_data)

    if timestamps is None:
        timestamps = np.arange(len(sensor_data))

    _apply_plot_style()
    colors = _get_colors()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot sensor data
    ax.plot(
        timestamps,
        sensor_data,
        color=colors["primary"],
        linewidth=1.5,
        label="Sensor Reading",
        alpha=0.8,
    )

    # Mark anomalies if provided
    if predictions is not None:
        anomaly_mask = predictions == 1
        if np.any(anomaly_mask):
            ax.scatter(
                timestamps[anomaly_mask],
                sensor_data[anomaly_mask],
                c=colors["anomaly"],
                s=100,
                marker="x",
                label="Anomaly",
                zorder=5,
                linewidths=1.5,
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Sensor Value")
    ax.set_title("Sensor Drift with Anomaly Detection")
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig
