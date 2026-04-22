"""Parity tests for primitives merged from the former Anomaly Detection Toolkit."""

import numpy as np
import pandas as pd
import pytest

from anomsmith.primitives.detectors.ensemble import (
    EnsembleDetector,
    ScoreCombiningEnsembleDetector,
    VotingEnsemble,
    VotingEnsembleDetector,
)
from anomsmith.primitives.detectors.wavelet import WaveletDenoiser
from anomsmith.primitives.scorers.robust_zscore import RobustZScoreScorer


def test_voting_ensemble_is_alias() -> None:
    assert VotingEnsemble is VotingEnsembleDetector


def test_ensemble_detector_is_alias() -> None:
    assert EnsembleDetector is ScoreCombiningEnsembleDetector


def test_score_combining_ensemble() -> None:
    y = pd.Series(np.random.randn(80))
    a = RobustZScoreScorer()
    b = RobustZScoreScorer()
    a.fit(y.values)
    b.fit(y.values)
    ens = ScoreCombiningEnsembleDetector([a, b], combination_method="mean")
    ens.fit(y)
    lv = ens.predict(y)
    assert len(lv.labels) == len(y)


def test_wavelet_denoiser() -> None:
    pytest.importorskip("pywt")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    den = WaveletDenoiser(level=3)
    out = den.denoise(x)
    assert out.shape == x.shape


def test_pytorch_autoencoder_smoke() -> None:
    pytest.importorskip("torch")
    from anomsmith.primitives.detectors.autoencoders import PyTorchAutoencoderDetector

    y = pd.Series(np.random.randn(60))
    m = PyTorchAutoencoderDetector(window_size=10, epochs=2, batch_size=8)
    m.fit(y)
    sv = m.score(y)
    assert len(sv.scores) == len(y)
    lv = m.predict(y)
    assert len(lv.labels) == len(y)


def test_lstm_autoencoder_smoke() -> None:
    pytest.importorskip("keras")
    from anomsmith.primitives.detectors.autoencoders import LSTMAutoencoderDetector

    y = pd.Series(np.random.randn(80))
    m = LSTMAutoencoderDetector(window_size=12, epochs=1, batch_size=16)
    m.fit(y)
    sv = m.score(y)
    assert len(sv.scores) == len(y)
