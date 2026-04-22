Retiring ``anomaly_detection_toolkit`` (``asset_health``)
=========================================================

The **Anomaly Detection Toolkit** repository (folder ``asset_health``, import name
``anomaly_detection_toolkit``) has been **fully superseded** by **anomsmith**. After
upgrading, you can archive or delete that repository.

Install and imports
-------------------

.. code-block:: text

   pip install anomsmith[deep,wavelet,plots,stats]

.. code-block:: python

   # Old
   # import anomaly_detection_toolkit as adt

   # New (examples)
   import anomsmith
   from anomsmith import IsolationForestDetector, VotingEnsemble, FeatureExtractor

Module map
----------

=============================================  =============================================
Old (``anomaly_detection_toolkit``)            New (``anomsmith``)
=============================================  =============================================
``BaseDetector``                               :class:`anomsmith.primitives.base.BaseDetector`
``ZScoreDetector``                             :class:`anomsmith.primitives.scorers.statistical.ZScoreScorer` + :func:`anomsmith.detect_anomalies`
``IQROutlierDetector``                         :class:`anomsmith.primitives.scorers.statistical.IQRScorer`
``SeasonalBaselineDetector``                   :class:`anomsmith.primitives.scorers.seasonal.SeasonalBaselineScorer`
``StatisticalDetector``                        Compose scorers / custom :class:`~anomsmith.primitives.base.BaseScorer`
``IsolationForestDetector``                    Same name under ``anomsmith.primitives.detectors.ml``
``LOFDetector`` / ``RobustCovarianceDetector`` Same names under ``anomsmith.primitives.detectors.ml``
``PCADetector``                                Same name under ``anomsmith.primitives.detectors.pca``
``WaveletDetector``                            Same name under ``anomsmith.primitives.detectors.wavelet``
``WaveletDenoiser``                            :class:`anomsmith.primitives.detectors.wavelet.WaveletDenoiser`
``VotingEnsemble``                             Alias → :class:`anomsmith.primitives.detectors.ensemble.VotingEnsembleDetector`
``EnsembleDetector``                           Alias → :class:`anomsmith.primitives.detectors.ensemble.ScoreCombiningEnsembleDetector`
``LSTMAutoencoderDetector``                    :class:`anomsmith.primitives.detectors.autoencoders.LSTMAutoencoderDetector` (needs TensorFlow)
``PyTorchAutoencoderDetector``                 :class:`anomsmith.primitives.detectors.autoencoders.PyTorchAutoencoderDetector` (needs PyTorch)
``predictive_maintenance`` module              :mod:`anomsmith.platform`
``evaluation`` module                          :mod:`anomsmith.platform.evaluation`
``visualization`` module                       :mod:`anomsmith.platform.visualization`
=============================================  =============================================

API differences
---------------

* **Labels:** anomsmith uses **``1`` = anomaly**, **``0`` = normal** in
  :class:`~anomsmith.objects.views.LabelView`. The old toolkit used **``-1`` / ``1``**
  in raw numpy arrays.
* **Fit signature:** primitives use ``fit(y, X=None)`` with time-series-first
  semantics, not only ``fit(X)``.
* **Scores / predictions:** use ``.score(y)`` → :class:`~anomsmith.objects.views.ScoreView`
  and ``.predict(y)`` → :class:`~anomsmith.objects.views.LabelView` instead of
  ``score_samples`` / ``predict`` returning bare arrays.

``prepare_pm_features`` renamed the boolean **``add_degradation_rates``** to
**``include_degradation_rates``** to avoid shadowing the helper function of the same
name.
