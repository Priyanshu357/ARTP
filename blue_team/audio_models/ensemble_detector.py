"""Ensemble detector for audio models.

Combines multiple audio detectors (Energy + Spectral) using voting or
score aggregation to produce a final adversarial detection decision.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .base_detector import BaseDetector
from .energy_detector import EnergyDetector
from .spectral_detector import SpectralDetector


class EnsembleDetector(BaseDetector):
    """Combine multiple audio detectors using voting or score aggregation.

    By default, uses both EnergyDetector and SpectralDetector.
    Supports two modes:
    - "vote": majority voting (detected if majority of detectors agree)
    - "average": average anomaly scores, threshold at 0.5
    """

    def __init__(
        self,
        detectors: Optional[List[BaseDetector]] = None,
        mode: str = "average",
    ):
        """Initialize ensemble detector.

        Args:
            detectors: List of detector instances. If None, uses default
                       [EnergyDetector, SpectralDetector].
            mode: Combination mode ("vote" or "average")
        """
        if detectors is None:
            detectors = [
                EnergyDetector(),
                SpectralDetector(),
            ]
        self.detectors = detectors
        self.mode = mode

    def detect(
        self,
        original_output: Any,
        adversarial_output: Any,
        original_audio: Optional[np.ndarray] = None,
        adversarial_audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Detect using ensemble of detectors.

        Runs all detectors and aggregates their results.

        Args:
            original_output: Model output on clean audio
            adversarial_output: Model output on adversarial audio
            original_audio: Clean audio waveform
            adversarial_audio: Adversarial audio waveform
            sample_rate: Audio sample rate in Hz

        Returns:
            Dict with aggregated anomaly_score, detected flag, and explanations
        """
        individual_results = []

        for detector in self.detectors:
            try:
                result = detector.detect(
                    original_output=original_output,
                    adversarial_output=adversarial_output,
                    original_audio=original_audio,
                    adversarial_audio=adversarial_audio,
                    sample_rate=sample_rate,
                )
                individual_results.append(result)
            except Exception as e:
                print(f"[WARNING] Detector {detector.__class__.__name__} failed: {e}")
                individual_results.append({
                    "anomaly_score": 0.0,
                    "detected": False,
                    "explanation": f"Detector error: {e}",
                })

        if not individual_results:
            return {
                "anomaly_score": 0.0,
                "detected": False,
                "is_attack": False,
                "explanation": "No detectors produced results",
            }

        # Aggregate based on mode
        scores = [r.get("anomaly_score", 0.0) for r in individual_results]
        detections = [r.get("detected", False) for r in individual_results]
        explanations = [r.get("explanation", "") for r in individual_results]

        if self.mode == "vote":
            # Majority voting
            detected = sum(detections) > len(detections) / 2
            anomaly_score = float(np.mean(scores))
        else:
            # Average score
            anomaly_score = float(np.mean(scores))
            detected = anomaly_score > 0.5

        # Combine explanations
        non_empty = [e for e in explanations if e and "No " not in e]
        combined_explanation = (
            " | ".join(non_empty) if non_empty else "No anomalies detected by ensemble"
        )

        return {
            "anomaly_score": anomaly_score,
            "detected": detected,
            "is_attack": any(r.get("is_attack", False) for r in individual_results),
            "explanation": combined_explanation,
            "individual_results": [
                {
                    "detector": self.detectors[i].__class__.__name__,
                    "anomaly_score": individual_results[i].get("anomaly_score", 0.0),
                    "detected": individual_results[i].get("detected", False),
                }
                for i in range(len(individual_results))
            ],
        }
