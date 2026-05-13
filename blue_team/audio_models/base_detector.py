"""Base class for audio adversarial detectors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseDetector(ABC):
    """Abstract base class for detecting adversarial audio inputs.

    TODO: Implement audio-specific detection interface.
    Audio detectors analyze spectral features, energy patterns, and model
    confidence to identify adversarial perturbations.
    """

    @abstractmethod
    def detect(
        self,
        original_output: Any,
        adversarial_output: Any,
        original_audio: Optional[np.ndarray] = None,
        adversarial_audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Detect adversarial audio examples.

        Args:
            original_output: Model output on clean audio
            adversarial_output: Model output on adversarial audio
            original_audio: Clean audio waveform (optional)
            adversarial_audio: Adversarial audio waveform (optional)
            sample_rate: Audio sample rate in Hz

        Returns:
            Dict with keys:
                - "anomaly_score": Float (higher = more likely adversarial)
                - "detected": Bool (whether flagged as adversarial)
                - "explanation": String explaining the detection signal

        TODO: Implement detection logic for audio models.
        """
        raise NotImplementedError("Audio detection not yet implemented")
