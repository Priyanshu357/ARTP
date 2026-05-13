"""Base class for audio adversarial attacks."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseAttack(ABC):
    """Abstract base class for audio adversarial attacks.

    All audio attacks work with raw waveforms (numpy arrays) and must
    preserve audio quality constraints while fooling the target model.
    """

    @abstractmethod
    def generate(
        self, model: Any, audio: np.ndarray, labels: Any, sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Generate adversarial audio examples.

        Args:
            model: Target audio model (callable)
            audio: Audio waveform(s) [batch, samples] or [samples]
            labels: Ground-truth labels
            sample_rate: Audio sample rate in Hz

        Returns:
            Dict containing:
                - adversarial_audio: Perturbed waveforms as np.ndarray
                - original_predictions: Model outputs on clean audio
                - adversarial_predictions: Model outputs on adversarial audio
                - attack_success_rate: Float 0–1
                - perturbation_metrics: Dict with SNR, L2 norm, etc.
        """
        ...
