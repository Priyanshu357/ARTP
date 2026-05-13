"""Noise injection attack for audio models."""

from typing import Any, Dict
import numpy as np
from .base_attack import BaseAttack


class NoiseInjectionAttack(BaseAttack):
    """Simple adversarial noise injection for audio models.

    Adds random noise to audio waveforms while maintaining a target SNR.
    This is a basic implementation similar to FGSM for audio.
    """

    def __init__(self, epsilon: float = 0.01, target_snr_db: float = 20.0):
        """Initialize noise injection attack.

        Args:
            epsilon: Perturbation magnitude (0.0 to 1.0)
            target_snr_db: Target signal-to-noise ratio in dB (higher = less noticeable)
        """
        self.epsilon = epsilon
        self.target_snr_db = target_snr_db

    def generate(
        self, model: Any, audio: np.ndarray, labels: Any, sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Generate adversarial audio by injecting noise.

        Args:
            model: Audio classification model
            audio: Audio batch [batch_size, samples] or [samples]
            labels: Ground truth labels
            sample_rate: Audio sample rate in Hz

        Returns:
            Dictionary with attack results and per-sample diagnostics
        """
        # Ensure audio is 2D [batch_size, samples]
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        batch_size = audio.shape[0]

        # Get original predictions
        original_predictions = []
        try:
            predictions = model([audio[i] for i in range(batch_size)])
            original_predictions = predictions
        except Exception as e:
            print(f"[WARNING] Model prediction failed: {e}")
            original_predictions = [{"label": "error", "score": 0.0}] * batch_size

        # Generate adversarial audio
        adversarial_audio = []
        adversarial_predictions = []
        successful_attacks = 0

        for i in range(batch_size):
            original_sample = audio[i]

            # Add random noise scaled by epsilon
            noise = np.random.randn(len(original_sample)) * self.epsilon

            # Scale noise to achieve target SNR
            signal_power = np.mean(original_sample ** 2)
            noise_power = np.mean(noise ** 2)

            if noise_power > 1e-10 and signal_power > 1e-10:
                # Calculate required noise scale for target SNR
                target_snr_linear = 10 ** (self.target_snr_db / 10)
                noise_scale = np.sqrt(signal_power / (noise_power * target_snr_linear))
                noise = noise * noise_scale

            # Add noise to audio
            perturbed = original_sample + noise

            # Clip to valid range [-1, 1] for audio
            perturbed = np.clip(perturbed, -1.0, 1.0)

            adversarial_audio.append(perturbed)

            # Get prediction for adversarial audio
            try:
                adv_pred = model([perturbed])[0]
                adversarial_predictions.append(adv_pred)

                # Check if attack was successful (prediction changed)
                if adv_pred["label"] != original_predictions[i]["label"]:
                    successful_attacks += 1

            except Exception as e:
                print(f"[WARNING] Adversarial prediction failed: {e}")
                adversarial_predictions.append({"label": "error", "score": 0.0})

        # Calculate attack success rate
        attack_success_rate = successful_attacks / batch_size if batch_size > 0 else 0.0

        # Compute average perturbation metrics
        avg_l2_norm = np.mean([
            np.linalg.norm(adversarial_audio[i] - audio[i])
            for i in range(batch_size)
        ])

        avg_snr = self._compute_avg_snr(audio, np.array(adversarial_audio))

        return {
            "attack": "NoiseInjectionAttack",
            "attack_success_rate": attack_success_rate,
            "adversarial_audio": np.array(adversarial_audio),
            "original_predictions": original_predictions,
            "adversarial_predictions": adversarial_predictions,
            "perturbation_metrics": {
                "avg_l2_norm": float(avg_l2_norm),
                "avg_snr_db": float(avg_snr),
                "epsilon": self.epsilon,
                "target_snr_db": self.target_snr_db
            },
            "successful_attacks": successful_attacks,
            "total_samples": batch_size
        }

    def _compute_avg_snr(self, original: np.ndarray, adversarial: np.ndarray) -> float:
        """Compute average SNR across batch.

        Args:
            original: Original audio [batch_size, samples]
            adversarial: Adversarial audio [batch_size, samples]

        Returns:
            Average SNR in dB
        """
        snr_values = []

        for i in range(original.shape[0]):
            signal_power = np.mean(original[i] ** 2)
            noise = adversarial[i] - original[i]
            noise_power = np.mean(noise ** 2)

            if noise_power > 1e-10 and signal_power > 1e-10:
                snr_db = 10 * np.log10(signal_power / noise_power)
                snr_values.append(snr_db)
            else:
                snr_values.append(float('inf'))

        return float(np.mean(snr_values)) if snr_values else float('inf')
