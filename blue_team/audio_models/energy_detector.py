"""Energy-based detector for adversarial audio.

Detects adversarial perturbations by analyzing energy distribution changes
between original and adversarial audio waveforms. Adversarial perturbations
often alter the energy profile in ways that are statistically detectable.
"""

from typing import Any, Dict, Optional

import numpy as np

from .base_detector import BaseDetector


class EnergyDetector(BaseDetector):
    """Detect adversarial audio using energy and SNR analysis.

    Detection signals:
    1. RMS energy change between original and adversarial
    2. Frame-level energy variance change
    3. Signal-to-Noise Ratio (SNR) of the perturbation
    4. Peak-to-average power ratio (crest factor) change

    Adversarial perturbations often introduce measurable energy shifts.
    """

    def __init__(
        self,
        energy_threshold: float = 0.05,
        snr_threshold: float = 30.0,
        frame_size: int = 512,
    ):
        """Initialize energy detector.

        Args:
            energy_threshold: Max allowed relative energy change (0-1)
            snr_threshold: SNR below this (in dB) flags as adversarial
            frame_size: Number of samples per analysis frame
        """
        self.energy_threshold = energy_threshold
        self.snr_threshold = snr_threshold
        self.frame_size = frame_size

    def detect(
        self,
        original_output: Any,
        adversarial_output: Any,
        original_audio: Optional[np.ndarray] = None,
        adversarial_audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Detect adversarial audio based on energy analysis.

        Args:
            original_output: Model output on clean audio
            adversarial_output: Model output on adversarial audio
            original_audio: Clean audio waveform
            adversarial_audio: Adversarial audio waveform
            sample_rate: Audio sample rate in Hz

        Returns:
            Dict with anomaly_score, detected flag, and explanation
        """
        if original_audio is None or adversarial_audio is None:
            return {
                "anomaly_score": 0.0,
                "detected": False,
                "is_attack": False,
                "explanation": "No audio waveforms provided for energy analysis",
            }

        scores = []
        reasons = []

        # 1. RMS energy change
        orig_rms = np.sqrt(np.mean(original_audio ** 2))
        adv_rms = np.sqrt(np.mean(adversarial_audio ** 2))

        if orig_rms > 1e-10:
            rms_change = abs(adv_rms - orig_rms) / orig_rms
        else:
            rms_change = 0.0

        if rms_change > self.energy_threshold:
            scores.append(min(rms_change / self.energy_threshold, 1.0))
            reasons.append(f"RMS energy changed by {rms_change:.1%}")

        # 2. SNR of perturbation
        noise = adversarial_audio - original_audio
        signal_power = np.mean(original_audio ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power > 1e-10 and signal_power > 1e-10:
            snr_db = 10 * np.log10(signal_power / noise_power)
            if snr_db < self.snr_threshold:
                snr_score = 1.0 - (snr_db / self.snr_threshold)
                scores.append(max(0.0, min(snr_score, 1.0)))
                reasons.append(f"Low SNR: {snr_db:.1f} dB (threshold: {self.snr_threshold} dB)")

        # 3. Frame-level energy variance change
        orig_frames = self._compute_frame_energies(original_audio)
        adv_frames = self._compute_frame_energies(adversarial_audio)

        if len(orig_frames) > 1 and len(adv_frames) > 1:
            orig_var = np.var(orig_frames)
            adv_var = np.var(adv_frames)

            if orig_var > 1e-10:
                var_change = abs(adv_var - orig_var) / orig_var
                if var_change > 0.1:
                    scores.append(min(var_change, 1.0))
                    reasons.append(f"Frame energy variance changed by {var_change:.1%}")

        # 4. Crest factor (peak-to-RMS ratio) change
        orig_crest = np.max(np.abs(original_audio)) / (orig_rms + 1e-10)
        adv_crest = np.max(np.abs(adversarial_audio)) / (adv_rms + 1e-10)
        crest_change = abs(adv_crest - orig_crest) / (orig_crest + 1e-10)

        if crest_change > 0.15:
            scores.append(min(crest_change, 1.0))
            reasons.append(f"Crest factor changed by {crest_change:.1%}")

        # Aggregate
        anomaly_score = float(np.mean(scores)) if scores else 0.0
        detected = anomaly_score > 0.5

        explanation = "; ".join(reasons) if reasons else "No energy anomalies detected"

        return {
            "anomaly_score": anomaly_score,
            "detected": detected,
            "is_attack": True,  # This was an attacked sample
            "explanation": explanation,
            "details": {
                "rms_change": float(rms_change),
                "snr_db": float(snr_db) if noise_power > 1e-10 else float("inf"),
                "crest_factor_change": float(crest_change),
            },
        }

    def _compute_frame_energies(self, audio: np.ndarray) -> np.ndarray:
        """Compute per-frame RMS energies.

        Args:
            audio: Audio waveform

        Returns:
            Array of frame energies
        """
        n_frames = len(audio) // self.frame_size
        if n_frames == 0:
            return np.array([np.sqrt(np.mean(audio ** 2))])

        frames = audio[: n_frames * self.frame_size].reshape(n_frames, self.frame_size)
        return np.sqrt(np.mean(frames ** 2, axis=1))
