"""Spectral anomaly detector for adversarial audio.

Detects adversarial perturbations by analyzing spectral (frequency domain)
differences between original and adversarial audio. Adversarial noise often
introduces high-frequency artifacts or alters the spectral envelope.
"""

from typing import Any, Dict, Optional

import numpy as np

from .base_detector import BaseDetector


class SpectralDetector(BaseDetector):
    """Detect adversarial audio using spectral analysis.

    Detection signals:
    1. Spectral centroid shift (center of mass of spectrum)
    2. Spectral rolloff change (frequency below which X% of energy lies)
    3. High-frequency energy ratio change
    4. Spectral flux (rate of spectral change)
    5. STFT magnitude difference (if available)
    """

    def __init__(
        self,
        anomaly_threshold: float = 0.1,
        n_fft: int = 2048,
        hop_length: int = 512,
        rolloff_percent: float = 0.85,
    ):
        """Initialize spectral detector.

        Args:
            anomaly_threshold: Threshold for spectral anomaly score (0-1)
            n_fft: FFT window size for spectrogram computation
            hop_length: Hop length for STFT
            rolloff_percent: Percentage for spectral rolloff calculation
        """
        self.threshold = anomaly_threshold
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.rolloff_percent = rolloff_percent

    def detect(
        self,
        original_output: Any,
        adversarial_output: Any,
        original_audio: Optional[np.ndarray] = None,
        adversarial_audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Detect adversarial audio based on spectral analysis.

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
                "explanation": "No audio waveforms provided for spectral analysis",
            }

        scores = []
        reasons = []
        details = {}

        # Compute magnitude spectra using numpy FFT (no librosa needed)
        orig_spec = np.abs(np.fft.rfft(original_audio, n=self.n_fft))
        adv_spec = np.abs(np.fft.rfft(adversarial_audio, n=self.n_fft))

        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / sample_rate)

        # 1. Spectral centroid shift
        orig_centroid = self._spectral_centroid(orig_spec, freqs)
        adv_centroid = self._spectral_centroid(adv_spec, freqs)
        nyquist = sample_rate / 2.0

        centroid_shift = abs(adv_centroid - orig_centroid) / (nyquist + 1e-10)
        details["centroid_shift_hz"] = float(abs(adv_centroid - orig_centroid))

        if centroid_shift > 0.01:
            scores.append(min(centroid_shift * 10, 1.0))
            reasons.append(
                f"Spectral centroid shifted by {abs(adv_centroid - orig_centroid):.0f} Hz"
            )

        # 2. Spectral rolloff change
        orig_rolloff = self._spectral_rolloff(orig_spec, freqs)
        adv_rolloff = self._spectral_rolloff(adv_spec, freqs)
        rolloff_change = abs(adv_rolloff - orig_rolloff) / (nyquist + 1e-10)
        details["rolloff_change_hz"] = float(abs(adv_rolloff - orig_rolloff))

        if rolloff_change > 0.02:
            scores.append(min(rolloff_change * 5, 1.0))
            reasons.append(
                f"Spectral rolloff shifted by {abs(adv_rolloff - orig_rolloff):.0f} Hz"
            )

        # 3. High-frequency energy ratio change
        hf_boundary = int(len(freqs) * 0.7)  # Top 30% of frequency range
        orig_hf_ratio = (
            np.sum(orig_spec[hf_boundary:] ** 2) / (np.sum(orig_spec ** 2) + 1e-10)
        )
        adv_hf_ratio = (
            np.sum(adv_spec[hf_boundary:] ** 2) / (np.sum(adv_spec ** 2) + 1e-10)
        )
        hf_change = abs(adv_hf_ratio - orig_hf_ratio)
        details["hf_energy_change"] = float(hf_change)

        if hf_change > 0.02:
            scores.append(min(hf_change * 10, 1.0))
            reasons.append(
                f"High-frequency energy ratio changed by {hf_change:.3f}"
            )

        # 4. Spectral distance (L2 between magnitude spectra, normalized)
        spec_distance = np.linalg.norm(adv_spec - orig_spec) / (
            np.linalg.norm(orig_spec) + 1e-10
        )
        details["spectral_distance"] = float(spec_distance)

        if spec_distance > 0.05:
            scores.append(min(spec_distance * 5, 1.0))
            reasons.append(f"Spectral distance: {spec_distance:.4f}")

        # Aggregate score
        anomaly_score = float(np.mean(scores)) if scores else 0.0
        detected = anomaly_score > 0.5

        explanation = (
            "; ".join(reasons) if reasons else "No spectral anomalies detected"
        )

        return {
            "anomaly_score": anomaly_score,
            "detected": detected,
            "is_attack": True,
            "explanation": explanation,
            "details": details,
        }

    def _spectral_centroid(
        self, magnitude: np.ndarray, freqs: np.ndarray
    ) -> float:
        """Compute spectral centroid (center of mass of spectrum).

        Args:
            magnitude: Magnitude spectrum
            freqs: Frequency bins

        Returns:
            Spectral centroid in Hz
        """
        total_energy = np.sum(magnitude)
        if total_energy < 1e-10:
            return 0.0
        return float(np.sum(freqs * magnitude) / total_energy)

    def _spectral_rolloff(
        self, magnitude: np.ndarray, freqs: np.ndarray
    ) -> float:
        """Compute spectral rolloff frequency.

        The frequency below which `rolloff_percent` of the spectral energy lies.

        Args:
            magnitude: Magnitude spectrum
            freqs: Frequency bins

        Returns:
            Rolloff frequency in Hz
        """
        total_energy = np.sum(magnitude ** 2)
        if total_energy < 1e-10:
            return 0.0

        cumulative = np.cumsum(magnitude ** 2)
        rolloff_idx = np.searchsorted(cumulative, self.rolloff_percent * total_energy)
        rolloff_idx = min(rolloff_idx, len(freqs) - 1)
        return float(freqs[rolloff_idx])
