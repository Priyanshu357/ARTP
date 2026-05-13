"""Psychoacoustic Hiding Attack for audio models.

This attack exploits the human auditory system's frequency masking property
(ISO 226 simplified model) to hide adversarial perturbations below the
masking threshold. The perturbations are inaudible to humans but remain
effective against machine learning classifiers.

Key Concept:
    The human ear cannot perceive a quieter sound (maskee) when a louder
    sound (masker) is present in the same or adjacent frequency band.
    This attack computes per-frequency masking thresholds and injects
    perturbation energy only where it will be masked (inaudible).

Reference:
    Psychoacoustic model based on ISO 226:2003 and the approach from:
    Schoenherr et al., "Spectral Subtraction for Robust Speech Recognition" (2019)
    Qin et al., "Imperceptible, Robust and Targeted Adversarial Examples for
    Automatic Speech Recognition" (ICML 2019)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from .base_attack import BaseAttack


class PsychoacousticAttack(BaseAttack):
    """Adversarial audio attack using psychoacoustic frequency masking.

    Exploits auditory masking to inject perturbations that are:
    - Imperceptible to human listeners
    - Effective at fooling audio classification models
    - Constrained by a target masking margin (dB)

    The attack computes per-band masking thresholds using a simplified
    ISO 226 psychoacoustic model, then injects noise shaped to stay below
    those thresholds.
    """

    # Frequency bands for critical band analysis (Bark scale boundaries in Hz)
    BARK_BANDS = [
        (0, 100), (100, 200), (200, 300), (300, 400), (400, 510),
        (510, 630), (630, 770), (770, 920), (920, 1080), (1080, 1270),
        (1270, 1480), (1480, 1720), (1720, 2000), (2000, 2320), (2320, 2700),
        (2700, 3150), (3150, 3700), (3700, 4400), (4400, 5300), (5300, 6400),
        (6400, 7700), (7700, 9500), (9500, 12000), (12000, 15500), (15500, 22050),
    ]

    def __init__(
        self,
        masking_margin_db: float = 6.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_iterations: int = 50,
        learning_rate: float = 0.005,
        epsilon: float = 0.05,
    ):
        """Initialize psychoacoustic attack.

        Args:
            masking_margin_db: How many dB below the masking threshold the
                perturbation is constrained (higher = more imperceptible)
            n_fft: FFT window size for frequency analysis
            hop_length: Hop length for STFT
            num_iterations: Number of iterative refinement steps
            learning_rate: Gradient step size for refinement
            epsilon: Hard L-inf clipping bound as a safety net
        """
        self.masking_margin_db = masking_margin_db
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def generate(
        self, model: Any, audio: np.ndarray, labels: Any, sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Generate psychoacoustically-hidden adversarial audio.

        Args:
            model: Audio classification model (callable)
            audio: Audio batch [batch_size, samples] or [samples]
            labels: Ground truth labels
            sample_rate: Audio sample rate in Hz

        Returns:
            Dictionary with attack results and psychoacoustic metrics
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        batch_size = audio.shape[0]

        # Get original predictions
        original_predictions = self._batch_predict(model, audio)

        adversarial_audio = []
        adversarial_predictions = []
        successful_attacks = 0
        per_sample_metrics = []

        for i in range(batch_size):
            original_sample = audio[i].astype(np.float64)
            original_label = original_predictions[i]["label"]

            # Compute psychoacoustic masking threshold
            masking_threshold = self._compute_masking_threshold(
                original_sample, sample_rate
            )

            # Generate shaped perturbation within masking budget
            best_adv, attack_info = self._attack_with_masking(
                model,
                original_sample,
                original_label,
                masking_threshold,
                sample_rate,
            )

            adversarial_audio.append(best_adv)
            adv_pred = self._single_predict(model, best_adv)
            adversarial_predictions.append(adv_pred)

            success = adv_pred["label"] != original_label
            if success:
                successful_attacks += 1

            perturbation = best_adv - original_sample
            l2_norm = float(np.linalg.norm(perturbation))
            linf_norm = float(np.max(np.abs(perturbation)))
            snr_db = self._compute_snr(original_sample, perturbation)
            avg_masking_headroom = float(attack_info.get("avg_masking_headroom_db", 0.0))

            per_sample_metrics.append({
                "success": success,
                "l2_norm": l2_norm,
                "linf_norm": linf_norm,
                "snr_db": snr_db,
                "masking_headroom_db": avg_masking_headroom,
                "original_label": original_label,
                "adversarial_label": adv_pred["label"],
                "imperceptibility_score": self._compute_imperceptibility(
                    original_sample, best_adv, sample_rate
                ),
            })

        attack_success_rate = successful_attacks / batch_size if batch_size > 0 else 0.0
        adversarial_array = np.array(adversarial_audio)

        avg_snr = float(np.mean([
            m["snr_db"] for m in per_sample_metrics if np.isfinite(m["snr_db"])
        ] or [float("inf")]))
        avg_imperceptibility = float(np.mean([
            m["imperceptibility_score"] for m in per_sample_metrics
        ]))

        return {
            "attack": "PsychoacousticAttack",
            "attack_success_rate": attack_success_rate,
            "adversarial_audio": adversarial_array,
            "original_predictions": original_predictions,
            "adversarial_predictions": adversarial_predictions,
            "perturbation_metrics": {
                "avg_l2_norm": float(np.mean([m["l2_norm"] for m in per_sample_metrics])),
                "avg_snr_db": avg_snr,
                "avg_imperceptibility_score": avg_imperceptibility,
                "masking_margin_db": self.masking_margin_db,
                "epsilon": self.epsilon,
            },
            "successful_attacks": successful_attacks,
            "total_samples": batch_size,
            "per_sample_metrics": per_sample_metrics,
        }

    def _compute_masking_threshold(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Compute per-frequency masking threshold using simplified ISO 226 model.

        The threshold is computed using:
        1. Absolute threshold of hearing (ATH) — minimum audible level per freq
        2. Simultaneous masking — loud tones mask nearby quieter tones
        3. Spreading function across Bark critical bands

        Args:
            audio: Audio waveform [samples]
            sample_rate: Sample rate in Hz

        Returns:
            Masking threshold in linear amplitude, same length as FFT magnitude
        """
        n = min(len(audio), self.n_fft)
        windowed = audio[:n] * np.hanning(n)
        spectrum = np.fft.rfft(windowed, n=self.n_fft)
        magnitude = np.abs(spectrum)
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / sample_rate)
        n_bins = len(freqs)

        # 1. Absolute Threshold of Hearing (ATH) in dB SPL (simplified)
        # Based on ISO 226, the ATH curve minimum is around 3-4 kHz
        ath_db = self._absolute_threshold_of_hearing(freqs)
        # Convert dB SPL to linear power (relative to signal scale)
        # We normalize so that 0 dB ATH ≈ our quiet floor
        ath_linear = 10 ** (ath_db / 20.0) * 1e-4  # scale to audio amplitude range

        # 2. Simultaneous masking via spreading function
        # For each frequency bin, compute the masking contribution from all other bins
        masking_threshold = np.copy(ath_linear)

        # Group bins into Bark bands and compute masking per band
        for f_low, f_high in self.BARK_BANDS:
            band_mask = (freqs >= f_low) & (freqs < f_high)
            if not np.any(band_mask):
                continue

            # Energy of this band
            band_energy = np.mean(magnitude[band_mask] ** 2)
            if band_energy < 1e-20:
                continue

            # Band masking level (in dB below band energy)
            masker_db = 10 * np.log10(band_energy + 1e-20)
            # Masking threshold is typically ~26 dB below the masker
            masking_level_db = masker_db - 26.0

            # Spread to ±1 Bark (adjacent bands get a reduced masking effect)
            spread_linear = 10 ** (masking_level_db / 20.0)
            masking_threshold[band_mask] = np.maximum(
                masking_threshold[band_mask], spread_linear
            )

        # Apply masking margin: perturbation must stay this many dB below threshold
        masking_threshold = masking_threshold * 10 ** (-self.masking_margin_db / 20.0)

        return masking_threshold  # shape: [n_fft // 2 + 1]

    def _absolute_threshold_of_hearing(self, freqs: np.ndarray) -> np.ndarray:
        """Compute ATH curve in dB SPL (ISO 226 simplified polynomial).

        The formula approximates the equal-loudness contour at 0 phons,
        which represents the threshold of human hearing.

        Args:
            freqs: Frequency bins in Hz

        Returns:
            ATH values in dB SPL for each frequency bin
        """
        f = np.maximum(freqs, 20.0)  # avoid log(0)
        f_khz = f / 1000.0

        # Terhardt (1979) approximation of the ATH
        ath = (
            3.64 * f_khz ** (-0.8)
            - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2)
            + 1e-3 * f_khz ** 4
        )
        return ath

    def _attack_with_masking(
        self,
        model: Any,
        original: np.ndarray,
        original_label: str,
        masking_threshold: np.ndarray,
        sample_rate: int,
    ) -> tuple:
        """Craft perturbation constrained to psychoacoustic masking threshold.

        Strategy:
        1. Start with threshold-shaped noise in frequency domain
        2. Iteratively refine using NES gradient estimation
        3. At each step, project back to masking constraint in freq domain

        Args:
            model: Target model
            original: Original audio waveform
            original_label: Ground truth label
            masking_threshold: Per-frequency masking threshold (linear amplitude)
            sample_rate: Audio sample rate

        Returns:
            Tuple of (adversarial_audio, info_dict)
        """
        n = len(original)

        # Initialize perturbation as shaped random noise in freq domain
        # Start at 50% of masking threshold amplitude
        random_phase = np.random.uniform(-np.pi, np.pi, size=len(masking_threshold))
        n_fft_half = len(masking_threshold)

        # Build frequency-domain perturbation
        freq_delta = masking_threshold * 0.5 * np.exp(1j * random_phase)
        # Inverse FFT to time domain (take real part only)
        time_delta = np.fft.irfft(freq_delta, n=n)[:n]

        best_adv = np.clip(original + time_delta, -1.0, 1.0)
        best_l2 = np.linalg.norm(time_delta)
        found_success = False
        iterations_used = 0

        # Check if initial noise already flips prediction
        pred = self._single_predict(model, best_adv)
        if pred["label"] != original_label:
            found_success = True

        for iteration in range(self.num_iterations):
            iterations_used += 1

            # Estimate gradient via NES (Natural Evolution Strategy)
            grad_freq = self._estimate_freq_gradient(
                model, original, time_delta, original_label, masking_threshold
            )

            # Gradient step in frequency domain
            # Step size scaled by masking threshold (adaptive per-frequency rate)
            time_delta_update = self.learning_rate * grad_freq

            # Project into time domain
            time_delta = time_delta - time_delta_update[:n]

            # Project back to masking constraint in frequency domain
            delta_spectrum = np.fft.rfft(time_delta, n=self.n_fft)
            delta_mag = np.abs(delta_spectrum)
            # Clip frequency-domain amplitude to masking threshold
            scale = np.where(
                delta_mag > masking_threshold,
                masking_threshold / (delta_mag + 1e-20),
                1.0
            )
            delta_spectrum_constrained = delta_spectrum * scale
            time_delta = np.fft.irfft(delta_spectrum_constrained, n=self.n_fft)[:n]

            # Hard L-inf clip as safety net
            time_delta = np.clip(time_delta, -self.epsilon, self.epsilon)

            # Ensure audio stays in valid range
            adv = np.clip(original + time_delta, -1.0, 1.0)
            time_delta = adv - original  # re-derive after clipping

            # Check success
            pred = self._single_predict(model, adv)
            current_l2 = np.linalg.norm(time_delta)

            if pred["label"] != original_label:
                if not found_success or current_l2 < best_l2:
                    best_adv = adv.copy()
                    best_l2 = current_l2
                    found_success = True

        # Compute average masking headroom
        final_delta_spectrum = np.fft.rfft(best_adv - original, n=self.n_fft)
        final_mag = np.abs(final_delta_spectrum)
        with np.errstate(divide="ignore", invalid="ignore"):
            headroom_db = np.where(
                final_mag > 1e-20,
                20 * np.log10(masking_threshold / (final_mag + 1e-20)),
                60.0,
            )
        avg_headroom_db = float(np.mean(headroom_db[headroom_db < 60.0]) if np.any(headroom_db < 60.0) else 60.0)

        return best_adv, {
            "iterations_used": iterations_used,
            "found_success": found_success,
            "avg_masking_headroom_db": avg_headroom_db,
        }

    def _estimate_freq_gradient(
        self,
        model: Any,
        original: np.ndarray,
        delta: np.ndarray,
        original_label: str,
        masking_threshold: np.ndarray,
        num_samples: int = 15,
        sigma: float = 0.001,
    ) -> np.ndarray:
        """Estimate gradient in time domain using NES with frequency-shaped noise.

        Args:
            model: Target model
            original: Original audio
            delta: Current perturbation (time domain)
            original_label: Label we want to flip
            masking_threshold: Per-frequency masking constraint
            num_samples: NES random direction samples
            sigma: Noise scale for gradient estimation

        Returns:
            Estimated gradient in time domain
        """
        grad = np.zeros_like(delta)
        n = len(original)

        for _ in range(num_samples):
            # Sample random direction shaped by masking threshold
            random_phase = np.random.uniform(-np.pi, np.pi, size=len(masking_threshold))
            noise_freq = sigma * masking_threshold * np.exp(1j * random_phase)
            noise_time = np.fft.irfft(noise_freq, n=self.n_fft)[:n]

            pos_adv = np.clip(original + delta + noise_time, -1.0, 1.0)
            neg_adv = np.clip(original + delta - noise_time, -1.0, 1.0)

            pos_pred = self._single_predict(model, pos_adv)
            neg_pred = self._single_predict(model, neg_adv)

            # Loss: high when model predicts original label (attack failed)
            pos_loss = pos_pred["score"] if pos_pred["label"] == original_label else -pos_pred["score"]
            neg_loss = neg_pred["score"] if neg_pred["label"] == original_label else -neg_pred["score"]

            # NES gradient estimate
            grad += (pos_loss - neg_loss) * noise_time / (2 * sigma)

        grad /= num_samples
        return grad

    def _compute_imperceptibility(
        self, original: np.ndarray, adversarial: np.ndarray, sample_rate: int
    ) -> float:
        """Compute an imperceptibility score (0-1, higher = more imperceptible).

        Based on the ratio of perturbation energy to the masking threshold energy.
        A score of 1.0 means the perturbation is fully below the masking threshold.

        Args:
            original: Original audio
            adversarial: Adversarial audio
            sample_rate: Sample rate in Hz

        Returns:
            Imperceptibility score between 0 and 1
        """
        perturbation = adversarial - original
        masking_threshold = self._compute_masking_threshold(original, sample_rate)

        n = min(len(perturbation), self.n_fft)
        pert_spectrum = np.abs(np.fft.rfft(perturbation[:n], n=self.n_fft))

        # Compute per-bin masking margin
        margins = masking_threshold / (pert_spectrum + 1e-20)
        # Score: fraction of bins where perturbation is below threshold
        imperceptible_fraction = float(np.mean(margins >= 1.0))
        return imperceptible_fraction

    def _batch_predict(
        self, model: Any, audio_batch: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Get model predictions for a batch of audio."""
        try:
            return model([audio_batch[i] for i in range(audio_batch.shape[0])])
        except Exception as e:
            print(f"[WARNING] Batch prediction failed: {e}")
            return [{"label": "error", "score": 0.0}] * audio_batch.shape[0]

    def _single_predict(self, model: Any, audio: np.ndarray) -> Dict[str, Any]:
        """Get model prediction for a single audio sample."""
        try:
            preds = model([audio])
            if preds and len(preds) > 0:
                p = preds[0]
                return p if isinstance(p, dict) else {"label": str(p), "score": 1.0}
            return {"label": "unknown", "score": 0.0}
        except Exception:
            return {"label": "error", "score": 0.0}

    @staticmethod
    def _compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
        """Compute Signal-to-Noise Ratio in dB."""
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power < 1e-10:
            return float("inf")
        if signal_power < 1e-10:
            return float("-inf")
        return float(10 * np.log10(signal_power / noise_power))
