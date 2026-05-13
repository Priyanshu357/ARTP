"""
Audio Attack Runner with Per-Sample Diagnostics

This module provides AudioAttackRunner, which executes adversarial audio attacks
and collects comprehensive per-sample diagnostics including SNR, spectral distance,
energy metrics, and prediction changes.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class AudioAttackRunner:
    """
    Run adversarial audio attacks and collect per-sample diagnostics.

    Similar to NLPAttackRunner but for audio waveforms, tracking audio-specific
    metrics like SNR (Signal-to-Noise Ratio), spectral distance, and energy changes.

    Args:
        model: Audio classification model (callable that takes np.ndarray)
        attacks: List of audio attack instances
        sample_rate: Audio sample rate in Hz (default: 16000)
    """

    def __init__(self, model, attacks: List, sample_rate: int = 16000):
        """Initialize audio attack runner."""
        self.model = model
        self.attacks = attacks
        self.sample_rate = sample_rate

    def run_batch(
        self,
        audio_batch: np.ndarray,  # Shape: [batch_size, samples]
        labels: List[int],
        audio_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run attacks on an audio batch and collect per-sample diagnostics.

        Args:
            audio_batch: Batch of audio waveforms [batch_size, samples]
            labels: Ground truth labels
            audio_ids: Optional identifiers for audio samples

        Returns:
            List of attack results with per_sample_diagnostics field
        """
        batch_results = []

        if audio_ids is None:
            audio_ids = [f"sample_{i}" for i in range(len(audio_batch))]

        for attack in self.attacks:
            attack_failed = False

            try:
                # Run attack
                attack_output = attack.generate(
                    self.model,
                    audio_batch,
                    labels,
                    sample_rate=self.sample_rate
                )
            except NotImplementedError:
                # Handle template attacks that aren't implemented yet
                print(f"[WARNING] {attack.__class__.__name__} not implemented")
                print(f"[INFO] Collecting baseline predictions for diagnostic analysis...")

                # Create minimal attack_output with baseline predictions (no perturbations)
                # This allows universal diagnostics (class bias, confidence) to still work
                original_preds = [self._get_prediction(audio_batch[i]) for i in range(len(audio_batch))]

                attack_output = {
                    "attack": attack.__class__.__name__,
                    "attack_success_rate": 0.0,
                    "adversarial_audio": audio_batch,  # Keep as numpy array internally
                    "original_predictions": original_preds,
                    "adversarial_predictions": original_preds,  # Same as original
                    "perturbation_metrics": {
                        "avg_l2_norm": 0.0,
                        "avg_snr_db": float('inf'),
                        "attack_not_implemented": True
                    },
                    "successful_attacks": 0,
                    "total_samples": len(audio_batch)
                }
                attack_failed = True

            # Collect per-sample diagnostics
            per_sample_diagnostics = []

            for i in range(len(audio_batch)):
                original_audio = audio_batch[i]
                original_label = labels[i]

                # Get predictions (from attack results if failed, or fresh call if succeeded)
                if attack_failed:
                    # Use pre-computed predictions when attack didn't run
                    original_pred = attack_output["original_predictions"][i]
                    adv_pred = original_pred  # No perturbation
                    adv_audio = original_audio
                else:
                    # Get fresh predictions when attack ran
                    original_pred = self._get_prediction(original_audio)

                    # Get adversarial audio
                    adv_audio_batch = attack_output.get('adversarial_audio', audio_batch)
                    adv_audio = adv_audio_batch[i] if len(adv_audio_batch) > i else original_audio

                    # Get adversarial prediction
                    adv_pred = self._get_prediction(adv_audio)

                # Check if perturbation was attempted
                perturbation_attempted = not attack_failed and not np.array_equal(original_audio, adv_audio)

                # Compute audio-specific metrics
                if perturbation_attempted:
                    snr_db = self._compute_snr(original_audio, adv_audio)
                    l2_norm = np.linalg.norm(adv_audio - original_audio)
                    linf_norm = np.max(np.abs(adv_audio - original_audio))

                    # Spectral distance (try-except for optional librosa dependency)
                    try:
                        spectral_distance = self._compute_spectral_distance(
                            original_audio, adv_audio
                        )
                    except Exception:
                        spectral_distance = 0.0

                    energy_change_db = self._compute_energy_change(
                        original_audio, adv_audio
                    )
                else:
                    snr_db = float('inf')  # No perturbation = infinite SNR
                    l2_norm = 0.0
                    linf_norm = 0.0
                    spectral_distance = 0.0
                    energy_change_db = 0.0

                # Build diagnostic record
                per_sample_diagnostics.append({
                    "audio_id": audio_ids[i],
                    "original_label": original_label,
                    "original_pred": original_pred,
                    "adversarial_pred": adv_pred,
                    "perturbation_attempted": perturbation_attempted,
                    "prediction_flipped": original_pred['label'] != adv_pred['label'],
                    "confidence_change": adv_pred['score'] - original_pred['score'],

                    # Audio-specific metrics
                    "snr_db": snr_db,
                    "l2_norm": l2_norm,
                    "linf_norm": linf_norm,
                    "spectral_distance": spectral_distance,
                    "energy_change_db": energy_change_db,
                    "is_silent": self._is_silent(original_audio),
                    "rms_energy": self._compute_rms(original_audio),
                })

            # Store results with diagnostics (always, even for failed attacks)
            batch_results.append({
                "attack": attack.__class__.__name__,
                "result": attack_output,
                "per_sample_diagnostics": per_sample_diagnostics
            })

        return batch_results

    def _get_prediction(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Get model prediction for a single audio sample.

        Returns:
            Dict with 'label' (str) and 'score' (float)
        """
        try:
            # Model should return list of predictions
            predictions = self.model([audio])
            if predictions and len(predictions) > 0:
                pred = predictions[0]
                if isinstance(pred, dict) and 'label' in pred and 'score' in pred:
                    return pred
                else:
                    # Fallback format
                    return {"label": str(pred), "score": 1.0}
            else:
                return {"label": "unknown", "score": 0.0}
        except Exception as e:
            print(f"[WARNING] Model prediction failed: {e}")
            return {"label": "error", "score": 0.0}

    def _compute_snr(self, original: np.ndarray, adversarial: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio in dB.

        SNR = 10 * log10(signal_power / noise_power)

        Returns:
            SNR in decibels (dB)
        """
        signal_power = np.mean(original ** 2)
        noise = adversarial - original
        noise_power = np.mean(noise ** 2)

        if noise_power < 1e-10:
            return float('inf')

        if signal_power < 1e-10:
            return float('-inf')

        return 10 * np.log10(signal_power / noise_power)

    def _compute_spectral_distance(
        self,
        original: np.ndarray,
        adversarial: np.ndarray
    ) -> float:
        """
        Compute spectral distance using STFT (Short-Time Fourier Transform).

        Returns:
            Normalized L2 distance between magnitude spectrograms
        """
        try:
            import librosa

            # Compute magnitude spectrograms
            orig_spec = np.abs(librosa.stft(original))
            adv_spec = np.abs(librosa.stft(adversarial))

            # L2 norm of difference, normalized by size
            distance = np.linalg.norm(orig_spec - adv_spec) / orig_spec.size
            return float(distance)
        except ImportError:
            # librosa not installed
            return 0.0
        except Exception as e:
            print(f"[WARNING] Spectral distance computation failed: {e}")
            return 0.0

    def _compute_energy_change(
        self,
        original: np.ndarray,
        adversarial: np.ndarray
    ) -> float:
        """
        Compute energy change in dB.

        Energy change = 10 * log10(adv_energy / orig_energy)

        Returns:
            Energy change in decibels (dB)
        """
        orig_energy = np.mean(original ** 2)
        adv_energy = np.mean(adversarial ** 2)

        if orig_energy < 1e-10:
            return 0.0

        return 10 * np.log10(adv_energy / orig_energy)

    def _is_silent(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Check if audio is effectively silent (low RMS energy).

        Args:
            audio: Audio waveform
            threshold: RMS energy threshold for silence (default: 0.01)

        Returns:
            True if audio is silent, False otherwise
        """
        rms = self._compute_rms(audio)
        return rms < threshold

    def _compute_rms(self, audio: np.ndarray) -> float:
        """
        Compute RMS (Root Mean Square) energy.

        Returns:
            RMS energy value
        """
        return float(np.sqrt(np.mean(audio ** 2)))

    def run_dataset(self, dataset) -> List[Dict[str, Any]]:
        """
        Run attacks on an entire dataset.

        Args:
            dataset: Iterable of (audio_batch, labels) tuples

        Returns:
            List of all attack results across batches
        """
        all_results = []

        for batch_audio, batch_labels in dataset:
            batch_results = self.run_batch(batch_audio, batch_labels)
            all_results.extend(batch_results)

        return all_results
