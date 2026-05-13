"""Environmental Robustness Testing for audio models.

Tests how well adversarial audio attacks survive real-world audio
degradations such as room reverberation and background noise.

Two attack types:
  - ReverbAttack: Convolves audio with a synthetic Room Impulse Response (RIR)
    simulating acoustic reflections in rooms of varying sizes.
  - BackgroundNoiseAttack: Overlays calibrated background noise (white noise,
    pink noise, crowd noise approximation) at a target SNR, simulating
    real-world recording conditions.

These attacks evaluate environmental robustness — a model may be fooled by
an adversarial example in ideal conditions, but does the attack survive when
the audio is played in a room, or recorded with ambient noise?
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
import numpy as np

from .base_attack import BaseAttack


# ─────────────────────────────────────────────────────────────────────────────
# Room Impulse Response Attack
# ─────────────────────────────────────────────────────────────────────────────

class ReverbAttack(BaseAttack):
    """Simulate room reverberation using a synthetic Room Impulse Response (RIR).

    Reverberation is one of the most common signal degradations in real-world
    audio. It occurs when sound reflects off room surfaces (walls, ceiling,
    floor) and arrives at the microphone with varying delays and attenuations.

    The synthetic RIR is modeled as an exponentially-decaying series of
    reflections, parameterized by:
      - Room size (small, medium, large) → controls RT60 (reverberation time)
      - Damping factor → controls how quickly reflections decay
      - Early reflections → simulate direct first-order reflections
    """

    RT60_PRESETS = {
        "small":  0.15,   # Small room (bathroom, studio booth) — 150 ms
        "medium": 0.40,   # Medium room (office, bedroom) — 400 ms
        "large":  0.80,   # Large room (lecture hall, church) — 800 ms
    }

    def __init__(
        self,
        room_size: Literal["small", "medium", "large"] = "medium",
        rt60: Optional[float] = None,
        wet_mix: float = 0.5,
        early_reflections: int = 5,
    ):
        """Initialize reverb attack.

        Args:
            room_size: Preset room size ('small', 'medium', 'large')
            rt60: Override RT60 reverberation time in seconds (overrides room_size)
            wet_mix: Mix ratio between original (dry) and reverb (wet) signal
                     0.0 = fully dry, 1.0 = fully wet (pure reverb)
            early_reflections: Number of early reflection pulses to simulate
        """
        if rt60 is not None:
            self.rt60 = rt60
        else:
            self.rt60 = self.RT60_PRESETS.get(room_size, 0.40)

        self.room_size = room_size
        self.wet_mix = wet_mix
        self.early_reflections = early_reflections

    def generate(
        self, model: Any, audio: np.ndarray, labels: Any, sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Apply room reverberation to audio and evaluate model robustness.

        Args:
            model: Audio classification model (callable)
            audio: Audio batch [batch_size, samples] or [samples]
            labels: Ground truth labels
            sample_rate: Audio sample rate in Hz

        Returns:
            Dictionary with attack results and reverberation metrics
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        batch_size = audio.shape[0]

        # Get original predictions
        original_predictions = self._batch_predict(model, audio)

        # Generate RIR for this sample rate
        rir = self._synthesize_rir(sample_rate)

        adversarial_audio = []
        adversarial_predictions = []
        successful_attacks = 0
        per_sample_metrics = []

        for i in range(batch_size):
            original_sample = audio[i].astype(np.float64)
            original_label = original_predictions[i]["label"]

            # Apply reverb via convolution
            reverbed = self._apply_reverb(original_sample, rir)
            adversarial_audio.append(reverbed)

            # Get prediction
            adv_pred = self._single_predict(model, reverbed)
            adversarial_predictions.append(adv_pred)

            success = adv_pred["label"] != original_label
            if success:
                successful_attacks += 1

            perturbation = reverbed - original_sample
            snr_db = self._compute_snr(original_sample, perturbation)

            per_sample_metrics.append({
                "success": success,
                "l2_norm": float(np.linalg.norm(perturbation)),
                "linf_norm": float(np.max(np.abs(perturbation))),
                "snr_db": snr_db,
                "rt60_s": self.rt60,
                "wet_mix": self.wet_mix,
                "original_label": original_label,
                "adversarial_label": adv_pred["label"],
            })

        asr = successful_attacks / batch_size if batch_size > 0 else 0.0
        avg_snr = float(np.mean([
            m["snr_db"] for m in per_sample_metrics if np.isfinite(m["snr_db"])
        ] or [float("inf")]))

        return {
            "attack": "ReverbAttack",
            "attack_success_rate": asr,
            "adversarial_audio": np.array(adversarial_audio),
            "original_predictions": original_predictions,
            "adversarial_predictions": adversarial_predictions,
            "perturbation_metrics": {
                "avg_l2_norm": float(np.mean([m["l2_norm"] for m in per_sample_metrics])),
                "avg_snr_db": avg_snr,
                "rt60_s": self.rt60,
                "room_size": self.room_size,
                "wet_mix": self.wet_mix,
                "rir_length_samples": len(rir),
            },
            "successful_attacks": successful_attacks,
            "total_samples": batch_size,
            "per_sample_metrics": per_sample_metrics,
        }

    def _synthesize_rir(self, sample_rate: int) -> np.ndarray:
        """Synthesize a Room Impulse Response using an exponential decay model.

        The RIR is modeled as:
            h[n] = sum_k (a_k * delta[n - tau_k]) * exp(-alpha * n / sample_rate)

        where:
            - alpha = -np.log(0.001) / rt60   (decay constant from RT60)
            - tau_k are arrival times of reflections
            - a_k are random reflection amplitudes

        Args:
            sample_rate: Audio sample rate in Hz

        Returns:
            Room Impulse Response as numpy array
        """
        # RIR duration = RT60 + a small buffer
        rir_duration = self.rt60 * 1.2
        rir_length = int(rir_duration * sample_rate)
        rir = np.zeros(rir_length)

        # Decay constant: after rt60 seconds, amplitude drops to -60 dB (0.001)
        alpha = -np.log(0.001) / self.rt60

        # Direct path (sample 0)
        rir[0] = 1.0

        # Early reflections (first 10% of RIR duration)
        early_end = max(2, int(rir_length * 0.10))
        np.random.seed(42)  # deterministic RIR for reproducibility
        for k in range(self.early_reflections):
            delay = np.random.randint(1, early_end)
            amplitude = np.random.uniform(0.3, 0.7) * np.exp(-alpha * delay / sample_rate)
            rir[delay] = amplitude

        # Late reverberation tail (exponentially decaying random noise)
        late_start = early_end
        t_tail = np.arange(late_start, rir_length) / sample_rate
        late_tail = np.random.randn(rir_length - late_start) * np.exp(-alpha * t_tail)
        rir[late_start:] = late_tail * 0.1  # scale late tail

        # Normalize RIR to unit energy
        rir_energy = np.sqrt(np.sum(rir ** 2))
        if rir_energy > 1e-10:
            rir = rir / rir_energy

        return rir

    def _apply_reverb(self, audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
        """Apply RIR to audio via linear convolution with wet/dry mixing.

        Args:
            audio: Dry audio waveform
            rir: Room Impulse Response

        Returns:
            Reverbed audio, same length as input
        """
        # Convolve audio with RIR (linear convolution)
        wet = np.convolve(audio, rir, mode="full")[: len(audio)]

        # Normalize wet signal to same RMS as dry
        dry_rms = np.sqrt(np.mean(audio ** 2))
        wet_rms = np.sqrt(np.mean(wet ** 2))
        if wet_rms > 1e-10 and dry_rms > 1e-10:
            wet = wet * (dry_rms / wet_rms)

        # Mix dry and wet
        reverbed = (1.0 - self.wet_mix) * audio + self.wet_mix * wet
        return np.clip(reverbed, -1.0, 1.0).astype(np.float32)

    def _batch_predict(self, model, audio_batch):
        try:
            return model([audio_batch[i] for i in range(audio_batch.shape[0])])
        except Exception:
            return [{"label": "error", "score": 0.0}] * audio_batch.shape[0]

    def _single_predict(self, model, audio):
        try:
            preds = model([audio])
            if preds and len(preds) > 0:
                p = preds[0]
                return p if isinstance(p, dict) else {"label": str(p), "score": 1.0}
            return {"label": "unknown", "score": 0.0}
        except Exception:
            return {"label": "error", "score": 0.0}

    @staticmethod
    def _compute_snr(signal, noise):
        sp = np.mean(signal ** 2)
        np_ = np.mean(noise ** 2)
        if np_ < 1e-10:
            return float("inf")
        if sp < 1e-10:
            return float("-inf")
        return float(10 * np.log10(sp / np_))


# ─────────────────────────────────────────────────────────────────────────────
# Background Noise Attack
# ─────────────────────────────────────────────────────────────────────────────

class BackgroundNoiseAttack(BaseAttack):
    """Overlay calibrated background noise to simulate real-world recording conditions.

    Supported noise types:
      - white:  Uniform-spectrum Gaussian noise (hardest for models)
      - pink:   1/f noise (more natural-sounding, realistic outdoor ambience)
      - crowd:  Simulated crowd babble (overlapping bandpass-filtered noise bursts)
      - traffic: Low-frequency dominant noise (simulates urban traffic)

    The noise is calibrated to a target SNR so the original signal remains
    dominant while the noise introduces realistic interference.
    """

    NOISE_TYPES = ["white", "pink", "crowd", "traffic"]

    def __init__(
        self,
        noise_type: Literal["white", "pink", "crowd", "traffic"] = "white",
        target_snr_db: float = 10.0,
        seed: int = 42,
    ):
        """Initialize background noise attack.

        Args:
            noise_type: Type of background noise ('white', 'pink', 'crowd', 'traffic')
            target_snr_db: Target Signal-to-Noise Ratio in dB
                           Lower = more noise, harder for model
                           10 dB ≈ clearly noisy, 20 dB ≈ slight background
            seed: Random seed for reproducible noise generation
        """
        if noise_type not in self.NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {self.NOISE_TYPES}")
        self.noise_type = noise_type
        self.target_snr_db = target_snr_db
        self.seed = seed

    def generate(
        self, model: Any, audio: np.ndarray, labels: Any, sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Overlay background noise on audio and evaluate model robustness.

        Args:
            model: Audio classification model (callable)
            audio: Audio batch [batch_size, samples] or [samples]
            labels: Ground truth labels
            sample_rate: Audio sample rate in Hz

        Returns:
            Dictionary with attack results and noise metrics
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        batch_size = audio.shape[0]
        n_samples = audio.shape[1]

        # Get original predictions
        original_predictions = self._batch_predict(model, audio)

        # Generate noise (same noise texture, different random seed per sample)
        rng = np.random.RandomState(self.seed)

        adversarial_audio = []
        adversarial_predictions = []
        successful_attacks = 0
        per_sample_metrics = []

        for i in range(batch_size):
            original_sample = audio[i].astype(np.float64)
            original_label = original_predictions[i]["label"]

            # Generate noise of same length as audio
            noise = self._generate_noise(n_samples, sample_rate, rng)

            # Scale noise to target SNR
            signal_power = np.mean(original_sample ** 2)
            noise_power = np.mean(noise ** 2)

            if noise_power > 1e-10 and signal_power > 1e-10:
                target_noise_power = signal_power / (10 ** (self.target_snr_db / 10.0))
                noise_scale = np.sqrt(target_noise_power / noise_power)
                noise = noise * noise_scale

            noisy = np.clip(original_sample + noise, -1.0, 1.0)
            adversarial_audio.append(noisy)

            # Get prediction
            adv_pred = self._single_predict(model, noisy)
            adversarial_predictions.append(adv_pred)

            success = adv_pred["label"] != original_label
            if success:
                successful_attacks += 1

            perturbation = noisy - original_sample
            actual_snr = self._compute_snr(original_sample, perturbation)

            per_sample_metrics.append({
                "success": success,
                "l2_norm": float(np.linalg.norm(perturbation)),
                "linf_norm": float(np.max(np.abs(perturbation))),
                "snr_db": actual_snr,
                "target_snr_db": self.target_snr_db,
                "noise_type": self.noise_type,
                "original_label": original_label,
                "adversarial_label": adv_pred["label"],
            })

        asr = successful_attacks / batch_size if batch_size > 0 else 0.0
        avg_snr = float(np.mean([
            m["snr_db"] for m in per_sample_metrics if np.isfinite(m["snr_db"])
        ] or [float("inf")]))

        return {
            "attack": "BackgroundNoiseAttack",
            "attack_success_rate": asr,
            "adversarial_audio": np.array(adversarial_audio),
            "original_predictions": original_predictions,
            "adversarial_predictions": adversarial_predictions,
            "perturbation_metrics": {
                "avg_l2_norm": float(np.mean([m["l2_norm"] for m in per_sample_metrics])),
                "avg_snr_db": avg_snr,
                "target_snr_db": self.target_snr_db,
                "noise_type": self.noise_type,
            },
            "successful_attacks": successful_attacks,
            "total_samples": batch_size,
            "per_sample_metrics": per_sample_metrics,
        }

    def _generate_noise(
        self, n_samples: int, sample_rate: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Generate noise of the configured type.

        Args:
            n_samples: Number of samples to generate
            sample_rate: Audio sample rate in Hz
            rng: RandomState for reproducibility

        Returns:
            Noise waveform [n_samples]
        """
        if self.noise_type == "white":
            return rng.randn(n_samples).astype(np.float64)

        elif self.noise_type == "pink":
            return self._generate_pink_noise(n_samples, rng)

        elif self.noise_type == "crowd":
            return self._generate_crowd_noise(n_samples, sample_rate, rng)

        elif self.noise_type == "traffic":
            return self._generate_traffic_noise(n_samples, sample_rate, rng)

        return rng.randn(n_samples).astype(np.float64)

    def _generate_pink_noise(
        self, n_samples: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Generate pink (1/f) noise via FFT shaping.

        Pink noise has equal energy per octave, making it sound more
        natural than white noise and a better simulation of environmental
        background sounds.

        Args:
            n_samples: Number of samples
            rng: Random state

        Returns:
            Pink noise waveform
        """
        white = rng.randn(n_samples)
        freq = np.fft.rfftfreq(n_samples)
        # 1/f shaping: divide by sqrt(f) in frequency domain
        freq[0] = 1.0  # avoid division by zero at DC
        pink_spectrum = np.fft.rfft(white) / np.sqrt(freq)
        pink = np.fft.irfft(pink_spectrum, n=n_samples)
        # Normalize
        pink_std = np.std(pink)
        if pink_std > 1e-10:
            pink = pink / pink_std
        return pink.astype(np.float64)

    def _generate_crowd_noise(
        self, n_samples: int, sample_rate: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Generate crowd babble noise approximation.

        Crowd noise is characterized by overlapping speech-like signals with
        energy concentrated in the speech frequency range (300–3400 Hz).
        Simulated by bandpass-filtered pink noise with random amplitude modulation.

        Args:
            n_samples: Number of samples
            sample_rate: Audio sample rate
            rng: Random state

        Returns:
            Crowd-like noise waveform
        """
        # Start with pink noise
        noise = self._generate_pink_noise(n_samples, rng)

        # Bandpass filter 300–3400 Hz (speech range) via FFT masking
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
        spectrum = np.fft.rfft(noise)
        # Create bandpass mask
        bandpass = (freqs >= 300) & (freqs <= 3400)
        spectrum_filtered = spectrum * bandpass.astype(np.float64)
        noise_filtered = np.fft.irfft(spectrum_filtered, n=n_samples)

        # Add amplitude modulation to simulate individual voices
        n_voices = 5
        modulated = np.zeros(n_samples)
        for _ in range(n_voices):
            mod_freq = rng.uniform(2.0, 8.0)  # 2–8 Hz modulation (speech rate)
            t = np.arange(n_samples) / sample_rate
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t + rng.uniform(0, 2 * np.pi))
            voice_noise = self._generate_pink_noise(n_samples, rng)
            modulated += envelope * voice_noise

        result = noise_filtered + 0.3 * modulated
        result_std = np.std(result)
        if result_std > 1e-10:
            result = result / result_std
        return result.astype(np.float64)

    def _generate_traffic_noise(
        self, n_samples: int, sample_rate: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Generate traffic-like noise simulation.

        Traffic noise is characterized by low-frequency dominance (engine rumble,
        tire rolling) with occasional mid-frequency peaks (horns, acceleration).

        Args:
            n_samples: Number of samples
            sample_rate: Audio sample rate
            rng: Random state

        Returns:
            Traffic-like noise waveform
        """
        # Base: pink noise with heavy low-frequency emphasis
        noise = self._generate_pink_noise(n_samples, rng)

        # Low-pass enhance: boost frequencies below 500 Hz
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
        spectrum = np.fft.rfft(noise)

        # Low-frequency boost (below 500 Hz gets 3x amplitude)
        low_freq_mask = freqs <= 500
        spectrum_shaped = spectrum.copy()
        spectrum_shaped[low_freq_mask] *= 3.0
        # High-frequency cut (above 4000 Hz)
        high_freq_mask = freqs > 4000
        spectrum_shaped[high_freq_mask] *= 0.1

        noise_shaped = np.fft.irfft(spectrum_shaped, n=n_samples)

        # Add occasional "engine rev" bursts (slow amplitude modulation)
        n_bursts = rng.randint(2, 6)
        for _ in range(n_bursts):
            burst_start = rng.randint(0, max(1, n_samples - sample_rate // 2))
            burst_len = rng.randint(sample_rate // 4, sample_rate // 2)
            burst_end = min(burst_start + burst_len, n_samples)
            burst_amplitude = rng.uniform(1.5, 3.0)
            noise_shaped[burst_start:burst_end] *= burst_amplitude

        ns = np.std(noise_shaped)
        if ns > 1e-10:
            noise_shaped = noise_shaped / ns
        return noise_shaped.astype(np.float64)

    def _batch_predict(self, model, audio_batch):
        try:
            return model([audio_batch[i] for i in range(audio_batch.shape[0])])
        except Exception:
            return [{"label": "error", "score": 0.0}] * audio_batch.shape[0]

    def _single_predict(self, model, audio):
        try:
            preds = model([audio])
            if preds and len(preds) > 0:
                p = preds[0]
                return p if isinstance(p, dict) else {"label": str(p), "score": 1.0}
            return {"label": "unknown", "score": 0.0}
        except Exception:
            return {"label": "error", "score": 0.0}

    @staticmethod
    def _compute_snr(signal, noise):
        sp = np.mean(signal ** 2)
        np_ = np.mean(noise ** 2)
        if np_ < 1e-10:
            return float("inf")
        if sp < 1e-10:
            return float("-inf")
        return float(10 * np.log10(sp / np_))
