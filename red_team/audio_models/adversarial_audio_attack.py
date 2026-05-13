"""Carlini & Wagner style adversarial audio attack.

Optimization-based attack that crafts adversarial audio perturbations by
minimizing perturbation magnitude while maximizing misclassification confidence.
Applies perceptual constraints to keep perturbations imperceptible.

Reference: Carlini & Wagner, "Audio Adversarial Examples" (2018)
           https://arxiv.org/abs/1801.01944
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base_attack import BaseAttack


class AdversarialAudioAttack(BaseAttack):
    """Optimization-based adversarial audio attack (Carlini & Wagner).

    Iteratively optimizes a perturbation delta to:
    1. Maximize model misclassification (targeted or untargeted)
    2. Minimize perturbation magnitude (L2 norm)
    3. Preserve audio quality via clipping and SNR constraints

    The optimization uses projected gradient descent with adaptive step size.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        num_iterations: int = 100,
        learning_rate: float = 0.01,
        confidence: float = 0.0,
        binary_search_steps: int = 5,
        initial_const: float = 1.0,
        max_perturbation_db: float = -20.0,
    ):
        """Initialize adversarial audio attack.

        Args:
            epsilon: Maximum perturbation magnitude (L∞ constraint)
            num_iterations: Number of optimization iterations per search step
            learning_rate: Step size for gradient descent
            confidence: Minimum confidence margin for successful attack
            binary_search_steps: Number of binary search steps for constant c
            initial_const: Initial value of the loss trade-off constant
            max_perturbation_db: Maximum perturbation power in dB relative to signal
        """
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.confidence = confidence
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.max_perturbation_db = max_perturbation_db

    def generate(
        self, model: Any, audio: np.ndarray, labels: Any, sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Generate adversarial audio using optimization-based attack.

        For each audio sample in the batch:
        1. Initialize perturbation delta = 0
        2. For each binary search step, find the optimal trade-off constant c
        3. For each iteration, compute loss and update delta via gradient estimation
        4. Project delta to satisfy L∞ and SNR constraints
        5. Return the best adversarial example found

        Args:
            model: Target audio classification model (callable)
            audio: Audio batch [batch_size, samples] or [samples]
            labels: Ground truth labels (list of ints)
            sample_rate: Audio sample rate in Hz

        Returns:
            Dictionary with attack results and per-sample diagnostics
        """
        # Ensure audio is 2D [batch_size, samples]
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        batch_size = audio.shape[0]

        # Get original predictions
        original_predictions = self._batch_predict(model, audio)

        # Attack each sample
        adversarial_audio = []
        adversarial_predictions = []
        successful_attacks = 0
        per_sample_metrics = []

        for i in range(batch_size):
            original_sample = audio[i]
            original_pred = original_predictions[i]
            original_label = original_pred["label"]

            # Run C&W attack on this sample
            best_adv, attack_info = self._attack_single(
                model, original_sample, original_label, sample_rate
            )

            adversarial_audio.append(best_adv)

            # Get adversarial prediction
            adv_pred = self._single_predict(model, best_adv)
            adversarial_predictions.append(adv_pred)

            # Check success
            success = adv_pred["label"] != original_label
            if success:
                successful_attacks += 1

            # Compute per-sample metrics
            perturbation = best_adv - original_sample
            l2_norm = float(np.linalg.norm(perturbation))
            linf_norm = float(np.max(np.abs(perturbation)))
            snr_db = self._compute_snr(original_sample, perturbation)

            per_sample_metrics.append({
                "success": success,
                "l2_norm": l2_norm,
                "linf_norm": linf_norm,
                "snr_db": snr_db,
                "iterations_used": attack_info["iterations_used"],
                "best_const": attack_info["best_const"],
                "original_label": original_label,
                "adversarial_label": adv_pred["label"],
                "confidence_change": adv_pred["score"] - original_pred["score"],
            })

        # Aggregate metrics
        attack_success_rate = successful_attacks / batch_size if batch_size > 0 else 0.0

        adversarial_array = np.array(adversarial_audio)
        avg_l2 = float(np.mean([m["l2_norm"] for m in per_sample_metrics]))
        avg_snr = float(np.mean([
            m["snr_db"] for m in per_sample_metrics
            if np.isfinite(m["snr_db"])
        ] or [float("inf")]))

        return {
            "attack": "AdversarialAudioAttack",
            "attack_success_rate": attack_success_rate,
            "adversarial_audio": adversarial_array,
            "original_predictions": original_predictions,
            "adversarial_predictions": adversarial_predictions,
            "perturbation_metrics": {
                "avg_l2_norm": avg_l2,
                "avg_snr_db": avg_snr,
                "epsilon": self.epsilon,
                "num_iterations": self.num_iterations,
                "learning_rate": self.learning_rate,
            },
            "successful_attacks": successful_attacks,
            "total_samples": batch_size,
            "per_sample_metrics": per_sample_metrics,
        }

    def _attack_single(
        self,
        model: Any,
        original: np.ndarray,
        original_label: str,
        sample_rate: int,
    ) -> tuple:
        """Run C&W attack on a single audio sample.

        Uses binary search over the constant c to find the optimal trade-off
        between perturbation size and attack confidence.

        Args:
            model: Target model
            original: Original audio waveform [samples]
            original_label: Original model prediction label
            sample_rate: Audio sample rate

        Returns:
            Tuple of (best_adversarial_audio, attack_info_dict)
        """
        best_adv = original.copy()
        best_l2 = float("inf")
        best_const = self.initial_const
        total_iters = 0

        # Binary search over constant c
        c_lower = 0.0
        c_upper = self.initial_const * 10
        c = self.initial_const

        for search_step in range(self.binary_search_steps):
            # Initialize perturbation in tanh-space for unconstrained optimization
            # w = arctanh(2 * x - 1) maps [-1, 1] -> (-inf, inf)
            # but we use a simpler projected approach for numerical stability
            delta = np.zeros_like(original, dtype=np.float64)

            # Track best result for this c value
            step_best_adv = original.copy()
            step_best_l2 = float("inf")
            found_success = False

            for iteration in range(self.num_iterations):
                total_iters += 1

                # Current adversarial example
                adv = np.clip(original + delta, -1.0, 1.0)

                # Get model prediction
                pred = self._single_predict(model, adv)

                # Check if attack succeeded
                if pred["label"] != original_label:
                    current_l2 = np.linalg.norm(delta)
                    if current_l2 < step_best_l2:
                        step_best_adv = adv.copy()
                        step_best_l2 = current_l2
                        found_success = True

                # Estimate gradient via finite differences
                # (model is a black box, no direct gradients available)
                grad = self._estimate_gradient(
                    model, original, delta, original_label, c
                )

                # Update delta with gradient descent
                delta -= self.learning_rate * grad

                # Project: L∞ constraint
                delta = np.clip(delta, -self.epsilon, self.epsilon)

                # Project: SNR constraint
                delta = self._apply_snr_constraint(original, delta)

                # Project: ensure audio stays in [-1, 1]
                delta = np.clip(original + delta, -1.0, 1.0) - original

            # Binary search update
            if found_success:
                # Attack succeeded — try smaller c (less perturbation)
                if step_best_l2 < best_l2:
                    best_adv = step_best_adv.copy()
                    best_l2 = step_best_l2
                    best_const = c
                c_upper = c
                c = (c_lower + c_upper) / 2
            else:
                # Attack failed — try larger c (more aggressive)
                c_lower = c
                if c_upper < 1e9:
                    c = (c_lower + c_upper) / 2
                else:
                    c *= 10

        return best_adv, {
            "iterations_used": total_iters,
            "best_const": best_const,
            "best_l2": best_l2,
        }

    def _estimate_gradient(
        self,
        model: Any,
        original: np.ndarray,
        delta: np.ndarray,
        original_label: str,
        c: float,
        num_samples: int = 20,
        sigma: float = 0.001,
    ) -> np.ndarray:
        """Estimate gradient using Natural Evolution Strategy (NES).

        Since the model is a black box (no direct gradients), we use
        random perturbations to estimate the gradient of the attack loss.

        The loss function is:
            L = c * f(x + delta) + ||delta||^2

        where f(x+delta) measures how close we are to misclassification.

        Args:
            model: Target model
            original: Original audio
            delta: Current perturbation
            original_label: Label we want to change
            c: Trade-off constant
            num_samples: Number of random directions for gradient estimation
            sigma: Standard deviation of random perturbations

        Returns:
            Estimated gradient as numpy array
        """
        grad = np.zeros_like(delta)
        current_adv = np.clip(original + delta, -1.0, 1.0)

        for _ in range(num_samples):
            # Random perturbation direction
            noise = np.random.randn(*delta.shape) * sigma

            # Forward evaluation: f(x + delta + noise)
            pos_adv = np.clip(original + delta + noise, -1.0, 1.0)
            neg_adv = np.clip(original + delta - noise, -1.0, 1.0)

            pos_pred = self._single_predict(model, pos_adv)
            neg_pred = self._single_predict(model, neg_adv)

            # Attack loss: we want to DECREASE confidence in original label
            # Higher loss = more confident in original label (bad for attack)
            pos_loss = self._attack_loss(pos_pred, original_label, c)
            neg_loss = self._attack_loss(neg_pred, original_label, c)

            # NES gradient estimate
            grad += (pos_loss - neg_loss) * noise / (2 * sigma)

        grad /= num_samples

        # Add L2 regularization gradient (pushes toward smaller perturbation)
        grad += 2 * delta

        return grad

    def _attack_loss(
        self, prediction: Dict[str, Any], original_label: str, c: float
    ) -> float:
        """Compute attack loss for a prediction.

        We want to maximize misclassification, so the loss is:
        - High when model predicts the original label (attack hasn't worked)
        - Low when model predicts a different label (attack succeeded)

        Args:
            prediction: Model prediction dict {"label": ..., "score": ...}
            original_label: The label we want to flip away from
            c: Trade-off constant

        Returns:
            Scalar loss value
        """
        if prediction["label"] == original_label:
            # Still predicting original label — high loss
            # Use confidence as loss (more confident = higher loss)
            return c * prediction["score"]
        else:
            # Successfully misclassified — negative loss (reward)
            return -c * (prediction["score"] + self.confidence)

    def _apply_snr_constraint(
        self, original: np.ndarray, delta: np.ndarray
    ) -> np.ndarray:
        """Apply SNR constraint to keep perturbation imperceptible.

        Scales delta down if the perturbation-to-signal ratio exceeds
        the maximum allowed perturbation power.

        Args:
            original: Original audio signal
            delta: Current perturbation

        Returns:
            Constrained perturbation
        """
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(delta ** 2)

        if noise_power < 1e-10 or signal_power < 1e-10:
            return delta

        # max_perturbation_db is negative (e.g., -20 dB)
        # Convert to linear scale: max_noise_power = signal_power * 10^(db/10)
        max_noise_power = signal_power * (10 ** (self.max_perturbation_db / 10))

        if noise_power > max_noise_power:
            scale = np.sqrt(max_noise_power / noise_power)
            delta = delta * scale

        return delta

    def _batch_predict(
        self, model: Any, audio_batch: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Get model predictions for a batch of audio.

        Args:
            model: Audio classification model
            audio_batch: Batch of audio [batch_size, samples]

        Returns:
            List of prediction dicts
        """
        try:
            predictions = model([audio_batch[i] for i in range(audio_batch.shape[0])])
            return predictions
        except Exception as e:
            print(f"[WARNING] Batch prediction failed: {e}")
            return [{"label": "error", "score": 0.0}] * audio_batch.shape[0]

    def _single_predict(self, model: Any, audio: np.ndarray) -> Dict[str, Any]:
        """Get model prediction for a single audio sample.

        Args:
            model: Audio classification model
            audio: Single audio waveform [samples]

        Returns:
            Prediction dict {"label": str, "score": float}
        """
        try:
            predictions = model([audio])
            if predictions and len(predictions) > 0:
                pred = predictions[0]
                if isinstance(pred, dict):
                    return pred
                return {"label": str(pred), "score": 1.0}
            return {"label": "unknown", "score": 0.0}
        except Exception as e:
            return {"label": "error", "score": 0.0}

    @staticmethod
    def _compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
        """Compute Signal-to-Noise Ratio in dB.

        Args:
            signal: Original signal
            noise: Noise (perturbation)

        Returns:
            SNR in decibels
        """
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power < 1e-10:
            return float("inf")
        if signal_power < 1e-10:
            return float("-inf")

        return float(10 * np.log10(signal_power / noise_power))
