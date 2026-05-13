"""Confidence-drop based adversarial detector."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .base_detector import BaseDetector


class ConfidenceDropDetector(BaseDetector):
    """Flags samples whose max softmax confidence drops by a relative threshold."""

    def __init__(self, drop_threshold: float = 0.2) -> None:
        """Args:
            drop_threshold: relative drop fraction; e.g., 0.2 flags if confidence falls by >20%.
        """
        self.drop_threshold = float(drop_threshold)

    def detect(self, original_output: Any, adversarial_output: Any) -> Dict[str, Any]:
        # Convert to tensors
        orig = self._to_probs(original_output)
        adv = self._to_probs(adversarial_output)

        # Compute max confidence per sample
        orig_conf, _ = orig.max(dim=1)
        adv_conf, _ = adv.max(dim=1)

        # Relative drop: (orig - adv) / max(orig, eps)
        eps = 1e-8
        rel_drop = (orig_conf - adv_conf) / (orig_conf.clamp(min=eps))

        # Aggregate: use mean drop as anomaly score
        anomaly_score = rel_drop.mean().item()
        detected = anomaly_score > self.drop_threshold
        explanation = f"Mean confidence drop {anomaly_score:.3f} (threshold {self.drop_threshold:.3f})"

        return {
            "anomaly_score": anomaly_score,
            "detected": detected,
            "explanation": explanation,
        }

    @staticmethod
    def _to_probs(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            if output.ndim == 1:
                output = output.unsqueeze(0)
            # assume logits
            return F.softmax(output.detach(), dim=1)
        raise TypeError("Outputs must be torch.Tensor logits or probs")
