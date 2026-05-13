"""Entropy-based adversarial detector."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .base_detector import BaseDetector


def _entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = probs.clamp(min=eps, max=1.0)
    return -torch.sum(probs * probs.log(), dim=1)


class EntropyIncreaseDetector(BaseDetector):
    """Flags samples whose prediction entropy rises beyond a threshold."""

    def __init__(self, entropy_threshold: float = 0.2) -> None:
        self.entropy_threshold = float(entropy_threshold)

    def detect(self, original_output: Any, adversarial_output: Any) -> Dict[str, Any]:
        clean_probs = self._to_probs(original_output)
        adv_probs = self._to_probs(adversarial_output)

        clean_ent = _entropy(clean_probs)
        adv_ent = _entropy(adv_probs)

        ent_increase = (adv_ent - clean_ent).mean().item()
        detected = ent_increase > self.entropy_threshold
        explanation = f"Mean entropy increase {ent_increase:.3f} (threshold {self.entropy_threshold:.3f})"

        return {
            "anomaly_score": ent_increase,
            "detected": detected,
            "explanation": explanation,
        }

    @staticmethod
    def _to_probs(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            if output.ndim == 1:
                output = output.unsqueeze(0)
            return F.softmax(output.detach(), dim=1)
        raise TypeError("Outputs must be torch.Tensor logits or probs")
