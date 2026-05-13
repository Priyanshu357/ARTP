"""Core abstractions for Red Team adversarial attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAttack(ABC):
    """Abstract base class for adversarial attacks.

    Subclasses implement concrete attack strategies that generate adversarial
    examples against a given model. The attack contract standardizes inputs and
    outputs to simplify evaluation and comparison.
    """

    @abstractmethod
    def generate(self, model: Any, inputs: Any, labels: Any) -> Dict[str, Any]:
        """Create adversarial inputs and report attack outcomes.

        Implementations should not mutate inputs in-place unless clearly
        documented. They must return a dictionary with the following keys:
        - "adversarial_inputs": crafted adversarial samples
        - "original_predictions": model outputs on clean inputs
        - "adversarial_predictions": model outputs on adversarial inputs
        - "attack_success_rate": float between 0 and 1
        - "perturbation_metrics": mapping containing at least "l2" and "linf"

        Args:
            model: Target model exposing a predict-like interface.
            inputs: Clean input samples.
            labels: Ground-truth labels for the inputs.

        Returns:
            Dict[str, Any]: Structured attack results.
        """

        raise NotImplementedError
