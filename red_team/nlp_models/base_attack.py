"""Base class for NLP adversarial attacks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseAttack(ABC):
    """Abstract base class for NLP adversarial attacks.

    TODO: Implement NLP-specific attack interface.
    NLP attacks typically work with text inputs (strings or token sequences)
    and need to preserve semantic meaning while fooling the model.
    """

    @abstractmethod
    def generate(self, model: Any, texts: List[str], labels: Any) -> Dict[str, Any]:
        """Generate adversarial text examples.

        Args:
            model: Target NLP model (e.g., transformers model)
            texts: List of input text strings
            labels: Ground-truth labels

        Returns:
            Dict containing:
                - "adversarial_texts": List of perturbed texts
                - "original_predictions": Model outputs on clean texts
                - "adversarial_predictions": Model outputs on adversarial texts
                - "attack_success_rate": Float between 0 and 1
                - "perturbation_metrics": Dict with metrics like edit_distance, word_changes

        TODO: Implement this method for your specific NLP attack strategy.
        """
        raise NotImplementedError("NLP attack generation not yet implemented")
