"""Base class for NLP adversarial detectors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseDetector(ABC):
    """Abstract base class for detecting adversarial NLP inputs.

    TODO: Implement NLP-specific detection interface.
    NLP detectors typically analyze text properties like perplexity,
    grammar, semantic coherence, and model confidence.
    """

    @abstractmethod
    def detect(
        self,
        original_output: Any,
        adversarial_output: Any,
        original_text: Optional[str] = None,
        adversarial_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect adversarial text examples.

        Args:
            original_output: Model output on clean text
            adversarial_output: Model output on adversarial text
            original_text: Clean text string (optional)
            adversarial_text: Adversarial text string (optional)

        Returns:
            Dict with keys:
                - "anomaly_score": Float (higher = more likely adversarial)
                - "detected": Bool (whether flagged as adversarial)
                - "explanation": String explaining the detection signal

        TODO: Implement detection logic for your NLP model.
        """
        raise NotImplementedError("NLP detection not yet implemented")
