"""Abstract detector interface for Blue Team defenses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDetector(ABC):
    """Base class for adversarial detection modules.

    Implementations should analyze original and adversarial model outputs and
    emit a structured detection result containing:
      - "anomaly_score": float (higher means more likely adversarial)
      - "detected": bool (whether the sample is flagged)
      - "explanation": str (brief rationale or signal source)
    """

    @abstractmethod
    def detect(self, original_output: Any, adversarial_output: Any) -> Dict[str, Any]:
        """Detect adversariality given clean vs. adversarial outputs.

        Args:
            original_output: Model output on the clean input (e.g., logits or probs).
            adversarial_output: Model output on the adversarial input.

        Returns:
            Dict[str, Any]: A mapping with keys "anomaly_score", "detected", and "explanation".
        """

        raise NotImplementedError
