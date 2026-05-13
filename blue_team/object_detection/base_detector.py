"""Base class for object detection adversarial detectors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseDetector(ABC):
    """Abstract base class for detecting adversarial examples in object detection.

    TODO: Implement object detection-specific detection interface.
    Detectors can analyze detection confidence, bounding box stability,
    IoU consistency, and other signals.
    """

    @abstractmethod
    def detect(
        self,
        original_detections: List[Dict],
        adversarial_detections: List[Dict],
        original_images: Optional[Any] = None,
        adversarial_images: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Detect adversarial examples for object detection.

        Args:
            original_detections: Detections on clean images (list of dicts with boxes, labels, scores)
            adversarial_detections: Detections on adversarial images
            original_images: Clean images (optional)
            adversarial_images: Adversarial images (optional)

        Returns:
            Dict with keys:
                - "anomaly_score": Float (higher = more likely adversarial)
                - "detected": Bool (whether flagged as adversarial)
                - "explanation": String explaining the detection signal

        TODO: Implement detection logic for object detection models.
        """
        raise NotImplementedError("Object detection detector not yet implemented")
