"""Base class for object detection adversarial attacks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseAttack(ABC):
    """Abstract base class for object detection adversarial attacks.

    TODO: Implement object detection-specific attack interface.
    Object detection attacks can target different aspects:
    - Hiding objects (reduce confidence)
    - Creating false detections
    - Misclassifying detected objects
    - Perturbing bounding boxes
    """

    @abstractmethod
    def generate(self, model: Any, images: Any, targets: List[Dict]) -> Dict[str, Any]:
        """Generate adversarial examples for object detection.

        Args:
            model: Target object detection model (e.g., YOLO, Faster R-CNN)
            images: Input images
            targets: List of dicts with keys like 'boxes', 'labels', 'scores'

        Returns:
            Dict containing:
                - "adversarial_images": Perturbed images
                - "original_detections": Detections on clean images
                - "adversarial_detections": Detections on adversarial images
                - "attack_success_rate": Float between 0 and 1
                - "perturbation_metrics": Dict with L2, Linf norms
                - "detection_metrics": Dict with IoU, mAP changes

        TODO: Implement object detection attack.
        """
        raise NotImplementedError("Object detection attack not yet implemented")
