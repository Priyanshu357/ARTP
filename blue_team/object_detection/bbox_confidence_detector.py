"""Bounding box confidence detector for object detection (template)."""

from typing import Any, Dict, List, Optional

from .base_detector import BaseDetector


class BBoxConfidenceDetector(BaseDetector):
    """Detect adversarial examples based on detection confidence anomalies.

    TODO: Implement confidence-based detection:
    1. Compare detection confidences between original and adversarial
    2. Flag significant drops in confidence scores
    3. Detect appearance/disappearance of detections
    4. Analyze confidence distribution changes

    Intuition: Adversarial perturbations typically cause confidence drops
    or unstable detections.
    """

    def __init__(self, confidence_drop_threshold: float = 0.3):
        """Initialize bbox confidence detector.

        Args:
            confidence_drop_threshold: Min confidence drop to flag as adversarial
        """
        self.threshold = confidence_drop_threshold

    def detect(
        self,
        original_detections: List[Dict],
        adversarial_detections: List[Dict],
        original_images: Optional[Any] = None,
        adversarial_images: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Detect based on confidence anomalies.

        TODO: Implement confidence-based detection logic.
        """
        raise NotImplementedError("BBox confidence detector not yet implemented")
