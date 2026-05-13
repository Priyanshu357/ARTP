"""IoU anomaly detector for object detection (template)."""

from typing import Any, Dict, List, Optional

from .base_detector import BaseDetector


class IoUAnomalyDetector(BaseDetector):
    """Detect adversarial examples based on bounding box IoU anomalies.

    TODO: Implement IoU-based detection:
    1. Compute IoU between original and adversarial detections
    2. Flag significant shifts in bounding box locations
    3. Detect unstable or jittery boxes
    4. Analyze bbox consistency across frames (for video)

    Intuition: Adversarial attacks may cause bounding boxes to shift
    or become unstable despite similar visual content.
    """

    def __init__(self, iou_threshold: float = 0.5):
        """Initialize IoU anomaly detector.

        Args:
            iou_threshold: Min IoU to consider detections consistent
        """
        self.threshold = iou_threshold

    def detect(
        self,
        original_detections: List[Dict],
        adversarial_detections: List[Dict],
        original_images: Optional[Any] = None,
        adversarial_images: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Detect based on IoU anomalies.

        TODO: Implement IoU-based detection logic.
        """
        raise NotImplementedError("IoU anomaly detector not yet implemented")
