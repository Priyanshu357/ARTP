"""
Object Detection Attack Runner with Per-Detection Diagnostics

This module provides ObjectDetectionAttackRunner, which executes adversarial attacks
on object detection models and collects comprehensive per-detection diagnostics including
IoU metrics, confidence changes, localization errors, and class-specific vulnerabilities.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class ObjectDetectionAttackRunner:
    """
    Run adversarial object detection attacks and collect per-detection diagnostics.

    Similar to AudioAttackRunner and NLPAttackRunner but for object detection models,
    tracking bounding box quality (IoU), confidence changes, missing detections, and
    false positives.

    Args:
        model: Object detection model (callable that takes images)
        attacks: List of object detection attack instances
    """

    def __init__(self, model, attacks: List):
        """Initialize object detection attack runner."""
        self.model = model
        self.attacks = attacks

    def run_batch(
        self,
        images: Any,  # torch.Tensor or np.ndarray [batch, H, W, C]
        targets: List[Dict],  # List of dicts with 'boxes', 'labels', 'scores'
        image_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run attacks on an image batch and collect per-detection diagnostics.

        Args:
            images: Batch of images (torch.Tensor or np.ndarray)
            targets: List of ground truth targets per image
                Each target dict contains:
                - 'boxes': np.ndarray of shape [N, 4] in (x1, y1, x2, y2) format
                - 'labels': np.ndarray of shape [N] with class indices
                - 'scores': np.ndarray of shape [N] with confidence scores (optional)
            image_ids: Optional identifiers for images

        Returns:
            List of attack results with per_detection_diagnostics field
        """
        batch_results = []

        if image_ids is None:
            image_ids = [f"image_{i}" for i in range(len(images))]

        for attack in self.attacks:
            try:
                # Run attack
                attack_output = attack.generate(self.model, images, targets)
            except NotImplementedError:
                # Handle template attacks that aren't implemented yet
                print(f"[WARNING] {attack.__class__.__name__} not implemented, skipping")
                continue

            # Collect per-detection diagnostics
            per_detection_diagnostics = []

            for i in range(len(images)):
                original_image = images[i]
                target = targets[i] if i < len(targets) else {}

                # Get original detections
                original_dets = self._get_detections(original_image)

                # Get adversarial image and detections
                adv_images = attack_output.get('adversarial_images', images)
                adv_image = adv_images[i] if len(adv_images) > i else original_image
                adv_dets = self._get_detections(adv_image)

                # Check if perturbation was attempted
                perturbation_attempted = not self._images_equal(original_image, adv_image)

                # Compute detection-level metrics
                if perturbation_attempted:
                    # Image-level perturbation metrics
                    l2_norm = float(np.linalg.norm(np.array(adv_image) - np.array(original_image)))
                    linf_norm = float(np.max(np.abs(np.array(adv_image) - np.array(original_image))))

                    # Detection quality metrics
                    iou_metrics = self._compute_iou_metrics(
                        original_dets.get('boxes', []),
                        adv_dets.get('boxes', [])
                    )

                    confidence_change = self._compute_confidence_change(
                        original_dets.get('scores', []),
                        adv_dets.get('scores', [])
                    )

                    # Missing detections (objects that disappeared)
                    missing_detections = len(original_dets.get('boxes', [])) - len(adv_dets.get('boxes', []))

                    # False positives (new spurious detections)
                    false_positives = max(0, len(adv_dets.get('boxes', [])) - len(original_dets.get('boxes', [])))

                else:
                    l2_norm = 0.0
                    linf_norm = 0.0
                    iou_metrics = {'mean_iou': 1.0, 'min_iou': 1.0, 'max_iou': 1.0}
                    confidence_change = 0.0
                    missing_detections = 0
                    false_positives = 0

                # Object size analysis
                object_sizes = self._compute_object_sizes(target.get('boxes', []))

                # Build diagnostic record
                per_detection_diagnostics.append({
                    "image_id": image_ids[i],
                    "num_ground_truth": len(target.get('boxes', [])),
                    "num_detections_original": len(original_dets.get('boxes', [])),
                    "num_detections_adversarial": len(adv_dets.get('boxes', [])),
                    "perturbation_attempted": perturbation_attempted,

                    # Perturbation metrics
                    "l2_norm": l2_norm,
                    "linf_norm": linf_norm,

                    # Detection quality metrics
                    "iou_metrics": iou_metrics,
                    "confidence_change": confidence_change,
                    "missing_detections": missing_detections,
                    "false_positives": false_positives,

                    # Object characteristics
                    "object_sizes": object_sizes,
                    "avg_object_size": np.mean(object_sizes) if object_sizes else 0.0,

                    # Attack success indicators
                    "detections_dropped": missing_detections > 0,
                    "confidence_dropped": confidence_change < -0.1,
                    "localization_degraded": iou_metrics.get('mean_iou', 1.0) < 0.5,
                })

            # Store results with diagnostics
            batch_results.append({
                "attack": attack.__class__.__name__,
                "result": attack_output,
                "per_detection_diagnostics": per_detection_diagnostics
            })

        return batch_results

    def _get_detections(self, image: Any) -> Dict[str, Any]:
        """
        Get model detections for a single image.

        Returns:
            Dict with 'boxes' (np.ndarray), 'labels' (np.ndarray), 'scores' (np.ndarray)
        """
        try:
            # Model should return dict with boxes, labels, scores
            detections = self.model([image])
            if detections and len(detections) > 0:
                det = detections[0]
                if isinstance(det, dict):
                    return {
                        'boxes': np.array(det.get('boxes', [])),
                        'labels': np.array(det.get('labels', [])),
                        'scores': np.array(det.get('scores', []))
                    }
            return {'boxes': np.array([]), 'labels': np.array([]), 'scores': np.array([])}
        except Exception as e:
            print(f"[WARNING] Model detection failed: {e}")
            return {'boxes': np.array([]), 'labels': np.array([]), 'scores': np.array([])}

    def _images_equal(self, img1: Any, img2: Any) -> bool:
        """Check if two images are identical."""
        try:
            return np.array_equal(np.array(img1), np.array(img2))
        except Exception:
            return False

    def _compute_iou_metrics(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute IoU (Intersection over Union) metrics between two sets of boxes.

        Args:
            boxes1: Original bounding boxes [N, 4] in (x1, y1, x2, y2) format
            boxes2: Adversarial bounding boxes [M, 4]

        Returns:
            Dict with mean_iou, min_iou, max_iou
        """
        if len(boxes1) == 0 or len(boxes2) == 0:
            return {'mean_iou': 0.0, 'min_iou': 0.0, 'max_iou': 0.0}

        ious = []
        # Compute IoU for each box in boxes1 with all boxes in boxes2
        for box1 in boxes1:
            max_iou = 0.0
            for box2 in boxes2:
                iou = self._compute_iou(box1, box2)
                max_iou = max(max_iou, iou)
            ious.append(max_iou)

        if not ious:
            return {'mean_iou': 0.0, 'min_iou': 0.0, 'max_iou': 0.0}

        return {
            'mean_iou': float(np.mean(ious)),
            'min_iou': float(np.min(ious)),
            'max_iou': float(np.max(ious))
        }

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute IoU between two boxes in (x1, y1, x2, y2) format.

        Returns:
            IoU score between 0 and 1
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Compute intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)
        inter_area = inter_width * inter_height

        # Compute union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return float(inter_area / union_area)

    def _compute_confidence_change(
        self,
        original_scores: np.ndarray,
        adversarial_scores: np.ndarray
    ) -> float:
        """
        Compute average confidence change.

        Returns:
            Mean change in confidence scores
        """
        if len(original_scores) == 0 or len(adversarial_scores) == 0:
            return 0.0

        # Take minimum length to handle missing detections
        min_len = min(len(original_scores), len(adversarial_scores))
        if min_len == 0:
            return 0.0

        orig_sorted = np.sort(original_scores)[::-1][:min_len]
        adv_sorted = np.sort(adversarial_scores)[::-1][:min_len]

        return float(np.mean(adv_sorted - orig_sorted))

    def _compute_object_sizes(self, boxes: np.ndarray) -> List[float]:
        """
        Compute sizes (areas) of bounding boxes.

        Args:
            boxes: Bounding boxes [N, 4] in (x1, y1, x2, y2) format

        Returns:
            List of box areas
        """
        if len(boxes) == 0:
            return []

        sizes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            sizes.append(float(area))

        return sizes

    def run_dataset(self, dataset) -> List[Dict[str, Any]]:
        """
        Run attacks on an entire dataset.

        Args:
            dataset: Iterable of (images, targets) tuples

        Returns:
            List of all attack results across batches
        """
        all_results = []

        for batch_images, batch_targets in dataset:
            batch_results = self.run_batch(batch_images, batch_targets)
            all_results.extend(batch_results)

        return all_results
