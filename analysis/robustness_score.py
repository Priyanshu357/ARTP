"""Robustness score computation (0-100)."""

from __future__ import annotations

import math
from typing import List, Dict


def compute_robustness_score(
    attack_success_rate: float,
    detection_accuracy: float,
    false_positive_rate: float,
    weight_attack: float = 0.5,
    weight_detection: float = 0.4,
    weight_fpr: float = 0.1,
    detection_used: bool = True,
) -> float:
    """Combine metrics into a 0–100 robustness score.

    Lower attack success and false positives increase the score; higher detection
    accuracy increases the score. Weights should sum to 1.0; if they do not, they
    are renormalized.

    When detection is not used (detection_used=False), the detection weight is
    redistributed to attack and FPR components:
    - Default with detection: attack=50%, detection=40%, FPR=10%
    - Without detection: attack=90%, detection=0%, FPR=10%

    Args:
        attack_success_rate: Attack success rate (0-1)
        detection_accuracy: Detection accuracy (0-1)
        false_positive_rate: False positive rate (0-1)
        weight_attack: Weight for attack resistance component
        weight_detection: Weight for detection accuracy component
        weight_fpr: Weight for false positive rate component
        detection_used: Whether detection was used in evaluation

    Returns:
        Robustness score (0-100)
    """
    # If detection not used, redistribute its weight to attack component
    if not detection_used:
        weight_attack += weight_detection  # 0.5 + 0.4 = 0.9
        weight_detection = 0.0

    weights_sum = weight_attack + weight_detection + weight_fpr
    if not math.isfinite(weights_sum) or weights_sum <= 0:
        weight_attack = 0.9 if not detection_used else 0.5
        weight_detection = 0.0 if not detection_used else 0.4
        weight_fpr = 0.1
        weights_sum = 1.0
    weight_attack /= weights_sum
    weight_detection /= weights_sum
    weight_fpr /= weights_sum

    # Normalize inputs to [0,1]
    asr = min(max(attack_success_rate, 0.0), 1.0)
    det_acc = min(max(detection_accuracy, 0.0), 1.0)
    fpr = min(max(false_positive_rate, 0.0), 1.0)

    raw = (
        (1.0 - asr) * weight_attack
        + det_acc * weight_detection
        + (1.0 - fpr) * weight_fpr
    )
    score = raw * 100.0
    return float(min(max(score, 0.0), 100.0))


def apply_diagnostic_penalties(base_score: float, diagnostics_list: List[Dict]) -> float:
    """Apply penalties to robustness score based on CRITICAL diagnostic findings.

    This prevents broken models (class imbalance, flat confidence, etc.) from
    receiving high robustness scores just because attacks can't flip predictions.

    Penalty multipliers:
    - class_imbalance_bias (CRITICAL): 0.3 (model stuck predicting one class)
    - flat_confidence_pattern (HIGH): 0.6 (model not learning features)
    - Multiple CRITICAL issues: penalties multiply (0.3 × 0.6 = 0.18)

    Args:
        base_score: Original robustness score (0-100)
        diagnostics_list: List of diagnostic dicts with 'diagnostic' and 'severity' keys

    Returns:
        Penalized robustness score (0-100)

    Example:
        Model with class_imbalance_bias:
        - Base score: 100.0 (0% ASR because model is stuck)
        - Penalized: 100.0 × 0.3 = 30.0 (reflects broken model)
    """
    if not diagnostics_list:
        return base_score

    penalty_multiplier = 1.0

    for diag in diagnostics_list:
        diag_type = diag.get('diagnostic', '')
        severity = diag.get('severity', '')

        # CRITICAL penalties (model is fundamentally broken)
        if diag_type == 'class_imbalance_bias' and severity == 'CRITICAL':
            # Model always predicts one class - apply severe penalty
            penalty_multiplier *= 0.3  # Reduce to 30% of original score

        # HIGH penalties (model has serious issues)
        elif diag_type == 'flat_confidence_pattern' and severity == 'HIGH':
            # Model not learning meaningful features - moderate penalty
            penalty_multiplier *= 0.6  # Reduce to 60% of original score

    # Apply penalty if any issues were found
    if penalty_multiplier < 1.0:
        penalized_score = base_score * penalty_multiplier
        return float(min(max(penalized_score, 0.0), 100.0))

    return base_score
