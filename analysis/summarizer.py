"""Summarize attack and detection metrics into a single report."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

from .metrics import (
    compute_attack_success_rate,
    compute_detection_accuracy,
    compute_false_positive_rate,
)
from .robustness_score import compute_robustness_score


def summarize(
    attack_results: Sequence[Mapping[str, object]],
    detection_results: Iterable[Mapping[str, object]],
) -> Dict[str, float]:
    """Compute core security metrics and robustness score.

    Returns a dict suitable for JSON serialization:
    {
        "attack_success_rate": float,
        "detection_accuracy": float,
        "false_positive_rate": float,
        "robustness_score": float,
    }
    """
    asr = compute_attack_success_rate(attack_results)
    det_acc = compute_detection_accuracy(detection_results)
    fpr = compute_false_positive_rate(detection_results)

    # Check if detection was used
    detection_results_list = list(detection_results) if detection_results else []
    detection_used = len(detection_results_list) > 0

    # Compute robustness score with auto-adjusted weights
    score = compute_robustness_score(asr, det_acc, fpr, detection_used=detection_used)
    return {
        "attack_success_rate": asr,
        "detection_accuracy": det_acc,
        "false_positive_rate": fpr,
        "robustness_score": score,
    }
