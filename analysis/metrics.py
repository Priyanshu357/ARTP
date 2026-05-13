"""Metrics computations for adversarial evaluation and detection."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence


def compute_attack_success_rate(attack_results: Sequence[Mapping[str, object]]) -> float:
    """Aggregate attack success rates from runner outputs.

    Expects entries shaped like:
    {"attack": "FGSMAttack", "result": {"attack_success_rate": float, ...}, ...}
    Returns the mean of provided batch-level success rates. If none provided, returns 0.0.
    """
    rates: List[float] = []
    for entry in attack_results:
        result = entry.get("result", {}) if isinstance(entry, Mapping) else {}
        rate = None
        if isinstance(result, Mapping):
            rate_obj = result.get("attack_success_rate")
            if isinstance(rate_obj, (float, int)):
                rate = float(rate_obj)
        if rate is not None:
            rates.append(rate)
    return float(sum(rates) / len(rates)) if rates else 0.0


def compute_detection_accuracy(detection_results: Iterable[Mapping[str, object]]) -> float:
    """Compute detection accuracy: correct detections / total.

    Each record should have:
      - "is_attack": bool (ground truth)
      - "detected": bool (detector decision)
    """
    total = 0
    correct = 0
    for entry in detection_results:
        is_attack = entry.get("is_attack") if isinstance(entry, Mapping) else None
        detected = entry.get("detected") if isinstance(entry, Mapping) else None
        if isinstance(is_attack, bool) and isinstance(detected, bool):
            total += 1
            correct += int(detected == is_attack)
    return float(correct / total) if total else 0.0


def compute_false_positive_rate(detection_results: Iterable[Mapping[str, object]]) -> float:
    """Compute FPR: false alarms / total benign.

    Uses the same schema as compute_detection_accuracy.
    """
    benign = 0
    false_positives = 0
    for entry in detection_results:
        is_attack = entry.get("is_attack") if isinstance(entry, Mapping) else None
        detected = entry.get("detected") if isinstance(entry, Mapping) else None
        if isinstance(is_attack, bool) and isinstance(detected, bool):
            if not is_attack:
                benign += 1
                if detected:
                    false_positives += 1
    return float(false_positives / benign) if benign else 0.0
