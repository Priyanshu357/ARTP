"""Content templates and defaults for PDF reports."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence


def build_recommendations(summary: Mapping[str, float]) -> List[str]:
    """Derive brief recommendations based on metrics."""
    recs: List[str] = []
    asr = float(summary.get("attack_success_rate", 0.0) or 0.0)
    det = float(summary.get("detection_accuracy", 0.0) or 0.0)
    fpr = float(summary.get("false_positive_rate", 0.0) or 0.0)

    if asr > 0.5:
        recs.append("Model is highly vulnerable (ASR > 50%). Reduce epsilon exposure, apply adversarial training, and add input normalization checks.")
    elif asr > 0.2:
        recs.append("Moderate vulnerability. Consider PGD-based adversarial training and gradient masking avoidance.")
    else:
        recs.append("Low observed vulnerability for tested budgets; broaden attack coverage and test adaptive attacks.")

    if det < 0.8:
        recs.append("Detection accuracy is low; retrain detectors with harder negatives and calibrate thresholds.")
    else:
        recs.append("Detection is acceptable; validate against unseen attack types to ensure generalization.")

    if fpr > 0.1:
        recs.append("High false positives; adjust thresholds and improve benign coverage to reduce alert fatigue.")

    recs.append("Add continuous evaluation in CI with periodic red-team runs and drift monitoring.")
    return recs


def format_model_details(model_info: Mapping[str, object]) -> Dict[str, str]:
    """Normalize model details for display."""
    return {
        "name": str(model_info.get("name", "Unknown Model")),
        "version": str(model_info.get("version", "n/a")),
        "framework": str(model_info.get("framework", "onnx/torch")),
        "notes": str(model_info.get("notes", "")),
    }
