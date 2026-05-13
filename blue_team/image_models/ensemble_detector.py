"""Ensemble detector that aggregates multiple detectors."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from .base_detector import BaseDetector


class EnsembleDetector(BaseDetector):
    """Combine multiple detectors via weighted voting."""

    def __init__(self, detectors: Sequence[BaseDetector], weights: Iterable[float] | None = None, mode: str = "vote") -> None:
        """Args:
            detectors: list of detector instances implementing BaseDetector.
            weights: optional per-detector weights; defaults to 1 for each.
            mode: "vote" (majority on detected) or "weighted" (weighted anomaly score > threshold).
        """
        self.detectors = list(detectors)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.detectors)
        self.mode = mode
        if len(self.weights) != len(self.detectors):
            raise ValueError("weights length must match detectors length")

    def detect(self, original_output: Any, adversarial_output: Any) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for det in self.detectors:
            results.append(det.detect(original_output, adversarial_output))

        if self.mode == "vote":
            return self._vote(results)
        return self._weighted(results)

    def _vote(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Majority vote on detected
        votes = []
        for res, w in zip(results, self.weights):
            detected = bool(res.get("detected"))
            votes.extend([detected] * int(max(1, round(w))))
        detected_final = votes.count(True) > votes.count(False)
        anomaly_score = sum(res.get("anomaly_score", 0.0) for res in results) / max(1, len(results))
        explanation = "; ".join(str(res.get("explanation", "")) for res in results)
        return {
            "anomaly_score": float(anomaly_score),
            "detected": detected_final,
            "explanation": f"Vote={detected_final}; {explanation}",
        }

    def _weighted(self, results: List[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, Any]:
        # Weighted mean anomaly score, then threshold
        total_w = sum(self.weights)
        if total_w <= 0:
            total_w = 1.0
        score = 0.0
        parts = []
        for res, w in zip(results, self.weights):
            s = float(res.get("anomaly_score", 0.0))
            score += w * s
            parts.append(f"{s:.3f}*{w:.2f}")
        score /= total_w
        detected_final = score > threshold
        explanation = f"Weighted score {score:.3f} (threshold {threshold:.3f}); parts: {', '.join(parts)}"
        return {
            "anomaly_score": float(score),
            "detected": detected_final,
            "explanation": explanation,
        }
