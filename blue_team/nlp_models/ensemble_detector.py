"""Ensemble detector for NLP models (implementation)."""

from typing import Any, Dict, List, Optional

from .base_detector import BaseDetector


class EnsembleDetector(BaseDetector):
    """Combine multiple NLP detectors using voting or score aggregation.

    Combines detection signals from multiple detectors to improve
    overall detection accuracy and reduce false positives/negatives.
    """

    def __init__(self, detectors: List[BaseDetector], mode: str = "vote"):
        """Initialize ensemble detector.

        Args:
            detectors: List of detector instances to combine
            mode: Combination mode ("vote" or "average")
                - "vote": Majority voting (detected if > 50% detectors flag it)
                - "average": Average anomaly scores and use threshold
        """
        self.detectors = detectors
        self.mode = mode

    def detect(
        self,
        original_output: Any,
        adversarial_output: Any,
        original_text: Optional[str] = None,
        adversarial_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect using ensemble of detectors.

        Args:
            original_output: Model output on clean text
            adversarial_output: Model output on adversarial text
            original_text: Clean text string (optional)
            adversarial_text: Adversarial text string (optional)

        Returns:
            Dict with anomaly_score, detected flag, and explanation
        """
        if not self.detectors:
            return {
                "anomaly_score": 0.0,
                "detected": False,
                "explanation": "No detectors configured"
            }

        # Run all detectors
        results = []
        for detector in self.detectors:
            try:
                result = detector.detect(
                    original_output,
                    adversarial_output,
                    original_text,
                    adversarial_text
                )
                results.append(result)
            except Exception as e:
                # Skip detector if it fails
                continue

        if not results:
            return {
                "anomaly_score": 0.0,
                "detected": False,
                "explanation": "All detectors failed"
            }

        # Combine results based on mode
        if self.mode == "vote":
            return self._vote_combination(results)
        else:  # average
            return self._average_combination(results)

    def _vote_combination(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine using majority voting.

        Args:
            results: List of detection results from individual detectors

        Returns:
            Combined detection result
        """
        votes_detected = sum(1 for r in results if r.get("detected", False))
        total_votes = len(results)

        # Majority voting
        detected = votes_detected > (total_votes / 2)

        # Average anomaly scores
        avg_score = sum(r.get("anomaly_score", 0.0) for r in results) / total_votes

        # Combine explanations
        explanations = [r.get("explanation", "") for r in results]
        explanation = f"Voting: {votes_detected}/{total_votes} detectors flagged. " + " | ".join(explanations)

        return {
            "anomaly_score": float(avg_score),
            "detected": bool(detected),
            "explanation": explanation
        }

    def _average_combination(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine using average anomaly scores.

        Args:
            results: List of detection results from individual detectors

        Returns:
            Combined detection result
        """
        total_votes = len(results)

        # Average anomaly scores
        avg_score = sum(r.get("anomaly_score", 0.0) for r in results) / total_votes

        # Detect if average score > 0.5
        detected = avg_score > 0.5

        # Combine explanations
        explanations = [r.get("explanation", "") for r in results]
        explanation = f"Average score: {avg_score:.3f}. " + " | ".join(explanations)

        return {
            "anomaly_score": float(avg_score),
            "detected": bool(detected),
            "explanation": explanation
        }

