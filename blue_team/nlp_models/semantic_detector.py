"""Semantic similarity detector for adversarial NLP inputs (simplified POC implementation)."""

from typing import Any, Dict, Optional
import numpy as np

from .base_detector import BaseDetector


class SemanticDetector(BaseDetector):
    """Detect adversarial texts using semantic similarity analysis.

    Uses sentence embeddings to measure semantic similarity between
    original and adversarial texts. Low similarity may indicate
    adversarial manipulation.

    Intuition: Adversarial texts may have different semantic meaning
    despite fooling the classifier.
    """

    def __init__(self, similarity_threshold: float = 0.7, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize semantic detector.

        Args:
            similarity_threshold: Min similarity score (0-1) to consider non-adversarial
            embedding_model: Sentence transformer model name
        """
        self.threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        self.encoder = None

    def _load_encoder(self):
        """Lazy-load sentence embedding model."""
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            print(f"Loading sentence embedding model: {self.embedding_model_name}...")
            self.encoder = SentenceTransformer(self.embedding_model_name)

    def detect(
        self,
        original_output: Any,
        adversarial_output: Any,
        original_text: Optional[str] = None,
        adversarial_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect based on semantic similarity.

        Args:
            original_output: Model output on clean text (unused)
            adversarial_output: Model output on adversarial text (unused)
            original_text: Clean text string (required)
            adversarial_text: Adversarial text string (required)

        Returns:
            Dict with anomaly_score, detected flag, and explanation
        """
        if original_text is None or adversarial_text is None:
            return {
                "anomaly_score": 0.0,
                "detected": False,
                "explanation": "Missing text for comparison"
            }

        # Load encoder if needed
        self._load_encoder()

        # Compute semantic similarity
        similarity = self._compute_similarity(original_text, adversarial_text)

        # Anomaly score is inverse of similarity
        # Low similarity = high anomaly score
        anomaly_score = 1.0 - similarity

        # Detect if similarity is below threshold
        detected = similarity < self.threshold

        return {
            "anomaly_score": float(anomaly_score),
            "detected": bool(detected),
            "explanation": f"Semantic similarity: {similarity:.3f} (threshold: {self.threshold:.3f})"
        }

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity (0-1)
        """
        try:
            # Encode both texts
            embeddings = self.encoder.encode([text1, text2])

            # Compute cosine similarity
            emb1 = embeddings[0]
            emb2 = embeddings[1]

            # Cosine similarity formula
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Ensure result is in [0, 1]
            similarity = max(0.0, min(1.0, similarity))

            return float(similarity)

        except Exception as e:
            # If computation fails, return neutral value
            return 1.0  # Assume similar to avoid false positives

