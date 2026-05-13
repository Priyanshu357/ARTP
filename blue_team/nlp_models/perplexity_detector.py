"""Perplexity-based detector for adversarial NLP inputs (simplified POC implementation)."""

from typing import Any, Dict, Optional
import torch
import math

from .base_detector import BaseDetector


class PerplexityDetector(BaseDetector):
    """Detect adversarial texts using perplexity anomalies.

    Perplexity measures how well a language model predicts text.
    Adversarial text tend to have higher perplexity as they may be ungrammatical
    or unnatural due to synonym substitutions.

    Intuition: Adversarial word substitutions may create unnatural text
    with higher perplexity than genuine human-written text.
    """

    def __init__(self, perplexity_threshold: float = 200.0, lm_model: str = "distilgpt2"):
        """Initialize perplexity detector.

        Args:
            perplexity_threshold: Perplexity threshold for detection
            lm_model: Language model for perplexity computation (default: distilgpt2)
        """
        self.threshold = perplexity_threshold
        self.lm_model_name = lm_model
        self.lm_model = None
        self.lm_tokenizer = None

    def _load_lm_model(self):
        """Lazy-load language model for perplexity computation."""
        if self.lm_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading language model: {self.lm_model_name}...")
            self.lm_tokenizer = AutoTokenizer.from_pretrained(self.lm_model_name)
            self.lm_model = AutoModelForCausalLM.from_pretrained(self.lm_model_name)
            self.lm_model.eval()

            # Set pad token if not available
            if self.lm_tokenizer.pad_token is None:
                self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token

    def detect(
        self,
        original_output: Any,
        adversarial_output: Any,
        original_text: Optional[str] = None,
        adversarial_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect based on perplexity score.

        Args:
            original_output: Model output on clean text (unused)
            adversarial_output: Model output on adversarial text (unused)
            original_text: Clean text string (optional, unused)
            adversarial_text: Adversarial text string (required)

        Returns:
            Dict with anomaly_score, detected flag, and explanation
        """
        if adversarial_text is None:
            return {
                "anomaly_score": 0.0,
                "detected": False,
                "explanation": "No text provided"
            }

        # Load LM if needed
        self._load_lm_model()

        # Compute perplexity
        perplexity = self._compute_perplexity(adversarial_text)

        # Normalize anomaly score to [0, 1]
        # Higher perplexity = higher anomaly score
        anomaly_score = min(perplexity / 500.0, 1.0)  # Normalize with max 500

        # Detect if perplexity exceeds threshold
        detected = perplexity > self.threshold

        return {
            "anomaly_score": float(anomaly_score),
            "detected": bool(detected),
            "explanation": f"Perplexity: {perplexity:.2f} (threshold: {self.threshold:.2f})"
        }

    def _compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text using the language model.

        Args:
            text: Input text

        Returns:
            Perplexity value (float)
        """
        try:
            # Tokenize
            inputs = self.lm_tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )

            # Get model outputs
            with torch.no_grad():
                outputs = self.lm_model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss

            # Perplexity = exp(loss)
            perplexity = math.exp(loss.item())

            return perplexity

        except Exception as e:
            # If computation fails return a default value
            return 0.0

