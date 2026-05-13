"""TextFooler attack for NLP models (simplified POC implementation)."""

from typing import Any, Dict, List, Optional
import torch

from .base_attack import BaseAttack
from .utils import (
    compute_word_importance,
    get_synonyms_wordnet,
    compute_edit_distance,
    compute_word_changes,
    download_nltk_data
)


class TextFoolerAttack(BaseAttack):
    """TextFooler: word substitution attack using WordNet synonyms.

    Simplified implementation:
    1. Identify important words using deletion-based importance ranking
    2. Find synonyms using NLTK WordNet (simplified from counter-fitted embeddings)
    3. Replace words greedily until prediction flips

    Reference: https://arxiv.org/abs/1907.11932
    """

    def __init__(self, max_candidates: int = 10, max_perturbed_words: Optional[int] = None):
        """Initialize TextFooler attack.

        Args:
            max_candidates: Maximum synonym candidates per word
            max_perturbed_words: Maximum words to perturb (None = unlimited)
        """
        self.max_candidates = max_candidates
        self.max_perturbed_words = max_perturbed_words

        # Download NLTK data if needed
        download_nltk_data()

    def generate(
        self,
        model: Any,
        texts: List[str],
        labels: List[int],
        tokenizer: Any = None
    ) -> Dict[str, Any]:
        """Generate adversarial texts using TextFooler algorithm.

        Args:
            model: Target NLP model (HuggingFace pipeline or model)
            texts: List of input text strings
            labels: Ground-truth labels
            tokenizer: Optional tokenizer (required for raw models)

        Returns:
            Dict containing adversarial_texts, predictions, success rate, and metrics
        """
        adversarial_texts = []
        original_predictions = []
        adversarial_predictions = []
        successful_attacks = 0
        total_edit_distances = []
        total_word_changes = []

        for text, label in zip(texts, labels):
            result = self._attack_single_text(text, label, model, tokenizer)

            adversarial_texts.append(result['adversarial_text'])
            original_predictions.append(result['original_pred'])
            adversarial_predictions.append(result['adversarial_pred'])

            if result['success']:
                successful_attacks += 1

            total_edit_distances.append(result['edit_distance'])
            total_word_changes.append(result['word_changes'])

        attack_success_rate = successful_attacks / len(texts) if texts else 0.0

        return {
            "adversarial_texts": adversarial_texts,
            "original_predictions": original_predictions,
            "adversarial_predictions": adversarial_predictions,
            "attack_success_rate": attack_success_rate,
            "perturbation_metrics": {
                "avg_edit_distance": sum(total_edit_distances) / len(total_edit_distances) if total_edit_distances else 0,
                "avg_word_changes": sum(total_word_changes) / len(total_word_changes) if total_word_changes else 0,
                "min_changes": min(total_word_changes) if total_word_changes else 0,
                "max_changes": max(total_word_changes) if total_word_changes else 0
            }
        }

    def _attack_single_text(
        self,
        text: str,
        label: int,
        model: Any,
        tokenizer: Any
    ) -> Dict[str, Any]:
        """Attack a single text example.

        Args:
            text: Input text
            label: Ground-truth label
            model: Target model
            tokenizer: Tokenizer

        Returns:
            Dict with attack results for this text
        """
        # Get original prediction
        original_pred = self._get_prediction(text, model, tokenizer)
        original_label = self._get_label_from_pred(original_pred)

        # If already misclassified, no attack needed
        if original_label != label:
            return {
                'adversarial_text': text,
                'original_pred': original_pred,
                'adversarial_pred': original_pred,
                'success': False,
                'edit_distance': 0,
                'word_changes': 0
            }

        # Compute word importance
        words = text.split()
        importance_scores = compute_word_importance(text, model, tokenizer, original_label)

        # Try to perturb important words
        adversarial_text = text
        words_perturbed = 0
        max_words = self.max_perturbed_words if self.max_perturbed_words else len(words)

        for word_idx, importance in importance_scores:
            if words_perturbed >= max_words:
                break

            if word_idx >= len(words):
                continue

            word = words[word_idx]

            # Get synonyms
            synonyms = get_synonyms_wordnet(word)[:self.max_candidates]

            if not synonyms:
                continue

            # Try each synonym
            for synonym in synonyms:
                # Create candidate text
                candidate_words = adversarial_text.split()
                if word_idx >= len(candidate_words):
                    continue

                candidate_words[word_idx] = synonym
                candidate_text = ' '.join(candidate_words)

                # Get prediction on candidate
                candidate_pred = self._get_prediction(candidate_text, model, tokenizer)
                candidate_label = self._get_label_from_pred(candidate_pred)

                # If prediction flipped, accept this perturbation
                if candidate_label != original_label:
                    adversarial_text = candidate_text
                    words_perturbed += 1
                    break  # Move to next word

        # Get final adversarial prediction
        adversarial_pred = self._get_prediction(adversarial_text, model, tokenizer)
        adversarial_label = self._get_label_from_pred(adversarial_pred)

        success = (adversarial_label != original_label)
        edit_dist = compute_edit_distance(text, adversarial_text)
        word_changes_count = compute_word_changes(text, adversarial_text)

        return {
            'adversarial_text': adversarial_text,
            'original_pred': original_pred,
            'adversarial_pred': adversarial_pred,
            'success': success,
            'edit_distance': edit_dist,
            'word_changes': word_changes_count
        }

    def _get_prediction(self, text: str, model: Any, tokenizer: Any) -> Any:
        """Get model prediction for a text.

        Args:
            text: Input text
            model: Model object
            tokenizer: Tokenizer object

        Returns:
            Model prediction output
        """
        # Check if it's a HuggingFace pipeline
        if hasattr(model, '__call__') and hasattr(model, 'model'):
            output = model(text)
            return output[0] if isinstance(output, list) else output

        # Raw model interface
        if tokenizer is not None:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs

        # Fallback
        return model(text)

    def _get_label_from_pred(self, pred: Any) -> int:
        """Extract predicted label from model output.

        Args:
            pred: Model prediction output

        Returns:
            Predicted label (integer)
        """
        # HuggingFace pipeline output
        if isinstance(pred, dict) and 'label' in pred:
            label_str = pred['label']
            # Handle labels like "LABEL_0", "LABEL_1", "POSITIVE", "NEGATIVE"
            if label_str.startswith('LABEL_'):
                return int(label_str.split('_')[1])
            elif label_str == 'POSITIVE':
                return 1
            elif label_str == 'NEGATIVE':
                return 0
            else:
                return 0

        # Raw logits
        if hasattr(pred, 'logits'):
            return int(torch.argmax(pred.logits, dim=-1)[0])

        # Probabilities
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        if hasattr(pred, 'argmax'):
            return int(pred.argmax())

        return 0
