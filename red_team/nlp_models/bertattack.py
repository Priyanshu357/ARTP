"""BERT-based adversarial attack for NLP models (simplified POC implementation)."""

from typing import Any, Dict, List, Optional
import torch

from .base_attack import BaseAttack
from .utils import (
    compute_word_importance,
    compute_edit_distance,
    compute_word_changes
)


class BERTAttack(BaseAttack):
    """BERT-Attack: Use BERT masked language model for word substitution.

    Simplified implementation:
    1. Use BERT MLM to generate context-aware word substitutions
    2. Rank words by importance
    3. Apply substitutions greedily until prediction flips

    Reference: https://arxiv.org/abs/2004.09984
    """

    def __init__(self, bert_model: str = "bert-base-uncased", top_k: int = 10):
        """Initialize BERT-Attack.

        Args:
            bert_model: HuggingFace BERT model name
            top_k: Number of top candidate words from BERT MLM
        """
        self.bert_model_name = bert_model
        self.top_k = top_k
        self.mlm_model = None
        self.mlm_tokenizer = None

    def _load_mlm_model(self):
        """Lazy-load BERT masked language model."""
        if self.mlm_model is None:
            import warnings
            from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as hf_logging

            # Suppress noisy HuggingFace warnings during model load
            original_verbosity = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()

            print(f"Loading BERT MLM model: {self.bert_model_name}...")
            self.mlm_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.mlm_model = AutoModelForMaskedLM.from_pretrained(self.bert_model_name)
            self.mlm_model.eval()

            # Restore original logging level
            hf_logging.set_verbosity(original_verbosity)

    def generate(
        self,
        model: Any,
        texts: List[str],
        labels: List[int],
        tokenizer: Any = None
    ) -> Dict[str, Any]:
        """Generate adversarial texts using BERT-Attack.

        Args:
            model: Target NLP model (HuggingFace pipeline or model)
            texts: List of input text strings
            labels: Ground-truth labels
            tokenizer: Optional tokenizer (required for raw models)

        Returns:
            Dict containing adversarial_texts, predictions, success rate, and metrics
        """
        # Load BERT MLM model
        self._load_mlm_model()

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

        for word_idx, importance in importance_scores:
            if word_idx >= len(words):
                continue

            # Get BERT MLM suggestions for this word
            candidates = self._get_bert_candidates(adversarial_text, word_idx)

            if not candidates:
                continue

            # Try each candidate
            for candidate in candidates:
                # Create candidate text
                candidate_words = adversarial_text.split()
                if word_idx >= len(candidate_words):
                    continue

                candidate_words[word_idx] = candidate
                candidate_text = ' '.join(candidate_words)

                # Get prediction on candidate
                candidate_pred = self._get_prediction(candidate_text, model, tokenizer)
                candidate_label = self._get_label_from_pred(candidate_pred)

                # If prediction flipped, accept this perturbation
                if candidate_label != original_label:
                    adversarial_text = candidate_text
                    break  # Move to next word

            # Check if we've succeeded
            adversarial_pred = self._get_prediction(adversarial_text, model, tokenizer)
            adversarial_label = self._get_label_from_pred(adversarial_pred)
            if adversarial_label != original_label:
                break  # Attack succeeded

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

    def _get_bert_candidates(self, text: str, word_idx: int) -> List[str]:
        """Get BERT MLM candidate replacements for a word.

        Args:
            text: Input text
            word_idx: Index of word to replace

        Returns:
            List of candidate words from BERT MLM
        """
        words = text.split()
        if word_idx >= len(words):
            return []

        # Create masked text
        masked_words = words.copy()
        masked_words[word_idx] = '[MASK]'
        masked_text = ' '.join(masked_words)

        # Get BERT predictions
        try:
            inputs = self.mlm_tokenizer(masked_text, return_tensors='pt')
            with torch.no_grad():
                outputs = self.mlm_model(**inputs)

            # Find the mask token position
            mask_token_id = self.mlm_tokenizer.mask_token_id
            mask_token_index = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]

            if len(mask_token_index) == 0:
                return []

            # Get top-k predictions
            logits = outputs.logits[0, mask_token_index[0]]
            top_k_ids = torch.topk(logits, self.top_k).indices.tolist()

            # Convert to words
            candidates = []
            for token_id in top_k_ids:
                candidate = self.mlm_tokenizer.decode([token_id]).strip()
                # Filter out special tokens and multi-word candidates
                if candidate and not candidate.startswith('[') and ' ' not in candidate:
                    candidates.append(candidate)

            return candidates[:self.top_k]
        except:
            return []

    def _get_prediction(self, text: str, model: Any, tokenizer: Any) -> Any:
        """Get model prediction for a text."""
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
        """Extract predicted label from model output."""
        # HuggingFace pipeline output
        if isinstance(pred, dict) and 'label' in pred:
            label_str = pred['label']
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
