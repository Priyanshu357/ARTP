"""Utility functions for NLP adversarial attacks."""

from typing import Any, List, Tuple
import re


def tokenize_text(text: str) -> List[str]:
    """Simple word tokenization.

    Args:
        text: Input text string

    Returns:
        List of word tokens
    """
    # Simple tokenization by spaces and punctuation
    words = re.findall(r'\b\w+\b', text.lower())
    return words


def compute_word_importance(
    text: str,
    model: Any,
    tokenizer: Any,
    original_label: int
) -> List[Tuple[int, float]]:
    """Compute importance score for each word by deletion method.

    Simplified importance: Try deleting each word and check if prediction changes.

    Args:
        text: Input text
        model: Target model
        tokenizer: Tokenizer for the model
        original_label: Original predicted label

    Returns:
        List of (word_index, importance_score) tuples sorted by importance
    """
    words = text.split()
    if len(words) == 0:
        return []

    # Get original prediction confidence
    try:
        original_output = _get_model_prediction(text, model, tokenizer)
        if hasattr(original_output, 'logits'):
            original_probs = original_output.logits.softmax(dim=-1)[0]
        elif isinstance(original_output, dict) and 'score' in original_output:
            original_conf = original_output['score']
        else:
            original_probs = original_output[0] if isinstance(original_output, (list, tuple)) else original_output
            if hasattr(original_probs, 'softmax'):
                original_probs = original_probs.softmax(dim=-1)

        original_conf = float(original_probs[original_label]) if hasattr(original_probs, '__getitem__') else original_conf
    except:
        # If we can't get confidence, assign uniform importance
        return [(i, 1.0 / len(words)) for i in range(len(words))]

    importance_scores = []

    for i in range(len(words)):
        # Create text with word i deleted
        modified_words = words[:i] + words[i+1:]
        modified_text = ' '.join(modified_words)

        if not modified_text.strip():
            importance_scores.append((i, 0.0))
            continue

        try:
            # Get prediction on modified text
            modified_output = _get_model_prediction(modified_text, model, tokenizer)
            if hasattr(modified_output, 'logits'):
                modified_probs = modified_output.logits.softmax(dim=-1)[0]
            elif isinstance(modified_output, dict) and 'score' in modified_output:
                modified_conf = modified_output['score']
            else:
                modified_probs = modified_output[0] if isinstance(modified_output, (list, tuple)) else modified_output
                if hasattr(modified_probs, 'softmax'):
                    modified_probs = modified_probs.softmax(dim=-1)

            modified_conf = float(modified_probs[original_label]) if hasattr(modified_probs, '__getitem__') else modified_conf

            # Importance = how much confidence drops when word is removed
            importance = abs(original_conf - modified_conf)
            importance_scores.append((i, importance))
        except:
            importance_scores.append((i, 0.0))

    # Sort by importance (descending)
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    return importance_scores


def get_synonyms_wordnet(word: str, pos: str = None) -> List[str]:
    """Get synonyms for a word using NLTK WordNet.

    Args:
        word: Word to find synonyms for
        pos: Part-of-speech tag (optional)

    Returns:
        List of synonym strings
    """
    try:
        from nltk.corpus import wordnet
        import nltk

        # Download wordnet if not available
        try:
            wordnet.synsets('test')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower() and len(synonym.split()) == 1:
                    synonyms.add(synonym)

        return list(synonyms)[:50]  # Limit to 50 synonyms
    except:
        return []


def compute_edit_distance(text1: str, text2: str) -> int:
    """Compute Levenshtein edit distance between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Edit distance (integer)
    """
    if len(text1) < len(text2):
        return compute_edit_distance(text2, text1)

    if len(text2) == 0:
        return len(text1)

    previous_row = range(len(text2) + 1)
    for i, c1 in enumerate(text1):
        current_row = [i + 1]
        for j, c2 in enumerate(text2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_word_changes(text1: str, text2: str) -> int:
    """Count number of words that changed between two texts.

    Args:
        text1: Original text
        text2: Modified text

    Returns:
        Number of word changes
    """
    words1 = text1.lower().split()
    words2 = text2.lower().split()

    # Handle different lengths
    max_len = max(len(words1), len(words2))
    min_len = min(len(words1), len(words2))

    changes = abs(len(words1) - len(words2))  # Length difference

    for i in range(min_len):
        if words1[i] != words2[i]:
            changes += 1

    return changes


def _get_model_prediction(text: str, model: Any, tokenizer: Any) -> Any:
    """Helper to get model prediction for a text.

    Handles different model interfaces (HuggingFace pipeline, raw model, etc.)

    Args:
        text: Input text
        model: Model object
        tokenizer: Tokenizer object

    Returns:
        Model output
    """
    import torch

    # Check if it's a HuggingFace pipeline
    if hasattr(model, '__call__') and hasattr(model, 'model'):
        # Pipeline interface
        output = model(text)
        return output[0] if isinstance(output, list) else output

    # Raw model interface
    if tokenizer is not None:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs

    # Fallback - try calling model directly
    return model(text)


def download_nltk_data():
    """Download required NLTK data packages."""
    import nltk

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK WordNet...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK Punkt tokenizer...")
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("Downloading NLTK POS tagger...")
        nltk.download('averaged_perceptron_tagger', quiet=True)
