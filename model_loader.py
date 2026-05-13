"""Universal NLP model loader with format auto-detection.

This module provides automatic detection and loading of NLP models in various formats:
- ONNX models (.onnx files)
- HuggingFace Hub models (model names like "distilbert-base-uncased")
- Local HuggingFace models (directories with config.json)
- PyTorch checkpoints (.pth, .pt, .bin files)

Example Usage:
    # Load from HuggingFace Hub
    model = load_nlp_model("distilbert-base-uncased-finetuned-sst-2-english")

    # Load ONNX model
    model = load_nlp_model(
        "models/nlp/model.onnx",
        tokenizer_name="distilbert-base-uncased",
        label_mapping={0: 'NEGATIVE', 1: 'POSITIVE'}
    )

    # Load local HuggingFace model
    model = load_nlp_model("./my_finetuned_model")
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os
import sys

import torch
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# NumPy compatibility fix for pickle files created with NumPy 2.0+
# NumPy 2.0 reorganized internal modules (_core), but older versions use 'core'
# This allows loading pickle files across different NumPy versions
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = sys.modules['numpy.core']
    sys.modules['numpy._core.multiarray'] = sys.modules['numpy.core.multiarray']
    sys.modules['numpy._core.umath'] = sys.modules['numpy.core']


class ModelFormat(Enum):
    """Supported model formats for NLP models."""
    ONNX = "onnx"
    HUGGINGFACE_HUB = "huggingface_hub"
    HUGGINGFACE_LOCAL = "huggingface_local"
    PYTORCH_LOCAL = "pytorch_local"
    UNKNOWN = "unknown"


class TFIDFONNXModelWrapper:
    """Wrapper for TF-IDF + sklearn models exported to ONNX.

    This handles models where text is converted to TF-IDF features first,
    then fed to a classifier (e.g., LogisticRegression, SVM).

    The ONNX model expects 'float_input' (TF-IDF features) instead of
    'input_ids' and 'attention_mask' (transformer inputs).

    Attributes:
        session: ONNX Runtime inference session
        vectorizer: Loaded TF-IDF vectorizer (from pickle file)
        label_mapping: Mapping from label indices to label names
    """

    def __init__(
        self,
        onnx_path: str,
        vectorizer_path: Optional[str] = None,
        label_mapping: Optional[Dict[int, str]] = None,
        device: str = "cpu"
    ):
        """Initialize TF-IDF ONNX model wrapper.

        Args:
            onnx_path: Path to .onnx model file
            vectorizer_path: Path to pickled TF-IDF vectorizer (.pkl)
                            If None, looks for file with same name as ONNX but .pkl extension
            label_mapping: Label index to name mapping
            device: Device for inference ('cpu' or 'cuda')

        Raises:
            ImportError: If onnxruntime is not installed
            FileNotFoundError: If vectorizer file not found
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime required for ONNX models. "
                "Install with: pip install onnxruntime"
            )

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        print(f"[TFIDFONNXWrapper] Loading TF-IDF + sklearn ONNX model from: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path)

        # Auto-detect vectorizer path if not provided
        if vectorizer_path is None:
            # Look for .pkl file with same base name
            base_path = Path(onnx_path).with_suffix('.pkl')
            model_dir = Path(onnx_path).parent

            possible_paths = [
                "models/nlp/tfidf_vectorizer_patched.pkl",
                str(base_path),
                str(model_dir / 'vectorizer.pkl'),
                str(model_dir.parent / 'models' / 'lr_tfidf_model.pkl'),
                str(model_dir.parent / 'models' / 'vectorizer.pkl'),
                "models/lr_tfidf_model.pkl",
                "models/vectorizer.pkl",
                "models/nlp/tfidf_vectorizer.pkl",
                "models/nlp/vectorizer.pkl"
            ]

            for path in possible_paths:
                if Path(path).exists():
                    vectorizer_path = path
                    break

            if vectorizer_path is None:
                raise FileNotFoundError(
                    f"TF-IDF vectorizer not found. Checked multiple locations including models/nlp/.\n"
                    f"Please provide vectorizer_path parameter or save the vectorizer to one of these locations."
                )
        # Load TF-IDF vectorizer
        print(f"[TFIDFONNXWrapper] Loading TF-IDF vectorizer from: {vectorizer_path}")
        import pickle
        
        class RenameUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith('numpy._core'):
                    module = module.replace('numpy._core', 'numpy.core')
                return super().find_class(module, name)
                
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = RenameUnpickler(f).load()

        self.label_mapping = label_mapping or {0: 'LABEL_0', 1: 'LABEL_1'}
        self.device = device
        self.model = self  # For compatibility

        # Initialize preprocessing components for TF-IDF models
        # Download NLTK data if not present
        try:
            nltk.data.find('corpora/stopwords.zip')
        except LookupError:
            print("[TFIDFONNXWrapper] Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("[TFIDFONNXWrapper] Downloading NLTK punkt...")
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("[TFIDFONNXWrapper] Downloading NLTK POS tagger...")
            nltk.download('averaged_perceptron_tagger', quiet=True)
        try:
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            print("[TFIDFONNXWrapper] Downloading NLTK wordnet...")
            nltk.download('wordnet', quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        print(f"[TFIDFONNXWrapper] Model loaded successfully")
        print(f"[TFIDFONNXWrapper]   Input names: {[inp.name for inp in self.session.get_inputs()]}")
        print(f"[TFIDFONNXWrapper]   Output names: {[out.name for out in self.session.get_outputs()]}")
        print(f"[TFIDFONNXWrapper]   Labels: {self.label_mapping}")
        print(f"[TFIDFONNXWrapper]   Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing: lowercase, remove HTML, punctuation, digits.

        This matches the preprocessing pipeline from the training notebook.
        """
        text = text.lower()
        text = text.strip()
        text = re.compile('<.*?>').sub('', text)  # Remove HTML tags
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'\[[0-9]*\]', ' ', text)  # Remove [numbers]
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())  # Remove non-alphanumeric
        text = re.sub(r'\d', ' ', text)  # Remove digits
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        return text.strip()

    def _remove_stopwords(self, text: str) -> str:
        """Remove English stopwords."""
        words = [word for word in text.split() if word not in self.stop_words]
        return ' '.join(words)

    def _get_wordnet_pos(self, tag: str) -> str:
        """Map NLTK POS tag to WordNet POS tag."""
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def _lemmatize_text(self, text: str) -> str:
        """Lemmatize text using POS tags."""
        word_pos_tags = pos_tag(word_tokenize(text))
        lemmatized = [
            self.lemmatizer.lemmatize(tag[0], self._get_wordnet_pos(tag[1]))
            for tag in word_pos_tags
        ]
        return " ".join(lemmatized)

    def _final_preprocess(self, text: str) -> str:
        """Complete preprocessing pipeline matching training notebook.

        Applies: lowercase, remove punctuation/HTML/digits, remove stopwords, lemmatization.
        This ensures the text is preprocessed the same way as during training.
        """
        text = self._preprocess_text(text)
        text = self._remove_stopwords(text)
        text = self._lemmatize_text(text)
        return text

    def __call__(self, texts: Union[str, List[str]], **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Run inference matching HuggingFace pipeline output format.

        Args:
            texts: Single text string or list of text strings
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Single prediction dict if input was a string, list of dicts otherwise
        """
        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Preprocess texts before TF-IDF transformation
        # This matches the preprocessing pipeline from the training notebook
        preprocessed_texts = [self._final_preprocess(text) for text in texts]

        # Transform preprocessed texts to TF-IDF features
        tfidf_features = self.vectorizer.transform(preprocessed_texts).toarray().astype(np.float32)

        # Get input name from ONNX model (usually 'float_input')
        input_name = self.session.get_inputs()[0].name

        # Run ONNX inference
        onnx_inputs = {input_name: tfidf_features}
        outputs = self.session.run(None, onnx_inputs)

        # Handle different output formats
        # Some sklearn models output (label, probability), others just probability
        if len(outputs) == 2:
            # Format: (output_label, output_probability)
            # Convert to numpy arrays in case ONNX returns lists
            pred_labels = np.array(outputs[0])
            probs_raw = outputs[1]

            # Handle dict-based probability output (sklearn ONNX format)
            if isinstance(probs_raw, (list, np.ndarray)) and len(probs_raw) > 0:
                # Check if first element is a dict
                first_elem = probs_raw[0] if isinstance(probs_raw, (list, np.ndarray)) else probs_raw
                if isinstance(first_elem, dict):
                    # Convert dict format to array: [{'0': 0.6, '1': 0.4}] -> [[0.6, 0.4]]
                    num_classes = len(first_elem)
                    probs = np.zeros((len(probs_raw), num_classes))
                    for i, prob_dict in enumerate(probs_raw):
                        for class_idx, prob_val in prob_dict.items():
                            probs[i, int(class_idx)] = float(prob_val)
                else:
                    # Regular array format
                    probs = np.array(probs_raw)
            else:
                probs = np.array(probs_raw)
        else:
            # Format: just logits/probabilities
            logits = np.array(outputs[0])

            # If logits, apply softmax
            if logits.dtype == np.float32 or logits.dtype == np.float64:
                # Check if already probabilities (sum to 1) or logits
                row_sums = np.abs(np.sum(logits, axis=-1) - 1.0)
                if np.any(row_sums > 0.01):  # Not probabilities, apply softmax
                    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                else:
                    probs = logits
            else:
                probs = logits

            pred_labels = np.argmax(probs, axis=-1)

        # Flatten if needed
        if pred_labels.ndim > 1:
            pred_labels = pred_labels.flatten()
        if probs.ndim > 2:
            probs = probs.reshape(-1, probs.shape[-1])

        # Format results like HuggingFace pipeline
        results = []
        for i, label_idx in enumerate(pred_labels):
            # Convert to Python int
            if isinstance(label_idx, np.ndarray):
                label_idx_int = int(label_idx.item())
            else:
                label_idx_int = int(label_idx)

            # Get probability score
            if probs.ndim > 1 and probs.shape[1] > label_idx_int:
                # Probability array for multiple classes
                prob_score = float(probs[i][label_idx_int])
            elif probs.ndim == 1:
                # Single probability value
                prob_score = float(probs[i])
            else:
                prob_score = 1.0  # Fallback

            results.append({
                'label': self.label_mapping.get(label_idx_int, f'LABEL_{label_idx_int}'),
                'score': prob_score
            })

        return results[0] if single_input else results


class ONNXModelWrapper:
    """Wrapper to make ONNX models compatible with HuggingFace pipeline interface.

    This class loads ONNX NLP models and provides a __call__() interface that
    matches HuggingFace pipelines, returning predictions in the same format:
    [{'label': 'POSITIVE', 'score': 0.98}, ...]

    Attributes:
        session: ONNX Runtime inference session
        tokenizer: HuggingFace tokenizer for text preprocessing
        label_mapping: Mapping from label indices to label names
        device: Device for inference ('cpu' or 'cuda')
    """

    def __init__(
        self,
        onnx_path: str,
        tokenizer_name: str,
        label_mapping: Optional[Dict[int, str]] = None,
        device: str = "cpu"
    ):
        """Initialize ONNX model wrapper.

        Args:
            onnx_path: Path to .onnx model file
            tokenizer_name: HuggingFace tokenizer identifier (e.g., 'distilbert-base-uncased')
            label_mapping: Label index to name mapping (e.g., {0: 'NEGATIVE', 1: 'POSITIVE'})
                          If None, uses generic labels like 'LABEL_0', 'LABEL_1'
            device: Device for inference ('cpu' or 'cuda')

        Raises:
            ImportError: If onnxruntime is not installed
            FileNotFoundError: If onnx_path does not exist
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime required for ONNX models. "
                "Install with: pip install onnxruntime"
            )

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        print(f"[ONNXModelWrapper] Loading ONNX model from: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path)

        print(f"[ONNXModelWrapper] Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.label_mapping = label_mapping or {0: 'LABEL_0', 1: 'LABEL_1'}
        self.device = device

        # Mimic HuggingFace pipeline attributes for compatibility checks
        self.model = self

        print(f"[ONNXModelWrapper] Model loaded successfully")
        print(f"[ONNXModelWrapper]   Input names: {[inp.name for inp in self.session.get_inputs()]}")
        print(f"[ONNXModelWrapper]   Output names: {[out.name for out in self.session.get_outputs()]}")
        print(f"[ONNXModelWrapper]   Labels: {self.label_mapping}")

    def __call__(self, texts: Union[str, List[str]], **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Run inference matching HuggingFace pipeline output format.

        Args:
            texts: Single text string or list of text strings
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Single prediction dict if input was a string:
                {'label': 'POSITIVE', 'score': 0.98}
            List of prediction dicts if input was a list:
                [{'label': 'POSITIVE', 'score': 0.98}, ...]
        """
        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Tokenize inputs
        # Check if model expects fixed sequence length
        input_shapes = [inp.shape for inp in self.session.get_inputs()]
        max_length = 512  # Default

        # Try to detect fixed sequence length from model inputs
        for shape in input_shapes:
            if len(shape) >= 2 and shape[1] is not None and isinstance(shape[1], int):
                max_length = shape[1]
                break

        inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='np'  # Return numpy arrays for ONNX
        )

        # Prepare ONNX inputs (handle different input name conventions)
        onnx_inputs = {}
        input_names = [inp.name for inp in self.session.get_inputs()]
        input_types = {inp.name: inp.type for inp in self.session.get_inputs()}

        for name in input_names:
            if name == 'input_ids' and 'input_ids' in inputs:
                # Convert to the expected type based on model requirements
                if 'int64' in input_types.get(name, ''):
                    onnx_inputs[name] = inputs['input_ids'].astype(np.int64)
                elif 'float' in input_types.get(name, ''):
                    onnx_inputs[name] = inputs['input_ids'].astype(np.float32)
                else:
                    onnx_inputs[name] = inputs['input_ids']
            elif name == 'attention_mask' and 'attention_mask' in inputs:
                if 'int64' in input_types.get(name, ''):
                    onnx_inputs[name] = inputs['attention_mask'].astype(np.int64)
                elif 'float' in input_types.get(name, ''):
                    onnx_inputs[name] = inputs['attention_mask'].astype(np.float32)
                else:
                    onnx_inputs[name] = inputs['attention_mask']
            elif name == 'token_type_ids' and 'token_type_ids' in inputs:
                if 'int64' in input_types.get(name, ''):
                    onnx_inputs[name] = inputs['token_type_ids'].astype(np.int64)
                elif 'float' in input_types.get(name, ''):
                    onnx_inputs[name] = inputs['token_type_ids'].astype(np.float32)
                else:
                    onnx_inputs[name] = inputs['token_type_ids']

        # Run ONNX inference
        outputs = self.session.run(None, onnx_inputs)
        logits = outputs[0]  # Assuming first output is logits

        # Convert logits to probabilities (softmax)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Get predicted labels
        pred_labels = np.argmax(logits, axis=-1)

        # Flatten if needed (handle cases where pred_labels might have extra dimensions)
        if pred_labels.ndim > 1:
            pred_labels = pred_labels.flatten()

        # Also reshape probs if needed to match
        if probs.ndim > 2:
            # Reshape to (batch_size, num_classes)
            probs = probs.reshape(-1, probs.shape[-1])

        # Format results like HuggingFace pipeline
        results = []
        for i, label_idx in enumerate(pred_labels):
            # Convert numpy scalar to Python int
            # Handle both 0-d arrays and numpy scalars
            if isinstance(label_idx, np.ndarray):
                label_idx_int = int(label_idx.item())
            else:
                label_idx_int = int(label_idx)

            # Get probability for this prediction
            if probs.ndim > 1:
                prob_score = float(probs[i][label_idx_int])
            else:
                prob_score = float(probs[label_idx_int])

            results.append({
                'label': self.label_mapping.get(label_idx_int, f'LABEL_{label_idx_int}'),
                'score': prob_score
            })

        # Return single result if single input, otherwise return list
        return results[0] if single_input else results


class UniversalModelLoader:
    """Universal loader for NLP models with format auto-detection.

    This class automatically detects the format of NLP models and loads them
    with the appropriate loader, providing a unified interface.
    """

    @staticmethod
    def detect_format(model_path: str) -> ModelFormat:
        """Auto-detect model format from path/name.

        Detection logic:
        1. Check file extension (.onnx, .pth, .pt, .bin)
        2. Check if directory contains config.json (HuggingFace local)
        3. Assume HuggingFace Hub name if no file exists

        Args:
            model_path: Path to model file/directory or HuggingFace Hub name

        Returns:
            ModelFormat enum indicating the detected format
        """
        # Check if it's a file
        if os.path.isfile(model_path):
            ext = Path(model_path).suffix.lower()
            if ext == '.onnx':
                return ModelFormat.ONNX
            elif ext in ['.pth', '.pt', '.bin']:
                return ModelFormat.PYTORCH_LOCAL

        # Check if it's a directory with config.json (HuggingFace local)
        if os.path.isdir(model_path):
            if (Path(model_path) / 'config.json').exists():
                return ModelFormat.HUGGINGFACE_LOCAL

        # Otherwise assume it's a HuggingFace Hub name
        if not os.path.exists(model_path):
            return ModelFormat.HUGGINGFACE_HUB

        return ModelFormat.UNKNOWN

    @staticmethod
    def load(
        model_path: str,
        task: str = "text-classification",
        tokenizer_name: Optional[str] = None,
        label_mapping: Optional[Dict[int, str]] = None,
        device: int = -1,
        **kwargs
    ) -> Any:
        """Load model with automatic format detection.

        This method detects the model format and loads it with the appropriate
        loader, returning a model with HuggingFace pipeline-compatible interface.

        Args:
            model_path: Path to model file/directory or HuggingFace Hub name
            task: NLP task type (text-classification, sentiment-analysis, etc.)
            tokenizer_name: Tokenizer name (required for ONNX, optional otherwise)
            label_mapping: Label mapping dict for ONNX models (e.g., {0: 'NEGATIVE', 1: 'POSITIVE'})
            device: Device ID (-1 for CPU, 0+ for GPU)
            **kwargs: Additional parameters for model loading

        Returns:
            Model with HuggingFace pipeline-compatible interface

        Raises:
            ValueError: If model format cannot be detected or tokenizer_name is missing for ONNX
            ImportError: If required packages are not installed

        Example:
            # Load HuggingFace Hub model
            model = UniversalModelLoader.load("distilbert-base-uncased-finetuned-sst-2-english")

            # Load ONNX model
            model = UniversalModelLoader.load(
                "models/nlp/model.onnx",
                tokenizer_name="distilbert-base-uncased",
                label_mapping={0: 'NEGATIVE', 1: 'POSITIVE'}
            )
        """
        format_type = UniversalModelLoader.detect_format(model_path)

        print(f"[ModelLoader] Detected format: {format_type.value}")
        print(f"[ModelLoader] Model path: {model_path}")

        if format_type == ModelFormat.ONNX:
            # Detect ONNX model type by checking input names
            import onnxruntime as ort
            try:
                temp_session = ort.InferenceSession(model_path)
                input_names = [inp.name for inp in temp_session.get_inputs()]

                # Check if it's a TF-IDF model (expects float_input) or transformer model
                is_tfidf_model = 'float_input' in input_names
                is_transformer_model = 'input_ids' in input_names or 'attention_mask' in input_names

                if is_tfidf_model:
                    # TF-IDF + sklearn model
                    print(f"[ModelLoader] Detected TF-IDF + sklearn ONNX model")

                    vectorizer_path = kwargs.get('vectorizer_path', None)

                    return TFIDFONNXModelWrapper(
                        onnx_path=model_path,
                        vectorizer_path=vectorizer_path,
                        label_mapping=label_mapping,
                        device="cpu" if device == -1 else f"cuda:{device}"
                    )

                elif is_transformer_model:
                    # Transformer model (BERT, DistilBERT, etc.)
                    print(f"[ModelLoader] Detected transformer ONNX model")

                    if tokenizer_name is None:
                        raise ValueError(
                            "tokenizer_name is required for transformer ONNX models. "
                            "Example: load(..., tokenizer_name='distilbert-base-uncased')"
                        )

                    return ONNXModelWrapper(
                        onnx_path=model_path,
                        tokenizer_name=tokenizer_name,
                        label_mapping=label_mapping,
                        device="cpu" if device == -1 else f"cuda:{device}"
                    )

                else:
                    raise ValueError(
                        f"Unknown ONNX model type. Input names: {input_names}\n"
                        "Expected either:\n"
                        "  - Transformer model: input_ids, attention_mask\n"
                        "  - TF-IDF model: float_input"
                    )

            except Exception as e:
                # If detection fails, fall back to original behavior
                print(f"[ModelLoader] Warning: Could not detect ONNX model type: {e}")
                print(f"[ModelLoader] Assuming transformer model")

                if tokenizer_name is None:
                    raise ValueError(
                        "tokenizer_name is required for ONNX models. "
                        "Example: load(..., tokenizer_name='distilbert-base-uncased')"
                    )

                return ONNXModelWrapper(
                    onnx_path=model_path,
                    tokenizer_name=tokenizer_name,
                    label_mapping=label_mapping,
                    device="cpu" if device == -1 else f"cuda:{device}"
                )

        elif format_type == ModelFormat.HUGGINGFACE_HUB:
            print(f"[ModelLoader] Loading from HuggingFace Hub...")
            return pipeline(task, model=model_path, device=device, **kwargs)

        elif format_type == ModelFormat.HUGGINGFACE_LOCAL:
            print(f"[ModelLoader] Loading local HuggingFace model...")
            return pipeline(task, model=model_path, device=device, **kwargs)

        elif format_type == ModelFormat.PYTORCH_LOCAL:
            # PyTorch checkpoint loading requires model architecture specification
            # This is complex and varies by model, so we provide a helpful error
            raise NotImplementedError(
                f"PyTorch checkpoint loading not yet implemented for: {model_path}\n"
                "Recommended approaches:\n"
                "1. Convert to HuggingFace format with model.save_pretrained()\n"
                "2. Save as ONNX format with torch.onnx.export()\n"
                "3. Provide a custom model wrapper class"
            )

        else:
            raise ValueError(
                f"Unknown model format for path: {model_path}\n"
                "Supported formats: .onnx, HuggingFace Hub names, HuggingFace local directories"
            )


def load_nlp_model(model_path: str, **kwargs) -> Any:
    """Convenience function to load any NLP model format.

    This is a simple wrapper around UniversalModelLoader.load() for
    easier usage. Automatically detects and loads ONNX, HuggingFace Hub,
    or local HuggingFace models.

    Args:
        model_path: Path to model or HuggingFace Hub name
        **kwargs: Additional arguments passed to UniversalModelLoader.load()

    Returns:
        Model with HuggingFace pipeline-compatible interface

    Examples:
        # HuggingFace Hub model (simplest)
        model = load_nlp_model("distilbert-base-uncased-finetuned-sst-2-english")
        results = model(["This is great!"])

        # Local ONNX model
        model = load_nlp_model(
            "models/nlp/distilbert.onnx",
            tokenizer_name="distilbert-base-uncased",
            label_mapping={0: 'NEGATIVE', 1: 'POSITIVE'}
        )
        results = model(["I love this!"])

        # Local HuggingFace model
        model = load_nlp_model("./my_finetuned_model")
        results = model(["Amazing product!"])
    """
    return UniversalModelLoader.load(model_path, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Universal NLP Model Loader - Example Usage")
    print("=" * 80)

    # Example 1: Detect format
    print("\nExample 1: Format Detection")
    print("-" * 40)
    test_paths = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "models/nlp/model.onnx",
        "./my_model",
        "checkpoint.pth"
    ]
    for path in test_paths:
        fmt = UniversalModelLoader.detect_format(path)
        print(f"  {path:<50} → {fmt.value}")

    # Example 2: Load HuggingFace model (if available)
    print("\nExample 2: Load HuggingFace Model")
    print("-" * 40)
    try:
        model = load_nlp_model("distilbert-base-uncased-finetuned-sst-2-english")
        test_text = "This movie is absolutely fantastic!"
        result = model(test_text)
        print(f"  Text: {test_text}")
        print(f"  Prediction: {result}")
    except Exception as e:
        print(f"  Skipped: {e}")

    print("\n" + "=" * 80)
    print("✓ Model loader module loaded successfully")
