"""
Helper script: How to export TF-IDF + sklearn models for use with the platform

This script demonstrates how to properly save TF-IDF models so they can be
loaded by the model_loader.py module.

IMPORTANT: You need BOTH files for TF-IDF models to work:
  1. model.onnx - The exported sklearn classifier
  2. model.pkl or vectorizer.pkl - The TF-IDF vectorizer

Example directory structure:
  models/nlp/
    ├── lr_tfidf_model.onnx      # Exported classifier
    └── lr_tfidf_model.pkl        # TF-IDF vectorizer (same base name)

  OR:

  models/nlp/
    ├── lr_tfidf_model.onnx      # Exported classifier
    └── vectorizer.pkl            # Named "vectorizer.pkl"
"""

import pickle
import numpy as np
from pathlib import Path

# =============================================================================
# STEP 1: Train sklearn model with TF-IDF (example)
# =============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Sample training data
texts = [
    "This is a disaster",
    "Emergency situation",
    "Beautiful sunny day",
    "Normal weather today"
]
labels = [1, 1, 0, 0]  # 1=disaster, 0=not_disaster

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000)

# Fit vectorizer
X_train = vectorizer.fit_transform(texts)

# Train classifier
classifier = LogisticRegression()
classifier.fit(X_train, labels)

print("✓ Model trained successfully")
print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"  Classes: {classifier.classes_}")

# =============================================================================
# STEP 2: Export to ONNX
# =============================================================================

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Define input shape (None = variable batch size, num_features)
num_features = len(vectorizer.vocabulary_)
initial_type = [('float_input', FloatTensorType([None, num_features]))]

# Convert sklearn model to ONNX
onnx_model = convert_sklearn(classifier, initial_types=initial_type)

# Save ONNX model
output_dir = Path("models/nlp")
output_dir.mkdir(parents=True, exist_ok=True)

onnx_path = output_dir / "lr_tfidf_model.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✓ ONNX model saved to: {onnx_path}")

# =============================================================================
# STEP 3: Save TF-IDF vectorizer (CRITICAL!)
# =============================================================================

# Option 1: Save with same base name as ONNX file (recommended)
vectorizer_path_option1 = onnx_path.with_suffix('.pkl')
with open(vectorizer_path_option1, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"✓ Vectorizer saved to: {vectorizer_path_option1}")

# Option 2: Save as "vectorizer.pkl" in same directory (alternative)
# vectorizer_path_option2 = output_dir / "vectorizer.pkl"
# with open(vectorizer_path_option2, 'wb') as f:
#     pickle.dump(vectorizer, f)
# print(f"✓ Vectorizer also saved to: {vectorizer_path_option2}")

# =============================================================================
# STEP 4: Verify the exports work
# =============================================================================

print("\n" + "="*70)
print("Testing exported model...")
print("="*70)

# Load and test
from model_loader import load_nlp_model

try:
    # Load model (will automatically find vectorizer)
    model = load_nlp_model(
        str(onnx_path),
        label_mapping={0: "Not Disaster", 1: "Disaster"}
    )

    # Test predictions
    test_texts = [
        "Fire spreading rapidly",
        "Nice weather today"
    ]

    predictions = model(test_texts)

    print("\nTest Results:")
    for text, pred in zip(test_texts, predictions):
        print(f"  Text: '{text}'")
        print(f"  Prediction: {pred['label']} (confidence: {pred['score']:.2f})")
        print()

    print("✓ Model loading and inference successful!")

except Exception as e:
    print(f"✗ Error: {e}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: How to use your TF-IDF model with the platform")
print("="*70)
print(f"""
1. Make sure you have both files:
   ✓ {onnx_path.name}        (ONNX classifier)
   ✓ {vectorizer_path_option1.name}  (TF-IDF vectorizer)

2. Run evaluation:

   python main_nlp.py \\
       --model models/nlp/lr_tfidf_model.onnx \\
       --labels '{{"0": "Not Disaster", "1": "Disaster"}}' \\
       --max-batches 2

3. The model_loader will automatically:
   - Detect it's a TF-IDF model (float_input)
   - Load the vectorizer from {vectorizer_path_option1.name}
   - Transform texts to TF-IDF features
   - Run ONNX inference
   - Return predictions in HuggingFace format

4. Diagnostics will work just like any other NLP model!
""")

print("="*70)
print("Dependencies needed:")
print("  pip install scikit-learn skl2onnx onnxruntime")
print("="*70)
