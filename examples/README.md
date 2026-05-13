# NLP Examples

This directory contains example scripts showing different ways to use the NLP adversarial evaluation system.

## Note

**These are examples only!** For actual usage, use the unified entry point:
```bash
python main_nlp.py --model YOUR_MODEL
```

## Example Scripts

### example_custom_model.py
Demonstrates testing custom trained models, including ONNX models.

**Features:**
- ONNX model support with auto-detection
- Custom tokenizer and label mapping
- TextFooler attack + Perplexity and Semantic detection

**Original file:** test_my_model.py

### example_huggingface.py
Shows how to test pre-trained HuggingFace Hub models.

**Features:**
- Simple HuggingFace Hub integration
- Multiple model testing
- TextFooler attack with quick results

**Original file:** test_imported_models.py

### example_quick_test.py
Comprehensive testing suite for all attack and detector components.

**Features:**
- Tests TextFooler and BERT-Attack
- Tests all detectors (Perplexity, Semantic, Ensemble)
- Modular test functions
- Good for understanding component interactions

**Original file:** test_nlp_models.py

## Migration Guide

These examples were part of the old structure with 6 redundant test files. They have been archived here for reference but are **no longer the recommended way** to run NLP adversarial evaluation.

### Old Way (6 files):
```bash
python test_my_model.py
python test_imported_models.py
python test_nlp_models.py
python run_nlp_standalone.py
python main_nlp.py
python test_distilbert_onnx.py  # (empty)
```

### New Way (1 file):
```bash
# Just use main_nlp.py with any model format
python main_nlp.py --model MODEL_PATH

# Examples:
python main_nlp.py --model distilbert-base-uncased-finetuned-sst-2-english
python main_nlp.py --model models/nlp/my_model.onnx --tokenizer distilbert-base-uncased
python main_nlp.py --model ./my_local_model --data test_data.json
```

## When to Use These Examples

Use these examples only if you need to:
- Understand how individual components work
- See alternative implementation patterns
- Debug specific functionality
- Learn the API before customizing

For production use or frontend integration, always use `main_nlp.py`.
