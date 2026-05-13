# Quick Start Commands

Copy-paste commands for common adversarial testing scenarios.

---

## Table of Contents

1. [Testing Your Own Models](#testing-your-own-models)
2. [Testing HuggingFace Models](#testing-huggingface-models)
3. [Custom Column Names](#custom-column-names)
4. [Different Attack Configurations](#different-attack-configurations)
5. [Data Preparation](#data-preparation)
6. [Troubleshooting](#troubleshooting)

---

## Testing Your Own Models

### Quick Test with Training Data

Test directly on train.csv with limited batches:

```bash
cd c:/Users/Hp/Desktop/Major2/adversarial-platform

# For TF-IDF model
python main_nlp.py \
    --model models/nlp/lr_tfidf_model.onnx \
    --tokenizer distilbert-base-uncased \
    --labels '{"0": "Not Disaster", "1": "Disaster"}' \
    --data models/nlp/train.csv \
    --max-batches 5

# For transformer model
python main_nlp.py \
    --model models/nlp/disaster_tweets_model.onnx \
    --tokenizer distilbert-base-uncased \
    --labels '{"0": "Not Disaster", "1": "Disaster"}' \
    --data models/nlp/train.csv \
    --max-batches 5
```

### Proper Evaluation with Train/Val Split

Split data and test on validation set (recommended):

```bash
# Step 1: Create train/validation split (run once)
python create_train_val_split.py

# Step 2: Train your model on train_80.csv
# (your training pipeline here)

# Step 3: Test adversarial robustness on validation set
python main_nlp.py \
    --model models/nlp/disaster_tweets_model.onnx \
    --tokenizer distilbert-base-uncased \
    --labels '{"0": "Not Disaster", "1": "Disaster"}' \
    --data models/nlp/validation_20.csv \
    --max-batches 10
```

### Comprehensive Evaluation

Test with more batches for thorough assessment:

```bash
python main_nlp.py \
    --model your_model.onnx \
    --tokenizer distilbert-base-uncased \
    --labels '{"0": "Class0", "1": "Class1"}' \
    --data models/nlp/validation_20.csv \
    --max-batches 50 \
    --batch-size 4
```

---

## Testing HuggingFace Models

### Sentiment Analysis Models

Test sentiment models with any sentiment dataset:

```bash
# DistilBERT SST-2 (sentiment classification)
python main_nlp.py \
    --model distilbert-base-uncased-finetuned-sst-2-english \
    --data your_sentiment_data.csv \
    --max-batches 5

# Cross-domain test (using disaster data for demo)
python main_nlp.py \
    --model distilbert-base-uncased-finetuned-sst-2-english \
    --data models/nlp/train.csv \
    --max-batches 2
```

### Disaster Classification Models

Test disaster models with disaster tweets:

```bash
python main_nlp.py \
    --model huggingface-user/disaster-classifier \
    --data models/nlp/train.csv \
    --max-batches 10
```

### Custom HuggingFace Models

Test any HF model with appropriate labeled data:

```bash
python main_nlp.py \
    --model your-username/your-model \
    --data your_labeled_data.csv \
    --max-batches 5
```

---

## Custom Column Names

### Auto-Detection

Platform auto-detects common column names:
- **Text:** text, tweet, sentence, content, comment, review, message
- **Label:** label, target, class, category, sentiment, y

```bash
# Works if columns are standard
python main_nlp.py \
    --model your_model \
    --data file.csv \
    --max-batches 5
```

### Manual Specification

Specify custom column names explicitly:

```bash
# Custom column names
python main_nlp.py \
    --model your_model.onnx \
    --tokenizer distilbert-base-uncased \
    --labels '{"0": "Class0", "1": "Class1"}' \
    --data custom_data.csv \
    --text-column message \
    --label-column classification \
    --max-batches 5
```

### Common Kaggle Format Examples

```bash
# Kaggle disaster tweets (id, keyword, location, text, target)
python main_nlp.py \
    --model your_model.onnx \
    --tokenizer distilbert-base-uncased \
    --data models/nlp/train.csv \
    --max-batches 5
# No column specification needed - auto-detects text & target

# Custom Kaggle format
python main_nlp.py \
    --model your_model \
    --data kaggle_data.csv \
    --text-column question \
    --label-column answer_label \
    --max-batches 5
```

---

## Different Attack Configurations

### Test Only TextFooler

Run only the TextFooler attack (faster):

```bash
python main_nlp.py \
    --model your_model \
    --data your_data.csv \
    --attacks textfooler \
    --max-batches 5
```

### Test Only BERT-Attack

Run only the BERT-Attack (slower but more sophisticated):

```bash
python main_nlp.py \
    --model your_model \
    --data your_data.csv \
    --attacks bertattack \
    --max-batches 3
```

### Test Both Attacks (Default)

Run both attacks in sequence:

```bash
python main_nlp.py \
    --model your_model \
    --data your_data.csv \
    --attacks textfooler bertattack \
    --max-batches 5
```

### Adjust Batch Configuration

```bash
# Smaller batches (less memory, slower)
python main_nlp.py \
    --model your_model \
    --data your_data.csv \
    --batch-size 2 \
    --max-batches 10

# Larger batches (more memory, faster)
python main_nlp.py \
    --model your_model \
    --data your_data.csv \
    --batch-size 8 \
    --max-batches 5
```

### Specify GPU/CPU

```bash
# Use CPU (safer, no GPU memory issues)
python main_nlp.py \
    --model your_model \
    --data your_data.csv \
    --device -1 \
    --max-batches 5

# Use GPU 0
python main_nlp.py \
    --model your_model \
    --data your_data.csv \
    --device 0 \
    --max-batches 5
```

---

## Data Preparation

### Create Train/Validation Split

```bash
# Run once to create train_80.csv and validation_20.csv
python create_train_val_split.py
```

**Output:**
- `models/nlp/train_80.csv` - 6,090 samples (80%)
- `models/nlp/validation_20.csv` - 1,523 samples (20%)

### Create Custom Test Dataset

```python
# create_custom_test.py
import pandas as pd

# Manual labeling
data = [
    {"text": "Earthquake hits California", "label": 1},
    {"text": "Beautiful sunny day", "label": 0},
    {"text": "Fire spreads through forest", "label": 1},
    {"text": "Great movie recommendation", "label": 0},
    # Add more samples...
]

df = pd.DataFrame(data)
df.to_csv("custom_test.csv", index=False)
print(f"Created custom_test.csv with {len(df)} samples")
```

Then test:

```bash
python main_nlp.py \
    --model your_model.onnx \
    --data custom_test.csv \
    --max-batches 2
```

### Sample Existing Dataset

```python
# sample_dataset.py
import pandas as pd

# Load large dataset
df = pd.read_csv("large_dataset.csv")

# Take random sample
sample = df.sample(n=100, random_state=42)
sample.to_csv("sample_100.csv", index=False)

print(f"Created sample with {len(sample)} samples")
print(f"Class distribution:\n{sample['label'].value_counts()}")
```

---

## Troubleshooting

### Verify Platform Works

Test with known-good model:

```bash
python diagnose_pipeline.py
```

**Expected output:** Attack success on sentiment examples.

### Check Model Functionality

Quick model sanity check:

```bash
python -c "
from model_loader import load_nlp_model

model = load_nlp_model(
    'models/nlp/disaster_tweets_model.onnx',
    tokenizer_name='distilbert-base-uncased',
    label_mapping={0: 'Not Disaster', 1: 'Disaster'}
)

print('Disaster test:', model('Earthquake destroys buildings'))
print('Non-disaster test:', model('I love sunny weather'))
"
```

**Expected:** Different predictions for disaster vs non-disaster text.

### Test with Minimal Data

Verify everything works with just 1 batch:

```bash
python main_nlp.py \
    --model your_model \
    --data your_data.csv \
    --max-batches 1
```

### Check Column Detection

See what columns the platform detects:

```bash
python main_nlp.py \
    --model your_model \
    --data your_data.csv \
    --max-batches 1 2>&1 | grep "INFO"
```

**Look for:**
```
[INFO] Using text column: 'text'
[INFO] Using label column: 'target'
```

### Fix scikit-learn Version Mismatch

If using TF-IDF models:

```bash
# Check current version
pip show scikit-learn

# Requires Python 3.9+ for newer scikit-learn
python --version

# If Python 3.8, use transformer models instead
python main_nlp.py \
    --model models/nlp/disaster_tweets_model.onnx \
    --data models/nlp/train.csv \
    --max-batches 5
```

---

## Command Templates

### Generic Template

```bash
python main_nlp.py \
    --model <MODEL_PATH_OR_HF_NAME> \
    [--tokenizer <TOKENIZER_NAME>] \
    [--labels '<JSON_LABEL_MAPPING>'] \
    --data <CSV_OR_JSON_FILE> \
    [--text-column <TEXT_COL>] \
    [--label-column <LABEL_COL>] \
    [--batch-size <INT>] \
    --max-batches <INT> \
    [--attacks textfooler bertattack] \
    [--device <-1_FOR_CPU_OR_GPU_ID>] \
    [--output <REPORTS_DIR>]
```

### Minimal Command (Auto-detect everything)

```bash
python main_nlp.py --model your_model --data file.csv --max-batches 2
```

### Full Command (Explicit everything)

```bash
python main_nlp.py \
    --model models/nlp/lr_tfidf_model.onnx \
    --tokenizer distilbert-base-uncased \
    --labels '{"0": "Not Disaster", "1": "Disaster"}' \
    --data models/nlp/train.csv \
    --text-column text \
    --label-column target \
    --batch-size 4 \
    --max-batches 10 \
    --attacks textfooler bertattack \
    --device -1 \
    --output reports \
    --seed 42
```

---

## Sample Sizes Guide

| Batches | Batch Size | Total Samples | Use Case |
|---------|-----------|---------------|----------|
| 1 | 4 | 4 | Quick verification |
| 2 | 4 | 8 | Initial testing |
| 5 | 4 | 20 | Development testing |
| 10 | 4 | 40 | Basic evaluation |
| 25 | 4 | 100 | Comprehensive testing |
| 50 | 4 | 200 | Production evaluation |
| 100 | 4 | 400 | Full assessment |

**Recommendation:** Start with `--max-batches 2`, then increase as needed.

---

## See Also

- **[ADVERSARIAL_TESTING_GUIDE.md](ADVERSARIAL_TESTING_GUIDE.md)** - Complete guide
- **[FLEXIBLE_DATA_LOADING.md](FLEXIBLE_DATA_LOADING.md)** - Column name flexibility
- **[INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md)** - Troubleshooting
- **[CLAUDE.md](../CLAUDE.md)** - Platform documentation

---

## Quick Reference Card

```bash
# Test your disaster model - Quick
python main_nlp.py --model models/nlp/lr_tfidf_model.onnx \
    --tokenizer distilbert-base-uncased \
    --labels '{"0": "Not Disaster", "1": "Disaster"}' \
    --data models/nlp/train.csv --max-batches 5

# Test your disaster model - Proper
python create_train_val_split.py
python main_nlp.py --model your_model.onnx \
    --tokenizer distilbert-base-uncased \
    --labels '{"0": "Not Disaster", "1": "Disaster"}' \
    --data models/nlp/validation_20.csv --max-batches 10

# Test HuggingFace model
python main_nlp.py \
    --model distilbert-base-uncased-finetuned-sst-2-english \
    --data your_sentiment_data.csv --max-batches 5

# Custom columns
python main_nlp.py --model your_model --data file.csv \
    --text-column message --label-column category --max-batches 5

# Verify platform works
python diagnose_pipeline.py
```

**Remember:** Always start with `--max-batches 1` or `2` for initial testing!
