# Adversarial Robustness Testing Guide

Complete guide for testing model robustness against adversarial attacks.

---

## Table of Contents

1. [Overview](#overview)
2. [Why Labels Are Required](#why-labels-are-required)
3. [Scenario A: Testing Your Own Models](#scenario-a-testing-your-own-models)
4. [Scenario B: Testing HuggingFace Models](#scenario-b-testing-huggingface-models)
5. [Data Requirements](#data-requirements)
6. [Understanding Results](#understanding-results)
7. [Common Issues and Solutions](#common-issues-and-solutions)
8. [Best Practices](#best-practices)

---

## Overview

### What is Adversarial Robustness Testing?

Adversarial robustness testing evaluates how easily a model can be fooled by **small, intentional perturbations** to inputs:

```
Original Input:  "Earthquake destroys buildings"
               ↓ Slight change
Perturbed Input: "Tremor destroys buildings"
               ↓
Model Changes Prediction: Disaster → Not Disaster ❌
```

**Purpose:** Identify vulnerabilities before deployment to ensure models are reliable in adversarial conditions.

### How It Differs from Traditional Testing

| Traditional Testing | Adversarial Testing |
|-------------------|-------------------|
| Predict on unlabeled data | Evaluate robustness on labeled data |
| Goal: Get predictions | Goal: Measure how easily model is fooled |
| No labels needed | **Labels required** |
| Example: Kaggle test.csv | Example: Validation set with labels |

---

## Why Labels Are Required

You might think: *"We're just changing inputs slightly and checking if predictions change - why do we need labels?"*

### The Critical Reason

**You need labels to know if an attack actually succeeded.**

### Example 1: Without Labels - Ambiguous Results

```python
Original: "Earthquake hits California"
Model prediction: Disaster

Perturbed: "Tremor hits California"
Model prediction: Not Disaster

❓ Question: Is this a successful attack?
- Prediction changed ✓
- But was the model correct originally? Unknown!
- Maybe it was already wrong? Unknown!
```

### Example 2: With Labels - Clear Success Metric

```python
Original: "Earthquake hits California"
Ground truth: Disaster (label=1)
Model prediction: Disaster ✓ CORRECT

Perturbed: "Tremor hits California"
Ground truth: Still Disaster (label=1)
Model prediction: Not Disaster ✗ WRONG

✅ Attack SUCCESS! Model fooled from correct → wrong
```

### Attack Success Rate Formula

```
ASR = (Samples correctly classified but fooled) / (Total correctly classified samples)
```

**Without labels:** Can't identify "correctly classified" samples
**With labels:** Clear measurement of robustness

---

## Scenario A: Testing Your Own Models

When you have trained your own models, you have access to the training data with labels.

### Option 1: Quick Test (Simplest)

Use the full training dataset with limited batches:

```bash
python main_nlp.py \
    --model models/nlp/lr_tfidf_model.onnx \
    --tokenizer distilbert-base-uncased \
    --labels '{"0": "Not Disaster", "1": "Disaster"}' \
    --data models/nlp/train.csv \
    --max-batches 5
```

**Pros:**
- ✓ Works immediately
- ✓ No additional setup
- ✓ Good for initial testing

**Cons:**
- ⚠️ Tests on data model was trained on
- ⚠️ May overestimate robustness

### Option 2: Proper Evaluation (Recommended)

Split training data into train/validation sets:

**Step 1: Create Split**
```bash
python create_train_val_split.py
```

This creates:
- `train_80.csv` - 80% for training (6,090 samples)
- `validation_20.csv` - 20% for testing (1,523 samples)

**Step 2: Train Your Model**
```bash
# Train on train_80.csv (your training pipeline)
python your_training_script.py --data models/nlp/train_80.csv
```

**Step 3: Test Adversarial Robustness**
```bash
python main_nlp.py \
    --model your_trained_model.onnx \
    --tokenizer distilbert-base-uncased \
    --labels '{"0": "Not Disaster", "1": "Disaster"}' \
    --data models/nlp/validation_20.csv \
    --max-batches 10
```

**Pros:**
- ✓ Tests on unseen data
- ✓ Realistic robustness estimates
- ✓ Standard ML practice

### Expected Output

```
[Results Summary]
  Attack Success Rate:  23.5%
  Detection Accuracy:   67.2%
  False Positive Rate:  12.3%
  Robustness Score:     71.4/100

[Diagnostics] Model Health: GOOD
```

---

## Scenario B: Testing HuggingFace Models

### Key Insight

**You don't need the original training data!**

Any labeled data in the **same domain** works for adversarial testing.

### Why This Works

Adversarial testing measures: *"Can this model be fooled on this type of data?"*

You're not checking if the model learned correctly - you're checking if it's **robust to perturbations**.

### Domain Matching Guidelines

| Model Type | Compatible Test Data |
|-----------|---------------------|
| **Sentiment Analysis** | Movie reviews, product reviews, tweets with sentiment labels, restaurant reviews |
| **Disaster Classification** | Disaster tweets, news headlines, emergency messages |
| **Named Entity Recognition (NER)** | CoNLL-2003, OntoNotes, WikiNER, custom entity-labeled text |
| **Question Answering** | SQuAD, Natural Questions, custom QA pairs |
| **Text Classification** | Any labeled text in the same categories |

### Examples

#### Example 1: Testing Sentiment Model

```bash
# Option A: Use any sentiment dataset
python main_nlp.py \
    --model distilbert-base-uncased-finetuned-sst-2-english \
    --data movie_reviews.csv \
    --max-batches 5

# Option B: Use disaster data (cross-domain, for demo)
python main_nlp.py \
    --model distilbert-base-uncased-finetuned-sst-2-english \
    --data models/nlp/train.csv \
    --max-batches 2
```

#### Example 2: Testing Disaster Model from HuggingFace

```bash
# Use your disaster training data
python main_nlp.py \
    --model huggingface-user/disaster-classifier \
    --data models/nlp/train.csv \
    --max-batches 10
```

#### Example 3: Testing with Custom Dataset

```bash
# Your custom labeled data (CSV format)
python main_nlp.py \
    --model any-hf-model \
    --data my_labeled_data.csv \
    --text-column content \
    --label-column category \
    --max-batches 5
```

### Finding Test Datasets

**Public Datasets:**
- **Sentiment:** [IMDb Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), [Amazon Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- **Disaster:** [Kaggle Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)
- **NER:** [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/), [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19)
- **QA:** [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), [Natural Questions](https://ai.google.com/research/NaturalQuestions)

**Create Your Own:**
- Manually label 50-100 samples relevant to your use case
- Use GPT/Claude to help label data (verify quality!)
- Extract and label data from your production logs

---

## Data Requirements

### Minimum Samples

| Testing Scope | Recommended Samples | Purpose |
|--------------|--------------------|---------|
| **Initial Assessment** | 50-100 | Quick robustness check |
| **Development Testing** | 200-500 | Iterative improvements |
| **Production Evaluation** | 500-1000+ | Comprehensive assessment |

### Data Quality

✅ **Good test data:**
- Balanced across classes (equal representation)
- Representative of production scenarios
- Clean labels (verified correct)
- Matches model's domain

❌ **Poor test data:**
- Heavily imbalanced (90% one class)
- Different domain than model
- Noisy/incorrect labels
- Too small (< 50 samples)

### File Formats

**CSV Format:**
```csv
text,label
"Earthquake hits California",1
"Beautiful sunny day",0
```

**JSON Format:**
```json
[
  {"text": "Earthquake hits California", "label": 1},
  {"text": "Beautiful sunny day", "label": 0}
]
```

**Custom column names:**
```bash
# Platform auto-detects: text, tweet, content, message
# Or specify manually:
python main_nlp.py --data file.csv \
    --text-column custom_text \
    --label-column custom_label
```

See [FLEXIBLE_DATA_LOADING.md](FLEXIBLE_DATA_LOADING.md) for details.

---

## Understanding Results

### Attack Success Rate (ASR)

**What it measures:** Percentage of correctly-classified samples that were fooled by attacks.

```
ASR = (Correct samples fooled) / (Total correct samples) × 100%
```

**Interpretation:**
- **0-20%:** Excellent robustness ✅
- **20-40%:** Good robustness ✓
- **40-60%:** Moderate robustness ⚠️
- **60-80%:** Poor robustness ❌
- **80-100%:** Critical vulnerability 🚨

**Special case - ASR = 0%:**
- Could mean: Model is perfectly robust ✓
- **Or:** Attacks failed to generate perturbations (check diagnostics!)

### Detection Accuracy (DA)

**What it measures:** Percentage of adversarial examples correctly identified by detectors.

```
DA = (Adversarial samples detected) / (Total adversarial samples) × 100%
```

**Interpretation:**
- **80-100%:** Excellent detection ✅
- **60-80%:** Good detection ✓
- **40-60%:** Moderate detection ⚠️
- **20-40%:** Poor detection ❌
- **0-20%:** Ineffective detection 🚨

**Special case - DA = 0%:**
- If ASR = 0%: No adversarial examples to detect (normal)
- If ASR > 0%: Detectors are not working ❌

### False Positive Rate (FPR)

**What it measures:** Percentage of benign samples incorrectly flagged as adversarial.

```
FPR = (Benign samples flagged) / (Total benign samples) × 100%
```

**Interpretation:**
- **0-10%:** Excellent precision ✅
- **10-20%:** Good precision ✓
- **20-30%:** Acceptable ⚠️
- **30-50%:** High false alarms ❌
- **50-100%:** Unusable 🚨

### Robustness Score

**What it measures:** Composite score combining ASR, DA, and FPR.

```
Score = (1 - ASR)×0.5 + DA×0.4 + (1-FPR)×0.1 × 100
```

**Interpretation:**
- **80-100:** Robust and well-protected ✅
- **70-80:** Good overall robustness ✓
- **60-70:** Moderate robustness ⚠️
- **50-60:** Vulnerable ❌
- **0-50:** Critical issues 🚨

**Special case - Score = 60:**
- Baseline when ASR=0%, DA=0%, FPR=0%
- Check diagnostics to understand why (see [INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md))

### Model Health Diagnostics

The platform analyzes model behavior and reports health status:

**GOOD:** Model functioning normally
**WARNING:** Minor issues detected
**CRITICAL:** Serious problems (bias, constant predictions, high error rate)

**Common critical issues:**
- Model predicts one class 100% → Model is broken/undertrained
- All predictions similar confidence → Model is not discriminating
- High misclassification rate → Model quality issue

---

## Common Issues and Solutions

### Issue 1: "CSV must have label column"

**Error message:**
```
[ERROR] Error loading data: Could not auto-detect label column.
Available columns: ['message', 'classification']
```

**Solution:**
```bash
# Specify custom column names
python main_nlp.py --data file.csv \
    --text-column message \
    --label-column classification
```

See [FLEXIBLE_DATA_LOADING.md](FLEXIBLE_DATA_LOADING.md) for details.

### Issue 2: "idf vector is not fitted"

**Error message:**
```
sklearn.exceptions.NotFittedError: idf vector is not fitted
```

**Cause:** scikit-learn version mismatch between pickle file and your environment.

**Solution:**
- Upgrade to Python 3.9+ and install compatible scikit-learn
- Or use a different model (transformer-based models don't have this issue)

### Issue 3: "Model predicts one class 100%"

**Diagnostic message:**
```
[CRITICAL] Model predicts 'Not Disaster' for 100% of inputs
```

**Cause:** Model is broken, undertrained, or has severe bias.

**Solutions:**
1. Check model training logs - did training succeed?
2. Test with a known-good model to verify platform works
3. Retrain with balanced data and proper hyperparameters

See [INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md) for detailed analysis.

### Issue 4: Robustness score always 60

**Cause:** Attacks aren't generating perturbations (ASR=0%, DA=0%, FPR=0%).

**Possible reasons:**
1. Model predictions never change (model broken)
2. All inputs already misclassified (check base accuracy)
3. Domain mismatch (testing sentiment model with disaster data)
4. Test data labels wrong (ground truth doesn't match model's task)

**Solution:**
```bash
# Verify attacks work with known-good model
python diagnose_pipeline.py

# Check if model is functional
python -c "
from model_loader import load_nlp_model
model = load_nlp_model('your_model')
print(model('test positive sentiment'))  # Should vary
print(model('test negative sentiment'))
"
```

### Issue 5: Out of memory

**Error:** CUDA out of memory or system memory exhausted

**Solutions:**
```bash
# Reduce batch size
python main_nlp.py --model your_model --data file.csv \
    --batch-size 2 --max-batches 10

# Use CPU instead of GPU
python main_nlp.py --model your_model --data file.csv \
    --device -1

# Test fewer samples initially
python main_nlp.py --model your_model --data file.csv \
    --max-batches 2
```

---

## Best Practices

### 1. Start Small

```bash
# Always test with 1-2 batches first
python main_nlp.py --model your_model --data file.csv --max-batches 1
```

**Why:** Quickly verify everything works before running full evaluation.

### 2. Use Validation Sets

```bash
# Split data (once)
python create_train_val_split.py

# Train on train_80.csv
# Test robustness on validation_20.csv
```

**Why:** Tests on data model hasn't seen (realistic evaluation).

### 3. Match Domain to Model

| Model | Good Test Data | Bad Test Data |
|-------|---------------|--------------|
| Sentiment | Movie reviews | Disaster tweets |
| Disaster | Emergency messages | Product reviews |
| NER | News articles | Sentiment reviews |

**Why:** In-domain testing provides meaningful robustness metrics.

### 4. Monitor Diagnostics

**Always check:**
- Model health status (GOOD/WARNING/CRITICAL)
- Base accuracy (% correctly classified before attacks)
- Diagnostic warnings about bias or constant predictions

**Why:** Catches model quality issues before evaluating robustness.

### 5. Verify with Known-Good Models

```bash
# Test platform with working model
python diagnose_pipeline.py
```

**Why:** Confirms attacks work correctly before testing your model.

### 6. Iterative Testing

```
Test (small) → Fix issues → Test (medium) → Refine → Test (large) → Deploy
```

**Why:** Incremental testing is faster and cheaper than full evaluation each time.

### 7. Document Your Results

Keep track of:
- Model version tested
- Dataset used (size, source, domain)
- Robustness scores over time
- Changes made after each test

**Why:** Track improvements and understand what works.

---

## Quick Reference

### Testing Your Own Models

```bash
# Quick test
python main_nlp.py --model your_model.onnx \
    --tokenizer distilbert-base-uncased \
    --labels '{"0": "Class0", "1": "Class1"}' \
    --data models/nlp/train.csv --max-batches 5

# Proper evaluation
python create_train_val_split.py
python main_nlp.py --model your_model.onnx \
    --data models/nlp/validation_20.csv --max-batches 10
```

### Testing HuggingFace Models

```bash
# Any labeled data in same domain
python main_nlp.py \
    --model distilbert-base-uncased-finetuned-sst-2-english \
    --data your_sentiment_data.csv --max-batches 5
```

### Custom Column Names

```bash
python main_nlp.py --model your_model --data file.csv \
    --text-column message --label-column category
```

---

## See Also

- **[QUICK_START_COMMANDS.md](QUICK_START_COMMANDS.md)** - Copy-paste commands
- **[FLEXIBLE_DATA_LOADING.md](FLEXIBLE_DATA_LOADING.md)** - Column name handling
- **[INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md)** - Why score is always 60
- **[CLAUDE.md](../CLAUDE.md)** - Platform architecture and design

---

## Need Help?

**Common workflows:**
1. Testing disaster models → Use train.csv with --max-batches
2. Testing HuggingFace models → Find any labeled data in same domain
3. Understanding results → Check diagnostics and health status
4. Debugging issues → Run diagnose_pipeline.py

**Still stuck?** Check [INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md) for troubleshooting the most common issues.
