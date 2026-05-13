# Flexible Data Loading Guide

The NLP adversarial evaluation platform now supports **flexible column detection** and works with any dataset regardless of column naming conventions.

## Features

### 1. **Auto-Detection of Column Names**

The platform automatically recognizes common column name variations:

**Text Columns (auto-detected):**
- `text`, `tweet`, `sentence`, `content`, `comment`, `review`, `message`

**Label Columns (auto-detected):**
- `label`, `target`, `class`, `category`, `sentiment`, `y`

### 2. **Custom Column Names**

If your dataset uses different column names, specify them manually:

```bash
python main_nlp.py \
    --model <your_model> \
    --data <your_data.csv> \
    --text-column <column_name> \
    --label-column <column_name>
```

### 3. **Helpful Error Messages**

If columns can't be detected, you'll see exactly what's available:

```
[ERROR] Could not auto-detect label column.
Available columns: ['message', 'classification']
Expected one of: ['label', 'target', 'class', 'category', 'sentiment', 'y']
Use --label-column <column_name> to specify manually.
```

---

## Usage Examples

### Example 1: Standard Kaggle Dataset (auto-detection)

```bash
# train.csv has columns: id, keyword, location, text, target
python main_nlp.py \
    --model distilbert-base-uncased-finetuned-sst-2-english \
    --data models/nlp/train.csv \
    --max-batches 2

# Output:
#   [INFO] Using text column: 'text'
#   [INFO] Using label column: 'target'
#   [OK] Loaded 7613 samples
```

### Example 2: Custom Column Names

```bash
# custom_data.csv has columns: message, classification
python main_nlp.py \
    --model my_model.onnx \
    --tokenizer distilbert-base-uncased \
    --data custom_data.csv \
    --text-column message \
    --label-column classification \
    --max-batches 2

# Output:
#   [INFO] Using text column: 'message'
#   [INFO] Using label column: 'classification'
#   [OK] Loaded 150 samples
```

### Example 3: TF-IDF Model with Kaggle Data

```bash
python main_nlp.py \
    --model models/nlp/lr_tfidf_model.onnx \
    --tokenizer distilbert-base-uncased \
    --labels '{"0": "Not Disaster", "1": "Disaster"}' \
    --data models/nlp/train.csv \
    --max-batches 2

# Auto-detects: text='text', label='target'
```

### Example 4: JSON File with Custom Structure

```json
// data.json
[
    {"review": "Great product!", "rating": 1},
    {"review": "Terrible quality", "rating": 0}
]
```

```bash
python main_nlp.py \
    --model my_sentiment_model \
    --data data.json \
    --text-column review \
    --label-column rating
```

---

## Supported File Formats

### CSV Files
```csv
text,label
"Sample text here",1
"Another example",0
```

- Requires: Text column + Label column
- Delimiter: Comma (,)
- Encoding: UTF-8

### JSON Files
```json
[
    {"text": "Sample text", "label": 1},
    {"text": "Another example", "label": 0}
]
```

- Format: Array of objects
- Requires: Text field + Label field
- Encoding: UTF-8

---

## Command-Line Arguments Reference

| Argument | Short | Type | Description |
|----------|-------|------|-------------|
| `--data` | `-d` | str | Path to data file (CSV or JSON) |
| `--text-column` | | str | Name of text column (optional, auto-detects) |
| `--label-column` | | str | Name of label column (optional, auto-detects) |
| `--model` | `-m` | str | Model path or HuggingFace name |
| `--tokenizer` | `-t` | str | Tokenizer name (for ONNX models) |
| `--labels` | `-l` | str | Label mapping as JSON |
| `--batch-size` | `-b` | int | Batch size (default: 4) |
| `--max-batches` | | int | Max batches to process (default: 2) |
| `--attacks` | | list | Attacks to run (default: textfooler, bertattack) |

---

## Troubleshooting

### Issue: "Could not find text column"

**Solution 1: Check available columns**
```bash
# The error message shows available columns
# Example: Available columns: ['content', 'label']
```

**Solution 2: Specify manually**
```bash
python main_nlp.py --data mydata.csv --text-column content
```

### Issue: "CSV must have either 'label' or 'target' column"

**Solution: Specify custom label column**
```bash
python main_nlp.py --data mydata.csv --label-column class
```

### Issue: Label values are strings instead of integers

**Solution: Ensure labels are numeric (0, 1, 2, ...)**
```csv
text,label
"Sample text",1    ✓ Correct
"Sample text","1"  ✗ Wrong (quoted)
```

---

## Testing Your Dataset

**Quick test with 1 batch:**
```bash
python main_nlp.py \
    --model distilbert-base-uncased-finetuned-sst-2-english \
    --data your_data.csv \
    --max-batches 1
```

**Check column detection:**
Look for these lines in the output:
```
[INFO] Using text column: 'text'
[INFO] Using label column: 'target'
[OK] Loaded 100 samples from your_data.csv
```

---

## Best Practices

1. **Always test with --max-batches 1 first** before running full evaluation
2. **Use standard column names** (`text`, `label`) when possible for automatic detection
3. **Ensure labels are integers** (0, 1, 2, ...) not strings
4. **Check data file encoding** - use UTF-8 for best compatibility
5. **Verify column detection** - check the [INFO] messages in the output

---

## Summary

The platform is now **fully flexible** and can handle:
- ✓ Any column names (with auto-detection or manual specification)
- ✓ Multiple file formats (CSV, JSON)
- ✓ Different naming conventions (Kaggle, custom, etc.)
- ✓ Clear error messages with actionable suggestions
- ✓ No confusion or hard-coded assumptions

**You can now test any model with any dataset without worrying about column names!**
