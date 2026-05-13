# Adversarial Robustness Testing Platform — File Structure & Workflow

## What This Platform Does

This platform tests how easily AI models can be **fooled** by adversarial attacks. It follows a **Red Team vs Blue Team** approach:

- **Red Team** = Attackers — generate slightly modified inputs that trick the model into wrong predictions
- **Blue Team** = Defenders — detect whether an input has been tampered with
- **Analysis** = Measure how well attacks worked and how well defenses caught them
- **Reports** = Generate PDF reports with metrics, charts, and recommendations

---

## Which File to Run for Which Model Type

### At a Glance

| I have a... | I run this file | It does what |
|-------------|----------------|--------------|
| Image classification model (CNN, ResNet, etc.) | `main.py` | Full pipeline: attacks + detection + analysis + PDF report |
| NLP / text classification model | `main_nlp.py` | Full pipeline: attacks + detection + analysis + PDF report |
| Audio classification model | `main_audio.py` | Full pipeline: attacks + detection + analysis + PDF report |
| ONNX image model (just want attacks, no report) | `run_red_team_on_onnx.py` | Attacks only, saves attack_results.json |

---

### 1. `main.py` — Image Classification Testing

**When to use:** You have an image classification model (like a CIFAR-10 CNN) in ONNX format.

**Has NO command-line arguments.** Everything is hardcoded inside the file. To change settings, you edit the file directly.

**What is hardcoded inside:**
- Model path: `models/cifar_net.onnx`
- Dataset: 64 dummy CIFAR-10-like images (32x32x3) — replace `build_dataloader()` for real data
- Attacks: `FGSMAttack(epsilon=0.007)`, `PGDAttack(epsilon=0.03, alpha=0.007, num_steps=10)`, `DeepFoolAttack(max_iter=20)`
- Detectors: `ConfidenceDropDetector(drop_threshold=0.1)` + `EntropyIncreaseDetector(entropy_threshold=0.05)` → `EnsembleDetector(mode="vote")`
- Max batches: 3

**Command:**
```bash
cd adversarial-platform
python main.py
```

**Files that execute when you run this:**
```
main.py                                          ← YOU RUN THIS
  ├── models/cifar_net.onnx                      ← loads this model
  ├── pipeline/config.py                         ← creates PipelineConfig, detects ModelType = IMAGE
  └── pipeline/orchestrator.py                   ← runs 4-stage pipeline:
        │
        ├── STAGE 1 — ATTACKS (Red Team)
        │   └── red_team/image_models/attack_runner.py
        │         ├── red_team/image_models/fgsm_attack.py       (FGSM attack)
        │         ├── red_team/image_models/pgd_attack.py        (PGD attack)
        │         └── red_team/image_models/deepfool_attack.py   (DeepFool attack)
        │
        ├── STAGE 2 — DETECTION (Blue Team)
        │   └── blue_team/image_models/ensemble_detector.py
        │         ├── blue_team/image_models/confidence_detector.py
        │         └── blue_team/image_models/entropy_detector.py
        │
        ├── STAGE 3 — ANALYSIS
        │   ├── analysis/summarizer.py → calls analysis/metrics.py + analysis/robustness_score.py
        │   └── analysis/diagnostics.py
        │
        └── STAGE 4 — REPORTING
              ├── reports/charts.py              → saves PNGs to reports/_charts/
              ├── reports/pdf_generator.py        → generates PDF
              └── reports/templates.py            → recommendations text
```

**Output files generated:**
```
reports/attack_results.json
reports/detection_results.json
reports/security_report.pdf
reports/cifar_net_summary.json
reports/cifar_net_diagnostics.json
reports/_charts/attack_rates.png
reports/_charts/det_confusion.png
```

---

### 2. `main_nlp.py` — NLP Text Classification Testing

**When to use:** You have a text classification model — either from HuggingFace Hub, a local ONNX file, or a local HuggingFace directory.

**Has full CLI arguments.** You control everything from the command line.

**All available arguments:**

| Argument | Short | Required? | Default | What It Does |
|----------|-------|-----------|---------|-------------|
| `--model` | `-m` | No | `distilbert-base-uncased-finetuned-sst-2-english` | Model path or HuggingFace name |
| `--tokenizer` | `-t` | Only for ONNX models | `None` | Tokenizer name (e.g., `distilbert-base-uncased`) |
| `--labels` | `-l` | Only for ONNX models | `None` | Label mapping as JSON string: `'{"0": "NEGATIVE", "1": "POSITIVE"}'` |
| `--data` | `-d` | No | Uses 8 built-in sample tweets | Path to CSV or JSON file with labeled text data |
| `--text-column` | — | No | Auto-detects (`text`, `tweet`, `sentence`, `content`, `comment`, `review`, `message`) | Column name for text in your data file |
| `--label-column` | — | No | Auto-detects (`label`, `target`, `class`, `category`, `sentiment`, `y`) | Column name for labels in your data file |
| `--batch-size` | `-b` | No | `4` | Number of samples per batch |
| `--max-batches` | — | No | `2` | How many batches to process (limits scope) |
| `--attacks` | — | No | `textfooler bertattack` | Which attacks to run. Choices: `textfooler`, `bertattack` |
| `--output` | `-o` | No | `reports` | Output directory for all generated files |
| `--seed` | — | No | `42` | Random seed for reproducibility |
| `--device` | — | No | `-1` (CPU) | `-1` = CPU, `0` = GPU 0, `1` = GPU 1, etc. |
| `--use-llm-diagnostics` | — | No | Off | Enable AI-powered diagnostic explanations (needs API key in `.env`) |
| `--llm-provider` | — | No | `openai` | Which LLM to use: `openai`, `anthropic`, or `github` |

**Example commands:**

```bash
cd adversarial-platform

# Simplest: test a HuggingFace sentiment model with built-in sample data
python main_nlp.py

# Test HuggingFace model with your own CSV data
python main_nlp.py --model distilbert-base-uncased-finetuned-sst-2-english \
    --data my_reviews.csv \
    --max-batches 5

# Test a local ONNX model (MUST specify --tokenizer and --labels)
python main_nlp.py --model models/nlp/disaster_tweets_model.onnx \
    --tokenizer distilbert-base-uncased \
    --labels '{"0": "Not Disaster", "1": "Disaster"}' \
    --data models/nlp/validation_20.csv \
    --max-batches 5

# Test local ONNX model with TF-IDF (no tokenizer needed, auto-detected)
python main_nlp.py --model models/nlp/lr_tfidf_model.onnx \
    --data models/nlp/validation_20.csv \
    --max-batches 3

# Run only TextFooler attack (skip BERTAttack which is slow)
python main_nlp.py --model distilbert-base-uncased-finetuned-sst-2-english \
    --attacks textfooler \
    --max-batches 3

# Custom CSV with non-standard column names
python main_nlp.py --model my_model \
    --data my_data.csv \
    --text-column "review_text" \
    --label-column "sentiment_score"

# With LLM-enhanced diagnostics
python main_nlp.py --model my_model \
    --data my_data.csv \
    --use-llm-diagnostics --llm-provider github
```

**Data file format requirements:**
- **CSV**: Must have a text column and a label column. Column names are auto-detected from common names (`text`, `tweet`, `sentence`, `content`, `comment`, `review`, `message` for text; `label`, `target`, `class`, `category`, `sentiment`, `y` for labels). Use `--text-column` and `--label-column` if your columns have different names.
- **JSON**: Array of objects, each with a text field and a label field. Same auto-detection as CSV.
- **Labels must be integers** (0, 1, 2, etc.)

**Files that execute when you run this:**
```
main_nlp.py                                      ← YOU RUN THIS
  ├── model_loader.py                            ← auto-detects model format, loads model + tokenizer
  ├── pipeline/config.py                         ← creates PipelineConfig, sets ModelType = NLP
  └── pipeline/orchestrator.py                   ← runs 4-stage pipeline:
        │
        ├── STAGE 1 — ATTACKS (Red Team)
        │   └── red_team/nlp_models/attack_runner.py
        │         ├── red_team/nlp_models/textfooler_attack.py   (TextFooler attack)
        │         ├── red_team/nlp_models/bertattack.py          (BERTAttack)
        │         └── red_team/nlp_models/utils.py               (shared NLP utilities)
        │
        ├── STAGE 2 — DETECTION (Blue Team)
        │   └── blue_team/nlp_models/ensemble_detector.py
        │         ├── blue_team/nlp_models/perplexity_detector.py  (loads distilgpt2)
        │         └── blue_team/nlp_models/semantic_detector.py    (loads all-MiniLM-L6-v2)
        │
        ├── STAGE 3 — ANALYSIS
        │   ├── analysis/summarizer.py → calls analysis/metrics.py + analysis/robustness_score.py
        │   └── analysis/diagnostics.py
        │
        └── STAGE 4 — REPORTING
              ├── reports/charts.py
              ├── reports/pdf_generator.py
              └── reports/templates.py
```

**Output files generated (example for disaster_tweets_model):**
```
reports/disaster_tweets_model_attack_results.json
reports/disaster_tweets_model_summary.json
reports/disaster_tweets_model_diagnostics.json
reports/disaster_tweets_model_security_report.pdf
reports/_charts/attack_rates.png
reports/_charts/det_confusion.png
```

---

### 3. `main_audio.py` — Audio Classification Testing

**When to use:** You have an audio classification model — from HuggingFace Hub, a local PyTorch file, or a local ONNX file.

**Has full CLI arguments.** You control everything from the command line.

**All available arguments:**

| Argument | Short | Required? | Default | What It Does |
|----------|-------|-----------|---------|-------------|
| `--model` | `-m` | No | `dummy` (synthetic test model) | Model path or HuggingFace name (e.g., `facebook/wav2vec2-base-960h`) |
| `--labels` | `-l` | No | `'{"0": "class_0", "1": "class_1"}'` | Label mapping as JSON string |
| `--data` | `-d` | No | Uses 8 synthetic audio samples | Directory containing `.wav` files (with class subdirectories) |
| `--sample-rate` | — | No | `16000` | Audio sample rate in Hz |
| `--batch-size` | `-b` | No | `4` | Number of samples per batch |
| `--max-batches` | — | No | `2` | How many batches to process (`-1` = all) |
| `--attacks` | — | No | `noise` | Which attacks to run. Choices: `noise`, `carlini` |
| `--epsilon` | — | No | `0.01` | Perturbation strength for attacks |
| `--target-snr` | — | No | `20.0` | Target Signal-to-Noise Ratio in dB for noise injection |
| `--output` | `-o` | No | `reports` | Output directory |
| `--seed` | — | No | `42` | Random seed |
| `--device` | — | No | `cuda` if available, else `cpu` | Device: `cpu` or `cuda` |
| `--use-llm-diagnostics` | — | No | Off | Enable AI-powered diagnostics |
| `--llm-provider` | — | No | `github` | LLM provider: `openai`, `anthropic`, `github` |

**Example commands:**

```bash
cd adversarial-platform

# Simplest: test with dummy model and synthetic audio
python main_audio.py

# Test a HuggingFace audio model
python main_audio.py --model facebook/wav2vec2-base-960h \
    --data audio_samples/ \
    --max-batches 3

# Test with custom labels and SNR
python main_audio.py --model models/audio/speech_classifier.pth \
    --labels '{"0": "speech", "1": "music", "2": "noise"}' \
    --data audio_samples/ \
    --sample-rate 16000 \
    --target-snr 15 \
    --epsilon 0.02

# Run both noise and carlini attacks
python main_audio.py --model my_model \
    --data audio/ \
    --attacks noise carlini
```

**Data directory structure:**
```
audio_samples/           ← pass this path to --data
  ├── class_0/           ← directory name = class label
  │   ├── audio1.wav
  │   └── audio2.wav
  ├── class_1/
  │   ├── audio3.wav
  │   └── audio4.wav
```
If no subdirectories exist, all `.wav` files are loaded with label `0`.

**Files that execute when you run this:**
```
main_audio.py                                    ← YOU RUN THIS
  ├── pipeline/config.py                         ← creates PipelineConfig, sets ModelType = AUDIO
  └── pipeline/orchestrator.py                   ← runs 4-stage pipeline:
        │
        ├── STAGE 1 — ATTACKS (Red Team)
        │   └── red_team/audio_models/attack_runner.py
        │         ├── red_team/audio_models/noise_injection_attack.py  (✅ works)
        │         └── red_team/audio_models/adversarial_audio_attack.py (❌ template, will error)
        │
        ├── STAGE 2 — DETECTION (Blue Team)
        │   └── blue_team/audio_models/ensemble_detector.py  (❌ template, may error)
        │         ├── blue_team/audio_models/energy_detector.py
        │         └── blue_team/audio_models/spectral_detector.py
        │
        ├── STAGE 3 — ANALYSIS
        │   ├── analysis/summarizer.py → calls analysis/metrics.py + analysis/robustness_score.py
        │   └── analysis/diagnostics.py
        │
        └── STAGE 4 — REPORTING
              ├── reports/charts.py
              ├── reports/pdf_generator.py
              └── reports/templates.py
```

**Output files generated:**
```
reports/audio_security_report.pdf
reports/audio_attack_results.json
reports/audio_detection_results.json
reports/diagnostics.json
reports/summary.json
```

---

### 4. `run_red_team_on_onnx.py` — Attack-Only (No Detection, No Report)

**When to use:** You only want to run attacks on the CIFAR-10 ONNX model and see raw attack results. No detection, no analysis, no PDF report.

**Has NO command-line arguments.** Everything is hardcoded inside the file.

**What is hardcoded inside:**
- Model path: `models/cifar_net.onnx`
- Dataset: CIFAR-10 test set (downloaded automatically via torchvision)
- Attacks: `FGSMAttack(epsilon=0.007)`, `PGDAttack(epsilon=0.03, num_steps=40)`, `DeepFoolAttack(max_iter=50)`
- Max batches: 5
- Batch size: 32

**Command:**
```bash
cd adversarial-platform
python run_red_team_on_onnx.py
```

**Files that execute when you run this:**
```
run_red_team_on_onnx.py                          ← YOU RUN THIS
  ├── models/cifar_net.onnx                      ← loads this model
  └── red_team/image_models/attack_runner.py     ← runs attacks directly (NO orchestrator)
        ├── red_team/image_models/fgsm_attack.py
        ├── red_team/image_models/pgd_attack.py
        └── red_team/image_models/deepfool_attack.py
```

**Output:** Only `reports/attack_results.json` — no detection, no summary, no PDF.

---

### Summary: What Each File Does vs Does NOT Do

| | `main.py` | `main_nlp.py` | `main_audio.py` | `run_red_team_on_onnx.py` |
|---|---|---|---|---|
| Has CLI arguments? | No (hardcoded) | Yes (full CLI) | Yes (full CLI) | No (hardcoded) |
| Runs attacks? | Yes | Yes | Yes | Yes |
| Runs detection? | Yes | Yes | Yes (if implemented) | **No** |
| Runs analysis? | Yes | Yes | Yes | **No** |
| Generates PDF? | Yes | Yes | Yes | **No** |
| Uses orchestrator? | Yes | Yes | Yes | **No** (direct runner) |
| Model format | ONNX only | ONNX / HuggingFace / PyTorch | ONNX / HuggingFace / PyTorch | ONNX only |

---

## Pipeline Workflow (How It All Connects)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                                 │
│  main.py (Image) / main_nlp.py (NLP) / main_audio.py (Audio)      │
│                                                                     │
│  1. Load model (ONNX / PyTorch / HuggingFace)                     │
│  2. Load dataset with labels                                        │
│  3. Configure attacks + detectors                                   │
│  4. Create PipelineConfig                                           │
│  5. Call Orchestrator.run(config)                                   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  pipeline/orchestrator.py                            │
│                                                                     │
│  Stage 1: _run_attacks()                                            │
│     ├─ Auto-detects model type (image/nlp/audio/object_detection)  │
│     ├─ Selects the right AttackRunner for that modality            │
│     └─ Runs all configured attacks batch-by-batch                  │
│           Output → attack_results.json                              │
│                                                                     │
│  Stage 2: _run_detection()                                          │
│     ├─ Passes attack results through detection_fn (Blue Team)      │
│     └─ Records: is_attack, detected, confidence per sample         │
│           Output → detection_results.json                           │
│                                                                     │
│  Stage 3: Analysis                                                  │
│     ├─ summarize() → ASR, Detection Accuracy, FPR, Robustness     │
│     └─ DiagnosticAnalyzer → class bias, confidence anomalies       │
│           Output → summary.json, diagnostics.json                   │
│                                                                     │
│  Stage 4: Reporting                                                 │
│     ├─ charts.py → PNG bar charts, confusion matrices              │
│     └─ pdf_generator.py → Full PDF security report                 │
│           Output → security_report.pdf                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Complete File Tree with Purpose of Every File

```
adversarial-platform/
│
├── main.py                             # ENTRY POINT: Image classification pipeline (CIFAR-10)
├── main_nlp.py                         # ENTRY POINT: NLP text classification pipeline
├── main_audio.py                       # ENTRY POINT: Audio classification pipeline
├── run_red_team_on_onnx.py             # ENTRY POINT: Attack-only runner for ONNX (no detection/reporting)
├── model_loader.py                     # Universal model loader — auto-detects ONNX/HuggingFace/PyTorch
├── requirements.txt                    # All Python dependencies
├── .env / .env.example                 # Environment variables (API keys for optional LLM enhancement)
├── labels.json                         # Label name mappings for models
│
├── pipeline/                           # ══════ ORCHESTRATION LAYER ══════
│   ├── orchestrator.py                 # CORE: Coordinates attack → detection → analysis → report
│   ├── config.py                       # PipelineConfig dataclass + ModelType enum + auto-detection
│   └── session_manager.py              # Tracks execution sessions (session_id, metadata, results)
│
├── red_team/                           # ══════ ATTACK LAYER (Red Team) ══════
│   ├── __init__.py                     # Re-exports all image attacks for convenience
│   │
│   ├── image_models/                   # --- Image Attacks (FULLY IMPLEMENTED) ---
│   │   ├── base_attack.py              # ABC: generate(model, inputs, labels) → Dict
│   │   ├── fgsm_attack.py              # FGSM — single-step gradient sign attack (L∞ bound)
│   │   ├── pgd_attack.py              # PGD — multi-step iterative FGSM with projection (L∞)
│   │   ├── deepfool_attack.py          # DeepFool — iterative L2-minimizing attack
│   │   ├── attack_runner.py            # AttackRunner: runs attacks over batches, outputs JSON
│   │   └── __init__.py                 # Exports: BaseAttack, AttackRunner, FGSM, PGD, DeepFool
│   │
│   ├── nlp_models/                     # --- NLP Attacks (FULLY IMPLEMENTED) ---
│   │   ├── base_attack.py              # ABC: generate(model, texts, labels) → Dict
│   │   ├── textfooler_attack.py        # TextFooler — word substitution via WordNet synonyms
│   │   ├── bertattack.py               # BERTAttack — context-aware substitution using BERT MLM
│   │   ├── attack_runner.py            # NLPAttackRunner: runs NLP attacks over text datasets
│   │   ├── utils.py                    # Helpers: tokenize, word importance, synonyms, edit distance
│   │   └── __init__.py                 # Exports: BaseAttack, TextFoolerAttack, BERTAttack, NLPAttackRunner
│   │
│   ├── audio_models/                   # --- Audio Attacks (PARTIALLY IMPLEMENTED) ---
│   │   ├── base_attack.py              # ABC: generate(model, audio, labels, sample_rate)
│   │   ├── noise_injection_attack.py   # NoiseInjection — adds noise at target SNR (✅ implemented)
│   │   ├── adversarial_audio_attack.py # C&W-style optimization attack (❌ template only)
│   │   ├── attack_runner.py            # AudioAttackRunner: per-sample SNR/spectral diagnostics
│   │   └── __init__.py
│   │
│   └── object_detection/               # --- Object Detection Attacks (❌ ALL TEMPLATES) ---
│       ├── base_attack.py              # ABC: generate(model, images, targets)
│       ├── patch_attack.py             # Adversarial patch optimization (❌ template)
│       ├── dapricot_attack.py          # DAPricot dense perturbation (❌ template)
│       ├── attack_runner.py            # ObjectDetectionAttackRunner
│       └── __init__.py
│
├── blue_team/                          # ══════ DETECTION LAYER (Blue Team) ══════
│   ├── __init__.py                     # Re-exports image detectors for convenience
│   │
│   ├── image_models/                   # --- Image Detectors (FULLY IMPLEMENTED) ---
│   │   ├── base_detector.py            # ABC: detect(original_output, adversarial_output) → Dict
│   │   ├── confidence_detector.py      # Flags samples with large confidence drop (threshold: 20%)
│   │   ├── entropy_detector.py         # Flags samples with entropy increase (threshold: 0.2)
│   │   ├── ensemble_detector.py        # Combines detectors via majority vote or weighted score
│   │   └── __init__.py
│   │
│   ├── nlp_models/                     # --- NLP Detectors (FULLY IMPLEMENTED) ---
│   │   ├── base_detector.py            # ABC: detect(orig_out, adv_out, original_text, adversarial_text)
│   │   ├── perplexity_detector.py      # High perplexity = likely adversarial (uses distilgpt2)
│   │   ├── semantic_detector.py        # Low embedding similarity = likely adversarial (MiniLM)
│   │   ├── ensemble_detector.py        # Combines NLP detectors via voting/averaging
│   │   └── __init__.py
│   │
│   ├── audio_models/                   # --- Audio Detectors (❌ ALL TEMPLATES) ---
│   │   ├── base_detector.py            # ABC: detect(orig_out, adv_out, orig_audio, adv_audio, sr)
│   │   ├── energy_detector.py          # SNR/energy analysis (❌ template)
│   │   ├── spectral_detector.py        # STFT spectral anomaly detection (❌ template)
│   │   ├── ensemble_detector.py        # Audio ensemble (❌ template)
│   │   └── __init__.py
│   │
│   └── object_detection/               # --- Object Detection Detectors (❌ ALL TEMPLATES) ---
│       ├── base_detector.py            # ABC for bounding box detection outputs
│       ├── bbox_confidence_detector.py # Confidence-based detection (❌ template)
│       ├── iou_anomaly_detector.py     # IoU stability analysis (❌ template)
│       ├── ensemble_detector.py        # Object detection ensemble (❌ template)
│       └── __init__.py
│
├── analysis/                           # ══════ ANALYSIS LAYER ══════
│   ├── metrics.py                      # Core: compute_attack_success_rate(), compute_detection_accuracy(), compute_false_positive_rate()
│   ├── summarizer.py                   # summarize() → aggregates ASR, DA, FPR, robustness_score
│   ├── robustness_score.py             # Composite: (1-ASR)×w1 + DA×w2 + (1-FPR)×w3, scaled 0-100
│   ├── diagnostics.py                  # DiagnosticAnalyzer: class bias, confidence anomalies, attack patterns
│   └── llm_enhancer.py                 # Optional LLM integration (OpenAI/Anthropic) for AI explanations
│
├── reports/                            # ══════ REPORTING LAYER ══════
│   ├── charts.py                       # plot_attack_success_bar() + plot_detection_confusion() → PNGs
│   ├── pdf_generator.py                # build_pdf() → comprehensive PDF via ReportLab
│   ├── templates.py                    # build_recommendations() + format_model_details()
│   ├── _charts/                        # Generated chart images
│   │   ├── attack_rates.png
│   │   └── det_confusion.png
│   └── *_security_report.pdf           # Generated per-model PDF reports
│
├── models/                             # ══════ TRAINED MODELS ══════
│   ├── cifar_net.py                    # CIFAR-10 CNN architecture definition (Python class)
│   ├── cifar_net.onnx                  # CIFAR-10 CNN in ONNX format
│   ├── cifar_net.pth                   # CIFAR-10 CNN PyTorch checkpoint
│   │
│   ├── nlp/
│   │   ├── disaster_tweets_model.onnx  # Disaster tweet classifier (ONNX)
│   │   ├── lr_tfidf_model.onnx        # Logistic Regression + TF-IDF (ONNX)
│   │   ├── tfidf_vectorizer.pkl       # Saved TF-IDF vectorizer
│   │   ├── train.csv / test.csv       # Original dataset
│   │   └── train_80.csv / validation_20.csv  # 80/20 split for proper evaluation
│   │
│   └── finbert/
│       ├── model.onnx                  # FinBERT financial sentiment model
│       ├── config.json                 # HuggingFace model config
│       ├── tokenizer.json / vocab.txt  # Tokenizer artifacts
│       └── tokenizer_config.json / special_tokens_map.json
│
├── data/                               # ══════ DATASETS ══════
│   └── cifar-10-batches-py/            # CIFAR-10 image dataset (10 classes, 60k images)
│
├── examples/                           # ══════ USAGE EXAMPLES ══════
│   ├── example_quick_test.py           # Quick demo: TextFooler + BERTAttack on HuggingFace model
│   ├── example_huggingface.py          # Test various HuggingFace models generically
│   ├── example_custom_model.py         # Template for your own ONNX/HuggingFace models
│   └── export_tfidf_model.py           # Guide: export sklearn TF-IDF model to ONNX
│
└── tf_idf model/                       # ══════ MODEL TRAINING REFERENCE ══════
    ├── nlp_text_classification_model_github.py   # TF-IDF training script (from Colab)
    └── NLP_text_classification_model_Github.ipynb # Original Jupyter notebook
```

---

## Detailed File Descriptions

### Entry Points

| File | What It Does |
|------|-------------|
| `main.py` | Loads ONNX CIFAR-10 model (with Reshape `allowzero` patching), builds CIFAR-10 dataloader, configures FGSM/PGD/DeepFool attacks + ConfidenceDrop/EntropyIncrease detectors, runs the full Orchestrator pipeline. |
| `main_nlp.py` | CLI-based NLP pipeline. Supports HuggingFace Hub models, local ONNX, and local HuggingFace dirs. Args: `--model`, `--data`, `--attacks`, `--max-batches`, `--batch-size`. Uses TextFooler + BERTAttack + Perplexity/Semantic detectors. |
| `main_audio.py` | CLI-based Audio pipeline. Supports HuggingFace, PyTorch, and ONNX audio models. Args: `--model`, `--data`, `--attacks`, `--epsilon`, `--target-snr`, `--max-batches`. Uses NoiseInjectionAttack. |
| `run_red_team_on_onnx.py` | Red-team-only runner. Loads/patches ONNX, runs FGSM/PGD/DeepFool attacks **without** detection or reporting. Useful for quick attack testing. |
| `model_loader.py` | Universal NLP model loader. Auto-detects format (ONNX, HuggingFace Hub, local HuggingFace, PyTorch). Handles TF-IDF vectorizers, NumPy 2.0 compatibility, NLTK data. Defines `ModelFormat` enum and `load_nlp_model()`. |

### Pipeline Layer

| File | What It Does |
|------|-------------|
| `pipeline/config.py` | Defines `ModelType` enum (`IMAGE`, `NLP`, `AUDIO`, `OBJECT_DETECTION`, `UNKNOWN`). `detect_model_type()` inspects PyTorch model layers: Conv2d → IMAGE, Embedding/Transformer → NLP, Conv1d → AUDIO. `PipelineConfig` dataclass holds model, dataloader, attacks, detection_fn, and settings. |
| `pipeline/orchestrator.py` | The **brain** of the platform. `Orchestrator.run(config)` executes the 4-stage pipeline: attacks → detection → analysis → reporting. Auto-selects the correct AttackRunner based on model type. Saves all artifacts to `reports/` with model-specific filenames. |
| `pipeline/session_manager.py` | `Session` dataclass (session_id, metadata, results) and `SessionManager` class. Manages run tracking with UUID-based session IDs. In-memory storage. |

### Red Team Layer — Attacks

#### Image Attacks (Fully Implemented)

| File | What It Does |
|------|-------------|
| `red_team/image_models/base_attack.py` | Abstract base class. Contract: `generate(model, inputs, labels) → Dict` returning adversarial_inputs, original_predictions, adversarial_predictions, attack_success_rate, perturbation_metrics. |
| `red_team/image_models/fgsm_attack.py` | **FGSM (Fast Gradient Sign Method)** — Single-step L∞ attack. Computes gradient of loss w.r.t. input, takes single step in sign direction. Params: `epsilon` (perturbation bound), `clip_min/max`. |
| `red_team/image_models/pgd_attack.py` | **PGD (Projected Gradient Descent)** — Multi-step iterative FGSM with L∞ projection back to epsilon-ball. Params: `epsilon`, `alpha` (step size), `num_steps` (default 40), `random_start`. Stronger than FGSM. |
| `red_team/image_models/deepfool_attack.py` | **DeepFool** — Iterative L2-minimizing attack. Finds minimal perturbation to cross decision boundary. Params: `max_iter` (50), `overshoot` (0.02). Produces smaller perturbations than FGSM/PGD. |
| `red_team/image_models/attack_runner.py` | `AttackRunner` class. Methods: `run_batch(model, inputs, labels, attacks)` and `run_dataset(model, dataloader, attacks)`. Iterates batches, runs each attack, collects results. `_to_jsonable()` converts tensors to Python lists for JSON. |

#### NLP Attacks (Fully Implemented)

| File | What It Does |
|------|-------------|
| `red_team/nlp_models/base_attack.py` | Abstract base class for NLP. Contract: `generate(model, texts: List[str], labels) → Dict`. |
| `red_team/nlp_models/textfooler_attack.py` | **TextFooler** — Word-level substitution attack. Ranks words by importance (deletion method), replaces with WordNet synonyms greedily until prediction flips. Params: `max_candidates`, `max_perturbed_words`. |
| `red_team/nlp_models/bertattack.py` | **BERTAttack** — Context-aware word substitution. Uses BERT's masked language model to find contextually appropriate replacements. Lazy-loads `bert-base-uncased`. More natural-sounding than TextFooler. Params: `bert_model`, `top_k`. |
| `red_team/nlp_models/attack_runner.py` | `NLPAttackRunner` — Same pattern as image AttackRunner but operates on `List[str]` text inputs instead of tensors. |
| `red_team/nlp_models/utils.py` | Shared utilities: `tokenize_text()`, `compute_word_importance()` (deletion-based), `get_synonyms_wordnet()`, `compute_edit_distance()`, `compute_word_changes()`, `download_nltk_data()`. |

#### Audio Attacks (Partially Implemented)

| File | What It Does |
|------|-------------|
| `red_team/audio_models/base_attack.py` | Abstract base class for audio. Contract: `generate(model, audio: np.ndarray, labels, sample_rate)`. |
| `red_team/audio_models/noise_injection_attack.py` | **NoiseInjection** — Adds random noise calibrated to a target Signal-to-Noise Ratio (SNR). Params: `epsilon`, `target_snr_db`. **Fully implemented.** |
| `red_team/audio_models/adversarial_audio_attack.py` | **AdversarialAudioAttack** — Planned Carlini & Wagner style optimization-based attack. **Not yet implemented (template only).** |
| `red_team/audio_models/attack_runner.py` | `AudioAttackRunner` — Runs audio attacks with per-sample diagnostics: SNR, spectral distance, energy metrics. |

#### Object Detection Attacks (Templates Only)

| File | What It Does |
|------|-------------|
| `red_team/object_detection/base_attack.py` | Abstract base class. Contract: `generate(model, images, targets: List[Dict])`. |
| `red_team/object_detection/patch_attack.py` | **PatchAttack** — Planned adversarial patch optimization. **Not implemented.** |
| `red_team/object_detection/dapricot_attack.py` | **DAPricotAttack** — Planned dense adversarial perturbation. **Not implemented.** |
| `red_team/object_detection/attack_runner.py` | `ObjectDetectionAttackRunner` — Per-detection diagnostics: IoU, confidence, localization. |

### Blue Team Layer — Detectors

#### Image Detectors (Fully Implemented)

| File | What It Does |
|------|-------------|
| `blue_team/image_models/base_detector.py` | Abstract base class. Contract: `detect(original_output, adversarial_output) → {anomaly_score, detected, explanation}`. |
| `blue_team/image_models/confidence_detector.py` | **ConfidenceDropDetector** — Flags samples where max softmax confidence drops by more than a relative threshold (default 20%). If the model suddenly becomes unsure, it may be under attack. |
| `blue_team/image_models/entropy_detector.py` | **EntropyIncreaseDetector** — Flags samples where prediction entropy (`-sum(p*log(p))`) increases beyond threshold (default 0.2). Higher entropy = more uncertain = likely adversarial. |
| `blue_team/image_models/ensemble_detector.py` | **EnsembleDetector** — Combines multiple detectors via majority voting or weighted anomaly score aggregation. The final "is this adversarial?" decision maker. |

#### NLP Detectors (Fully Implemented)

| File | What It Does |
|------|-------------|
| `blue_team/nlp_models/base_detector.py` | Abstract base class. Contract: `detect(orig_out, adv_out, original_text, adversarial_text)`. Takes both model outputs AND raw text. |
| `blue_team/nlp_models/perplexity_detector.py` | **PerplexityDetector** — Measures how "weird" adversarial text looks to a language model (distilgpt2). High perplexity = grammatically unusual = likely adversarial. Lazy-loads the LM. |
| `blue_team/nlp_models/semantic_detector.py` | **SemanticDetector** — Compares sentence embeddings (all-MiniLM-L6-v2) via cosine similarity. If the meaning changed a lot but words only changed slightly, it's likely adversarial. |
| `blue_team/nlp_models/ensemble_detector.py` | **EnsembleDetector** — Combines NLP detectors via voting or average score. |

#### Audio Detectors (Templates Only)

| File | What It Does |
|------|-------------|
| `blue_team/audio_models/base_detector.py` | Abstract base class. Accepts original/adversarial waveforms + sample_rate. |
| `blue_team/audio_models/energy_detector.py` | **EnergyDetector** — Planned: SNR and energy distribution analysis. **Not implemented.** |
| `blue_team/audio_models/spectral_detector.py` | **SpectralDetector** — Planned: STFT-based spectral anomaly detection. **Not implemented.** |
| `blue_team/audio_models/ensemble_detector.py` | Audio ensemble. **Not implemented.** |

#### Object Detection Detectors (Templates Only)

| File | What It Does |
|------|-------------|
| `blue_team/object_detection/base_detector.py` | Abstract base class for bounding box detection outputs. |
| `blue_team/object_detection/bbox_confidence_detector.py` | **BBoxConfidenceDetector** — Planned: confidence-based detection for bounding boxes. **Not implemented.** |
| `blue_team/object_detection/iou_anomaly_detector.py` | **IoUAnomalyDetector** — Planned: IoU-based bounding box stability analysis. **Not implemented.** |
| `blue_team/object_detection/ensemble_detector.py` | Object detection ensemble. **Not implemented.** |

### Analysis Layer

| File | What It Does |
|------|-------------|
| `analysis/metrics.py` | Three core metric functions: `compute_attack_success_rate()` (mean of per-batch ASRs), `compute_detection_accuracy()` (correctly detected / total), `compute_false_positive_rate()` (false alarms / total benign). |
| `analysis/summarizer.py` | `summarize(attack_results, detection_results)` — Aggregates everything into a single dict: ASR, DA, FPR, robustness_score, per-attack breakdowns. JSON-serializable output. |
| `analysis/robustness_score.py` | `compute_robustness_score()` — Weighted composite formula: `(1-ASR)×w1 + DA×w2 + (1-FPR)×w3`, scaled 0-100. Higher = more robust. Auto-adjusts weights when detection is not used. |
| `analysis/diagnostics.py` | `DiagnosticAnalyzer` class — Runs multiple diagnostic checks: class bias (some classes attacked more), confidence anomalies (flat/overconfident/underconfident distributions), attack failure analysis, label consistency, prediction distribution. Severity levels: CRITICAL, HIGH, MEDIUM, WARNING, LOW, INFO. |
| `analysis/llm_enhancer.py` | `LLMEnhancer` — Optional integration with OpenAI/Anthropic/GitHub Models API. Takes CRITICAL/HIGH severity diagnostics and generates AI-powered explanations and recommendations. In-memory caching, graceful fallback if API unavailable. |

### Reports Layer

| File | What It Does |
|------|-------------|
| `reports/charts.py` | Two chart generators: `plot_attack_success_bar()` (bar chart of ASR per attack) and `plot_detection_confusion()` (confusion matrix of detection results). Saves as PNG to `_charts/`. |
| `reports/pdf_generator.py` | `build_pdf()` — Generates a comprehensive PDF security report using ReportLab. Includes: model info, attack results table, detection results, robustness score, charts, and optional diagnostics section. |
| `reports/templates.py` | `build_recommendations()` — Derives actionable recommendations based on ASR/DA/FPR thresholds. `format_model_details()` — Normalizes model info dict for clean display in reports. |

### Examples

| File | What It Does |
|------|-------------|
| `examples/example_quick_test.py` | Quick demo showing TextFooler + BERTAttack + Perplexity/Semantic detectors on a HuggingFace sentiment model. Good starting point to understand the platform. |
| `examples/example_huggingface.py` | Generic tester for any HuggingFace model. Shows how to use `test_model()` function pattern. |
| `examples/example_custom_model.py` | Template for testing your own ONNX or HuggingFace models with configurable paths and label mappings. |
| `examples/export_tfidf_model.py` | Step-by-step guide for exporting sklearn TF-IDF + classifier models to ONNX format for use with this platform. |

---

## Key Metrics Explained

| Metric | Formula | Meaning |
|--------|---------|---------|
| **ASR** (Attack Success Rate) | `predictions_flipped / total_correct_samples` | % of correctly-classified samples that the attack fooled into wrong predictions |
| **DA** (Detection Accuracy) | `correctly_detected / total_samples` | % of adversarial samples the detector correctly identified as adversarial |
| **FPR** (False Positive Rate) | `false_alarms / total_benign_samples` | % of clean (unmodified) samples wrongly flagged as adversarial |
| **Robustness Score** | `(1-ASR)×w1 + DA×w2 + (1-FPR)×w3 × 100` | Overall model safety score (0-100). Higher = more robust against attacks |

---

## Implementation Status Summary

| Modality | Red Team (Attacks) | Blue Team (Detectors) | Status |
|----------|-------------------|----------------------|--------|
| **Image** | FGSM, PGD, DeepFool | ConfidenceDrop, Entropy, Ensemble | **Fully Working** |
| **NLP** | TextFooler, BERTAttack | Perplexity, Semantic, Ensemble | **Fully Working** |
| **Audio** | NoiseInjection (done), C&W (template) | Energy, Spectral, Ensemble (templates) | **Partial** |
| **Object Detection** | Patch, DAPricot (templates) | BBoxConfidence, IoUAnomaly (templates) | **Templates Only** |

---

## How Each Entry Point Works

### `main.py` — Image Pipeline

```
1. Load ONNX model → patch Reshape allowzero attribute → convert to PyTorch via onnx2torch
2. Load CIFAR-10 dataset with ToTensor() + Normalize() transforms
3. Create attacks: FGSMAttack(eps=0.03), PGDAttack(eps=0.03), DeepFoolAttack()
4. Create detectors: ConfidenceDropDetector + EntropyIncreaseDetector → EnsembleDetector
5. Wrap detectors in detection_fn_factory() → callable for orchestrator
6. Build PipelineConfig → Orchestrator.run() → saves JSON + PDF to reports/
```

### `main_nlp.py` — NLP Pipeline

```
1. Parse CLI args: --model, --data, --attacks, --max-batches, --batch-size
2. Auto-detect model format: HuggingFace Hub / local ONNX / local HuggingFace directory
3. Load model + tokenizer via model_loader.py (handles TF-IDF vectorizers too)
4. Load labeled CSV dataset (text + label columns)
5. Create attacks: TextFoolerAttack(), BERTAttack()
6. Create detectors: PerplexityDetector + SemanticDetector → EnsembleDetector
7. Build PipelineConfig → Orchestrator.run() → saves JSON + PDF to reports/
```

### `main_audio.py` — Audio Pipeline

```
1. Parse CLI args: --model, --data, --attacks, --epsilon, --target-snr, --max-batches
2. Load audio model (HuggingFace / PyTorch / ONNX)
3. Load audio files from directory (wav/mp3/flac)
4. Create attacks: NoiseInjectionAttack(target_snr_db=target_snr)
5. Build PipelineConfig → Orchestrator.run() → saves JSON + PDF to reports/
```

---

## Key Design Patterns

### 1. Factory Pattern for Detection Functions

In each `main_*.py`, a `detection_fn_factory()` wraps the ensemble detector into a callable that the orchestrator invokes per-batch. It converts raw attack results (with predictions) into tensors and calls `detector.detect()`, returning structured records with `is_attack`, `detected`, `confidence` fields.

### 2. Modality Auto-Detection

`config.py:detect_model_type()` inspects PyTorch model layers to determine modality:
- `Conv2d` layers → IMAGE
- `Embedding` or `Transformer` layers → NLP
- `Conv1d` layers → AUDIO
- Multiple output heads → OBJECT_DETECTION

### 3. Batch-wise Processing

All attack runners iterate data batch-by-batch. `max_batches` parameter limits scope for quick testing. Every result includes `batch_index` for traceability.

### 4. JSON Serialization

- `AttackRunner._to_jsonable()` converts PyTorch tensors → Python lists
- `Orchestrator._dump_json()` uses `sanitize_for_json()` to strip large arrays and convert numpy types
- Audio waveform arrays are explicitly excluded from JSON to avoid 100MB+ files

### 5. Abstract Base Classes

Every modality has its own `BaseAttack` and `BaseDetector` ABCs. To add a new attack or detector, subclass the appropriate base and implement the required `generate()` or `detect()` method.

---

## Output Artifacts

Every pipeline run generates these files in `adversarial-platform/reports/`:

| File | Contents |
|------|----------|
| `{model_name}_attack_results.json` | Per-attack, per-batch results with predictions and success metrics |
| `{model_name}_detection_results.json` | Detection flags and confidence scores per sample |
| `{model_name}_summary.json` | Aggregated metrics: ASR, DA, FPR, robustness score |
| `{model_name}_diagnostics.json` | Model behavior analysis and diagnostic insights |
| `{model_name}_security_report.pdf` | Comprehensive PDF with all findings, charts, recommendations |
| `_charts/attack_rates.png` | Bar chart of attack success rates |
| `_charts/det_confusion.png` | Detection confusion matrix |
