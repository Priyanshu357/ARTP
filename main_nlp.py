"""Unified NLP adversarial evaluation entry point.

This script provides a consolidated interface for testing NLP models
against adversarial attacks, supporting multiple model formats with
automatic detection.

Supported Model Formats:
  - HuggingFace Hub models (by name)
  - Local ONNX models (.onnx files)
  - Local HuggingFace models (directories with config.json)

Usage Examples:
  # Test HuggingFace model (simplest)
  python main_nlp.py --model distilbert-base-uncased-finetuned-sst-2-english

  # Test local ONNX model
  python main_nlp.py --model models/nlp/distilbert.onnx \\
      --tokenizer distilbert-base-uncased-finetuned-sst-2-english \\
      --labels '{"0": "NEGATIVE", "1": "POSITIVE"}'

  # Load data from file
  python main_nlp.py --model my_model --data test_data.json

  # Configure attacks and batches
  python main_nlp.py --model my_model \\
      --attacks textfooler \\
      --max-batches 3 \\
      --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch

from model_loader import load_nlp_model
from red_team.nlp_models import TextFoolerAttack, BERTAttack
from blue_team.nlp_models import PerplexityDetector, SemanticDetector, EnsembleDetector
from pipeline.config import PipelineConfig
from pipeline.orchestrator import Orchestrator


def sanitize_model_name(model_path: str) -> str:
    """Extract and sanitize model name for use in filenames.

    Args:
        model_path: Path or name of the model

    Returns:
        Sanitized model name suitable for filenames

    Examples:
        "models/nlp/lr_tfidf_model.onnx" -> "lr_tfidf_model"
        "distilbert-base-uncased" -> "distilbert_base_uncased"
        "/path/to/my-model.onnx" -> "my_model"
    """
    # Get filename from path
    name = Path(model_path).stem  # Removes extension and gets filename

    # Replace hyphens and special characters with underscores
    name = name.replace('-', '_').replace(' ', '_')

    # Remove any remaining special characters except underscores
    name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)

    # Remove consecutive underscores
    while '__' in name:
        name = name.replace('__', '_')

    # Remove leading/trailing underscores
    name = name.strip('_')

    return name


def set_deterministic(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data_from_file(
    file_path: str,
    text_column: str = None,
    label_column: str = None
) -> Tuple[List[str], List[int]]:
    """Load test data from JSON or CSV file with flexible column detection.

    Automatically detects common column name variations or uses custom names.

    Common text column names: text, tweet, sentence, content, comment, review
    Common label column names: label, target, class, category, sentiment

    Args:
        file_path: Path to JSON or CSV file
        text_column: Custom name for text column (optional, auto-detects if None)
        label_column: Custom name for label column (optional, auto-detects if None)

    Returns:
        Tuple of (texts, labels)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns not found
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Common column name variations
    TEXT_COLUMN_NAMES = ['text', 'tweet', 'sentence', 'content', 'comment', 'review', 'message']
    LABEL_COLUMN_NAMES = ['label', 'target', 'class', 'category', 'sentiment', 'y']

    if path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            raise ValueError(f"JSON file is empty: {file_path}")

        # Get available keys
        available_keys = list(data[0].keys()) if isinstance(data, list) else []

        # Detect text column
        if text_column:
            text_col = text_column
        else:
            text_col = next((col for col in TEXT_COLUMN_NAMES if col in available_keys), None)
            if not text_col:
                raise ValueError(
                    f"Could not find text column. Available columns: {available_keys}\n"
                    f"Use --text-column to specify custom column name."
                )

        # Detect label column
        if label_column:
            lbl_col = label_column
        else:
            lbl_col = next((col for col in LABEL_COLUMN_NAMES if col in available_keys), None)
            if not lbl_col:
                raise ValueError(
                    f"Could not find label column. Available columns: {available_keys}\n"
                    f"Use --label-column to specify custom column name."
                )

        texts = [item[text_col] for item in data]
        labels = [int(item[lbl_col]) for item in data]

    elif path.suffix == '.csv':
        texts, labels = [], []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Read first row to detect columns
            first_row = None
            rows = []
            for row in reader:
                if first_row is None:
                    first_row = row
                rows.append(row)

            if not first_row:
                raise ValueError(f"CSV file is empty: {file_path}")

            available_cols = list(first_row.keys())

            # Detect text column
            if text_column:
                text_col = text_column
                if text_col not in available_cols:
                    raise ValueError(
                        f"Text column '{text_col}' not found.\n"
                        f"Available columns: {available_cols}"
                    )
            else:
                text_col = next((col for col in TEXT_COLUMN_NAMES if col in available_cols), None)
                if not text_col:
                    raise ValueError(
                        f"Could not auto-detect text column.\n"
                        f"Available columns: {available_cols}\n"
                        f"Expected one of: {TEXT_COLUMN_NAMES}\n"
                        f"Use --text-column <column_name> to specify manually."
                    )

            # Detect label column
            if label_column:
                lbl_col = label_column
                if lbl_col not in available_cols:
                    raise ValueError(
                        f"Label column '{lbl_col}' not found.\n"
                        f"Available columns: {available_cols}"
                    )
            else:
                lbl_col = next((col for col in LABEL_COLUMN_NAMES if col in available_cols), None)
                if not lbl_col:
                    raise ValueError(
                        f"Could not auto-detect label column.\n"
                        f"Available columns: {available_cols}\n"
                        f"Expected one of: {LABEL_COLUMN_NAMES}\n"
                        f"Use --label-column <column_name> to specify manually."
                    )

            print(f"      [INFO] Using text column: '{text_col}'")
            print(f"      [INFO] Using label column: '{lbl_col}'")

            # Extract data
            for row in rows:
                texts.append(row[text_col])
                labels.append(int(row[lbl_col]))

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .csv")

    return texts, labels


def create_default_test_data() -> Tuple[List[str], List[int]]:
    """Create default test data for disaster tweet classification.

    Returns balanced set of disaster (1) and non-disaster (0) tweets.
    """
    texts = [
        # Disaster tweets (label=1)
        "Just happened a terrible car crash on Highway 101",
        "Earthquake shakes California, buildings damaged",
        "Forest fire spreading rapidly, evacuations underway",
        "Tornado warning issued for the county",

        # Non-disaster tweets (label=0)
        "I love this beautiful sunny day!",
        "Great movie, highly recommended to everyone",
        "The new restaurant downtown has amazing food",
        "What a wonderful time at the park today",
    ]
    # Balanced labels: 4 disasters, 4 non-disasters
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    return texts, labels


def create_nlp_dataloader(
    texts: List[str],
    labels: List[int],
    batch_size: int = 4
) -> List[Tuple[List[str], List[int]]]:
    """Create iterable NLP dataloader.

    Args:
        texts: List of text strings
        labels: List of labels
        batch_size: Number of samples per batch

    Returns:
        List of (text_batch, label_batch) tuples
    """
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append((
            texts[i:i+batch_size],
            labels[i:i+batch_size]
        ))
    return batches


def create_detection_fn(detectors):
    """Create detection function for the pipeline.

    Args:
        detectors: Ensemble detector instance

    Returns:
        Detection function compatible with pipeline
    """
    def detection_fn(attack_results: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
        """Run detection on attack results."""
        for result in attack_results:
            original_text = result.get('original_text', '')
            adversarial_text = result.get('adversarial_text', '')
            original_pred = result.get('original_prediction')
            adversarial_pred = result.get('adversarial_prediction')

            detection_result = detectors.detect(
                original_output=original_pred,
                adversarial_output=adversarial_pred,
                original_text=original_text,
                adversarial_text=adversarial_text
            )

            yield {**result, **detection_result}

    return detection_fn


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified NLP adversarial evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='distilbert-base-uncased-finetuned-sst-2-english',
        help='Model path or HuggingFace Hub name (default: DistilBERT SST-2)'
    )
    parser.add_argument(
        '--tokenizer', '-t',
        type=str,
        default=None,
        help='Tokenizer name (required for ONNX models)'
    )
    parser.add_argument(
        '--labels', '-l',
        type=str,
        default=None,
        help='Label mapping as JSON string (e.g., \'{"0": "NEGATIVE", "1": "POSITIVE"}\')'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Path to test data file (JSON or CSV)'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default=None,
        help='Name of text column in data file (auto-detects if not specified)'
    )
    parser.add_argument(
        '--label-column',
        type=str,
        default=None,
        help='Name of label column in data file (auto-detects if not specified)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=4,
        help='Batch size for evaluation (default: 4)'
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=2,
        help='Maximum number of batches to process (default: 2)'
    )
    parser.add_argument(
        '--attacks',
        nargs='+',
        choices=['textfooler', 'bertattack'],
        default=['textfooler', 'bertattack'],
        help='Attacks to run (default: textfooler, bertattack)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='reports',
        help='Output directory for reports (default: reports)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=-1,
        help='Device for inference: -1 for CPU, 0+ for GPU (default: -1)'
    )
    parser.add_argument(
        '--use-llm-diagnostics',
        action='store_true',
        help='Enable LLM-enhanced diagnostic explanations (requires API key)'
    )
    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['openai', 'anthropic', 'github'],
        default='openai',
        help='LLM provider for enhanced diagnostics (default: openai, options: openai, anthropic, github)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("NLP Adversarial Evaluation Pipeline")
    print("=" * 80)

    # Set seed
    set_deterministic(args.seed)
    print(f"\n[Config] Random seed: {args.seed}")

    # Parse label mapping if provided
    label_mapping = None
    if args.labels:
        try:
            label_mapping = {int(k): v for k, v in json.loads(args.labels).items()}
            print(f"[Config] Label mapping: {label_mapping}")
        except Exception as e:
            print(f"[WARNING] Could not parse label mapping: {e}")

    # Load model
    print(f"\n[1/5] Loading model: {args.model}")
    try:
        model = load_nlp_model(
            args.model,
            tokenizer_name=args.tokenizer,
            label_mapping=label_mapping,
            device=args.device
        )
        print("      [OK] Model loaded successfully")
    except Exception as e:
        print(f"      [ERROR] Error loading model: {e}")
        print("\nUsage tips:")
        print("  - For ONNX models: Specify --tokenizer")
        print("  - For HuggingFace: Just provide model name")
        print("  - Example: python main_nlp.py --model distilbert-base-uncased-finetuned-sst-2-english")
        return 1

    # Load data
    print(f"\n[2/5] Loading test data")
    try:
        if args.data:
            texts, labels = load_data_from_file(
                args.data,
                text_column=args.text_column,
                label_column=args.label_column
            )
            print(f"      [OK] Loaded {len(texts)} samples from {args.data}")
        else:
            texts, labels = create_default_test_data()
            print(f"      [OK] Using default test data ({len(texts)} samples)")
    except Exception as e:
        print(f"      [ERROR] Error loading data: {e}")
        return 1

    dataloader = create_nlp_dataloader(texts, labels, args.batch_size)
    print(f"      [OK] Created {len(dataloader)} batches (batch_size={args.batch_size})")

    # Configure attacks
    print(f"\n[3/5] Configuring attacks: {', '.join(args.attacks)}")
    attacks = []
    if 'textfooler' in args.attacks:
        attacks.append(TextFoolerAttack(max_candidates=10, max_perturbed_words=3))
        print("      [OK] TextFooler attack configured")
    if 'bertattack' in args.attacks:
        attacks.append(BERTAttack(bert_model="bert-base-uncased", top_k=10))
        print("      [OK] BERT-Attack configured (this may be slow)")

    # Configure detectors
    print(f"\n[4/5] Configuring detectors")
    detectors = EnsembleDetector(
        detectors=[
            PerplexityDetector(perplexity_threshold=200.0, lm_model="distilgpt2"),
            SemanticDetector(similarity_threshold=0.7, embedding_model="all-MiniLM-L6-v2"),
        ],
        mode="vote"
    )
    detection_fn = create_detection_fn(detectors)
    print("      [OK] Ensemble detector configured (Perplexity + Semantic)")

    # Configure pipeline
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate model-specific filenames
    model_name = sanitize_model_name(args.model)

    config = PipelineConfig(
        model=model,
        dataloader=dataloader,
        attacks=attacks,
        report_path=output_dir / f"{model_name}_security_report.pdf",
        attack_results_path=output_dir / f"{model_name}_attack_results.json",
        detection_results_path=output_dir / f"{model_name}_detection_results.json",
        project_title=f"Adversarial Security Report - {model_name}",
        model_info={
            "name": args.model,
            "framework": "NLP (Auto-detected)",
            "device": "CPU" if args.device == -1 else f"GPU:{args.device}",
            "label_mapping": label_mapping,  # Add label mapping for diagnostics
        },
        detection_fn=detection_fn,
        max_batches=args.max_batches,
        model_type="nlp"
    )

    # Add LLM diagnostic settings to config
    config.use_llm_diagnostics = args.use_llm_diagnostics
    config.llm_provider = args.llm_provider

    # Run pipeline
    print(f"\n[5/5] Running evaluation pipeline")
    print("=" * 80)

    orchestrator = Orchestrator()

    try:
        results = orchestrator.run(config)

        # Display results
        print("\n" + "=" * 80)
        print("Evaluation Complete!")
        print("=" * 80)

        summary = results.get('summary', {})
        print(f"\n[Results Summary]")
        print(f"  Attack Success Rate:  {summary.get('attack_success_rate', 0):.2%}")
        print(f"  Detection Accuracy:   {summary.get('detection_accuracy', 0):.2%}")
        print(f"  False Positive Rate:  {summary.get('false_positive_rate', 0):.2%}")
        print(f"  Robustness Score:     {summary.get('robustness_score', 0):.2f}/100")

        perturbation = summary.get('perturbation_metrics', {})
        if perturbation:
            print(f"\n[Perturbation Metrics]")
            print(f"  Avg Word Changes:     {perturbation.get('avg_word_changes', 0):.2f}")
            print(f"  Avg Edit Distance:    {perturbation.get('avg_edit_distance', 0):.2f}")

        print(f"\n[Reports saved to]")
        print(f"  - PDF Report:         {config.report_path}")
        print(f"  - Attack Results:     {config.attack_results_path}")
        print(f"  - Detection Results:  {config.detection_results_path}")
        print(f"  - Summary:            {output_dir / 'summary.json'}")

    except Exception as e:
        print(f"\n[ERROR] Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 80)
    print("[SUCCESS] NLP Adversarial Evaluation Complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
