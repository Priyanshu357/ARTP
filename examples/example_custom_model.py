"""Quick test script for your custom trained model."""

import os
from transformers import pipeline
from red_team.nlp_models import TextFoolerAttack
from blue_team.nlp_models import PerplexityDetector, SemanticDetector


def main():
    print("=" * 80)
    print("Testing Your Custom Model")
    print("=" * 80)

    # ============================================
    # MODIFY THIS SECTION FOR YOUR MODEL
    # ============================================

    # Your model path
    MODEL_PATH = "adversarial-platform\\models\\nlp\\distilbert_Opset17.onnx"  # ← CHANGE THIS

    # Load model
    print(f"\n📦 Loading model from: {MODEL_PATH}")

    # Check if ONNX model
    if MODEL_PATH.endswith('.onnx'):
        print("   Detected ONNX format - using ONNX wrapper...")
        from onnx_model_wrapper import load_onnx_model

        # Load ONNX model with appropriate tokenizer
        # Change tokenizer_name to match your model
        classifier = load_onnx_model(
            onnx_path=MODEL_PATH,
            tokenizer_name='distilbert-base-uncased-finetuned-sst-2-english',  # ← CHANGE if needed
            label_mapping={0: 'NEGATIVE', 1: 'POSITIVE'}  # ← CHANGE for your labels
        )
        print("   ✓ ONNX model loaded successfully!")

    else:
        # HuggingFace model (local path or Hub name)
        classifier = pipeline(
            "text-classification",
            model=MODEL_PATH,
            device=-1  # CPU
        )
        print("   ✓ Model loaded successfully!")

    # ============================================
    # YOUR TEST DATA
    # ============================================

    # Replace with your test examples
    test_texts = [
        "This is a great product!",
        "I love this movie!",
        "Excellent service and quality.",
    ]

    # Ground truth labels (adjust for your task)
    # For binary: 0 = negative, 1 = positive
    # For multi-class: 0, 1, 2, 3, ...
    labels = [1, 1, 1]

    print(f"\n📝 Testing with {len(test_texts)} samples")

    # ============================================
    # RUN ATTACK
    # ============================================

    print("\n" + "=" * 80)
    print("🔴 PHASE 1: Running TextFooler Attack")
    print("=" * 80)

    attack = TextFoolerAttack(max_candidates=10, max_perturbed_words=3)

    # For HuggingFace pipeline (Option 1)
    results = attack.generate(classifier, test_texts, labels)

    # For raw model (Option 2) - uncomment and pass tokenizer
    # results = attack.generate(model, test_texts, labels, tokenizer=tokenizer)

    print(f"\n   Attack Success Rate: {results['attack_success_rate']:.2%}")
    print(f"   Avg Word Changes: {results['perturbation_metrics']['avg_word_changes']:.2f}")

    print("\n   Examples:")
    for i in range(len(test_texts)):
        print(f"\n   [{i+1}] Original:    {test_texts[i]}")
        print(f"       Adversarial: {results['adversarial_texts'][i]}")
        orig = results['original_predictions'][i]
        adv = results['adversarial_predictions'][i]
        orig_label = orig.get('label', 'N/A') if isinstance(orig, dict) else 'N/A'
        adv_label = adv.get('label', 'N/A') if isinstance(adv, dict) else 'N/A'
        print(f"       Prediction: {orig_label} → {adv_label}")

    # ============================================
    # RUN DETECTION
    # ============================================

    print("\n" + "=" * 80)
    print("🔵 PHASE 2: Running Detection")
    print("=" * 80)

    perplexity_detector = PerplexityDetector(perplexity_threshold=200.0)
    semantic_detector = SemanticDetector(similarity_threshold=0.7)

    detected_count = 0
    for i in range(len(test_texts)):
        detection = perplexity_detector.detect(
            original_output=results['original_predictions'][i],
            adversarial_output=results['adversarial_predictions'][i],
            original_text=test_texts[i],
            adversarial_text=results['adversarial_texts'][i]
        )

        if detection['detected']:
            detected_count += 1

        print(f"\n   [{i+1}] Detected: {detection['detected']}")
        print(f"       {detection['explanation']}")

    detection_rate = detected_count / len(test_texts) if test_texts else 0
    print(f"\n   Detection Rate: {detection_rate:.2%}")

    print("\n" + "=" * 80)
    print("✅ Testing Complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
