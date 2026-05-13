"""Test script for NLP adversarial attacks and detectors.

This script demonstrates how to use the NLP models implementation
with a HuggingFace sentiment analysis model.
"""

from transformers import pipeline
from red_team.nlp_models import TextFoolerAttack, BERTAttack
from blue_team.nlp_models import PerplexityDetector, SemanticDetector, EnsembleDetector


def test_textfooler_attack():
    """Test TextFooler attack on sentiment classifier."""
    print("=" * 80)
    print("Testing TextFooler Attack")
    print("=" * 80)

    # Load sentiment analysis model
    print("Loading sentiment analysis model...")
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Test texts (positive sentiment)
    texts = [
        "I really enjoyed this movie, it was fantastic!",
        "The performance was absolutely brilliant and captivating.",
        "This is wonderful, I loved every minute of it."
    ]
    labels = [1, 1, 1]  # All POSITIVE

    # Create attack
    attack = TextFoolerAttack(max_candidates=10, max_perturbed_words=3)

    # Run attack
    print("\nRunning TextFooler attack...")
    result = attack.generate(classifier, texts, labels)

    # Display results
    print(f"\nAttack Success Rate: {result['attack_success_rate']:.2%}")
    print(f"Average Edit Distance: {result['perturbation_metrics']['avg_edit_distance']:.2f}")
    print(f"Average Word Changes: {result['perturbation_metrics']['avg_word_changes']:.2f}")

    print("\n" + "-" * 80)
    print("Examples:")
    for i, (orig, adv) in enumerate(zip(texts, result['adversarial_texts'])):
        print(f"\n[{i+1}] Original: {orig}")
        print(f"    Adversarial: {adv}")
        print(f"    Original pred: {result['original_predictions'][i]}")
        print(f"    Adversarial pred: {result['adversarial_predictions'][i]}")


def test_bertattack():
    """Test BERT-Attack on sentiment classifier."""
    print("\n" + "=" * 80)
    print("Testing BERT-Attack")
    print("=" * 80)

    # Load sentiment analysis model
    print("Loading sentiment analysis model...")
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Test texts
    texts = ["This product is amazing and works perfectly!"]
    labels = [1]  # POSITIVE

    # Create attack
    attack = BERTAttack(bert_model="bert-base-uncased", top_k=10)

    # Run attack
    print("\nRunning BERT-Attack...")
    result = attack.generate(classifier, texts, labels)

    # Display results
    print(f"\nAttack Success Rate: {result['attack_success_rate']:.2%}")
    print(f"Average Word Changes: {result['perturbation_metrics']['avg_word_changes']:.2f}")
    print(f"\nOriginal: {texts[0]}")
    print(f"Adversarial: {result['adversarial_texts'][0]}")


def test_detectors():
    """Test detection of adversarial texts."""
    print("\n" + "=" * 80)
    print("Testing Detectors")
    print("=" * 80)

    # Example texts
    original_text = "I really enjoyed this movie, it was fantastic!"
    adversarial_text = "I really hated this movie, it was terrible!"

    # Test Perplexity Detector
    print("\n1. Perplexity Detector")
    print("-" * 40)
    perplexity_detector = PerplexityDetector(perplexity_threshold=200.0)
    result = perplexity_detector.detect(None, None, original_text, adversarial_text)
    print(f"Detected: {result['detected']}")
    print(f"Anomaly Score: {result['anomaly_score']:.3f}")
    print(f"Explanation: {result['explanation']}")

    # Test Semantic Detector
    print("\n2. Semantic Detector")
    print("-" * 40)
    semantic_detector = SemanticDetector(similarity_threshold=0.7)
    result = semantic_detector.detect(None, None, original_text, adversarial_text)
    print(f"Detected: {result['detected']}")
    print(f"Anomaly Score: {result['anomaly_score']:.3f}")
    print(f"Explanation: {result['explanation']}")

    # Test Ensemble Detector
    print("\n3. Ensemble Detector")
    print("-" * 40)
    ensemble = EnsembleDetector(
        detectors=[perplexity_detector, semantic_detector],
        mode="vote"
    )
    result = ensemble.detect(None, None, original_text, adversarial_text)
    print(f"Detected: {result['detected']}")
    print(f"Anomaly Score: {result['anomaly_score']:.3f}")
    print(f"Explanation: {result['explanation']}")


def main():
    """Run all tests."""
    print("\nNLP Adversarial Models Test Suite")
    print("=" * 80)

    try:
        # Test attacks
        test_textfooler_attack()
        # test_bertattack()  # Uncomment to test BERT-Attack (slower)

        # Test detectors
        test_detectors()

        print("\n" + "=" * 80)
        print("All tests completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
