"""Test script for various imported/pre-trained models."""

from transformers import pipeline
from red_team.nlp_models import TextFoolerAttack
from blue_team.nlp_models import PerplexityDetector


def test_model(model_name, test_texts, labels):
    """Test any HuggingFace model."""
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print('='*80)

    # Load model
    print(f"📦 Loading {model_name}...")
    try:
        classifier = pipeline("text-classification", model=model_name, device=-1)
        print("   ✓ Model loaded!")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return

    # Run attack
    print("\n🔴 Running TextFooler attack...")
    attack = TextFoolerAttack(max_candidates=5, max_perturbed_words=2)

    try:
        results = attack.generate(classifier, test_texts, labels)
        print(f"   Attack Success Rate: {results['attack_success_rate']:.2%}")

        # Show example
        if results['adversarial_texts']:
            print(f"\n   Example:")
            print(f"   Original:    {test_texts[0]}")
            print(f"   Adversarial: {results['adversarial_texts'][0]}")

        return results
    except Exception as e:
        print(f"   ✗ Error during attack: {e}")
        return None


def main():
    print("="*80)
    print("Testing Imported/Pre-trained Models")
    print("="*80)

    # Test data
    test_texts = [
        "This movie is absolutely fantastic!",
        "I really enjoyed this product.",
    ]
    labels = [1, 1]  # Positive sentiment

    # ============================================
    # Example 1: DistilBERT (Sentiment Analysis)
    # ============================================
    test_model(
        "distilbert-base-uncased-finetuned-sst-2-english",
        test_texts,
        labels
    )

    # ============================================
    # Example 2: RoBERTa (Sentiment)
    # ============================================
    # Uncomment to test:
    # test_model(
    #     "cardiffnlp/twitter-roberta-base-sentiment",
    #     test_texts,
    #     labels
    # )

    # ============================================
    # Example 3: BERT Base
    # ============================================
    # Uncomment to test:
    # test_model(
    #     "textattack/bert-base-uncased-SST-2",
    #     test_texts,
    #     labels
    # )

    # ============================================
    # Example 4: Your Imported Model
    # ============================================
    # Add your model here:
    # test_model(
    #     "your-username/your-model-name",  # HuggingFace Hub
    #     test_texts,
    #     labels
    # )

    print("\n" + "="*80)
    print("✅ Testing Complete!")
    print("="*80)
    print("\nTo test your own imported model:")
    print("1. Find your model on HuggingFace Hub")
    print("2. Copy the model name (e.g., 'bert-base-uncased')")
    print("3. Add it to this script in Example 4")
    print("4. Run: python test_imported_models.py")


if __name__ == "__main__":
    main()
