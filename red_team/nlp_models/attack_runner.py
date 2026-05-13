"""NLP-specific attack runner for text-based models.

This module provides the NLP equivalent of red_team.image_models.AttackRunner.
It handles text inputs (List[str]) instead of tensors, while maintaining the
same interface pattern for consistency with the pipeline architecture.

Example Usage:
    from red_team.nlp_models import NLPAttackRunner, TextFoolerAttack
    from transformers import pipeline

    # Load model
    model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Setup attacks
    attacks = [TextFoolerAttack(max_candidates=10, max_perturbed_words=3)]

    # Create runner
    runner = NLPAttackRunner(model, attacks)

    # Run on batch
    texts = ["This is great!", "I love it!"]
    labels = [1, 1]
    results = runner.run_batch(texts, labels)
"""

from typing import Any, Dict, List, Sequence

from .base_attack import BaseAttack


class NLPAttackRunner:
    """Run NLP attacks over text datasets.

    This class is the NLP equivalent of red_team.image_models.AttackRunner.
    It provides the same interface but handles text inputs instead of tensors,
    making it compatible with the existing pipeline infrastructure.

    Attributes:
        model: NLP model (HuggingFace pipeline or compatible wrapper)
        attacks: List of NLP attack instances
        results: Accumulated results from all batches
    """

    def __init__(self, model: Any, attacks: Sequence[BaseAttack]):
        """Initialize runner with model and attacks.

        Args:
            model: NLP model with HuggingFace pipeline interface
                  (must have __call__(texts) method returning predictions)
            attacks: Sequence of NLP attack instances (e.g., TextFoolerAttack, BERTAttack)
        """
        self.model = model
        self.attacks = list(attacks)
        self.results: List[Dict[str, Any]] = []

    def run_batch(self, texts: List[str], labels: List[int]) -> List[Dict[str, Any]]:
        """Run all attacks on a batch of text inputs.

        This method runs each configured attack on the provided text batch
        and returns the results. The interface matches image AttackRunner
        for consistency.

        Args:
            texts: List of text strings to attack
            labels: List of ground-truth integer labels

        Returns:
            List of attack result dictionaries, one per attack:
            [
                {
                    "attack": "TextFoolerAttack",
                    "result": {
                        "adversarial_texts": ["perturbed text 1", ...],
                        "original_predictions": [...],
                        "adversarial_predictions": [...],
                        "attack_success_rate": 0.75,
                        "perturbation_metrics": {...}
                    },
                    "per_sample_diagnostics": [
                        {
                            "original_text": "...",
                            "original_label": 1,
                            "original_pred": {...},
                            "adversarial_text": "...",
                            "adversarial_pred": {...},
                            "perturbation_attempted": True,
                            "prediction_flipped": False,
                            "confidence_change": -0.05,
                            "words_changed": 2
                        },
                        ...
                    ]
                },
                ...
            ]

        Example:
            runner = NLPAttackRunner(model, [TextFoolerAttack()])
            texts = ["This is great!", "Amazing product!"]
            labels = [1, 1]
            results = runner.run_batch(texts, labels)
            print(f"Attack success: {results[0]['result']['attack_success_rate']}")
        """
        batch_results = []

        for attack in self.attacks:
            # Run attack on the batch
            attack_output = attack.generate(self.model, texts, labels)

            # NEW: Collect per-sample diagnostics for detailed analysis
            per_sample_diagnostics = []
            for i, text in enumerate(texts):
                # Get original prediction
                original_pred = self.model([text])[0]

                # Get adversarial text (use original if no perturbation)
                adv_text = attack_output.get('adversarial_texts', [text])[i]

                # Get adversarial prediction
                adv_pred = self.model([adv_text])[0]

                # Calculate diagnostic metrics
                per_sample_diagnostics.append({
                    "original_text": text,
                    "original_label": labels[i],
                    "original_pred": original_pred,
                    "adversarial_text": adv_text,
                    "adversarial_pred": adv_pred,
                    "perturbation_attempted": text != adv_text,
                    "prediction_flipped": original_pred['label'] != adv_pred['label'],
                    "confidence_change": adv_pred['score'] - original_pred['score'],
                    "words_changed": self._count_word_differences(text, adv_text)
                })

            # Package result with attack name and diagnostics
            batch_results.append({
                "attack": attack.__class__.__name__,
                "result": attack_output,  # Already JSON-serializable
                "per_sample_diagnostics": per_sample_diagnostics
            })

        # Accumulate results
        self.results.extend(batch_results)

        return batch_results

    def _count_word_differences(self, text1: str, text2: str) -> int:
        """Count word-level differences between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Number of words that differ between texts
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        return len(words1.symmetric_difference(words2))

    def run_dataset(self, dataset: List[tuple]) -> List[Dict[str, Any]]:
        """Run all attacks over an iterable of (texts, labels) batches.

        This method processes multiple batches and aggregates the results.
        Useful for running attacks on larger datasets.

        Args:
            dataset: Iterable of (texts, labels) tuples, where:
                    - texts is a List[str] of text inputs
                    - labels is a List[int] of ground-truth labels

        Returns:
            Aggregated list of attack results across all batches,
            with batch index added to each result

        Example:
            # Create dataset
            dataset = [
                (["Great movie!",  "Bad film."], [1, 0]),
                (["Love it!", "Terrible."], [1, 0]),
            ]

            # Run attacks
            runner = NLPAttackRunner(model, [TextFoolerAttack()])
            results = runner.run_dataset(dataset)
            print(f"Total batches processed: {len(set(r['batch_index'] for r in results))}")
        """
        aggregated = []

        for batch_index, (texts, labels) in enumerate(dataset):
            # Run attacks on this batch
            batch_res = self.run_batch(texts, labels)

            # Add batch index to each result
            for res in batch_res:
                res_with_batch = dict(res)
                res_with_batch["batch_index"] = batch_index
                aggregated.append(res_with_batch)

        return aggregated

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all accumulated results.

        Returns:
            List of all attack results accumulated across all run_batch calls
        """
        return self.results

    def clear_results(self):
        """Clear accumulated results.

        Useful for running multiple independent evaluations with the same runner.
        """
        self.results = []


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("NLP Attack Runner - Example Usage")
    print("=" * 80)

    # This example requires transformers and red_team.nlp_models imports
    try:
        from transformers import pipeline
        from .textfooler_attack import TextFoolerAttack

        print("\nLoading sentiment analysis model...")
        model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )

        print("\nSetting up attacks...")
        attacks = [TextFoolerAttack(max_candidates=5, max_perturbed_words=2)]

        print("\nCreating attack runner...")
        runner = NLPAttackRunner(model, attacks)

        print("\nRunning attacks on test batch...")
        texts = [
            "This movie is absolutely fantastic!",
            "I really enjoyed this product.",
        ]
        labels = [1, 1]  # All positive

        results = runner.run_batch(texts, labels)

        print("\nResults:")
        for i, result in enumerate(results):
            print(f"\n  Attack {i+1}: {result['attack']}")
            attack_result = result['result']
            print(f"    Success Rate: {attack_result['attack_success_rate']:.2%}")
            print(f"    Adversarial Examples:")
            for j, adv_text in enumerate(attack_result['adversarial_texts'][:2]):
                print(f"      [{j+1}] {texts[j]}")
                print(f"          → {adv_text}")

        print("\n" + "=" * 80)
        print("✓ Attack runner example completed")

    except ImportError as e:
        print(f"\nSkipped example (missing dependencies): {e}")
        print("\nTo run example, ensure you have:")
        print("  - transformers library installed")
        print("  - red_team.nlp_models attacks implemented")
