"""Comprehensive diagnostic analysis for model behavior and attack results.

This module provides automated diagnosis of common model issues including:
- Class bias (model always predicts one class)
- Confidence pattern anomalies (flat, overconfident, underconfident)
- Attack failure analysis (why attacks succeeded/failed)
- Label consistency issues
- Prediction distribution problems

The diagnostics help users understand WHAT is happening with their model,
not just surface-level metrics.
"""

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List
import numpy as np


# Severity levels ordered by importance
SEVERITY_ORDER = {
    'CRITICAL': 0,
    'HIGH': 1,
    'MEDIUM': 2,
    'WARNING': 3,
    'LOW': 4,
    'INFO': 5
}


class DiagnosticAnalyzer:
    """Analyzes model behavior and attack results for issues.

    This class runs multiple diagnostic checks on evaluation results
    and generates structured diagnostic reports with severity levels,
    evidence, root causes, and recommendations.

    Example:
        analyzer = DiagnosticAnalyzer()
        diagnostics = analyzer.analyze(attack_results, detection_results, summary, config)
        print(f"Model Health: {diagnostics['overall_health']}")
        for diag in diagnostics['diagnostics']:
            print(f"[{diag['severity']}] {diag['description']}")
    """

    def __init__(self, use_llm: bool = False, llm_provider: str = "openai"):
        """Initialize diagnostic analyzer with optional LLM enhancement.

        Args:
            use_llm: Enable LLM enhancement for CRITICAL/HIGH diagnostics
            llm_provider: "openai" or "anthropic"
        """
        self.use_llm = use_llm
        self.llm_enhancer = None

        if use_llm:
            try:
                from .llm_enhancer import LLMEnhancer
                self.llm_enhancer = LLMEnhancer(provider=llm_provider)
                if not self.llm_enhancer.enabled:
                    print("[WARNING] LLM enhancement requested but API key not found")
                    key_name = {
                        'openai': 'OPENAI_API_KEY',
                        'anthropic': 'ANTHROPIC_API_KEY',
                        'github': 'GITHUB_TOKEN'
                    }.get(llm_provider, 'API_KEY')
                    print(f"[WARNING] Set {key_name} environment variable")
                    self.llm_enhancer = None
            except ImportError as e:
                print(f"[WARNING] Could not import LLM enhancer: {e}")
                self.llm_enhancer = None

    def analyze(
        self,
        attack_results: List[Dict],
        detection_results: List[Dict],
        summary: Dict,
        config: Any
    ) -> Dict[str, Any]:
        """Run all diagnostic checks and return structured report.

        Args:
            attack_results: Results from attack execution
            detection_results: Results from detection pipeline
            summary: Summary metrics dict
            config: PipelineConfig instance

        Returns:
            Diagnostic report dict with structure:
            {
                "timestamp": "ISO format timestamp",
                "model_info": {...},
                "diagnostics": [
                    {
                        "diagnostic": "class_imbalance_bias",
                        "severity": "CRITICAL",
                        "description": "Model predicts 'DISASTER' for 100% of inputs",
                        "evidence": {...},
                        "root_causes": [...],
                        "recommendations": [...],
                        "interpretation": "...",
                        "verdict": "..."
                    },
                    ...
                ],
                "severity_counts": {"CRITICAL": 1, "HIGH": 2, ...},
                "overall_health": "CRITICAL" | "CONCERNING" | "HEALTHY"
            }
        """
        diagnostics = []

        # Run universal diagnostic checks (work for all model types)
        diagnostics.extend(self.check_class_bias(attack_results))
        diagnostics.extend(self.check_confidence_patterns(attack_results))
        diagnostics.extend(self.check_label_consistency(attack_results, config))

        # Model-type specific checks
        model_type = getattr(config, 'model_type', None)

        if model_type == 'audio' or self._is_audio_data(attack_results):
            # Audio-specific diagnostics
            diagnostics.extend(self.check_audio_quality(attack_results))
            diagnostics.extend(self.check_audio_attack_failures(attack_results))
        elif model_type == 'object_detection' or self._is_object_detection_data(attack_results):
            # Object detection-specific diagnostics
            diagnostics.extend(self.check_object_detection_quality(attack_results))
            diagnostics.extend(self.check_object_detection_failures(attack_results))
        else:
            # NLP/Image attack failure analysis
            diagnostics.extend(self.analyze_attack_failures(attack_results))

            # NLP-specific: Check for preprocessing-based defenses
            diagnostics.extend(self.check_nlp_preprocessing_defense(attack_results, config))

        # Sort by severity (most critical first)
        diagnostics.sort(key=lambda d: SEVERITY_ORDER[d['severity']])

        # NEW: Enhance CRITICAL/HIGH severity diagnostics with LLM (if enabled)
        if self.llm_enhancer and self.llm_enhancer.enabled:
            print("\n[LLM] Enhancing critical diagnostics with AI analysis...")

            # Build context for LLM
            context = {
                "model_name": config.normalized_model_info().get('name', 'Unknown') if hasattr(config, 'normalized_model_info') else 'Unknown',
                "task": "Text Classification",
                "attacks": [r['attack'] for r in attack_results],
                "num_samples": sum(len(r.get('per_sample_diagnostics', []))
                                  for r in attack_results)
            }

            # Enhance only CRITICAL/HIGH severity diagnostics (cost optimization)
            enhanced_count = 0
            for i, diag in enumerate(diagnostics):
                if diag['severity'] in ['CRITICAL', 'HIGH']:
                    diagnostics[i] = self.llm_enhancer.enhance_diagnostic(diag, context)
                    if diagnostics[i].get('llm_enhanced'):
                        enhanced_count += 1

            print(f"[LLM] Enhanced {enhanced_count} diagnostic(s)")

        return {
            "timestamp": datetime.now().isoformat(),
            "model_info": config.normalized_model_info() if hasattr(config, 'normalized_model_info') else {},
            "diagnostics": diagnostics,
            "severity_counts": self._count_by_severity(diagnostics),
            "overall_health": self._compute_health_score(diagnostics),
            "llm_enhanced": self.llm_enhancer is not None and self.llm_enhancer.enabled
        }

    def check_class_bias(self, attack_results: List[Dict]) -> List[Dict]:
        """Detect if model always predicts one class.

        Args:
            attack_results: Attack execution results

        Returns:
            List of diagnostic dicts (empty if no bias detected)
        """
        all_predictions = self._extract_all_predictions(attack_results)
        if not all_predictions:
            return []

        class_counts = Counter(p['label'] for p in all_predictions)
        total = len(all_predictions)
        diagnostics = []

        for label, count in class_counts.items():
            ratio = count / total
            if ratio > 0.9:  # 90% or more predictions are same class
                # Format label for display (handle both string and integer labels)
                label_str = str(label) if isinstance(label, int) else label

                diagnostics.append({
                    "diagnostic": "class_imbalance_bias",
                    "severity": "CRITICAL",
                    "description": f"Model predicts '{label_str}' for {ratio*100:.0f}% of inputs ({count}/{total} samples)",
                    "evidence": {
                        "class_distribution": {k: v/total for k, v in class_counts.items()},
                        "dominant_class": label,
                        "dominance_ratio": ratio,
                        "samples_examined": total
                    },
                    "root_causes": [
                        "Training data severely imbalanced",
                        "Model underfitting - learned to predict majority class only",
                        "Incorrect label mapping during model export/conversion",
                        "Model not properly trained or checkpoint corrupted"
                    ],
                    "recommendations": [
                        f"Verify training data has balanced classes (not {ratio*100:.0f}% '{label}')",
                        "Check if model checkpoint loaded correctly",
                        "Try swapping label mapping (0↔1) to test for export error",
                        "Re-train model with class weighting or oversampling"
                    ],
                    "interpretation": f"Model is stuck predicting '{label}' regardless of input. This indicates a serious training or configuration issue, not genuine robustness."
                })

        return diagnostics

    def check_confidence_patterns(self, attack_results: List[Dict]) -> List[Dict]:
        """Analyze confidence score patterns.

        Args:
            attack_results: Attack execution results

        Returns:
            List of diagnostic dicts (empty if no anomalies detected)
        """
        all_predictions = self._extract_all_predictions(attack_results)
        if not all_predictions:
            return []

        confidences = [p['score'] for p in all_predictions]
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)

        diagnostics = []

        # Flat confidence (low variance)
        if std_conf < 0.05:
            diagnostics.append({
                "diagnostic": "flat_confidence_pattern",
                "severity": "HIGH",
                "description": f"All predictions have similar confidence ({mean_conf:.3f} ± {std_conf:.3f})",
                "evidence": {
                    "mean_confidence": float(mean_conf),
                    "std_confidence": float(std_conf),
                    "min_confidence": float(min(confidences)),
                    "max_confidence": float(max(confidences)),
                    "confidence_range": float(max(confidences) - min(confidences)),
                    "expected_std": ">0.15 for healthy model"
                },
                "interpretation": "Model is not learning meaningful features from inputs. Predictions are nearly uniform, indicating poor training convergence or lack of discriminative power.",
                "recommendations": [
                    "Review model training logs for convergence issues",
                    "Verify model is not a random/untrained baseline",
                    "Check if fine-tuning actually modified model weights",
                    "Validate that model architecture is appropriate for the task"
                ]
            })

        # Overconfidence
        if mean_conf > 0.95:
            diagnostics.append({
                "diagnostic": "overconfident_predictions",
                "severity": "MEDIUM",
                "description": f"Model is overconfident (mean confidence: {mean_conf:.3f})",
                "evidence": {
                    "mean_confidence": float(mean_conf),
                    "confidence_threshold": 0.95
                },
                "interpretation": "May indicate overfitting or temperature calibration issues. Model assigns very high probabilities even to uncertain predictions.",
                "recommendations": [
                    "Consider applying temperature scaling to calibrate confidence",
                    "Check for overfitting on training data",
                    "Add regularization or validation-based early stopping"
                ]
            })

        # Underconfidence
        if mean_conf < 0.55:
            diagnostics.append({
                "diagnostic": "underconfident_predictions",
                "severity": "MEDIUM",
                "description": f"Model is underconfident (mean confidence: {mean_conf:.3f})",
                "evidence": {
                    "mean_confidence": float(mean_conf),
                    "confidence_threshold": 0.55
                },
                "interpretation": "Model is uncertain about predictions. May need more training, better features, or different architecture.",
                "recommendations": [
                    "Increase model capacity or training epochs",
                    "Improve feature engineering or data preprocessing",
                    "Check if task complexity matches model capability"
                ]
            })

        return diagnostics

    def analyze_attack_failures(self, attack_results: List[Dict]) -> List[Dict]:
        """Explain why attacks succeeded or failed.

        Args:
            attack_results: Attack execution results

        Returns:
            List of diagnostic dicts explaining attack outcomes (ONE per attack type)
        """
        diagnostics = []

        # Group results by attack name (aggregate across all batches)
        from collections import defaultdict
        attacks_by_name = defaultdict(list)

        for result in attack_results:
            attack_name = result['attack']
            attacks_by_name[attack_name].append(result)

        # Generate ONE diagnostic per attack type
        for attack_name, attack_batch_results in attacks_by_name.items():
            # Aggregate all samples from all batches for this attack
            all_samples = []
            total_asr = 0.0

            for result in attack_batch_results:
                per_sample = result.get('per_sample_diagnostics', [])
                all_samples.extend(per_sample)
                attack_data = result['result']
                total_asr += attack_data.get('attack_success_rate', 0.0)

            # Average ASR across batches
            avg_asr = total_asr / len(attack_batch_results) if attack_batch_results else 0.0

            # Categorize failure reasons across ALL samples
            failure_reasons = Counter()
            for sample in all_samples:
                if not sample['prediction_flipped']:
                    if not sample['perturbation_attempted']:
                        failure_reasons['no_perturbation_generated'] += 1
                    elif sample['words_changed'] == 0:
                        failure_reasons['model_bias_prevents_flip'] += 1
                    else:
                        failure_reasons['perturbation_insufficient'] += 1

            # Generate diagnostic based on dominant failure reason
            if failure_reasons:
                dominant_reason, count = failure_reasons.most_common(1)[0]
                total_samples = len(all_samples)

                if dominant_reason == 'model_bias_prevents_flip' and count / total_samples > 0.8:
                    diagnostics.append({
                        "diagnostic": "attack_failure_model_bias",
                        "attack": attack_name,
                        "severity": "MEDIUM",
                        "description": f"{attack_name}: {avg_asr*100:.0f}% success rate due to model bias, NOT robustness",
                        "explanation": f"Attack generated perturbations for {count}/{total_samples} samples, but model never changed predictions. This indicates frozen model bias preventing attacks from succeeding, not genuine adversarial robustness.",
                        "evidence": {
                            "success_rate": float(avg_asr),
                            "perturbations_attempted": len(all_samples),
                            "prediction_flips": sum(1 for s in all_samples if s['prediction_flipped']),
                            "dominant_failure": dominant_reason,
                            "failure_count": count,
                            "batches_tested": len(attack_batch_results)
                        },
                        "verdict": "Low attack success is NOT an indication of model security. Model has underlying issues that prevent meaningful evaluation.",
                        "recommendations": [
                            "Fix model bias issue first (see class imbalance diagnostic)",
                            "Re-run attacks after model is properly trained",
                            "Do not deploy this model - it's not functioning correctly"
                        ]
                    })
                elif dominant_reason == 'no_perturbation_generated' and count / total_samples > 0.5:
                    # Enhanced analysis for NLP preprocessing-based defense
                    percentage = (count / total_samples) * 100

                    # Detect if it's likely a preprocessing-based defense (high attack failure rate)
                    is_preprocessing_defense = count / total_samples >= 0.9  # 90%+ failure rate

                    base_diagnostic = {
                        "diagnostic": "attack_not_finding_perturbations",
                        "attack": attack_name,
                        "severity": "INFO" if is_preprocessing_defense else "LOW",
                        "description": f"{attack_name}: Could not generate perturbations for {count}/{total_samples} samples ({percentage:.0f}%)",
                        "evidence": {
                            "attack_name": attack_name,
                            "samples_tested": total_samples,
                            "perturbations_failed": count,
                            "failure_rate": float(count / total_samples),
                            "attack_success_rate": float(avg_asr),
                            "batches_tested": len(attack_batch_results)
                        }
                    }

                    if is_preprocessing_defense:
                        # Provide detailed preprocessing-based defense analysis
                        base_diagnostic.update({
                            "explanation": "Attack algorithm unable to find valid perturbations. This exceptionally high failure rate (≥90%) suggests the model has strong natural defenses against word-level adversarial attacks.",
                            "interpretation": f"The model is highly resistant to {attack_name} attacks. This is likely due to preprocessing or architectural defenses that neutralize synonym-based perturbations.",
                            "root_causes": [
                                "Text Preprocessing: Lemmatization converts synonyms to the same root form (e.g., 'explode'/'blast' → 'explode')",
                                "Stopword Removal: Eliminates grammatical variations that don't affect meaning",
                                "Limited Vocabulary: Out-of-vocabulary synonyms are ignored during inference",
                                "Feature Extraction: TF-IDF or BoW models focus on semantic content, not surface-level word choice"
                            ],
                            "recommendations": [
                                "✅ GOOD NEWS: Your model has natural adversarial defenses built-in through preprocessing",
                                "To test other vulnerabilities, try:",
                                "  • Character-level attacks (typos, homoglyphs) if input validation is weak",
                                "  • Semantic attacks that preserve preprocessing (add/remove key content words)",
                                "  • Test on raw user input without preprocessing to assess input validation",
                                "Consider this a SUCCESS: preprocessing-based defense is a legitimate robustness strategy"
                            ],
                            "verdict": f"Model exhibits exceptional resistance ({percentage:.0f}% attack failure). This is a POSITIVE finding indicating effective preprocessing-based defense mechanisms."
                        })
                    else:
                        # Standard case (moderate failure rate)
                        base_diagnostic.update({
                            "explanation": "Attack algorithm unable to find valid word replacements for many samples. This may indicate genuinely robust inputs, limitation of attack method, or short/simple inputs.",
                            "interpretation": "Model shows partial resistance to this attack type. Attacks may succeed on different samples or with different attack methods."
                        })

                    diagnostics.append(base_diagnostic)

        return diagnostics

    def check_nlp_preprocessing_defense(self, attack_results: List[Dict], config: Any) -> List[Dict]:
        """Analyze if NLP model has preprocessing-based adversarial defenses.

        This diagnostic detects when attacks fail due to text preprocessing
        (lemmatization, stopword removal, etc.) that neutralizes perturbations.

        Args:
            attack_results: Attack execution results
            config: Pipeline configuration

        Returns:
            List of diagnostic dicts explaining preprocessing defenses
        """
        diagnostics = []

        # Check if this is an NLP model
        model_type = getattr(config, 'model_type', None)
        if model_type != 'NLP':
            return []

        # Get model name and check if it's TF-IDF based
        model_name = ""
        is_tfidf_model = False
        if hasattr(config, 'model_info'):
            model_name = config.model_info.get('name', '')
            is_tfidf_model = 'tfidf' in model_name.lower() or 'tf-idf' in model_name.lower() or 'tf_idf' in model_name.lower()

        # Count attack failures
        total_samples = 0
        failed_perturbations = 0
        attack_names = []

        for result in attack_results:
            attack_names.append(result['attack'])
            per_sample = result.get('per_sample_diagnostics', [])
            total_samples += len(per_sample)

            for sample in per_sample:
                if not sample.get('perturbation_attempted', False):
                    failed_perturbations += 1

        if total_samples == 0:
            return []

        failure_rate = failed_perturbations / total_samples

        # If high failure rate (>50%), explain preprocessing-based defense
        if failure_rate > 0.5:
            explanation_parts = []

            if is_tfidf_model:
                explanation_parts.append(
                    "**TF-IDF Model with Preprocessing Pipeline**\n\n"
                    "Your model uses text preprocessing that provides exceptional adversarial defense:\n\n"
                    "1. **Lemmatization**: Converts words to their root form\n"
                    "   - 'explodes' → 'explode', 'burning' → 'burn', 'fires' → 'fire'\n"
                    "   - Synonym attacks fail because 'blaze'/'fire' become identical after lemmatization\n"
                    "   - Grammatical variations (tense, plurality) are normalized\n\n"
                    "2. **Stopword Removal**: Eliminates function words\n"
                    "   - Removes 'the', 'a', 'is', 'was', 'been', etc.\n"
                    "   - Attacks that change sentence structure fail\n"
                    "   - Only content words (nouns, verbs, adjectives) matter\n\n"
                    "3. **Limited Vocabulary**: TF-IDF uses fixed vocabulary from training\n"
                    "   - Rare synonym substitutions get ignored (out-of-vocabulary)\n"
                    "   - Only common words contribute to classification\n"
                    "   - Adversarial perturbations using uncommon words have no effect"
                )
            else:
                explanation_parts.append(
                    "**NLP Model with Text Preprocessing**\n\n"
                    "Your model appears to use text preprocessing that neutralizes adversarial attacks:\n\n"
                    "1. **Text Normalization**: Lowercasing, punctuation removal, etc.\n"
                    "   - Perturbations that rely on capitalization or punctuation fail\n\n"
                    "2. **Token-level Processing**: Likely lemmatization or stemming\n"
                    "   - Synonym substitutions become identical after processing\n"
                    "   - Grammatical variations are normalized\n\n"
                    "3. **Vocabulary Constraints**: Limited or fixed vocabulary\n"
                    "   - Out-of-vocabulary substitutions get ignored or mapped to UNK token\n"
                    "   - Reduces attack surface significantly"
                )

            explanation = "\n".join(explanation_parts)

            diagnostics.append({
                "diagnostic": "nlp_preprocessing_defense",
                "severity": "INFO",
                "description": f"Attacks failed on {failure_rate*100:.0f}% of samples ({failed_perturbations}/{total_samples}) - Model has preprocessing-based defense",
                "explanation": explanation,
                "evidence": {
                    "failure_rate": float(failure_rate),
                    "total_samples": total_samples,
                    "failed_perturbations": failed_perturbations,
                    "attacks_tested": attack_names,
                    "model_type": "TF-IDF" if is_tfidf_model else "NLP",
                    "model_name": model_name
                },
                "interpretation": "This is GENUINE adversarial robustness. The preprocessing pipeline acts as a natural defense mechanism by normalizing adversarial perturbations before they reach the model.",
                "verdict": "Model demonstrates strong adversarial resistance through preprocessing. This is a valid and effective defense strategy.",
                "recommendations": [
                    "This preprocessing-based defense is a strength, not a weakness",
                    "Consider testing with character-level attacks (typos, insertions, deletions)",
                    "Test with semantic-preserving paraphrasing attacks",
                    "For production: document this as an adversarial defense mechanism",
                    "Periodically verify preprocessing pipeline remains active in deployment"
                ]
            })

        return diagnostics

    def check_label_consistency(self, attack_results: List[Dict], config: Any) -> List[Dict]:
        """Check if predictions match ground truth labels.

        Args:
            attack_results: Attack execution results
            config: Pipeline configuration

        Returns:
            List of diagnostic dicts about prediction accuracy
        """
        diagnostics = []

        mismatches = []
        all_samples = []

        for result in attack_results:
            per_sample = result.get('per_sample_diagnostics', [])
            all_samples.extend(per_sample)

        if not all_samples:
            return []

        # Get label mapping
        label_mapping = {}
        if hasattr(config, 'label_mapping'):
            label_mapping = config.label_mapping
        elif hasattr(config, 'model_info') and 'label_mapping' in config.model_info:
            label_mapping = config.model_info['label_mapping']

        reverse_mapping = {v: k for k, v in label_mapping.items()} if label_mapping else {}

        for sample in all_samples:
            original_label = sample['original_label']
            predicted_label_text = sample['original_pred']['label']

            # Map label text back to int (if possible)
            if reverse_mapping:
                predicted_label = reverse_mapping.get(predicted_label_text, -1)
            else:
                # Try to extract integer from label text (e.g., "LABEL_1" -> 1)
                try:
                    predicted_label = int(predicted_label_text.split('_')[-1]) if '_' in predicted_label_text else -1
                except (ValueError, IndexError):
                    predicted_label = -1

            if predicted_label != original_label:
                # Get sample identifier (different for NLP vs audio)
                if 'original_text' in sample:
                    # NLP model
                    sample_id = sample['original_text'][:50] + "..." if len(sample['original_text']) > 50 else sample['original_text']
                elif 'audio_id' in sample:
                    # Audio model
                    sample_id = sample['audio_id']
                else:
                    # Generic fallback
                    sample_id = f"sample_{len(mismatches)}"

                mismatches.append({
                    "sample_id": sample_id,
                    "ground_truth": original_label,
                    "predicted": predicted_label_text,
                    "confidence": sample['original_pred']['score']
                })

        if mismatches:
            mismatch_rate = len(mismatches) / len(all_samples)
            if mismatch_rate > 0.3:  # More than 30% mismatches
                diagnostics.append({
                    "diagnostic": "high_prediction_error_rate",
                    "severity": "WARNING",
                    "description": f"Model misclassifies {mismatch_rate*100:.0f}% of test inputs ({len(mismatches)}/{len(all_samples)} samples)",
                    "evidence": {
                        "mismatch_rate": float(mismatch_rate),
                        "total_samples": len(all_samples),
                        "mismatches": len(mismatches),
                        "sample_mismatches": mismatches[:5]  # Show first 5
                    },
                    "interpretation": "Model has low accuracy on test data. May indicate wrong label mapping, poor training, or domain mismatch between training and test data.",
                    "recommendations": [
                        "Verify label mapping matches training configuration",
                        "Check if test data is from same distribution as training data",
                        "Review model performance metrics on validation set",
                        "Consider re-training with better quality data"
                    ]
                })

        return diagnostics

    def check_audio_quality(self, attack_results: List[Dict]) -> List[Dict]:
        """
        Audio-specific diagnostics: SNR, energy patterns, spectral anomalies.

        Args:
            attack_results: Attack execution results with per_sample_diagnostics

        Returns:
            List of audio-specific diagnostic dicts
        """
        diagnostics = []

        # Extract all per-sample diagnostics
        all_samples = []
        for result in attack_results:
            all_samples.extend(result.get('per_sample_diagnostics', []))

        if not all_samples:
            return diagnostics

        # 1. CRITICAL: Silence Detection
        silent_samples = [s for s in all_samples if s.get('is_silent', False)]
        if len(silent_samples) > len(all_samples) * 0.2:  # >20% silent
            diagnostics.append({
                "diagnostic": "silent_audio_bias",
                "severity": "CRITICAL",
                "description": f"Model tested on {len(silent_samples)}/{len(all_samples)} silent/low-energy samples",
                "evidence": {
                    "silent_ratio": len(silent_samples) / len(all_samples),
                    "avg_rms_all": np.mean([s.get('rms_energy', 0) for s in all_samples]),
                    "avg_rms_silent": np.mean([s['rms_energy'] for s in silent_samples]) if silent_samples else 0.0,
                    "samples_examined": len(all_samples),
                    "silent_samples": len(silent_samples)
                },
                "root_causes": [
                    "Test data contains too much silence or background noise",
                    "Audio preprocessing may be zeroing out signals",
                    "Sample rate mismatches causing near-zero amplitudes",
                    "Incorrect audio file loading (reading as zeros)"
                ],
                "recommendations": [
                    "Check audio loading pipeline for errors",
                    "Verify sample rate conversions are correct",
                    "Apply Voice Activity Detection (VAD) preprocessing",
                    "Filter out silent samples from test set"
                ],
                "interpretation": "High proportion of silent audio samples indicates data quality issues that may mask model vulnerabilities."
            })

        # 2. HIGH: Low SNR Perturbations
        perturbed_samples = [s for s in all_samples if s.get('perturbation_attempted', False)]
        if perturbed_samples:
            snr_values = [s['snr_db'] for s in perturbed_samples if s['snr_db'] not in [float('inf'), float('-inf')]]
            if snr_values:
                avg_snr = np.mean(snr_values)
                if avg_snr < 15:  # Low SNR threshold
                    diagnostics.append({
                        "diagnostic": "low_snr_perturbations",
                        "severity": "HIGH",
                        "description": f"Adversarial perturbations have low SNR ({avg_snr:.1f} dB)",
                        "evidence": {
                            "avg_snr_db": avg_snr,
                            "min_snr_db": min(snr_values),
                            "max_snr_db": max(snr_values),
                            "perceptibility_threshold": "20 dB (imperceptible)",
                            "samples_perturbed": len(perturbed_samples)
                        },
                        "interpretation": "Low SNR indicates perceptually obvious perturbations that humans would easily detect. Attacks are 'successful' but not stealthy.",
                        "recommendations": [
                            "Increase SNR constraints in attack configuration (target >20 dB)",
                            "Apply psychoacoustic masking to hide perturbations",
                            "Use perceptual loss functions (PESQ, STOI)",
                            "Reduce attack epsilon parameter"
                        ]
                    })

        # 3. HIGH: Spectral Anomalies
        if perturbed_samples:
            spectral_dists = [s['spectral_distance'] for s in perturbed_samples if s.get('spectral_distance', 0.0) > 0]
            if spectral_dists:
                avg_spectral_dist = np.mean(spectral_dists)

                if avg_spectral_dist > 0.1:  # Threshold for visible spectral changes
                    diagnostics.append({
                        "diagnostic": "high_spectral_distortion",
                        "severity": "HIGH",
                        "description": f"Perturbations cause significant spectral changes (dist: {avg_spectral_dist:.3f})",
                        "evidence": {
                            "avg_spectral_distance": avg_spectral_dist,
                            "max_spectral_distance": max(spectral_dists),
                            "samples_affected": len(spectral_dists)
                        },
                        "root_causes": [
                            "Gradient-based attacks creating high-frequency artifacts",
                            "Insufficient spectral smoothness constraints",
                            "Model vulnerable to imperceptible high-frequency noise"
                        ],
                        "recommendations": [
                            "Add spectral smoothness regularization to attacks",
                            "Use band-pass filtering as preprocessing defense",
                            "Train model with spectral augmentation",
                            "Apply temporal smoothing to perturbations"
                        ],
                        "interpretation": "High spectral distortion indicates perturbations are detectable via frequency analysis."
                    })

        # 4. MEDIUM: Energy Distribution Changes
        if perturbed_samples:
            energy_changes = [abs(s['energy_change_db']) for s in perturbed_samples if s.get('energy_change_db') is not None]
            if energy_changes:
                avg_energy_change = np.mean(energy_changes)

                if avg_energy_change > 3.0:  # >3 dB energy change is noticeable
                    diagnostics.append({
                        "diagnostic": "energy_distribution_mismatch",
                        "severity": "MEDIUM",
                        "description": f"Perturbations alter audio energy significantly ({avg_energy_change:.1f} dB)",
                        "evidence": {
                            "avg_energy_change_db": avg_energy_change,
                            "max_energy_change_db": max(energy_changes),
                            "samples_affected": len(energy_changes)
                        },
                        "interpretation": "Large energy changes are perceptually detectable and indicate non-stealthy perturbations.",
                        "recommendations": [
                            "Add energy preservation constraints to attacks",
                            "Normalize perturbations to maintain RMS energy",
                            "Use windowed perturbations for localized changes"
                        ]
                    })

        return diagnostics

    def check_audio_attack_failures(self, attack_results: List[Dict]) -> List[Dict]:
        """
        Analyze why audio attacks succeeded or failed.

        Args:
            attack_results: Attack execution results

        Returns:
            List of diagnostic dicts explaining attack failures
        """
        diagnostics = []

        for result in attack_results:
            attack_name = result['attack']
            per_sample = result.get('per_sample_diagnostics', [])

            if not per_sample:
                continue

            # Count failure reasons
            no_perturbations = sum(1 for s in per_sample if not s['perturbation_attempted'])
            no_flips = sum(1 for s in per_sample if s['perturbation_attempted'] and not s['prediction_flipped'])

            # Diagnosis: Attack couldn't generate perturbations
            if no_perturbations > len(per_sample) * 0.5:
                diagnostics.append({
                    "diagnostic": "attack_not_finding_perturbations",
                    "attack": attack_name,
                    "severity": "LOW",
                    "description": f"{attack_name}: Could not generate perturbations for {no_perturbations}/{len(per_sample)} samples",
                    "explanation": "Attack algorithm unable to find valid audio perturbations within constraints (epsilon, SNR limits).",
                    "interpretation": "Model may be genuinely robust to this attack type, or attack constraints are too strict."
                })

            # Diagnosis: Perturbations don't flip predictions (model bias)
            elif no_flips > len(per_sample) * 0.5:
                diagnostics.append({
                    "diagnostic": "attack_failure_model_bias",
                    "attack": attack_name,
                    "severity": "MEDIUM",
                    "description": f"{attack_name}: Perturbations generated but predictions never changed ({no_flips}/{len(per_sample)} samples)",
                    "explanation": "Attack created perturbations but model prediction remained constant, indicating frozen model bias.",
                    "verdict": "Low attack success is NOT security - model is broken, not robust"
                })

        return diagnostics

    def check_object_detection_quality(self, attack_results: List[Dict]) -> List[Dict]:
        """
        Object detection-specific diagnostics: IoU quality, missing detections, false positives.

        Args:
            attack_results: Attack execution results with per_detection_diagnostics

        Returns:
            List of object detection-specific diagnostic dicts
        """
        diagnostics = []

        # Extract all per-detection diagnostics
        all_detections = []
        for result in attack_results:
            all_detections.extend(result.get('per_detection_diagnostics', []))

        if not all_detections:
            return diagnostics

        # 1. CRITICAL: Excessive Missing Detections
        total_missing = sum(d.get('missing_detections', 0) for d in all_detections)
        images_with_missing = sum(1 for d in all_detections if d.get('missing_detections', 0) > 0)

        if images_with_missing > len(all_detections) * 0.3:  # >30% images have missing detections
            diagnostics.append({
                "diagnostic": "excessive_missing_detections",
                "severity": "CRITICAL",
                "description": f"Attacks cause missing detections in {images_with_missing}/{len(all_detections)} images (total: {total_missing} objects)",
                "evidence": {
                    "images_affected_ratio": images_with_missing / len(all_detections),
                    "total_missing_detections": total_missing,
                    "avg_missing_per_image": total_missing / len(all_detections),
                    "images_examined": len(all_detections)
                },
                "root_causes": [
                    "Model confidence thresholds too sensitive to perturbations",
                    "Feature extraction vulnerable to small pixel changes",
                    "Non-maximum suppression (NMS) being disrupted",
                    "Model underfitting fails to detect objects robustly"
                ],
                "recommendations": [
                    "Lower confidence threshold for detections",
                    "Train with adversarial examples to improve robustness",
                    "Add data augmentation during training (noise, brightness)",
                    "Use ensemble of detection heads for redundancy"
                ],
                "interpretation": "High rate of missing detections indicates model fails to detect objects under adversarial perturbations."
            })

        # 2. HIGH: Poor IoU Quality
        perturbed_detections = [d for d in all_detections if d.get('perturbation_attempted', False)]
        if perturbed_detections:
            mean_ious = [d['iou_metrics']['mean_iou'] for d in perturbed_detections
                        if 'iou_metrics' in d]

            if mean_ious:
                avg_iou = np.mean(mean_ious)

                if avg_iou < 0.5:  # IoU < 0.5 indicates poor localization
                    diagnostics.append({
                        "diagnostic": "poor_localization_quality",
                        "severity": "HIGH",
                        "description": f"Adversarial perturbations degrade bounding box IoU (mean: {avg_iou:.3f})",
                        "evidence": {
                            "mean_iou": avg_iou,
                            "min_iou": min(mean_ious),
                            "max_iou": max(mean_ious),
                            "images_affected": len(perturbed_detections),
                            "iou_threshold": "0.5 (standard detection threshold)"
                        },
                        "interpretation": "Low IoU indicates localization errors - bounding boxes don't align well with objects under attack.",
                        "recommendations": [
                            "Improve regression head robustness in object detector",
                            "Use IoU-based loss functions during training",
                            "Add localization-specific adversarial training",
                            "Consider anchor-free detection methods (e.g., FCOS)"
                        ]
                    })

        # 3. HIGH: Confidence Degradation
        if perturbed_detections:
            confidence_drops = [d['confidence_change'] for d in perturbed_detections
                               if d.get('confidence_change', 0) < -0.1]

            if len(confidence_drops) > len(perturbed_detections) * 0.5:
                avg_drop = np.mean(confidence_drops)
                diagnostics.append({
                    "diagnostic": "severe_confidence_degradation",
                    "severity": "HIGH",
                    "description": f"Detection confidence drops significantly under attack (mean: {avg_drop:.3f})",
                    "evidence": {
                        "avg_confidence_drop": avg_drop,
                        "images_affected": len(confidence_drops),
                        "images_affected_ratio": len(confidence_drops) / len(perturbed_detections)
                    },
                    "root_causes": [
                        "Classification head vulnerable to adversarial noise",
                        "Feature representations not robust to perturbations",
                        "Overconfident predictions on clean data"
                    ],
                    "recommendations": [
                        "Apply confidence calibration techniques",
                        "Use label smoothing during training",
                        "Train with adversarial examples",
                        "Add confidence-aware loss functions"
                    ],
                    "interpretation": "Confidence drops indicate classification uncertainty introduced by perturbations."
                })

        # 4. MEDIUM: False Positive Generation
        total_false_positives = sum(d.get('false_positives', 0) for d in perturbed_detections)
        images_with_fp = sum(1 for d in perturbed_detections if d.get('false_positives', 0) > 0)

        if total_false_positives > 0 and images_with_fp > len(perturbed_detections) * 0.2:
            diagnostics.append({
                "diagnostic": "false_positive_generation",
                "severity": "MEDIUM",
                "description": f"Attacks create {total_false_positives} spurious detections in {images_with_fp} images",
                "evidence": {
                    "total_false_positives": total_false_positives,
                    "images_affected": images_with_fp,
                    "avg_fp_per_image": total_false_positives / len(perturbed_detections)
                },
                "interpretation": "Adversarial perturbations trigger false object detections not present in original image.",
                "recommendations": [
                    "Increase NMS IoU threshold to suppress spurious boxes",
                    "Raise confidence threshold for detections",
                    "Add background class supervision during training",
                    "Use hard negative mining to reduce false positives"
                ]
            })

        # 5. MEDIUM: Small Object Vulnerability
        small_objects = [d for d in all_detections if d.get('avg_object_size', float('inf')) < 32*32]
        if small_objects:
            small_obj_missing = sum(d.get('missing_detections', 0) for d in small_objects)
            if small_obj_missing > 0:
                diagnostics.append({
                    "diagnostic": "small_object_vulnerability",
                    "severity": "MEDIUM",
                    "description": f"Small objects particularly vulnerable to attacks ({small_obj_missing} missing from {len(small_objects)} images)",
                    "evidence": {
                        "small_object_images": len(small_objects),
                        "missing_from_small": small_obj_missing,
                        "size_threshold": "32x32 pixels"
                    },
                    "interpretation": "Small objects harder to detect robustly under adversarial perturbations.",
                    "recommendations": [
                        "Use feature pyramid networks (FPN) for multi-scale detection",
                        "Increase training data with small object examples",
                        "Apply resolution-preserving augmentations",
                        "Use specialized small object detectors"
                    ]
                })

        return diagnostics

    def check_object_detection_failures(self, attack_results: List[Dict]) -> List[Dict]:
        """
        Analyze why object detection attacks succeeded or failed.

        Args:
            attack_results: Attack execution results

        Returns:
            List of diagnostic dicts explaining attack failures
        """
        diagnostics = []

        for result in attack_results:
            attack_name = result['attack']
            per_detection = result.get('per_detection_diagnostics', [])

            if not per_detection:
                continue

            # Count success patterns
            no_perturbations = sum(1 for d in per_detection if not d['perturbation_attempted'])
            successful_attacks = sum(1 for d in per_detection if d.get('detections_dropped', False) or d.get('confidence_dropped', False))

            # Diagnosis: Attack couldn't generate perturbations
            if no_perturbations > len(per_detection) * 0.5:
                diagnostics.append({
                    "diagnostic": "attack_not_finding_perturbations",
                    "attack": attack_name,
                    "severity": "LOW",
                    "description": f"{attack_name}: Could not generate perturbations for {no_perturbations}/{len(per_detection)} images",
                    "explanation": "Attack algorithm unable to find valid perturbations within constraints (epsilon limits, patch size).",
                    "interpretation": "Model may be robust to this attack type, or attack constraints are too strict."
                })

            # Diagnosis: Attacks highly effective
            elif successful_attacks > len(per_detection) * 0.7:
                diagnostics.append({
                    "diagnostic": "attack_highly_effective",
                    "attack": attack_name,
                    "severity": "INFO",
                    "description": f"{attack_name}: Successfully attacked {successful_attacks}/{len(per_detection)} images",
                    "explanation": "High attack success rate indicates model vulnerability to this attack type.",
                    "verdict": "Model requires adversarial robustness improvements"
                })

        return diagnostics

    def _is_object_detection_data(self, attack_results: List[Dict]) -> bool:
        """
        Detect if attack results are from object detection model.

        Args:
            attack_results: Attack execution results

        Returns:
            True if object detection data detected, False otherwise
        """
        if not attack_results:
            return False

        # Check if per_detection_diagnostics exists (unique to object detection)
        for result in attack_results:
            detections = result.get('per_detection_diagnostics', [])
            if detections and len(detections) > 0:
                detection = detections[0]
                # Object detection has unique fields like iou_metrics, missing_detections
                if 'iou_metrics' in detection or 'missing_detections' in detection or 'false_positives' in detection:
                    return True

        return False

    def _is_audio_data(self, attack_results: List[Dict]) -> bool:
        """
        Detect if attack results are from audio model by checking for audio-specific fields.

        Args:
            attack_results: Attack execution results

        Returns:
            True if audio data detected, False otherwise
        """
        if not attack_results:
            return False

        # Check if per_sample_diagnostics has audio-specific fields
        for result in attack_results:
            samples = result.get('per_sample_diagnostics', [])
            if samples and len(samples) > 0:
                sample = samples[0]
                # Audio samples have unique fields like snr_db, rms_energy, spectral_distance
                if 'snr_db' in sample or 'rms_energy' in sample or 'spectral_distance' in sample:
                    return True

        return False

    def _extract_all_predictions(self, attack_results: List[Dict]) -> List[Dict]:
        """Extract all original predictions from attack results.

        Handles two formats:
        - NLP models: Predictions already in dict format with 'label' and 'score'
        - Image models: Predictions as probability vectors (lists), converted to dict format

        Args:
            attack_results: Attack execution results

        Returns:
            List of prediction dicts with 'label' and 'score' keys
        """
        predictions = []
        for result in attack_results:
            # Get from per_sample_diagnostics if available (NLP path)
            per_sample = result.get('per_sample_diagnostics', [])
            if per_sample:
                predictions.extend([s['original_pred'] for s in per_sample])
            # Fallback to result dict (IMAGE path)
            elif 'result' in result:
                original_preds = result['result'].get('original_predictions', [])

                # Convert probability vectors to dict format for image models
                for pred in original_preds:
                    if isinstance(pred, list):
                        # pred is a probability distribution [0.1, 0.2, 0.3, ...]
                        pred_array = np.array(pred)
                        predicted_class = int(np.argmax(pred_array))
                        confidence_score = float(pred_array[predicted_class])

                        predictions.append({
                            "label": predicted_class,  # Integer class index
                            "score": confidence_score   # Probability of predicted class
                        })
                    else:
                        # Already in dict format
                        predictions.append(pred)

        return predictions

    def _count_by_severity(self, diagnostics: List[Dict]) -> Dict[str, int]:
        """Count diagnostics by severity level.

        Args:
            diagnostics: List of diagnostic dicts

        Returns:
            Dict mapping severity to count
        """
        return dict(Counter(d['severity'] for d in diagnostics))

    def _compute_health_score(self, diagnostics: List[Dict]) -> str:
        """Compute overall model health status.

        Args:
            diagnostics: List of diagnostic dicts

        Returns:
            Health status: "HEALTHY", "CONCERNING", or "CRITICAL"
        """
        severity_weights = {
            'CRITICAL': 10,
            'HIGH': 5,
            'MEDIUM': 2,
            'WARNING': 1,
            'LOW': 1,
            'INFO': 0
        }

        total_score = sum(severity_weights.get(d['severity'], 0) for d in diagnostics)

        if total_score == 0:
            return "HEALTHY"
        elif total_score < 5:
            return "CONCERNING"
        else:
            return "CRITICAL"
