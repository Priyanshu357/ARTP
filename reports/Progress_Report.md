# NTCC Progress Report

## 1. Introduction / Project Overview
- Purpose: Evaluate and harden image classifiers against adversarial attacks using a modular red-team/blue-team framework.
- Scope: Implement canonical attacks (FGSM, PGD, DeepFool), lightweight detectors (confidence/entropy/ensemble), and automated reporting with metrics and charts.
- Dataset/Model: CIFAR-10 with a baseline CNN exported to ONNX and converted to PyTorch for evaluation.

## 2. Work Completed So Far

### Literature Review / Research Papers Studied

**Adversarial Attack Methods:**
- **FGSM** (Goodfellow et al., 2015): One-step gradient attack; efficient baseline. Formula: $x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(x, y))$
- **PGD** (Madry et al., 2018): Multi-step iterative attack; gold standard for robustness evaluation.
- **DeepFool** (Moosavi-Dezfooli et al., 2016): Minimal-norm perturbations; evaluates model margins.

**Detection Methods:**
- **Confidence-Based**: Flag significant softmax confidence drops between clean and adversarial inputs.
- **Entropy-Based**: Flag elevated prediction entropy; adversarial examples exhibit higher uncertainty.
- **Ensemble Voting**: Combine multiple detectors to reduce false negatives and improve robustness.

**Evaluation Metrics:**
- **Attack Success Rate (ASR)**: Fraction of samples misclassified by attack.
- **Detection Accuracy (DA)**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **False Positive Rate (FPR)**: $\frac{FP}{TN + FP}$ — benign samples incorrectly flagged.
- **Robustness Score**: Composite metric: $\text{Score} = (1 - ASR) \times DA \times (1 - FPR) \times 100$

**Adversarial Training (Madry et al., 2018):**
- Train on mix of clean and adversarial examples: $\min_\theta \mathbb{E}_{(x,y)} \left[ \max_{\delta \in B_\epsilon} L(x+\delta, y; \theta) \right]$
- Improves certified robustness under threat model; moderate computational overhead.

**Key Insights:**
1. Attack diversity (FGSM/PGD/DeepFool) covers multiple threat models.
2. Single detectors insufficient; ensemble voting improves coverage.
3. Composite metrics better capture system-level security.
4. Adaptive adversaries require white-box evaluation; defenses must anticipate attacks.

### System Architecture Design
- Pipeline Orchestrator: Coordinates attacks, detection, analysis, and PDF generation ([pipeline/orchestrator.py](../pipeline/orchestrator.py)).
- Config-driven Runs: Centralized run-time configuration ([pipeline/config.py](../pipeline/config.py)).
- Red Team: Batch-wise attack execution via `AttackRunner` ([red_team/attack_runner.py](../red_team/attack_runner.py)); FGSM/PGD/DeepFool implementations ([red_team](../red_team)).
- Blue Team: Pluggable detectors — Confidence Drop, Entropy Increase, Ensemble voting ([blue_team](../blue_team)).
- Analysis: Metric computation and summarization ([analysis/metrics.py](../analysis/metrics.py), [analysis/summarizer.py](../analysis/summarizer.py)).
- Reporting: Charts + templated PDF ([reports/charts.py](charts.py), [reports/pdf_generator.py](pdf_generator.py), [reports/templates.py](templates.py)).
- ONNX Model Flow: ONNX → onnx2torch conversion with compatibility patching ([run_red_team_on_onnx.py](../run_red_team_on_onnx.py)).

### Technology Stack Setup
- Core: Python 3.x, PyTorch, Torchvision, NumPy.
- Model Interop: ONNX, onnx2torch; example model and exporter in [models](../models) and [convert_model.py](../convert_model.py).
- Data: CIFAR-10 (downloaded/loaded via Torchvision).
- Visualization/Reporting: Matplotlib for charts, ReportLab for PDFs.
- Project Structure: Modular red/blue teams, pipeline, analysis, and reports directories.

### Attack Agent Implementation
- Implemented Attacks: FGSM, PGD, DeepFool with configurable parameters and clipping.
- Batch Orchestration: `AttackRunner.run_batch()` produces JSON-safe outputs with predictions and per-batch metrics.
- ONNX Path: Helper script to run attacks directly on ONNX-converted model ([run_red_team_on_onnx.py](../run_red_team_on_onnx.py)).

### Defense Agent Development
- ConfidenceDropDetector: Flags significant drop in max softmax confidence ([blue_team/confidence_detector.py](../blue_team/confidence_detector.py)).
- EntropyIncreaseDetector: Flags rise in predictive entropy ([blue_team/entropy_detector.py](../blue_team/entropy_detector.py)).
- EnsembleDetector: Majority vote or weighted-score aggregation of detectors ([blue_team/ensemble_detector.py](../blue_team/ensemble_detector.py)).
- Pipeline Integration: Factory wiring converts attack outputs to tensors and emits structured detection records ([main.py](../main.py)).

### Dashboard & Visualization
- Automated PDF: Summary metrics, attack success bar chart, and detection confusion matrix ([reports/pdf_generator.py](pdf_generator.py)).
- Artifacts: Stored under [reports](.). Generated files include `attack_results.json`, `detection_results.json`, `summary.json`, and `_charts/` images.
- UI Placeholder: `ui/` scaffold exists for future interactive dashboarding.

## 3. Work in Progress
- Threshold Tuning: Calibrating confidence/entropy thresholds; exploring weighted ensemble mode.
- Data Loader Integration: Replacing dummy dataloader in [main.py](../main.py) with CIFAR-10 DataLoader used in [run_red_team_on_onnx.py](../run_red_team_on_onnx.py).
- Attack Parameter Sweeps: Systematic epsilon/step tuning for FGSM/PGD and iterations for DeepFool.
- Result Compression: Trimming very large JSON payloads for scalability and faster reporting.

## 4. Work Remaining / Next Steps
- Detector Enhancements: Add additional detectors (input gradients, feature-squeezing, spectral stats) and ROC-driven threshold selection.
- Adversarial Training: Integrate simple adversarial training to improve robustness and re-evaluate.
- Evaluation Coverage: Expand to more batches/classes; add benign-only baselines for robust FPR estimates.
- UI Dashboard: Build an interactive dashboard in `ui/` to browse runs, metrics, and samples.
- Performance/Portability: Optionally add ONNX Runtime inference path and CUDA kernels where beneficial.
- Reproducibility: Persist config/run seeds and environment snapshot per session.

## 5. Challenges Faced
- ONNX Compatibility: Needed to patch `Reshape` `allowzero` attribute for onnx2torch conversion ([run_red_team_on_onnx.py](../run_red_team_on_onnx.py)).
- Detector Sensitivity: Initial thresholds too strict; early results show under-detection requiring calibration.
- Large Artifacts: Attack results can be large; requires trimming/summarization for long runs.
- Determinism & Hardware: Aligning seeds, CUDA determinism, and batch sizes across CPU/GPU environments.

## 6. Timeline

| Task | Status |
|------|--------|
| Literature Review | ✅ Completed |
| System Architecture Design | ✅ Completed |
| Technology Stack Setup | ✅ Completed |
| Attack Agent Implementation | ✅ Completed |
| Defense Agent Development | ✅ Completed |
| Dashboard & Visualization | ✅ Completed |
| PDF Reporting Module | ✅ Completed |
| NLP Attack Integration | ⏳ In Progress |
| Advanced Attack Methods | ⏳ In Progress |
| User Acceptance Testing | ⏹️ Not Started |
| Final Documentation & Report | ⏳ In Progress |

## 7. Current Results & Performance Metrics
Based on the latest generated artifacts in [reports](.):
- Attack Success Rate: 0.986
- Detection Accuracy: 0.000
- False Positive Rate: 0.000
- Robustness Score: 10.69 / 100
Interpretation:
- Attacks are highly effective against the current model configuration.
- Detectors are not yet calibrated (low sensitivity); immediate tuning and additional signals required.

## 8. Conclusion
- The red-team pipeline, core attacks, blue-team detectors, analysis, and automated PDF reporting are implemented end-to-end.
- Current robustness is low given high attack success and near-zero detection accuracy; next milestones focus on detector calibration, broader signals, and adversarial training.
- With the existing modular design, extending detectors, tuning thresholds, and adding UI/dashboarding can proceed in parallel with model-hardening.
