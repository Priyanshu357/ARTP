"""Pipeline orchestrator for red/blue evaluation, analysis, and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from analysis.summarizer import summarize
from analysis.diagnostics import DiagnosticAnalyzer
from red_team import AttackRunner
from reports.pdf_generator import build_pdf

from .config import PipelineConfig
from .session_manager import SessionManager


class Orchestrator:
    """Coordinate attacks, detection, analysis, and reporting."""

    def __init__(self, session_manager: SessionManager | None = None) -> None:
        self.session_manager = session_manager or SessionManager()

    @staticmethod
    def _get_model_specific_filename(config: PipelineConfig, base_name: str, extension: str = ".json") -> Path:
        """Generate model-specific filename for reports.

        Args:
            config: Pipeline configuration
            base_name: Base name for the file (e.g., "attack_results", "summary")
            extension: File extension (default: ".json")

        Returns:
            Path with model-specific filename (e.g., "lr_tfidf_model_attack_results.json")
        """
        # Extract model name from config
        model_name = config.model_info.get("name")
        if not model_name:
            # Fallback to model class name
            model_name = getattr(config.model, "__class__", type("", (), {})).__name__

        # Extract filename from path and remove extension
        model_name = Path(model_name).stem  # e.g., "models/nlp/lr_tfidf_model.onnx" -> "lr_tfidf_model"

        # Clean model name for filename (replace hyphens/spaces with underscores)
        import re
        clean_name = model_name.replace('-', '_').replace(' ', '_')
        # Remove any remaining special characters except underscores
        clean_name = re.sub(r'[^\w]+', '_', clean_name)
        # Remove consecutive underscores
        while '__' in clean_name:
            clean_name = clean_name.replace('__', '_')
        # Remove leading/trailing underscores
        clean_name = clean_name.strip('_').lower()

        # Generate filename: modelname_basename.extension
        filename = f"{clean_name}_{base_name}{extension}"

        # Use the same parent directory as report_path
        report_dir = Path(config.report_path).parent
        return report_dir / filename

    def run(self, config: PipelineConfig) -> Dict[str, Any]:
        """Execute the full pipeline and return a summary object."""
        # Detect and log model type
        model_type = config.get_model_type()
        print(f"[Orchestrator] Detected model type: {model_type.value}")

        session = self.session_manager.start({"project_title": config.project_title})

        attack_results = self._run_attacks(config)
        attack_path = config.attack_results_path or self._get_model_specific_filename(config, "attack_results", ".json")
        self._dump_json(attack_results, attack_path)

        detection_results = self._run_detection(config, attack_results)
        detection_path = None
        if detection_results:
            detection_path = config.detection_results_path or self._get_model_specific_filename(config, "detection_results", ".json")
            self._dump_json(detection_results, detection_path)

        summary =summarize(attack_results, detection_results)
        summary_path = self._get_model_specific_filename(config, "summary", ".json")

        # NEW: Run diagnostic analysis
        print("\n[Diagnostics] Running model behavior analysis...")
        use_llm = getattr(config, 'use_llm_diagnostics', False)
        llm_provider = getattr(config, 'llm_provider', 'openai')
        analyzer = DiagnosticAnalyzer(use_llm=use_llm, llm_provider=llm_provider)
        diagnostics = analyzer.analyze(attack_results, detection_results, summary, config)

        # IMPORTANT: Re-compute robustness score with diagnostic penalties
        from analysis.robustness_score import apply_diagnostic_penalties
        penalized_score = apply_diagnostic_penalties(
            base_score=summary['robustness_score'],
            diagnostics_list=diagnostics.get('diagnostics', [])
        )

        # Update summary with penalized score
        if penalized_score != summary['robustness_score']:
            print(f"[Diagnostics] Applying penalty for model issues: {summary['robustness_score']:.1f} -> {penalized_score:.1f}")
            summary['robustness_score'] = penalized_score
            summary['score_penalty_applied'] = True

        # Save updated summary
        self._dump_json(summary, summary_path)

        diagnostics_path = self._get_model_specific_filename(config, "diagnostics", ".json")
        self._dump_json(diagnostics, diagnostics_path)

        print(f"[Diagnostics] Model Health: {diagnostics['overall_health']}")
        print(f"[Diagnostics] Issues Found: {len(diagnostics['diagnostics'])}")
        for d in diagnostics['diagnostics'][:3]:  # Show top 3
            print(f"  - [{d['severity']}] {d['description']}")

        # Generate model-specific PDF report path
        pdf_report_path = self._get_model_specific_filename(config, "security_report", ".pdf")
        report_path = build_pdf(
            output_path=pdf_report_path,
            project_title=config.project_title,
            model_info=config.normalized_model_info(),
            attack_results=attack_results,
            detection_results=detection_results,
            summary=summary,
            diagnostics=diagnostics,  # NEW: Pass diagnostics to PDF generator
        )

        final = {
            "session_id": session.session_id,
            "summary": summary,
            "diagnostics": diagnostics,  # NEW: Include diagnostics
            "report_path": str(report_path),
            "attack_results_path": str(attack_path),
            "detection_results_path": str(detection_path) if detection_path else None,
            "summary_path": str(summary_path),
            "diagnostics_path": str(diagnostics_path),  # NEW: Include diagnostics path
        }
        self.session_manager.store_results(session.session_id, "final", final)
        return final

    def _run_attacks(self, config: PipelineConfig) -> list[dict]:
        # Auto-select appropriate attack runner based on model type
        model_type = config.get_model_type()

        # Import NLPAttackRunner only when needed
        if model_type.value == 'nlp':
            from red_team.nlp_models import NLPAttackRunner
            runner = NLPAttackRunner(config.model, config.attacks)
        elif model_type.value == 'audio':
            from red_team.audio_models import AudioAttackRunner
            sample_rate = getattr(config, 'sample_rate', 16000)
            runner = AudioAttackRunner(config.model, config.attacks, sample_rate=sample_rate)
        elif model_type.value == 'object_detection':
            from red_team.object_detection import ObjectDetectionAttackRunner
            runner = ObjectDetectionAttackRunner(config.model, config.attacks)
        else:
            runner = AttackRunner(config.model, config.attacks)

        results: list[dict] = []
        for batch_idx, (inputs, labels) in enumerate(config.dataloader):
            if batch_idx >= config.max_batches:
                break
            batch_res = runner.run_batch(inputs, labels)
            for entry in batch_res:
                entry["batch_index"] = batch_idx
            results.extend(batch_res)
        return results

    def _run_detection(self, config: PipelineConfig, attack_results: Sequence[Mapping[str, Any]]) -> list[dict]:
        if not config.detection_fn:
            return []
        detections: list[dict] = []
        for attack_res in attack_results:
            try:
                # detection_fn should return iterable of dicts with keys is_attack/detected
                for det in config.detection_fn(attack_res):
                    if isinstance(det, Mapping):
                        detections.append(dict(det))
            except Exception:
                continue
        return detections

    @staticmethod
    def _dump_json(data: Any, path: Path) -> None:
        """Dump data to JSON file, handling numpy arrays."""
        import numpy as np

        def sanitize_for_json(obj):
            """Recursively remove/convert non-JSON-serializable objects."""
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()
                       if k != 'adversarial_audio'}  # Skip large audio arrays
            elif isinstance(obj, (list, tuple)):
                return [sanitize_for_json(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return f"<numpy array shape={obj.shape}>"  # Don't serialize large arrays
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            sanitized = sanitize_for_json(data)
            json.dump(sanitized, f, indent=2)
