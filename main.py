"""Main entry point to run attacks, detection, analysis, and report generation."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import onnx
import torch
from onnx2torch import convert
from torch.utils.data import DataLoader, TensorDataset

from blue_team import ConfidenceDropDetector, EnsembleDetector, EntropyIncreaseDetector
from pipeline.config import PipelineConfig
from pipeline.orchestrator import Orchestrator
from red_team import DeepFoolAttack, FGSMAttack, PGDAttack


def _patch_reshape_allowzero(model: onnx.ModelProto) -> None:
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        for attr in node.attribute:
            if attr.name == "allowzero":
                attr.i = 0


def load_model(onnx_path: Path, device: torch.device) -> torch.nn.Module:
    onnx_model = onnx.load(onnx_path)
    _patch_reshape_allowzero(onnx_model)
    patched_path = onnx_path.with_name(onnx_path.stem + "_patched.onnx")
    onnx.save_model(onnx_model, patched_path, save_as_external_data=False)
    patched_model = onnx.load(patched_path)
    torch_model = convert(patched_model)
    return torch_model.to(device).eval()


def build_dataloader(device: torch.device) -> DataLoader:
    # Placeholder dataloader; replace with your real evaluation data.
    dummy_x = torch.rand(64, 3, 32, 32, device=device)
    dummy_y = torch.randint(0, 10, (64,), device=device)
    dataset = TensorDataset(dummy_x, dummy_y)
    return DataLoader(dataset, batch_size=16, shuffle=False)


def set_deterministic(seed: int = 42) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detection_fn_factory(detectors: EnsembleDetector):
    def detect(attack_res: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
        res = attack_res.get("result") if isinstance(attack_res, Mapping) else None
        if not isinstance(res, Mapping):
            return []
        orig = res.get("original_predictions")
        adv = res.get("adversarial_predictions")
        if orig is None or adv is None:
            return []
        try:
            orig_t = torch.tensor(orig, dtype=torch.float32)
            adv_t = torch.tensor(adv, dtype=torch.float32)
        except Exception:
            return []

        det_out = detectors.detect(orig_t, adv_t)
        det_out.update({
            "is_attack": True,
            "attack": attack_res.get("attack"),
            "batch_index": attack_res.get("batch_index"),
        })
        return [det_out]

    return detect


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adversarial attacks on image models")
    parser.add_argument("--model", type=str, default="models/cifar_net.onnx",
                        help="Path to model file (.onnx)")
    parser.add_argument("--epsilon", type=float, default=0.03,
                        help="Perturbation budget for attacks")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--max-batches", type=int, default=3,
                        help="Maximum batches to process")
    args = parser.parse_args()

    set_deterministic()
    project_root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    print(f"[1/5] Loading model from {model_path}")
    model = load_model(model_path, device)

    print(f"[2/5] Loading test data")
    dataloader = build_dataloader(device)

    print(f"[3/5] Configuring attacks (epsilon={args.epsilon})")
    attacks = [
        FGSMAttack(epsilon=args.epsilon, clip_min=0.0, clip_max=1.0),
        PGDAttack(epsilon=args.epsilon, alpha=0.007, num_steps=10, clip_min=0.0, clip_max=1.0, random_start=False),
        DeepFoolAttack(max_iter=20, overshoot=0.02, clip_min=0.0, clip_max=1.0),
    ]

    print(f"[4/5] Configuring detectors")
    detectors = EnsembleDetector(
        detectors=[
            ConfidenceDropDetector(drop_threshold=0.1),
            EntropyIncreaseDetector(entropy_threshold=0.05),
        ],
        mode="vote",
    )

    # Use model stem for report names so each model gets its own reports
    model_key = model_path.stem.replace("-", "_").lower()
    reports_dir = project_root / "reports"
    cfg = PipelineConfig(
        model=model,
        dataloader=dataloader,
        attacks=attacks,
        report_path=reports_dir / f"{model_key}_security_report.pdf",
        attack_results_path=reports_dir / f"{model_key}_attack_results.json",
        detection_results_path=reports_dir / f"{model_key}_detection_results.json",
        project_title=f"Adversarial Security Report — {model_key}",
        model_info={"name": model_path.name, "framework": "onnx/torch"},
        detection_fn=detection_fn_factory(detectors),
        max_batches=args.max_batches,
    )

    print(f"[5/5] Running pipeline for {model_key}")
    orchestrator = Orchestrator()
    final = orchestrator.run(cfg)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
