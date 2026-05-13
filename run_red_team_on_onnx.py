"""Run red_team attacks against the ONNX CIFAR model."""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import torch
from onnx2torch import convert
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from red_team import AttackRunner, DeepFoolAttack, FGSMAttack, PGDAttack


def _patch_reshape_allowzero(model: onnx.ModelProto) -> None:
    """Set allowzero=0 on Reshape nodes to satisfy onnx2torch."""
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        for attr in node.attribute:
            if attr.name == "allowzero":
                attr.i = 0


def load_model(onnx_path: Path, device: torch.device) -> torch.nn.Module:
    onnx_model = onnx.load(onnx_path)
    _patch_reshape_allowzero(onnx_model)

    # Inline external data to avoid .onnx.data dependency.
    patched_path = onnx_path.with_name(onnx_path.stem + "_patched.onnx")
    onnx.save_model(onnx_model, patched_path, save_as_external_data=False)

    patched_model = onnx.load(patched_path)
    torch_model = convert(patched_model)
    return torch_model.to(device).eval()


def make_dataloader(data_root: Path, batch_size: int = 32) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),  # keeps inputs in [0, 1]
    ])
    dataset = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def run_attacks(model: torch.nn.Module, dataloader: DataLoader, device: torch.device,
                num_batches: int = 5) -> list[dict]:
    attacks = [
        FGSMAttack(epsilon=0.007, clip_min=0.0, clip_max=1.0),
        PGDAttack(epsilon=0.03, alpha=0.007, num_steps=40, clip_min=0.0, clip_max=1.0),
        DeepFoolAttack(max_iter=50, overshoot=0.02, clip_min=0.0, clip_max=1.0),
    ]
    runner = AttackRunner(model, attacks)

    results: list[dict] = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        batch_res = runner.run_batch(inputs.to(device), labels.to(device))
        for entry in batch_res:
            entry["batch_index"] = batch_idx
        results.extend(batch_res)
    return results


def main() -> None:
    project_root = Path(__file__).resolve().parent
    onnx_path = project_root / "models" / "cifar_net.onnx"
    data_root = project_root / "data"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(onnx_path, device)
    dataloader = make_dataloader(data_root)
    results = run_attacks(model, dataloader, device)

    out_path = reports_dir / "attack_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
