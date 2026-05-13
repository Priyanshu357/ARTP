"""AttackRunner orchestrates execution of multiple adversarial attacks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from .base_attack import BaseAttack


class AttackRunner:
    """Run one or more attacks against a model over a dataset."""

    def __init__(self, model: Any, attacks: Sequence[BaseAttack]):
        """Initialize the runner with a model and a collection of attacks.

        Args:
            model: PyTorch model to evaluate.
            attacks: Sequence of attack instances implementing BaseAttack.
        """
        self.model = model
        self.attacks = list(attacks)
        self.results: List[Dict[str, Any]] = []

    def run_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> List[Dict[str, Any]]:
        """Run all attacks on a single batch of inputs and labels."""
        batch_results: List[Dict[str, Any]] = []
        for attack in self.attacks:
            attack_output = attack.generate(self.model, inputs, labels)
            batch_results.append({
                "attack": attack.__class__.__name__,
                "result": self._to_jsonable(attack_output),
            })
        self.results.extend(batch_results)
        return batch_results

    def run_dataset(self, dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[Dict[str, Any]]:
        """Run all attacks over an iterable of (inputs, labels) batches."""
        aggregated: List[Dict[str, Any]] = []
        for batch_index, (inputs, labels) in enumerate(dataset):
            batch_res = self.run_batch(inputs, labels)
            for res in batch_res:
                res_with_batch = dict(res)
                res_with_batch["batch_index"] = batch_index
                aggregated.append(res_with_batch)
        return aggregated

    def _to_jsonable(self, obj: Any) -> Any:
        """Convert common PyTorch/Numpy structures into JSON-safe Python types."""
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            return obj.detach().cpu().tolist()
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        if isinstance(obj, (float, int, bool, type(None), str)):
            return obj
        # Fallback to string representation for unsupported types.
        return str(obj)
