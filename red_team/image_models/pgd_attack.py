"""Projected Gradient Descent (PGD) adversarial attack implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base_attack import BaseAttack


class PGDAttack(BaseAttack):
    """Iterative FGSM (PGD) attack for PyTorch models with Linf constraints."""

    def __init__(self, epsilon: float = 0.03, alpha: float = 0.007, num_steps: int = 40,
                 clip_min: float = 0.0, clip_max: float = 1.0, loss_fn: Optional[nn.Module] = None,
                 random_start: bool = True) -> None:
        """Configure PGD attack parameters.

        Args:
            epsilon: Linf perturbation budget.
            alpha: Step size for each gradient ascent step.
            num_steps: Number of iterative steps.
            clip_min: Minimum value for input clamping.
            clip_max: Maximum value for input clamping.
            loss_fn: Loss function; defaults to cross-entropy if not provided.
            random_start: If True, start within the epsilon-ball uniformly; else start at the clean point.
        """
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.num_steps = int(num_steps)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.random_start = bool(random_start)

    def generate(self, model: Any, inputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """Run PGD to craft adversarial examples and report metrics."""
        if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("inputs and labels must be torch.Tensor instances")

        device = next(model.parameters()).device
        model.eval()

        clean_inputs = inputs.detach().to(device)
        labels = labels.detach().to(device)

        with torch.no_grad():
            clean_logits = model(clean_inputs)
            original_predictions = torch.softmax(clean_logits, dim=1)

        if self.random_start:
            adv = clean_inputs + torch.empty_like(clean_inputs).uniform_(-self.epsilon, self.epsilon)
            adv = torch.clamp(adv, self.clip_min, self.clip_max)
        else:
            adv = clean_inputs.clone().detach()

        for _ in range(max(self.num_steps, 1)):
            adv.requires_grad_(True)
            logits = model(adv)
            loss = self.loss_fn(logits, labels)
            model.zero_grad(set_to_none=True)
            loss.backward()

            grad_sign = adv.grad.sign()
            adv = adv + self.alpha * grad_sign

            perturbation = torch.clamp(adv - clean_inputs, -self.epsilon, self.epsilon)
            adv = clean_inputs + perturbation
            adv = torch.clamp(adv, self.clip_min, self.clip_max).detach()

        with torch.no_grad():
            adv_logits = model(adv)
            adversarial_predictions = torch.softmax(adv_logits, dim=1)
            adv_labels = adversarial_predictions.argmax(dim=1)
            success_mask = adv_labels.ne(labels)
            attack_success_rate = success_mask.float().mean().item()

        diff = (adv - clean_inputs).view(adv.size(0), -1)
        perturbation_metrics = {
            "l2": diff.norm(p=2, dim=1).mean().item(),
            "linf": diff.abs().max(dim=1).values.mean().item(),
        }

        return {
            "adversarial_inputs": adv,
            "original_predictions": original_predictions,
            "adversarial_predictions": adversarial_predictions,
            "attack_success_rate": attack_success_rate,
            "perturbation_metrics": perturbation_metrics,
        }
