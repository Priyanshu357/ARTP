"""Fast Gradient Sign Method (FGSM) adversarial attack implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base_attack import BaseAttack


class FGSMAttack(BaseAttack):
    """Single-step FGSM attack for PyTorch models.

    This attack perturbs inputs in the direction of the sign of the gradient of
    the loss with respect to the inputs. It is untargeted by default and aims to
    increase the loss on the true labels.
    """

    def __init__(self, epsilon: float = 0.007, clip_min: float = 0.0, clip_max: float = 1.0,
                 loss_fn: Optional[nn.Module] = None) -> None:
        """Initialize the FGSM attack.

        Args:
            epsilon: Maximum perturbation per input dimension (Linf bound).
            clip_min: Minimum value for input clamping.
            clip_max: Maximum value for input clamping.
            loss_fn: Loss function; defaults to cross-entropy if not provided.
        """
        self.epsilon = float(epsilon)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def generate(self, model: Any, inputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """Run FGSM to craft adversarial examples and report metrics.

        Args:
            model: PyTorch model with callable forward (logits output) and parameters on a device.
            inputs: Clean input tensor with shape (N, ...).
            labels: Ground-truth labels as tensor of shape (N,).

        Returns:
            Dict[str, Any]: Attack results following the BaseAttack contract.
        """
        if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("inputs and labels must be torch.Tensor instances")

        device = next(model.parameters()).device
        model.eval()

        clean_inputs = inputs.detach().to(device)
        labels = labels.detach().to(device)

        clean_inputs.requires_grad_(True)

        logits = model(clean_inputs)
        loss = self.loss_fn(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()

        grad_sign = clean_inputs.grad.sign()
        adv_inputs = clean_inputs + self.epsilon * grad_sign
        adv_inputs = torch.clamp(adv_inputs, self.clip_min, self.clip_max).detach()

        with torch.no_grad():
            original_predictions = torch.softmax(logits, dim=1)
            adv_logits = model(adv_inputs)
            adversarial_predictions = torch.softmax(adv_logits, dim=1)

            orig_labels = original_predictions.argmax(dim=1)
            adv_labels = adversarial_predictions.argmax(dim=1)
            success_mask = adv_labels.ne(labels)
            attack_success_rate = success_mask.float().mean().item()

        diff = (adv_inputs - clean_inputs.detach()).view(adv_inputs.size(0), -1)
        perturbation_metrics = {
            "l2": diff.norm(p=2, dim=1).mean().item(),
            "linf": diff.abs().max(dim=1).values.mean().item(),
        }

        return {
            "adversarial_inputs": adv_inputs,
            "original_predictions": original_predictions,
            "adversarial_predictions": adversarial_predictions,
            "attack_success_rate": attack_success_rate,
            "perturbation_metrics": perturbation_metrics,
        }
