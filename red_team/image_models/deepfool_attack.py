"""DeepFool L2-minimizing adversarial attack implementation."""

from __future__ import annotations

from typing import Any, Dict

import torch

from .base_attack import BaseAttack


class DeepFoolAttack(BaseAttack):
    """Iterative DeepFool attack for PyTorch classifiers (L2)."""

    def __init__(self, max_iter: int = 50, overshoot: float = 0.02,
                 clip_min: float = 0.0, clip_max: float = 1.0, eps: float = 1e-8) -> None:
        """Configure DeepFool attack parameters.

        Args:
            max_iter: Maximum number of iterations before giving up.
            overshoot: Scaling factor to cross the decision boundary.
            clip_min: Minimum value for input clamping.
            clip_max: Maximum value for input clamping.
            eps: Numerical stability term for norm divisions.
        """
        self.max_iter = int(max_iter)
        self.overshoot = float(overshoot)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.eps = float(eps)

    def generate(self, model: Any, inputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """Run DeepFool to craft adversarial examples and report metrics."""
        if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("inputs and labels must be torch.Tensor instances")

        device = next(model.parameters()).device
        model.eval()

        clean_inputs = inputs.detach().to(device)
        labels = labels.detach().to(device)

        with torch.no_grad():
            clean_logits = model(clean_inputs)
            original_predictions = torch.softmax(clean_logits, dim=1)

        adv = clean_inputs.clone().detach()
        batch_size = adv.size(0)
        succeeded = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max(self.max_iter, 1)):
            all_succeeded = True
            adv_list = []
            for idx in range(batch_size):
                if succeeded[idx]:
                    adv_list.append(adv[idx:idx+1])
                    continue

                x = adv[idx:idx+1].clone().detach().requires_grad_(True)
                logits = model(x)
                pred_label = logits.argmax(dim=1)

                if pred_label.item() != labels[idx].item():
                    succeeded[idx] = True
                    adv_list.append(x.detach())
                    continue

                gradients = []
                values = []
                for k in range(logits.size(1)):
                    model.zero_grad(set_to_none=True)
                    logits[0, k].backward(retain_graph=True)
                    gradients.append(x.grad.detach().clone())
                    values.append(logits[0, k].detach())
                    x.grad.zero_()

                label = labels[idx].item()
                grad_orig = gradients[label]
                f_orig = values[label]

                min_pert = None
                w_star = None

                for k in range(logits.size(1)):
                    if k == label:
                        continue
                    w_k = gradients[k] - grad_orig
                    f_k = values[k] - f_orig
                    w_k_norm = torch.norm(w_k.flatten(), p=2).item() + self.eps
                    pert_k = abs(f_k.item()) / w_k_norm
                    if (min_pert is None) or (pert_k < min_pert):
                        min_pert = pert_k
                        w_star = w_k

                if w_star is None or min_pert is None:
                    adv_list.append(x.detach())
                    continue

                r_i = (min_pert * w_star / (torch.norm(w_star.flatten(), p=2) + self.eps))
                x_adv = x + (1.0 + self.overshoot) * r_i
                x_adv = torch.clamp(x_adv.detach(), self.clip_min, self.clip_max)

                adv_list.append(x_adv)
                all_succeeded = False if pred_label.item() == labels[idx].item() else all_succeeded

            adv = torch.cat(adv_list, dim=0).detach()
            if all_succeeded:
                break

        with torch.no_grad():
            adv_logits = model(adv)
            adversarial_predictions = torch.softmax(adv_logits, dim=1)
            adv_labels = adversarial_predictions.argmax(dim=1)
            success_mask = adv_labels.ne(labels)
            attack_success_rate = success_mask.float().mean().item()

        diff = (adv - clean_inputs).view(batch_size, -1)
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
