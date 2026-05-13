"""Chart helpers for PDF reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


def plot_attack_success_bar(attack_results: Sequence[Mapping[str, object]], out_path: Path) -> Optional[Path]:
    """Create a bar chart of attack success rates.

    attack_results: list of dicts containing keys "attack" and nested result.attack_success_rate.
    Returns the output path on success, None on failure.
    """
    try:
        names = []
        rates = []
        for entry in attack_results:
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("attack") or "Attack"
            res = entry.get("result")
            rate = None
            if isinstance(res, Mapping):
                val = res.get("attack_success_rate")
                if isinstance(val, (float, int)):
                    rate = float(val)
            if rate is not None:
                names.append(str(name))
                rates.append(rate)
        if not names:
            return None
        plt.figure(figsize=(6, 3))
        plt.bar(names, rates, color="#d9534f")
        plt.ylim(0, 1)
        plt.ylabel("Success Rate")
        plt.title("Attack Success Rates")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path
    except Exception:
        return None


def plot_detection_confusion(detection_results: Iterable[Mapping[str, object]], out_path: Path) -> Optional[Path]:
    """Plot a simple 2x2 confusion matrix heatmap for detection results."""
    try:
        tp = fp = tn = fn = 0
        for entry in detection_results:
            if not isinstance(entry, Mapping):
                continue
            is_attack = entry.get("is_attack")
            detected = entry.get("detected")
            if not isinstance(is_attack, bool) or not isinstance(detected, bool):
                continue
            if is_attack and detected:
                tp += 1
            elif is_attack and not detected:
                fn += 1
            elif (not is_attack) and detected:
                fp += 1
            else:
                tn += 1
        total = tp + fp + tn + fn
        if total == 0:
            return None
        import numpy as np  # local import to avoid hard dependency in unused cases

        data = np.array([[tp, fp], [fn, tn]], dtype=float)
        plt.figure(figsize=(3, 3))
        plt.imshow(data, cmap="Blues")
        for (i, j), val in np.ndenumerate(data):
            plt.text(j, i, f"{int(val)}", ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred Attack", "Pred Benign"])
        plt.yticks([0, 1], ["True Attack", "True Benign"])
        plt.title("Detection Confusion")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path
    except Exception:
        return None
