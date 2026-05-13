"""Configuration definitions for the pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch

from red_team import BaseAttack

DetectionFn = Callable[[Mapping[str, Any]], Iterable[Mapping[str, Any]]]


class ModelType(Enum):
    """Supported model types for adversarial evaluation."""

    IMAGE = "image"
    NLP = "nlp"
    AUDIO = "audio"
    OBJECT_DETECTION = "object_detection"
    UNKNOWN = "unknown"


def detect_model_type(
    model: torch.nn.Module, dataloader: Optional[Iterable] = None
) -> ModelType:
    """Auto-detect model type based on architecture and input shape.

    Uses heuristics to identify model type:
    - Image: Conv2d layers, typical for image classification
    - NLP: Embedding layers, Transformer blocks
    - Audio: 1D convolutions
    - Object Detection: Multiple output heads

    Args:
        model: PyTorch model to analyze
        dataloader: Optional dataloader to inspect input shapes

    Returns:
        ModelType enum indicating detected model type
    """
    # Check for convolutional layers (vision models)
    has_conv2d = any(isinstance(m, torch.nn.Conv2d) for m in model.modules())
    has_conv1d = any(isinstance(m, torch.nn.Conv1d) for m in model.modules())

    # Check for embeddings (NLP models)
    has_embedding = any(isinstance(m, torch.nn.Embedding) for m in model.modules())

    # Check for transformer layers (NLP models)
    has_transformer = any(
        "transformer" in type(m).__name__.lower() for m in model.modules()
    )

    # Heuristic detection
    if has_conv2d:
        # Could be image classification or object detection
        # For now, default to image classification
        # TODO: Add more sophisticated detection for object detection
        return ModelType.IMAGE
    elif has_embedding or has_transformer:
        return ModelType.NLP
    elif has_conv1d:
        return ModelType.AUDIO
    else:
        return ModelType.UNKNOWN


@dataclass
class PipelineConfig:
    """User-supplied configuration for a pipeline run."""

    model: torch.nn.Module
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]]
    attacks: Sequence[BaseAttack]
    report_path: Path
    attack_results_path: Optional[Path] = None
    detection_results_path: Optional[Path] = None
    project_title: str = "Adversarial Security Report"
    model_info: Dict[str, Any] = field(default_factory=dict)
    detection_fn: Optional[DetectionFn] = None
    max_batches: int = 5
    model_type: Optional[Union[ModelType, str]] = None  # Optional manual override

    def get_model_type(self) -> ModelType:
        """Get or auto-detect model type.

        If model_type is explicitly set in config, use that value.
        Otherwise, auto-detect based on model architecture.

        Returns:
            ModelType enum indicating the model type
        """
        if self.model_type is not None:
            if isinstance(self.model_type, str):
                return ModelType(self.model_type)
            return self.model_type
        return detect_model_type(self.model, self.dataloader)

    def normalized_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_info.get("name", getattr(self.model, "__class__", type("", (), {})).__name__),
            "version": self.model_info.get("version", "n/a"),
            "framework": self.model_info.get("framework", "torch/onnx"),
            "notes": self.model_info.get("notes", ""),
        }
