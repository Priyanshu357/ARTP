"""Blue team defenses for multiple model types."""

# Image classification detectors (backward compatibility)
from .image_models import (
    BaseDetector,
    ConfidenceDropDetector,
    EntropyIncreaseDetector,
    EnsembleDetector,
)

__all__ = [
    "BaseDetector",
    "ConfidenceDropDetector",
    "EntropyIncreaseDetector",
    "EnsembleDetector",
]
