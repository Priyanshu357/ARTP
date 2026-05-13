"""Red team adversarial attacks for multiple model types."""

# Image classification attacks (backward compatibility)
from .image_models import (
    BaseAttack,
    AttackRunner,
    FGSMAttack,
    PGDAttack,
    DeepFoolAttack,
)

__all__ = [
    "BaseAttack",
    "AttackRunner",
    "FGSMAttack",
    "PGDAttack",
    "DeepFoolAttack",
]
