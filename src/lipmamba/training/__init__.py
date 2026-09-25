"""Training entry points (objective + trainer + optimiser)."""
from .adv_objective import adversarial_loss
from .optim import build_optimizer, build_scheduler
from .pac_objective import pac_bayes_total_loss
from .trainer import LipMambaTrainer, TrainerConfig

__all__ = [
    "adversarial_loss",
    "build_optimizer",
    "build_scheduler",
    "pac_bayes_total_loss",
    "LipMambaTrainer",
    "TrainerConfig",
]
