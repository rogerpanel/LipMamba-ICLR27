"""LipMamba — Lipschitz-constrained selective state-space models.

Public API re-exports the most common symbols so that user code can simply

    from lipmamba import LipMambaModel, LipMambaConfig
    from lipmamba.certificates import certified_radius, pac_bayes_bound
    from lipmamba.attacks import HiSPAAttack
"""
from .models.lipmamba_model import LipMambaConfig, LipMambaModel  # noqa: F401
from .models.lipmamba_block import LipMambaBlock  # noqa: F401
from .models.selective_ssm import SelectiveSSM  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "LipMambaConfig",
    "LipMambaModel",
    "LipMambaBlock",
    "SelectiveSSM",
    "__version__",
]
