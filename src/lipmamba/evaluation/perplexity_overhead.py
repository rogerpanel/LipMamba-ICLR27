"""Clean-perplexity overhead relative to a stated base model (Figure 3).

The overhead is only interpretable when the base model, its checkpoint and
the evaluation corpus are stated (TODO 3 of the manuscript).  This helper
therefore *requires* all three and records them in its output.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch.nn as nn
from torch.utils.data import DataLoader

from .perplexity import perplexity


@dataclass
class PerplexityProvenance:
    base_model_id: str            # e.g. "state-spaces/mamba-130m"
    base_checkpoint: str          # HF revision or local path
    corpus: str                   # e.g. "WikiText-103 validation (raw), GPT-NeoX tokenizer"
    tokenizer: str
    block_size: int
    n_tokens_evaluated: int


def perplexity_overhead(base: nn.Module, constrained: nn.Module, loader: DataLoader,
                        provenance: PerplexityProvenance) -> dict:
    ppl_base = perplexity(base, loader)
    ppl_lip = perplexity(constrained, loader)
    return {**asdict(provenance), "ppl_base": ppl_base, "ppl_constrained": ppl_lip,
            "overhead_percent": 100.0 * (ppl_lip - ppl_base) / ppl_base}
