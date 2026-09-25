#!/usr/bin/env python
"""Perplexity overhead versus a *named* base checkpoint (TODO 3 / Figure 3).

    python scripts/perplexity_overhead.py --base state-spaces/mamba-130m \
        --lip runs/lipmamba_130m/final.pt --config configs/lipmamba_130m.yaml \
        --tokens data_cache/wikitext103_val.bin --block 1024

The base model is evaluated through the *same* LipMamba code path with all
constraints disabled (preset "unconstrained_mamba") after loading the same
HF tensors, so the only difference between the two numbers is the
constraint set.  The output JSON records base id, revision, corpus,
tokenizer and the number of evaluated tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from lipmamba import LipMambaModel
from lipmamba.baselines import preset
from lipmamba.data.language import LanguageModellingDataset, collate_lm
from lipmamba.evaluation import PerplexityProvenance, perplexity_overhead
from lipmamba.utils import load_checkpoint
from todo3_init_from_hf_mamba import BASES, CORPUS, convert, load_hf_state_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="state-spaces/mamba-130m", choices=sorted(BASES))
    ap.add_argument("--revision", default=None)
    ap.add_argument("--lip", required=True, help="constrained checkpoint (.pt)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", required=True, help="flat int32 token file of the eval corpus")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--out", default="runs/perplexity_overhead.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    from train import build_model
    lip = build_model(dict(cfg["model"])); load_checkpoint(args.lip, lip)

    base_cfg = preset("unconstrained_mamba", **BASES[args.base])
    sd = load_hf_state_dict(args.base, args.revision)
    base, _ = convert(sd, base_cfg)

    ds = LanguageModellingDataset(args.tokens, block_size=args.block)
    if args.max_blocks:
        ds = torch.utils.data.Subset(ds, range(min(args.max_blocks, len(ds))))
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_lm)
    if torch.cuda.is_available():
        lip.cuda(); base.cuda()
    prov = PerplexityProvenance(base_model_id=args.base, base_checkpoint=args.revision or "main",
                                corpus=CORPUS["perplexity_eval"], tokenizer=CORPUS["tokenizer"],
                                block_size=args.block, n_tokens_evaluated=len(ds) * args.block)
    res = perplexity_overhead(base, lip, loader, prov)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
