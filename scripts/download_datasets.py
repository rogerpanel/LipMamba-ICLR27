#!/usr/bin/env python
"""Print canonical download URLs (and run free downloads where possible).

Many datasets are gated behind a click-wrap or research request — for those
the script just emits the canonical URL and a short instruction.  Datasets on
the public HuggingFace hub are downloaded into ``data_cache/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lipmamba.data.registry import DATASETS

DATA_CACHE = Path("data_cache")

HF_HUB = {
    "wikitext103": ("wikitext", "wikitext-103-raw-v1"),
    "c4": ("allenai/c4", "en"),
    "slimpajama": ("cerebras/SlimPajama-627B", None),
    "wildjailbreak": ("allenai/wildjailbreak", None),
}


def hf_download(name: str) -> bool:
    try:
        from datasets import load_dataset
    except ImportError:
        print("⚠ datasets package not installed — install via 'pip install datasets'.")
        return False
    if name not in HF_HUB:
        return False
    repo, subset = HF_HUB[name]
    print(f"→ downloading {name} ({repo} {subset or ''}) into HuggingFace cache ...")
    load_dataset(repo, subset, cache_dir=str(DATA_CACHE / name))
    print(f"✓ {name} cached under {DATA_CACHE / name}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=sorted(DATASETS))
    args = ap.parse_args()
    DATA_CACHE.mkdir(exist_ok=True)
    for name in args.datasets:
        if name not in DATASETS:
            print(f"× unknown dataset {name} — skipping")
            continue
        spec = DATASETS[name]
        print(f"\n=== {spec.name} ===")
        print(f"License : {spec.license}")
        print(f"URL     : {spec.url}")
        print(f"Notes   : {spec.description}")
        if hf_download(name):
            continue
        print(f"⤷ This dataset must be downloaded manually from {spec.url}.")


if __name__ == "__main__":
    main()
