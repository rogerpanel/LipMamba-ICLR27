#!/usr/bin/env python
"""Regenerate every table and figure of the ICLR manuscript from the seeds
{42, 137, 2026} (Reproducibility Statement).

    python scripts/regenerate_all.py --config configs/lipmamba_130m.yaml \
        --checkpoint runs/lipmamba_130m/final.pt --tokens data_cache/wikitext103_val.bin
    python scripts/regenerate_all.py --demo      # smoke run on a random model (minutes, CPU)

Order:
  0. report_constants        — closed-form constants (sanity: prints the ℓ*≈1 / 10^193 facts)
  1. todo1_ell_star          — Fig. 4 shaded region, Remark 5 numbers
  2. todo2_adaptive_attack   — adaptive-attack paragraph
  3. todo5_fig2              — Fig. 2 (all four curves)
  4. todo4 baselines         — Table 1 rows incl. GloRo-Mamba, Friedman/Holm
  5. fill_paper_numbers      — LaTeX snippets
(TODO 3 needs network access to HuggingFace: run scripts/todo3_init_from_hf_mamba.py separately.)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(HERE / script), *args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--checkpoint"); ap.add_argument("--tokens")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2026])
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    common = ["--demo"] if a.demo else []
    if not a.demo:
        common = ["--config", a.config, "--checkpoint", a.checkpoint]
    run("report_constants.py", *(["--config", a.config] if a.config else []))
    run("todo1_ell_star.py", *common, "--out", f"{a.runs}/todo1_ell_star.json")
    run("todo2_adaptive_attack.py", *common, *([] if a.demo else ["--data", a.tokens]), "--out", f"{a.runs}/todo2_adaptive.json")
    run("todo5_fig2_lipschitz_depth.py", *common, *([] if a.demo else ["--tokens", a.tokens]), "--out", f"{a.runs}/todo5_fig2.json")
    run("todo4_gloro_mamba_baseline.py", *(["--demo"] if a.demo else ["--config", a.config]), "--seeds", *map(str, a.seeds), "--out", f"{a.runs}/todo4_baselines.json")
    run("fill_paper_numbers.py", "--runs", a.runs, "--out", "paper/todo_snippets.tex")


if __name__ == "__main__":
    main()
