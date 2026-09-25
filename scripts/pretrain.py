#!/usr/bin/env python
"""Convenience wrapper around ``scripts/train.py`` for large-scale pre-training.

Identical to ``train.py`` but defaults to disabling the margin-augmented
classification objective (the model has no classification head during
pre-training).
"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve().parent / "train.py"
    runpy.run_path(str(here), run_name="__main__")
