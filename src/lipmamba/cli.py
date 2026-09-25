"""Console-script entry points used by ``setup.py``.

Each entry point delegates to the matching script under ``scripts/``.
"""
from __future__ import annotations

import runpy
from pathlib import Path


def _run(script: str) -> None:
    here = Path(__file__).resolve().parent.parent.parent / "scripts" / script
    runpy.run_path(str(here), run_name="__main__")


def train_main() -> None:
    _run("train.py")


def evaluate_main() -> None:
    _run("evaluate.py")


def certify_main() -> None:
    _run("certify.py")


def attack_main() -> None:
    _run("attack.py")
