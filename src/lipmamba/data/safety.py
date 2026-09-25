"""Safety / jailbreak dataset wrappers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class SafetySample:
    prompt: str
    behavior: str
    label: str  # "harmful" | "benign"
    metadata: dict


class SafetyPromptDataset:
    """Reads HarmBench / JailbreakBench / WildJailbreak / AdvBench JSONL.

    Each line is expected to be a JSON object with at least ``prompt`` and
    ``label`` keys; unknown keys are preserved in ``metadata``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Safety dataset {self.path} not found.")

    def __iter__(self) -> Iterator[SafetySample]:
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                yield SafetySample(
                    prompt=obj["prompt"],
                    behavior=obj.get("behavior", obj.get("category", "")),
                    label=obj.get("label", "harmful"),
                    metadata={k: v for k, v in obj.items() if k not in {"prompt", "label", "behavior"}},
                )

    def to_list(self) -> list[SafetySample]:
        return list(self)
