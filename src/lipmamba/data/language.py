"""Language-modelling dataset wrappers.

The actual tokenisation pipeline is intentionally minimal — we expect users
to plug in their own (`tokenizers`, `sentencepiece`, ...).  These wrappers
expose a uniform iterator interface returning fixed-length token blocks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import torch
from torch.utils.data import Dataset

from .registry import LM_EVAL, PRETRAIN


def list_language_corpora() -> list[str]:
    """Return all known language datasets (pre-training + evaluation)."""
    return sorted({**PRETRAIN, **LM_EVAL}.keys())


class LanguageModellingDataset(Dataset):
    """Block-pack a tokenised numpy array on disk.

    The on-disk format is a flat ``int32`` array containing the entire
    tokenised corpus.  The constructor lazily memory-maps it and returns
    contiguous ``(block_size,)`` slices.
    """

    def __init__(
        self,
        token_path: str | Path,
        block_size: int = 1024,
        dtype: torch.dtype = torch.int32,
    ) -> None:
        path = Path(token_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Token file {path} not found. Run scripts/download_datasets.py first."
            )
        self._tokens = torch.from_file(
            str(path), shared=True, size=path.stat().st_size // 4, dtype=dtype
        )
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, (self._tokens.numel() - 1) // self.block_size)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size + 1
        chunk = self._tokens[start:end].long()
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


def collate_lm(samples: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Default collator for :class:`LanguageModellingDataset`."""
    samples = list(samples)
    return {
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "labels": torch.stack([s["labels"] for s in samples]),
    }


def text_files_to_token_array(
    text_paths: Iterable[str | Path],
    encoder: Callable[[str], list[int]],
    out_path: str | Path,
) -> int:
    """Helper: tokenise raw text files into a flat ``int32`` array."""
    import numpy as np

    chunks: list[list[int]] = []
    for p in text_paths:
        with open(p, "r", encoding="utf-8") as fh:
            chunks.append(encoder(fh.read()))
    arr = np.concatenate([np.asarray(c, dtype=np.int32) for c in chunks])
    arr.tofile(str(out_path))
    return int(arr.size)
