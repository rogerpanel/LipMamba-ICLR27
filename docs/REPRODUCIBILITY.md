# Reproducibility Guide

Exact command sequence to regenerate every number of the ICLR manuscript.
Python 3.10+, CUDA 12 for the 130M/370M runs (the harness runs on CPU for
the demo path).

## 1. Environment

```bash
git clone https://anonymous.4open.science/r/LipMamba-ICLR27 && cd CV/lipmamba
python -m venv .venv && source .venv/bin/activate
pip install -e ".[training,dev]"
pytest -q                                       # 36 tests
python scripts/regenerate_all.py --demo         # end-to-end smoke run on a random model
```

## 2. Constants first

```bash
python scripts/report_constants.py --config configs/lipmamba_130m.yaml
```

Read the three lines under "Readings".  If ℓ\* is < 1 or L_block is ≫ 10,
decide (docs/ICLR2027_AUDIT_RESPONSE.md §4) whether to change Δ_min, Δ_max,
λ_min, λ_max before spending GPU time.

## 3. Base checkpoints and data

```bash
python scripts/todo3_init_from_hf_mamba.py --base state-spaces/mamba-130m --revision <commit> \
    --out runs/lipmamba-130m_init.pt
python scripts/download_datasets.py --datasets wikitext103
python - <<'EOF'
from transformers import AutoTokenizer; import numpy as np, datasets
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
for split in ("train", "validation"):
    ds = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    ids = np.asarray(tok("\n".join(ds["text"]))["input_ids"], dtype=np.int32)
    ids.tofile(f"data_cache/wikitext103_{'val' if split=='validation' else 'train'}.bin")
EOF
```

RoBench-25: obtain the HiSPA authors' anonymised release, export to JSONL
(`abstract_id, abstract, question, answer`), tokenise prompts with the same
tokenizer.

## 4. Train

```bash
python scripts/finetune.py --config configs/lipmamba_130m.yaml --init-from runs/lipmamba-130m_init.pt
python scripts/finetune.py --config configs/lipmamba_370m.yaml --init-from runs/lipmamba-370m_init.pt
```

## 5. Regenerate every table and figure (three seeds)

```bash
python scripts/regenerate_all.py --config configs/lipmamba_130m.yaml \
    --checkpoint runs/lipmamba_130m/final.pt --tokens data_cache/wikitext103_val.bin
python scripts/perplexity_overhead.py --base state-spaces/mamba-130m --lip runs/lipmamba_130m/final.pt \
    --config configs/lipmamba_130m.yaml --tokens data_cache/wikitext103_val.bin
python scripts/fill_paper_numbers.py --runs runs --out paper/todo_snippets.tex
```

Outputs: `runs/todo{1,2,4,5}_*.json` (+ `.png`), `runs/perplexity_overhead.json`,
`paper/todo_snippets.tex`.

## 6. Anonymised bundle

```bash
python scripts/todo6_make_anonymous_bundle.py --out LipMamba-ICLR27-anon.zip
```

Aborts if any identifying string remains.  Upload to anonymous.4open.science
and set `\anonrepo`.

## 7. What "reproduced" means here

The manuscript's Table 1 numbers were not produced by this repository at the
time of writing; §5 regenerates them.  The demo path
(`--demo`) validates the pipeline on a random model and its numbers are
meaningless as results.
