# Matched fine-tuning at 130M parameters: results

Outputs of `scripts/scale_study.py` (see `docs/REPRODUCIBILITY.md`, section 8).

* `kaggle_t4/`: the study as reported. Two Tesla T4, preset `t4` (two accumulated
  batches of 8, 400 steps, 48 test examples per seed and trigger length, 40 search
  iterations of 32 candidates). Fourteen finished runs: `mamba_released`, `mamba`,
  `lipcal`, `mamba_at` for seeds 0-2 and `lipcal_at` for seeds 0-1.
  `*.partial.json` are two single-constraint ablations (seed 0) that were stopped
  after trigger lengths 8 and 16. `summary.json` / `summary.md` aggregate the
  finished runs.
* `colab_a100_training_only/`: an earlier session on one A100 (preset `a100`, batch
  16) in which every run completed fine-tuning and the clean evaluation but stopped
  before the attacks because of a bug fixed in commit 846ac1f. It is the source of
  the paper-constant rows (`lip`, `lip_at`): initial and final perplexity and
  accuracy. `diagnose.json` holds the calibrated constants and the starting
  perplexity of every constrained variant.

Not run: `lipcal_at` seed 2, trigger length 32 of the ablations, the other two
ablations, and attacks on the paper-constant arms.
