# Datasets

All datasets referenced in the manuscripts, with canonical URLs and licence
notes; also exposed in `lipmamba.data.registry`.  Nothing is redistributed.

## Pre-training / language modelling

| Dataset | Tokens | License | Link |
| --- | --- | --- | --- |
| The Pile (corpus of the `state-spaces/mamba-*` base checkpoints) | 825 GiB | various per subset | <https://pile.eleuther.ai/> |
| SlimPajama-627B | 627 B | various | <https://huggingface.co/datasets/cerebras/SlimPajama-627B> |
| C4 (en) | ~365 B | ODC-BY | <https://huggingface.co/datasets/allenai/c4> |
| WikiText-103 (perplexity-overhead evaluation) | 103 M | CC BY-SA 3.0 | <https://huggingface.co/datasets/wikitext> |

Base checkpoints (TODO 3): `state-spaces/mamba-130m`, `state-spaces/mamba-370m`
(<https://huggingface.co/state-spaces>), GPT-NeoX tokenizer
(`EleutherAI/gpt-neox-20b`).  Record the HF revision hash you used.

## Hidden-state poisoning and safety

| Dataset | Content | Status / License | Link |
| --- | --- | --- | --- |
| **RoBench-25** | 120 abstracts of accepted NeurIPS-2025 papers + 240 true/false questions (2 per abstract), evaluated with and without trigger insertion | 2026 **preprint**, unrefereed; anonymised 4open.science release by the HiSPA authors, public release pending publication | <https://arxiv.org/abs/2601.01972> |
| HarmBench | 510 behaviours × 18 red-team methods; HarmBench-CLS scorer | research only | <https://www.harmbench.org/> |
| JailbreakBench | 100 misuse + 100 benign behaviours | MIT | <https://github.com/JailbreakBench/jailbreakbench> |
| AdvBench | 520 harmful prompts | MIT | <https://github.com/llm-attacks/llm-attacks> |
| WildJailbreak | 2 210 adversarial-evaluation prompts | ODC-BY | <https://huggingface.co/datasets/allenai/wildjailbreak> |

**Attribution notes (from the ICLR audit).**  HiSPA / RoBench-25
(arXiv:2601.01972) and CLASP (arXiv:2603.12206) are by Le Mercier, Develder
& Demeester (Ghent–imec).  **SpectralGuard** (arXiv:2603.12414) is by
**Davi Bonetto**, not Le Mercier.  All are unrefereed 2026 preprints.
RoBench-25 is *not* "1 050 triggers across nine families"; triggers are
generated at evaluation time by Z-/M-HiSPA (`lipmamba.attacks.hispa`).

## Network intrusion detection (deployment-domain transfer, Appendix F)

| Dataset | Size | License | Link |
| --- | --- | --- | --- |
| CIC-IDS2017 | 2.8 M flows | research only | <https://www.unb.ca/cic/datasets/ids-2017.html> |
| CIC-IoT-2023 | 1.2 M flows | research only | <https://www.unb.ca/cic/datasets/iotdataset-2023.html> |
| CIC-DDoS-2019 | 80 M+ flows | research only | <https://www.unb.ca/cic/datasets/ddos-2019.html> |
| Edge-IIoTset | 20.8 M flows | CC BY-SA 4.0 | <https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot> |
| UNSW-NB15 | 2.5 M flows | research only | <https://research.unsw.edu.au/projects/unsw-nb15-dataset> |
| TON_IoT | telemetry + flows | research only | <https://research.unsw.edu.au/projects/toniot-datasets> |
| NSL-KDD | 148 K records | open | <https://www.unb.ca/cic/datasets/nsl.html> |
| PQC traffic | post-quantum captures | CC0 | <https://doi.org/10.34740/kaggle/dsv/15424420> |

## Independent robustness benchmarks suggested by the audit (Task 5)

* Qi et al., *Exploring Adversarial Robustness of Deep State Space Models*, NeurIPS 2024 (arXiv:2406.05532) — CIFAR-10/100, Tiny-ImageNet protocol; the key prior art for input-dependent SSM error bounds.
* BackdoorLLM (Li et al., NeurIPS 2025, arXiv:2408.12798) — includes a hidden-state modality.
* BadVim (Lee et al., arXiv:2408.11679) / BadViM (Wu & Zhang, arXiv:2507.00577) — Vision-Mamba backdoors.
* Long Range Arena — long-context SSM classification under certified perturbation.

## Downloading

```bash
python scripts/download_datasets.py --datasets wikitext103 cicids2017
```

Gated datasets (CIC, UNSW, PQC DOI, RoBench-25 release) print the canonical
URL; download manually into `data_cache/<name>/`.
