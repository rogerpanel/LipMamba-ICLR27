"""Canonical URLs and metadata for every dataset used in the paper."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    license: str
    description: str


# Pre-training corpora -------------------------------------------------------
PRETRAIN: dict[str, DatasetSpec] = {
    "the_pile": DatasetSpec(
        name="The Pile",
        url="https://pile.eleuther.ai/",
        license="various (per subset)",
        description="825 GiB English text used for LM pre-training (EleutherAI).",
    ),
    "slimpajama": DatasetSpec(
        name="SlimPajama-627B",
        url="https://huggingface.co/datasets/cerebras/SlimPajama-627B",
        license="various",
        description="627 B deduplicated tokens, mixture of seven sources.",
    ),
    "c4": DatasetSpec(
        name="C4",
        url="https://huggingface.co/datasets/allenai/c4",
        license="ODC-BY",
        description="Colossal Clean Crawled Corpus, English.",
    ),
}

# Language-modelling evaluation ---------------------------------------------
LM_EVAL: dict[str, DatasetSpec] = {
    "wikitext103": DatasetSpec(
        name="WikiText-103",
        url="https://huggingface.co/datasets/wikitext",
        license="CC BY-SA 3.0",
        description="103 M token Wikipedia LM benchmark.",
    ),
}

# Safety / robustness evaluation --------------------------------------------
SAFETY: dict[str, DatasetSpec] = {
    "harmbench": DatasetSpec(
        name="HarmBench",
        url="https://www.harmbench.org/",
        license="Research only",
        description="510 harmful behaviours × 18 red-team methods.",
    ),
    "jailbreakbench": DatasetSpec(
        name="JailbreakBench",
        url="https://github.com/JailbreakBench/jailbreakbench",
        license="MIT",
        description="100 misuse + 100 benign behaviours.",
    ),
    "advbench": DatasetSpec(
        name="AdvBench",
        url="https://github.com/llm-attacks/llm-attacks",
        license="MIT",
        description="Harmful prompts from llm-attacks.",
    ),
    "wildjailbreak": DatasetSpec(
        name="WildJailbreak",
        url="https://huggingface.co/datasets/allenai/wildjailbreak",
        license="ODC-BY",
        description="2,210 adversarial-evaluation prompts.",
    ),
    "robench25": DatasetSpec(
        name="RoBench-25",
        url="https://arxiv.org/abs/2601.01972",
        license="Research only (anonymised 4open.science release by the HiSPA authors; public release pending publication)",
        description=(
            "Long-context retrieval benchmark of the HiSPA preprint: 120 abstracts of accepted "
            "NeurIPS-2025 papers with 240 true/false questions, evaluated with/without trigger insertion. "
            "NOT a separate trigger collection."
        ),
    ),
}

# Network intrusion datasets (the deployment platform integration) --------------------
IDS: dict[str, DatasetSpec] = {
    "cicids2017": DatasetSpec(
        name="CIC-IDS2017",
        url="https://www.unb.ca/cic/datasets/ids-2017.html",
        license="Custom (research)",
        description="2.8 M flows, 80 features, 15 attack classes.",
    ),
    "cic_iot_2023": DatasetSpec(
        name="CIC-IoT-2023",
        url="https://www.unb.ca/cic/datasets/iotdataset-2023.html",
        license="Custom (research)",
        description="1.2 M flows, 33 attack types, IoT environment.",
    ),
    "cic_ddos_2019": DatasetSpec(
        name="CIC-DDoS-2019",
        url="https://www.unb.ca/cic/datasets/ddos-2019.html",
        license="Custom (research)",
        description="80 M+ flows, modern DDoS attacks.",
    ),
    "edge_iiotset": DatasetSpec(
        name="Edge-IIoTset",
        url=(
            "https://www.kaggle.com/datasets/mohamedamineferrag/"
            "edgeiiotset-cyber-security-dataset-of-iot-iiot"
        ),
        license="CC BY-SA 4.0",
        description="20.8 M flows, 61 features, 14 attacks (edge IIoT).",
    ),
    "unsw_nb15": DatasetSpec(
        name="UNSW-NB15",
        url="https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        license="Research only",
        description="2.5 M flows, 9 attack families.",
    ),
    "ton_iot": DatasetSpec(
        name="TON_IoT",
        url="https://research.unsw.edu.au/projects/toniot-datasets",
        license="Research only",
        description="Heterogeneous IoT/IIoT telemetry + network logs.",
    ),
    "nsl_kdd": DatasetSpec(
        name="NSL-KDD",
        url="https://www.unb.ca/cic/datasets/nsl.html",
        license="Open",
        description="148 K records, 4 attack categories.",
    ),
    "pqc_traffic": DatasetSpec(
        name="PQC Traffic",
        url="https://doi.org/10.34740/kaggle/dsv/15424420",
        license="CC0 (per Kaggle DOI)",
        description="Post-quantum cryptography handshake captures.",
    ),
}

DATASETS: dict[str, DatasetSpec] = {**PRETRAIN, **LM_EVAL, **SAFETY, **IDS}


def dataset_url(name: str) -> str:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name].url
