"""Network-intrusion-detection datasets (deployment-domain transfer, Appendix F).

Supports the eight datasets listed in :mod:`registry`.  All datasets are
flat CSV/Parquet flow records with a single label column; we standardise
features per-column with the running mean / std.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Canonical column lists (per-dataset).  Only the union of features needed
# by LipMamba's IDS classifier is enumerated here; users may extend.
IDS_FEATURE_COLUMNS: dict[str, list[str]] = {
    "cicids2017": [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
        "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
        "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
        "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
        "Flow IAT Min",
    ],
    "edge_iiotset": [
        "frame.time_epoch", "frame.len", "tcp.dstport", "tcp.srcport",
        "tcp.flags", "tcp.window_size_value", "ip.proto", "ip.ttl",
        "icmp.type", "udp.length", "http.request.method",
    ],
    "unsw_nb15": [
        "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes",
        "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss",
        "sinpkt", "dinpkt", "sjit", "djit",
    ],
    "ton_iot": [
        "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "dst_pkts",
        "src_ip_bytes", "dst_ip_bytes", "duration", "conn_state",
    ],
    "nsl_kdd": [
        "duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    ],
}


def ids_feature_columns(name: str) -> list[str]:
    if name not in IDS_FEATURE_COLUMNS:
        raise KeyError(
            f"Unknown IDS dataset {name!r}; known: {sorted(IDS_FEATURE_COLUMNS)}"
        )
    return list(IDS_FEATURE_COLUMNS[name])


@dataclass
class IDSDatasetConfig:
    name: str
    csv_path: str
    label_col: str = "Label"
    standardise: bool = True
    drop_na: bool = True


class IDSDataset(Dataset):
    """Generic flow-CSV → tensor dataset."""

    def __init__(self, cfg: IDSDatasetConfig) -> None:
        path = Path(cfg.csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"IDS dataset {path} missing — see scripts/download_datasets.py"
            )
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        if cfg.drop_na:
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

        cols = [c for c in ids_feature_columns(cfg.name) if c in df.columns]
        if not cols:
            raise ValueError(
                f"None of the canonical columns for {cfg.name} are present in {path}."
            )

        x = df[cols].astype(np.float32).to_numpy()
        if cfg.standardise:
            mu = x.mean(axis=0, keepdims=True)
            sd = x.std(axis=0, keepdims=True) + 1e-6
            x = (x - mu) / sd

        y_raw = df[cfg.label_col].astype(str).to_numpy()
        labels, y = np.unique(y_raw, return_inverse=True)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.classes_ = labels.tolist()
        self.feature_cols = cols

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"features": self.x[idx], "label": self.y[idx]}

    @property
    def num_classes(self) -> int:
        return len(self.classes_)
