"""Dataset registry for LipMamba experiments.

The actual datasets are gated by their respective licences and must be
downloaded from the canonical URLs listed in :mod:`registry`.  This package
provides a uniform :class:`DatasetSpec` schema and lightweight PyTorch
``Dataset`` wrappers around each on-disk format.
"""
from .ids import IDSDataset, ids_feature_columns
from .language import LanguageModellingDataset, list_language_corpora
from .registry import DATASETS, DatasetSpec, dataset_url
from .robench import RoBenchDataset
from .safety import SafetyPromptDataset

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "dataset_url",
    "IDSDataset",
    "ids_feature_columns",
    "LanguageModellingDataset",
    "list_language_corpora",
    "RoBenchDataset",
    "SafetyPromptDataset",
]
