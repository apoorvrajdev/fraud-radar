"""Dataset adapters and the canonical representation they produce."""
from __future__ import annotations

from ml.datasets.base import (
    CANONICAL_SCHEMA_VERSION,
    CanonicalDataset,
    DataOrigin,
    DatasetAdapter,
    DatasetContractError,
    DatasetProvenance,
    Subsample,
)
from ml.datasets.registry import available_datasets, get_adapter, register_adapter

__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "CanonicalDataset",
    "DataOrigin",
    "DatasetAdapter",
    "DatasetContractError",
    "DatasetProvenance",
    "Subsample",
    "available_datasets",
    "get_adapter",
    "register_adapter",
]
