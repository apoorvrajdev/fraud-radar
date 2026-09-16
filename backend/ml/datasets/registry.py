"""Adapter registry — lookup by dataset name, no dataset-specific imports.

Training, quality reporting and evaluation should not grow an `if dataset ==
"sparkov"` branch each time a benchmark is added. They ask the registry for an
adapter by name and get something satisfying `DatasetAdapter`; adding a
dataset means adding one module that registers itself.
"""
from __future__ import annotations

from ml.datasets.base import DatasetAdapter

_ADAPTERS: dict[str, DatasetAdapter] = {}


def register_adapter(adapter: DatasetAdapter) -> None:
    """Register `adapter` under its own `name`.

    Refuses to overwrite an existing name: a silent replacement would mean a
    run's `dataset_name` no longer identifies what actually produced it.
    """
    name = adapter.name
    if not name:
        raise ValueError("Adapter name must be non-empty.")
    existing = _ADAPTERS.get(name)
    if existing is not None and existing is not adapter:
        raise ValueError(f"A different adapter is already registered as {name!r}.")
    _ADAPTERS[name] = adapter


def get_adapter(name: str) -> DatasetAdapter:
    """Return the adapter registered as `name`."""
    try:
        return _ADAPTERS[name]
    except KeyError:
        known = ", ".join(available_datasets()) or "none registered"
        raise ValueError(f"Unknown dataset {name!r}. Available: {known}.") from None


def available_datasets() -> list[str]:
    """Registered dataset names, sorted."""
    return sorted(_ADAPTERS)
