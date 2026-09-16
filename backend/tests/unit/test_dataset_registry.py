"""Phase 5A — adapter lookup stays dataset-agnostic."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ml.datasets import base, registry
from ml.datasets.base import CanonicalDataset, DataOrigin, DatasetProvenance


class FixtureAdapter:
    """Minimal conforming adapter — loads nothing, registers like a real one."""

    def __init__(self, name: str = "fixture") -> None:
        self.name = name

    def load(
        self,
        root: Path,
        *,
        max_entities: int | None = None,
        seed: int = 42,
    ) -> CanonicalDataset:
        return CanonicalDataset(
            customers={},
            merchants={},
            transactions=[],
            labels={},
            provenance=DatasetProvenance(
                name=self.name,
                version="v1",
                origin=DataOrigin.SYNTHETIC,
                source_url="https://example.test",
                citation="fixture",
                license="CC0",
                label_field="is_fraud",
                label_definition="1 = fraud",
                subsample=base.Subsample(strategy="none", seed=seed),
            ),
        )


@pytest.fixture(autouse=True)
def isolated_registry() -> Iterator[None]:
    """Each test gets a clean registry; module imports elsewhere may populate it."""
    saved = dict(registry._ADAPTERS)
    registry._ADAPTERS.clear()
    try:
        yield
    finally:
        registry._ADAPTERS.clear()
        registry._ADAPTERS.update(saved)


def test_registered_adapter_is_returned_by_name() -> None:
    adapter = FixtureAdapter()
    registry.register_adapter(adapter)
    assert registry.get_adapter("fixture") is adapter


def test_available_datasets_lists_names_sorted() -> None:
    registry.register_adapter(FixtureAdapter("zeta"))
    registry.register_adapter(FixtureAdapter("alpha"))
    assert registry.available_datasets() == ["alpha", "zeta"]


def test_empty_registry_reports_no_datasets() -> None:
    assert registry.available_datasets() == []


def test_unknown_dataset_error_names_what_is_available() -> None:
    registry.register_adapter(FixtureAdapter("alpha"))
    with pytest.raises(ValueError, match=r"Unknown dataset 'ghost'\. Available: alpha"):
        registry.get_adapter("ghost")


def test_unknown_dataset_error_is_explicit_when_nothing_is_registered() -> None:
    with pytest.raises(ValueError, match="none registered"):
        registry.get_adapter("ghost")


def test_registering_a_different_adapter_under_a_taken_name_is_refused() -> None:
    """A run's dataset_name must keep identifying what produced it."""
    registry.register_adapter(FixtureAdapter("alpha"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register_adapter(FixtureAdapter("alpha"))


def test_registering_the_same_adapter_twice_is_idempotent() -> None:
    """Module-level registration runs again on re-import; that is harmless."""
    adapter = FixtureAdapter("alpha")
    registry.register_adapter(adapter)
    registry.register_adapter(adapter)
    assert registry.available_datasets() == ["alpha"]


def test_unnamed_adapter_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        registry.register_adapter(FixtureAdapter(""))
