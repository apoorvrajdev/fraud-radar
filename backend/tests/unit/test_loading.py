"""Phase 5D — one loader for training and analysis, parameterised by a run record.

A run is analysed on the matrix it was trained on only if both read it through
the same code with the same inputs. These tests pin how a `DataRequest` is
built from a run's `run.json`, and that each dataset kind is loaded the way
training has always loaded it.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ml import loading
from ml.data import LabelledDataset
from ml.datasets.base import DataOrigin, DatasetProvenance, Subsample
from ml.paths import RAW_DATA_DIR
from ml.runs import RunMetadata

START = datetime(2019, 1, 1, tzinfo=UTC)


def _matrix(n_rows: int = 6) -> LabelledDataset:
    return LabelledDataset(
        X=np.arange(n_rows * 2, dtype=np.float64).reshape(n_rows, 2),
        y=np.array([index % 2 for index in range(n_rows)], dtype=np.int64),
        timestamps=np.asarray([START + timedelta(hours=h) for h in range(n_rows)], dtype=object),
        transaction_ids=[f"tx{index}" for index in range(n_rows)],
        feature_names=["a", "b"],
    )


def _record(name: str, subsample: Subsample | None, featureset: str = "v1") -> RunMetadata:
    return RunMetadata(
        run_name="fixture-run",
        dataset=DatasetProvenance(
            name=name,
            version="v1",
            origin=DataOrigin.SYNTHETIC,
            source_url="",
            citation="fixture",
            license="fixture",
            label_field="is_fraud",
            label_definition="1 = fraud",
            subsample=subsample,
        ),
        featureset_version=featureset,
    )


# ---------------------------------------------------------------------------
# Requests built from a run record
# ---------------------------------------------------------------------------


def test_a_benchmark_record_supplies_the_dataset_featureset_limit_and_seed() -> None:
    record = _record("sparkov", Subsample(strategy="cards", seed=11, max_entities=200))

    request = loading.DataRequest.from_run_record(
        record, root=Path("raw"), cache_root=Path("cache"), full_corpus=True
    )

    assert request == loading.DataRequest(
        dataset="sparkov",
        featureset="v1",
        root=Path("raw"),
        max_entities=200,
        seed=11,
        cache_root=Path("cache"),
        full_corpus=True,
    )
    assert not request.is_synthetic


def test_an_unsampled_benchmark_record_keeps_the_seed_it_recorded() -> None:
    """The seed is part of the cache fingerprint even when nothing was sampled."""
    record = _record("sparkov", Subsample(strategy="none", seed=7, selected_entities=983))

    request = loading.DataRequest.from_run_record(record)

    assert (request.max_entities, request.seed) == (None, 7)


def test_a_synthetic_record_takes_the_label_csv_and_row_limit_from_the_caller() -> None:
    record = _record("synthetic", None)

    request = loading.DataRequest.from_run_record(record, csv_path=Path("labels.csv"), limit=500)

    assert request == loading.DataRequest(
        dataset="synthetic", featureset="v1", csv_path=Path("labels.csv"), limit=500
    )
    assert request.is_synthetic


def test_a_row_limit_is_refused_for_a_benchmark_record() -> None:
    record = _record("sparkov", Subsample(strategy="cards", seed=42, max_entities=200))

    with pytest.raises(ValueError, match="synthetic dataset only"):
        loading.DataRequest.from_run_record(record, limit=10)


# ---------------------------------------------------------------------------
# The synthetic dataset
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_loads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str | None, int | None]]:
    loads: list[tuple[str | None, int | None]] = []

    @contextmanager
    def stub_session() -> Iterator[object]:
        yield object()

    def stub_load(db: object, csv_path: str | None = None, *, limit: int | None = None) -> LabelledDataset:
        loads.append((csv_path, limit))
        return _matrix()

    monkeypatch.setattr(loading, "SessionLocal", stub_session)
    monkeypatch.setattr(loading, "load_dataset_with_csv_labels", stub_load)
    return loads


def test_the_synthetic_dataset_is_read_from_the_database_and_label_csv(
    synthetic_loads: list[tuple[str | None, int | None]],
) -> None:
    request = loading.DataRequest(csv_path=Path("missing.csv"), limit=4)

    data = loading.load_run_data(request, with_provenance=False)

    assert synthetic_loads == [("missing.csv", 4)]
    # Without a provenance record the CSV is never hashed, so it need not exist.
    assert data.provenance is None
    assert (data.cache_fingerprint, data.cache_hit) == (None, None)
    np.testing.assert_array_equal(data.ds.X, _matrix().X)


def test_a_synthetic_load_with_provenance_records_the_label_csv(
    synthetic_loads: list[tuple[str | None, int | None]], tmp_path: Path
) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text("id,is_fraud\ntx0,False\n", encoding="utf-8")

    data = loading.load_run_data(loading.DataRequest(csv_path=labels, limit=4))

    assert data.provenance is not None
    assert data.provenance.name == "synthetic"
    assert list(data.provenance.files) == ["labels.csv"]
    assert data.provenance.row_count == _matrix().n_rows
    assert any("first 4" in step for step in data.provenance.preprocessing)


def test_synthetic_countries_come_from_the_label_csv_in_row_order(
    synthetic_loads: list[tuple[str | None, int | None]], tmp_path: Path
) -> None:
    labels = tmp_path / "labels.csv"
    countries = ["US", "GB", "BR", "US", "GB", "BR"]
    rows = [f"tx{index},False,{country}" for index, country in enumerate(countries)]
    # Written out of row order: countries are matched by id, not by position.
    labels.write_text(
        "\n".join(["id,is_fraud,country", *reversed(rows)]) + "\n", encoding="utf-8"
    )

    data = loading.load_run_data(
        loading.DataRequest(csv_path=labels), with_provenance=False, with_countries=True
    )

    assert data.countries == tuple(countries)
    assert loading.synthetic_countries(labels)["tx4"] == "GB"


def test_a_row_without_a_recorded_country_is_refused(
    synthetic_loads: list[tuple[str | None, int | None]], tmp_path: Path
) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text("id,is_fraud,country\ntx0,False,US\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No country recorded for 5 transaction"):
        loading.load_run_data(
            loading.DataRequest(csv_path=labels), with_provenance=False, with_countries=True
        )


def test_countries_are_not_read_unless_asked_for(
    synthetic_loads: list[tuple[str | None, int | None]],
) -> None:
    data = loading.load_run_data(
        loading.DataRequest(csv_path=Path("missing.csv")), with_provenance=False
    )

    assert data.countries is None


def test_the_synthetic_dataset_has_no_canonical_dataset_to_return(
    synthetic_loads: list[tuple[str | None, int | None]],
) -> None:
    with pytest.raises(ValueError, match="no canonical dataset"):
        loading.load_run_data(loading.DataRequest(), with_provenance=False, with_canonical=True)
    assert synthetic_loads == []


def test_the_synthetic_dataset_refuses_another_featureset(
    synthetic_loads: list[tuple[str | None, int | None]],
) -> None:
    with pytest.raises(ValueError, match="featureset 'v1' only"):
        loading.load_run_data(loading.DataRequest(featureset="v2"), with_provenance=False)
    assert synthetic_loads == []


# ---------------------------------------------------------------------------
# Registered datasets
# ---------------------------------------------------------------------------


class _Adapter:
    name = "fixture"

    def __init__(self) -> None:
        self.allow_full_corpus = False
        self.loads: list[tuple[Path, int | None, int]] = []

    def load(self, root: Path, *, max_entities: int | None = None, seed: int = 42) -> Any:
        self.loads.append((root, max_entities, seed))
        # Out of row order: countries are matched to matrix rows by id.
        transactions = [
            SimpleNamespace(id=f"tx{index}", country="US" if index % 2 else "CA")
            for index in reversed(range(6))
        ]
        return SimpleNamespace(provenance="the adapter's provenance", transactions=transactions)


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> tuple[_Adapter, list[dict[str, Any]]]:
    fixture = _Adapter()
    builds: list[dict[str, Any]] = []

    def stub_build(dataset: Any, **options: Any) -> tuple[LabelledDataset, Any, bool]:
        builds.append(options)
        return _matrix(), SimpleNamespace(fingerprint="abc123"), True

    monkeypatch.setattr(loading, "get_adapter", lambda name: fixture)
    monkeypatch.setattr(loading, "build_or_load", stub_build)
    return fixture, builds


def test_a_registered_dataset_is_read_through_its_adapter_and_the_feature_cache(
    adapter: tuple[_Adapter, list[dict[str, Any]]],
) -> None:
    fixture, builds = adapter
    request = loading.DataRequest(
        dataset="fixture", max_entities=3, seed=11, cache_root=Path("cache"), refresh=True
    )

    data = loading.load_run_data(request)

    assert fixture.loads == [(RAW_DATA_DIR / "fixture", 3, 11)]
    assert builds == [{"featureset": "v1", "cache_root": Path("cache"), "refresh": True}]
    assert data.provenance == "the adapter's provenance"
    assert (data.cache_fingerprint, data.cache_hit) == ("abc123", True)
    assert fixture.allow_full_corpus is False


def test_registered_countries_come_from_the_canonical_transactions(
    adapter: tuple[_Adapter, list[dict[str, Any]]],
) -> None:
    data = loading.load_run_data(loading.DataRequest(dataset="fixture"), with_countries=True)

    assert data.countries == ("CA", "US", "CA", "US", "CA", "US")


def test_the_canonical_dataset_is_returned_only_when_asked_for(
    adapter: tuple[_Adapter, list[dict[str, Any]]],
) -> None:
    fixture, _ = adapter

    without = loading.load_run_data(loading.DataRequest(dataset="fixture"))
    with_it = loading.load_run_data(loading.DataRequest(dataset="fixture"), with_canonical=True)

    assert without.canonical is None
    assert with_it.canonical is not None
    assert [tx.id for tx in with_it.canonical.transactions] == [f"tx{i}" for i in range(5, -1, -1)]
    assert len(fixture.loads) == 2, "each load reads the adapter exactly once"


def test_the_full_corpus_opt_in_is_set_only_when_asked(
    adapter: tuple[_Adapter, list[dict[str, Any]]],
) -> None:
    fixture, _ = adapter

    loading.load_run_data(
        loading.DataRequest(dataset="fixture", root=Path("elsewhere"), full_corpus=True)
    )

    assert fixture.allow_full_corpus is True
    assert fixture.loads == [(Path("elsewhere"), None, loading.DEFAULT_SUBSAMPLE_SEED)]
