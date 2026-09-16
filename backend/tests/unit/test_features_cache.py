"""Phase 5C — the feature cache must prove what it holds.

A cache that returns the wrong matrix is worse than no cache: the run still
produces numbers, and nothing announces that they came from different data.
So every mismatch below is a refusal, not a warning.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from app.fraud.feature_spec import FEATURESETS
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction
from ml.datasets.base import CanonicalDataset, DataOrigin, DatasetProvenance, Subsample
from ml.features.batch import build_feature_matrix
from ml.features.cache import (
    FEATURE_CACHE_VERSION,
    FeatureCacheError,
    build_or_load,
    cache_path,
    dataset_fingerprint,
    load_feature_cache,
    save_feature_cache,
)

ANCHOR = datetime(2025, 6, 1, 9, 0, tzinfo=UTC)
DIGEST = "c" * 64


def _dataset(
    *,
    name: str = "fixture",
    version: str = "v1",
    files: dict[str, str] | None = None,
    subsample: Subsample | None = None,
    rows: int = 6,
) -> CanonicalDataset:
    customers = {
        "c1": Customer(
            id="c1",
            email="c1@example.invalid",
            full_name="Customer c1",
            country="US",
            risk_tier="LOW",
            account_age_days=120,
        )
    }
    merchants = {
        "m1": Merchant(
            id="m1",
            name="Merchant m1",
            category="GROCERY",
            mcc="5411",
            country="US",
            risk_rating="LOW",
        )
    }
    transactions = [
        Transaction(
            id=f"tx-{index}",
            idempotency_key=f"tx-{index}",
            customer_id="c1",
            merchant_id="m1",
            amount=Decimal(f"{10 + index * 7}.50"),
            currency="USD",
            status="APPROVED",
            payment_method="CARD",
            country="US",
            is_card_present=index % 2 == 0,
            created_at=ANCHOR + timedelta(hours=index * 5),
        )
        for index in range(rows)
    ]
    provenance = DatasetProvenance(
        name=name,
        version=version,
        origin=DataOrigin.SYNTHETIC,
        source_url="https://example.test",
        citation="fixture",
        license="CC0",
        label_field="is_fraud",
        label_definition="1 = fraud",
        files=files if files is not None else {"data.csv": DIGEST},
        subsample=subsample,
    )
    return CanonicalDataset(
        customers=customers,
        merchants=merchants,
        transactions=transactions,
        labels={tx.id: int(tx.id.endswith("3")) for tx in transactions},
        provenance=provenance,
    ).validate()


# ---------------------------------------------------------------------------
# Fingerprint and path
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_for_identical_inputs() -> None:
    assert dataset_fingerprint(_dataset().provenance) == dataset_fingerprint(
        _dataset().provenance
    )


def test_fingerprint_changes_with_the_source_bytes() -> None:
    other = _dataset(files={"data.csv": "d" * 64})
    assert dataset_fingerprint(_dataset().provenance) != dataset_fingerprint(other.provenance)


def test_fingerprint_changes_with_the_subsample() -> None:
    sampled = _dataset(subsample=Subsample(strategy="cards", seed=1, max_entities=10))
    assert dataset_fingerprint(_dataset().provenance) != dataset_fingerprint(sampled.provenance)


def test_fingerprint_changes_with_the_dataset_version() -> None:
    assert dataset_fingerprint(_dataset().provenance) != dataset_fingerprint(
        _dataset(version="v2").provenance
    )


def test_fingerprint_is_independent_of_row_count() -> None:
    """Row count is a consequence of the inputs; including it would miss hits."""
    assert dataset_fingerprint(_dataset(rows=6).provenance) == dataset_fingerprint(
        _dataset(rows=12).provenance
    )


def test_cache_path_names_the_dataset_featureset_and_fingerprint(tmp_path: Path) -> None:
    path = cache_path(_dataset().provenance, cache_root=tmp_path)
    assert path.parent == tmp_path
    assert path.name.startswith("fixture_v1_")
    assert path.suffix == ".npz"


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_the_matrix_exactly(tmp_path: Path) -> None:
    dataset = _dataset()
    original = build_feature_matrix(dataset)
    path = tmp_path / "cache.npz"
    save_feature_cache(path, original, dataset=dataset)

    restored, metadata = load_feature_cache(path)

    np.testing.assert_array_equal(restored.X, original.X)
    np.testing.assert_array_equal(restored.y, original.y)
    assert restored.transaction_ids == original.transaction_ids
    assert restored.feature_names == original.feature_names
    assert metadata.row_count == original.n_rows


def test_round_trip_preserves_timestamps_to_the_microsecond(tmp_path: Path) -> None:
    dataset = _dataset()
    original = build_feature_matrix(dataset)
    path = tmp_path / "cache.npz"
    save_feature_cache(path, original, dataset=dataset)

    restored, _ = load_feature_cache(path)
    assert list(restored.timestamps) == list(original.timestamps)


def test_cache_carries_the_dataset_provenance(tmp_path: Path) -> None:
    dataset = _dataset(subsample=Subsample(strategy="cards", seed=7, max_entities=5))
    path = tmp_path / "cache.npz"
    metadata = save_feature_cache(path, build_feature_matrix(dataset), dataset=dataset)

    assert metadata.dataset_name == "fixture"
    assert metadata.dataset_origin == "synthetic"
    assert metadata.dataset_files == {"data.csv": DIGEST}
    assert metadata.subsample is not None
    assert metadata.subsample["seed"] == 7
    assert metadata.feature_names == tuple(FEATURESETS["v1"])
    assert metadata.cache_version == FEATURE_CACHE_VERSION


def test_labels_live_in_their_own_array(tmp_path: Path) -> None:
    dataset = _dataset()
    path = tmp_path / "cache.npz"
    save_feature_cache(path, build_feature_matrix(dataset), dataset=dataset)

    with np.load(path, allow_pickle=False) as archive:
        assert "y" in archive.files
        assert archive["y"].shape[0] == len(dataset.transactions)
        assert archive["X"].shape[1] == len(FEATURESETS["v1"])


def test_archive_needs_no_pickle_to_read(tmp_path: Path) -> None:
    """allow_pickle=False keeps a cache file from being an execution vector."""
    dataset = _dataset()
    path = tmp_path / "cache.npz"
    save_feature_cache(path, build_feature_matrix(dataset), dataset=dataset)

    with np.load(path, allow_pickle=False) as archive:
        assert sorted(archive.files) == [
            "X",
            "feature_names",
            "metadata_json",
            "timestamps",
            "transaction_ids",
            "y",
        ]


def test_rows_remain_traceable_after_a_round_trip(tmp_path: Path) -> None:
    dataset = _dataset()
    path = tmp_path / "cache.npz"
    save_feature_cache(path, build_feature_matrix(dataset), dataset=dataset)

    restored, _ = load_feature_cache(path)
    by_id = {tx.id: tx for tx in dataset.transactions}
    for row_index, tx_id in enumerate(restored.transaction_ids):
        assert restored.timestamps[row_index] == by_id[tx_id].created_at


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_missing_cache_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FeatureCacheError, match="No feature cache at"):
        load_feature_cache(tmp_path / "absent.npz")


def test_wrong_cache_version_is_refused(tmp_path: Path) -> None:
    path = _tampered_cache(tmp_path, {"cache_version": "0"})
    with pytest.raises(FeatureCacheError, match="written by different code"):
        load_feature_cache(path)


def test_featureset_mismatch_is_refused(tmp_path: Path) -> None:
    dataset = _dataset()
    path = tmp_path / "cache.npz"
    save_feature_cache(path, build_feature_matrix(dataset), dataset=dataset)

    with pytest.raises(FeatureCacheError, match="but 'v99' was requested"):
        load_feature_cache(path, featureset="v99")


def test_reordered_feature_columns_are_refused(tmp_path: Path) -> None:
    """The exact failure a version bump alone would not catch."""
    dataset = _dataset()
    matrix = build_feature_matrix(dataset)
    rotated = [*matrix.feature_names[1:], matrix.feature_names[0]]
    path = tmp_path / "cache.npz"
    save_feature_cache(path, matrix, dataset=dataset)
    _rewrite(path, feature_names=np.asarray(rotated, dtype=np.str_))

    with pytest.raises(FeatureCacheError, match="Cached feature order does not match"):
        load_feature_cache(path)


def test_metadata_feature_order_is_checked_too(tmp_path: Path) -> None:
    names = list(FEATURESETS["v1"])
    path = _tampered_cache(tmp_path, {"feature_names": [*names[1:], names[0]]})
    with pytest.raises(FeatureCacheError, match="metadata feature order"):
        load_feature_cache(path)


def test_row_count_disagreement_is_refused(tmp_path: Path) -> None:
    path = _tampered_cache(tmp_path, {"row_count": 999})
    with pytest.raises(FeatureCacheError, match="Cache is inconsistent"):
        load_feature_cache(path)


def test_truncated_archive_is_refused(tmp_path: Path) -> None:
    dataset = _dataset()
    matrix = build_feature_matrix(dataset)
    path = tmp_path / "cache.npz"
    save_feature_cache(path, matrix, dataset=dataset)

    with np.load(path, allow_pickle=False) as archive:
        kept = {key: archive[key] for key in archive.files if key != "transaction_ids"}
    np.savez_compressed(path, **kept)

    with pytest.raises(FeatureCacheError, match="missing arrays: transaction_ids"):
        load_feature_cache(path)


def test_duplicate_ids_are_refused(tmp_path: Path) -> None:
    dataset = _dataset()
    matrix = build_feature_matrix(dataset)
    path = tmp_path / "cache.npz"
    save_feature_cache(path, matrix, dataset=dataset)
    duplicated = [matrix.transaction_ids[0]] * matrix.n_rows
    _rewrite(path, transaction_ids=np.asarray(duplicated, dtype=np.str_))

    with pytest.raises(FeatureCacheError, match="not unique"):
        load_feature_cache(path)


def test_malformed_metadata_is_refused(tmp_path: Path) -> None:
    dataset = _dataset()
    path = tmp_path / "cache.npz"
    save_feature_cache(path, build_feature_matrix(dataset), dataset=dataset)
    _rewrite(path, metadata_json=np.asarray("{not json"))

    with pytest.raises(FeatureCacheError, match="could not be read"):
        load_feature_cache(path)


def test_metadata_missing_a_field_is_refused(tmp_path: Path) -> None:
    path = _tampered_cache(tmp_path, {}, drop=("fingerprint",))
    with pytest.raises(FeatureCacheError, match="metadata is malformed"):
        load_feature_cache(path)


def test_corrupt_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    path.write_bytes(b"not an npz archive")
    with pytest.raises(FeatureCacheError, match="could not be read"):
        load_feature_cache(path)


# ---------------------------------------------------------------------------
# build_or_load
# ---------------------------------------------------------------------------


def test_first_call_builds_and_second_call_hits(tmp_path: Path) -> None:
    dataset = _dataset()

    built, _, hit_first = build_or_load(dataset, cache_root=tmp_path)
    loaded, _, hit_second = build_or_load(dataset, cache_root=tmp_path)

    assert hit_first is False
    assert hit_second is True
    np.testing.assert_array_equal(loaded.X, built.X)
    assert loaded.transaction_ids == built.transaction_ids


def test_refresh_rebuilds_even_when_a_cache_exists(tmp_path: Path) -> None:
    dataset = _dataset()
    build_or_load(dataset, cache_root=tmp_path)
    _, _, hit = build_or_load(dataset, cache_root=tmp_path, refresh=True)
    assert hit is False


def test_different_source_bytes_do_not_hit_the_same_cache(tmp_path: Path) -> None:
    """The cache must never answer for data it was not built from."""
    build_or_load(_dataset(), cache_root=tmp_path)
    _, metadata, hit = build_or_load(_dataset(files={"data.csv": "e" * 64}), cache_root=tmp_path)

    assert hit is False
    assert len(list(tmp_path.glob("*.npz"))) == 2
    assert metadata.dataset_files == {"data.csv": "e" * 64}


def _rewrite(path: Path, **replacements: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays.update(replacements)
    np.savez_compressed(path, **arrays)


def _tampered_cache(
    tmp_path: Path, changes: dict[str, object], *, drop: tuple[str, ...] = ()
) -> Path:
    dataset = _dataset()
    path = tmp_path / "cache.npz"
    save_feature_cache(path, build_feature_matrix(dataset), dataset=dataset)

    with np.load(path, allow_pickle=False) as archive:
        payload = json.loads(str(archive["metadata_json"]))
    payload.update(changes)
    for key in drop:
        payload.pop(key, None)
    _rewrite(path, metadata_json=np.asarray(json.dumps(payload, sort_keys=True)))
    return path
