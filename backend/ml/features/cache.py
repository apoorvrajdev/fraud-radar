"""Versioned `.npz` cache for extracted feature matrices.

Extraction runs the production extractor once per row in pure Python, which is
the price of having no second implementation. That cost is worth paying once,
not on every experiment, so the matrix is cached — and a cache is only safe if
it can prove what it holds.

The fingerprint in the filename is derived from the dataset's identity (name,
version, source file digests, subsample record) and the featureset version.
Change any of those and you get a different file rather than a stale hit, so
there is no way to silently evaluate new data with an old matrix.

Everything needed to reconstruct the run travels inside the archive: the
matrix, the labels, the transaction ids, the timestamps, the feature names in
their frozen order, and the provenance of the dataset they came from. Labels
stay in their own array and are never written onto a transaction.

Caches are derived data and are gitignored (`ml/data/cache/`). Deleting them
costs time and nothing else.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.fraud.feature_spec import DEFAULT_FEATURESET, feature_names
from ml.data import LabelledDataset
from ml.datasets.base import CanonicalDataset, DatasetContractError, DatasetProvenance
from ml.features.batch import build_feature_matrix
from ml.paths import FEATURE_CACHE_DIR, ensure_dir

log = logging.getLogger("ml.features.cache")

# Bumped when the archive layout changes. An older cache is then rejected on
# load rather than misread by newer code.
FEATURE_CACHE_VERSION = "1"

_FINGERPRINT_LENGTH = 16
_METADATA_KEY = "metadata_json"


class FeatureCacheError(DatasetContractError):
    """A cache file is missing, malformed, or does not match what was asked for."""


@dataclass(frozen=True)
class FeatureCacheMetadata:
    """What a cached matrix is, and what it was built from."""

    cache_version: str
    featureset_version: str
    feature_names: tuple[str, ...]
    dataset_name: str
    dataset_version: str
    dataset_origin: str
    dataset_files: Mapping[str, str]
    subsample: Mapping[str, Any] | None
    row_count: int
    fraud_count: int
    fingerprint: str
    built_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_version": self.cache_version,
            "featureset_version": self.featureset_version,
            "feature_names": list(self.feature_names),
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_origin": self.dataset_origin,
            "dataset_files": dict(self.dataset_files),
            "subsample": dict(self.subsample) if self.subsample else None,
            "row_count": self.row_count,
            "fraud_count": self.fraud_count,
            "fingerprint": self.fingerprint,
            "built_at_utc": self.built_at_utc,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureCacheMetadata:
        try:
            return cls(
                cache_version=str(payload["cache_version"]),
                featureset_version=str(payload["featureset_version"]),
                feature_names=tuple(str(name) for name in payload["feature_names"]),
                dataset_name=str(payload["dataset_name"]),
                dataset_version=str(payload["dataset_version"]),
                dataset_origin=str(payload["dataset_origin"]),
                dataset_files=dict(payload.get("dataset_files") or {}),
                subsample=payload.get("subsample"),
                row_count=int(payload["row_count"]),
                fraud_count=int(payload["fraud_count"]),
                fingerprint=str(payload["fingerprint"]),
                built_at_utc=str(payload["built_at_utc"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FeatureCacheError(f"Cache metadata is malformed: {exc}") from exc


def dataset_fingerprint(
    provenance: DatasetProvenance, *, featureset: str = DEFAULT_FEATURESET
) -> str:
    """Short, stable digest of everything that changes a feature matrix.

    Deliberately excludes row counts and timestamps: those are consequences of
    the inputs, not inputs themselves, and including them would let an
    identical build miss its own cache.
    """
    payload = {
        "dataset": provenance.name,
        "version": provenance.version,
        "schema_version": provenance.schema_version,
        "files": dict(sorted(provenance.files.items())),
        "subsample": provenance.subsample.to_dict() if provenance.subsample else None,
        "featureset": featureset,
        "cache_version": FEATURE_CACHE_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:_FINGERPRINT_LENGTH]


def cache_path(
    provenance: DatasetProvenance,
    *,
    featureset: str = DEFAULT_FEATURESET,
    cache_root: Path = FEATURE_CACHE_DIR,
) -> Path:
    """Where this dataset/featureset combination caches its matrix."""
    fingerprint = dataset_fingerprint(provenance, featureset=featureset)
    return cache_root / f"{provenance.name}_{featureset}_{fingerprint}.npz"


def save_feature_cache(
    path: Path,
    matrix: LabelledDataset,
    *,
    dataset: CanonicalDataset,
    featureset: str = DEFAULT_FEATURESET,
) -> FeatureCacheMetadata:
    """Write the matrix and its provenance to `path`."""
    provenance = dataset.provenance
    metadata = FeatureCacheMetadata(
        cache_version=FEATURE_CACHE_VERSION,
        featureset_version=featureset,
        feature_names=tuple(matrix.feature_names),
        dataset_name=provenance.name,
        dataset_version=provenance.version,
        dataset_origin=provenance.origin.value,
        dataset_files=dict(provenance.files),
        subsample=provenance.subsample.to_dict() if provenance.subsample else None,
        row_count=matrix.n_rows,
        fraud_count=int(matrix.y.sum()),
        fingerprint=dataset_fingerprint(provenance, featureset=featureset),
        built_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )

    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        X=matrix.X,
        y=matrix.y,
        transaction_ids=np.asarray(matrix.transaction_ids, dtype=np.str_),
        timestamps=_timestamps_to_epoch_us(matrix.timestamps),
        feature_names=np.asarray(matrix.feature_names, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(metadata.to_dict(), sort_keys=True)),
    )
    log.info("Cached %d x %d features at %s", matrix.n_rows, matrix.X.shape[1], path)
    return metadata


def load_feature_cache(
    path: Path, *, featureset: str = DEFAULT_FEATURESET
) -> tuple[LabelledDataset, FeatureCacheMetadata]:
    """Read a cached matrix back, refusing anything that does not match."""
    if not path.exists():
        raise FeatureCacheError(f"No feature cache at {path}.")

    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = {
                "X",
                "y",
                "transaction_ids",
                "timestamps",
                "feature_names",
                _METADATA_KEY,
            } - set(archive.files)
            if missing:
                raise FeatureCacheError(
                    f"Cache at {path} is missing arrays: {', '.join(sorted(missing))}."
                )
            payload = json.loads(str(archive[_METADATA_KEY]))
            X = np.asarray(archive["X"], dtype=np.float64)
            y = np.asarray(archive["y"], dtype=np.int64)
            ids = [str(value) for value in archive["transaction_ids"]]
            epoch_us = np.asarray(archive["timestamps"], dtype=np.int64)
            stored_names = [str(value) for value in archive["feature_names"]]
    except FeatureCacheError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FeatureCacheError(f"Cache at {path} could not be read: {exc}") from exc

    metadata = FeatureCacheMetadata.from_dict(payload)
    _validate(metadata, featureset=featureset, stored_names=stored_names, X=X, y=y, ids=ids)

    return (
        LabelledDataset(
            X=X,
            y=y,
            timestamps=_epoch_us_to_timestamps(epoch_us),
            transaction_ids=ids,
            feature_names=stored_names,
        ),
        metadata,
    )


def build_or_load(
    dataset: CanonicalDataset,
    *,
    featureset: str = DEFAULT_FEATURESET,
    cache_root: Path = FEATURE_CACHE_DIR,
    refresh: bool = False,
) -> tuple[LabelledDataset, FeatureCacheMetadata, bool]:
    """Return the feature matrix, its metadata, and whether the cache was hit.

    A miss extracts and writes; a hit reads and validates. `refresh` forces
    a rebuild without requiring anyone to delete files by hand.
    """
    path = cache_path(dataset.provenance, featureset=featureset, cache_root=cache_root)

    if not refresh and path.exists():
        matrix, metadata = load_feature_cache(path, featureset=featureset)
        log.info("Feature cache hit: %s (%d rows)", path.name, metadata.row_count)
        return matrix, metadata, True

    matrix = build_feature_matrix(dataset, featureset=featureset)
    metadata = save_feature_cache(path, matrix, dataset=dataset, featureset=featureset)
    return matrix, metadata, False


def _validate(
    metadata: FeatureCacheMetadata,
    *,
    featureset: str,
    stored_names: Sequence[str],
    X: np.ndarray,
    y: np.ndarray,
    ids: Sequence[str],
) -> None:
    if metadata.cache_version != FEATURE_CACHE_VERSION:
        raise FeatureCacheError(
            f"Cache version {metadata.cache_version!r} was written by different code; "
            f"this build reads {FEATURE_CACHE_VERSION!r}. Rebuild the cache."
        )
    if metadata.featureset_version != featureset:
        raise FeatureCacheError(
            f"Cache holds featureset {metadata.featureset_version!r} but "
            f"{featureset!r} was requested."
        )

    expected_names = feature_names(featureset)
    if list(stored_names) != expected_names:
        raise FeatureCacheError(
            "Cached feature order does not match the registry — the featureset "
            "changed without its version changing, so the matrix columns no "
            "longer mean what the registry says they mean."
        )
    if list(metadata.feature_names) != expected_names:
        raise FeatureCacheError("Cache metadata feature order does not match the registry.")

    if X.ndim != 2 or X.shape[1] != len(expected_names):
        raise FeatureCacheError(
            f"Cached matrix has shape {X.shape}; expected (n, {len(expected_names)})."
        )
    if not (X.shape[0] == y.shape[0] == len(ids) == metadata.row_count):
        raise FeatureCacheError(
            f"Cache is inconsistent: {X.shape[0]} rows, {y.shape[0]} labels, "
            f"{len(ids)} ids, metadata says {metadata.row_count}."
        )
    if len(set(ids)) != len(ids):
        raise FeatureCacheError("Cached transaction ids are not unique; rows cannot be traced.")


def _timestamps_to_epoch_us(timestamps: np.ndarray) -> np.ndarray:
    """Datetimes to integer microseconds — exact, and no pickle needed."""
    return np.asarray(
        [int(ts.timestamp() * 1_000_000) for ts in timestamps], dtype=np.int64
    )


def _epoch_us_to_timestamps(epoch_us: np.ndarray) -> np.ndarray:
    return np.asarray(
        [datetime.fromtimestamp(int(value) / 1_000_000, tz=UTC) for value in epoch_us],
        dtype=object,
    )
