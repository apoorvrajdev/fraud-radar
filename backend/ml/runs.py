"""The reproducibility record for a single benchmark run.

`run.json` answers "what would I have to hold fixed to get this number again?"
— which dataset bytes, which slice of them and the seed that drew it, which
feature contract, which code, which libraries, which rows each fold covers. It
is written per run directory, next to that run's metrics.

This is deliberately separate from `training_metadata.json` (see
`ml/artifacts.py`), which records the *model fit*: fold sizes, fraud rates,
chosen hyperparameters, and the fit's settings, its random state among them.
One describes the inputs, the other the fit; a run directory carries both,
and neither has to grow the other's fields.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ml.datasets.base import DatasetContractError, DatasetProvenance
from ml.paths import RUNS_ROOT, ensure_dir, run_dir, validate_run_name
from ml.splits import SplitIndices

RUN_METADATA_FILENAME = "run.json"

# Bumped when the shape of run.json changes, so an old run stays readable
# instead of being silently misparsed by newer code.
#   "1"  seed was the subsample seed, or the model random state when the
#        dataset had no subsample record.
#   "2"  seed is only ever the subsample seed, and null without a subsample.
#   "3"  a record that records a test fold also identifies its transactions.
RUN_METADATA_VERSION = "3"

# Versions written before a record identified its test fold's transactions.
# Such a record is still read, but its test fold cannot be identified from it.
_VERSIONS_WITHOUT_FOLD_IDENTITY = frozenset({"1", "2"})

# Names the chronological folds are recorded under, in time order.
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"

# How a fold's transaction ids are digested. The record states it beside the
# digest, so the digest can be recomputed without reading this code.
TRANSACTION_IDS_DIGEST_DEFINITION = (
    "SHA-256 of the fold's transaction ids, in fold order, encoded as a JSON array of "
    "strings with no whitespace, in UTF-8"
)

_SHA256_HEX_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")

_GIT_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class SplitPeriod:
    """The time span a fold covers.

    Recorded because chronological splitting is the only defensible split for
    a fraud model, and "chronological" is a claim a reader should be able to
    check rather than take on trust.
    """

    name: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if not self.name:
            raise DatasetContractError("SplitPeriod.name must be non-empty.")
        for attribute in ("start", "end"):
            value: datetime = getattr(self, attribute)
            if value.tzinfo is None:
                raise DatasetContractError(f"SplitPeriod.{attribute} must be timezone-aware.")
        if self.start > self.end:
            raise DatasetContractError(
                f"SplitPeriod {self.name!r} starts ({self.start.isoformat()}) "
                f"after it ends ({self.end.isoformat()})."
            )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "start": self.start.isoformat(), "end": self.end.isoformat()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SplitPeriod:
        return cls(
            name=str(payload["name"]),
            start=datetime.fromisoformat(str(payload["start"])),
            end=datetime.fromisoformat(str(payload["end"])),
        )


def split_periods(timestamps: np.ndarray, splits: SplitIndices) -> tuple[SplitPeriod, ...]:
    """The period each chronological fold actually covers, train → val → test.

    Read from the timestamps the folds hold, not from the split fractions, so
    the record states what was trained and scored on. A fold that is empty,
    naive, or out of time order has no honest period to record, and is refused.
    Adjacent folds may share a boundary timestamp: transactions at the same
    instant can fall on both sides of a split.
    """
    values = np.asarray(timestamps, dtype=object)
    periods: list[SplitPeriod] = []
    for name, indices in (
        (TRAIN_SPLIT, splits.train),
        (VAL_SPLIT, splits.val),
        (TEST_SPLIT, splits.test),
    ):
        if len(indices) == 0:
            raise DatasetContractError(f"Split {name!r} is empty, so it covers no period.")
        fold = values[indices]
        periods.append(SplitPeriod(name=name, start=min(fold), end=max(fold)))

    for earlier, later in pairwise(periods):
        if earlier.end > later.start:
            raise DatasetContractError(
                f"Split {later.name!r} starts ({later.start.isoformat()}) before "
                f"{earlier.name!r} ends ({earlier.end.isoformat()}); the folds are "
                "not in time order."
            )
    return tuple(periods)


def transaction_ids_digest(transaction_ids: Sequence[str]) -> str:
    """The SHA-256 of `transaction_ids` in the given order.

    Computed as `TRANSACTION_IDS_DIGEST_DEFINITION` states. A JSON array keeps
    the encoding unambiguous: no id can be mistaken for a separator, so two
    different id sequences never encode to the same bytes.
    """
    encoded = json.dumps(list(transaction_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FoldIdentity:
    """Exactly which transactions a fold holds, and in what order.

    A fold period says when a fold starts and ends, and a size says how many
    rows it holds; neither says which rows. Two loads can agree on both and
    still hold different transactions, so the ids themselves are recorded, as
    a count and a digest of the ordered sequence.
    """

    fold: str
    transaction_count: int
    transaction_ids_sha256: str

    def __post_init__(self) -> None:
        if not self.fold:
            raise DatasetContractError("FoldIdentity.fold must be non-empty.")
        if self.transaction_count <= 0:
            raise DatasetContractError(
                f"FoldIdentity for fold {self.fold!r} must count at least one transaction, "
                f"got {self.transaction_count}."
            )
        digest = self.transaction_ids_sha256
        if len(digest) != _SHA256_HEX_LENGTH or not set(digest) <= _HEX_DIGITS:
            raise DatasetContractError(
                f"FoldIdentity for fold {self.fold!r} must carry a lowercase 64-character hex "
                f"SHA-256 digest, got {digest!r}."
            )

    @classmethod
    def of(cls, fold: str, transaction_ids: Sequence[str]) -> FoldIdentity:
        """The identity of a fold holding `transaction_ids`, in that order."""
        return cls(
            fold=fold,
            transaction_count=len(transaction_ids),
            transaction_ids_sha256=transaction_ids_digest(transaction_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "transaction_count": self.transaction_count,
            "transaction_ids_sha256": self.transaction_ids_sha256,
            "digest_definition": TRANSACTION_IDS_DIGEST_DEFINITION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FoldIdentity:
        """Inverse of `to_dict`, refusing a digest computed some other way."""
        definition = payload.get("digest_definition")
        if definition != TRANSACTION_IDS_DIGEST_DEFINITION:
            raise DatasetContractError(
                f"Fold identity digest was computed as {definition!r}, not as "
                f"{TRANSACTION_IDS_DIGEST_DEFINITION!r}, so it cannot be checked."
            )
        return cls(
            fold=str(payload["fold"]),
            transaction_count=int(payload["transaction_count"]),
            transaction_ids_sha256=str(payload["transaction_ids_sha256"]),
        )


def identify_test_fold(transaction_ids: Sequence[str], splits: SplitIndices) -> FoldIdentity:
    """The identity of the test fold `splits` selects from `transaction_ids`, in fold order."""
    return FoldIdentity.of(TEST_SPLIT, [transaction_ids[index] for index in splits.test])


@dataclass(frozen=True)
class RunMetadata:
    """Everything needed to reproduce a benchmark run's inputs.

    `test_fold_identity` names the test fold's transactions. From version 3, a
    record that records a test fold period must carry it; records of earlier
    versions have none, and their test fold cannot be identified from them.
    """

    run_name: str
    dataset: DatasetProvenance
    featureset_version: str
    created_at_utc: str = field(default_factory=lambda: _utc_now_iso())
    code_version: str | None = None
    library_versions: Mapping[str, str] = field(default_factory=dict)
    splits: tuple[SplitPeriod, ...] = ()
    test_fold_identity: FoldIdentity | None = None
    notes: str = ""
    metadata_version: str = RUN_METADATA_VERSION

    def __post_init__(self) -> None:
        validate_run_name(self.run_name)
        if not self.featureset_version.strip():
            raise DatasetContractError("RunMetadata.featureset_version must be non-empty.")
        names = [split.name for split in self.splits]
        if len(set(names)) != len(names):
            raise DatasetContractError(f"Duplicate split names in run {self.run_name!r}: {names}.")
        identity = self.test_fold_identity
        if identity is not None and identity.fold != TEST_SPLIT:
            raise DatasetContractError(
                f"Run {self.run_name!r}: test_fold_identity describes fold {identity.fold!r}, "
                f"not {TEST_SPLIT!r}."
            )
        if (
            identity is None
            and TEST_SPLIT in names
            and self.metadata_version not in _VERSIONS_WITHOUT_FOLD_IDENTITY
        ):
            raise DatasetContractError(
                f"Run {self.run_name!r}: a version {self.metadata_version} record that records a "
                "test fold must identify the fold's transactions."
            )
        object.__setattr__(self, "library_versions", MappingProxyType(dict(self.library_versions)))
        object.__setattr__(self, "splits", tuple(self.splits))

    @property
    def seed(self) -> int | None:
        """The seed the dataset's subsample was drawn with; None without a subsample record.

        Derived rather than stored, so it cannot mean anything else. The model's
        random state is a fact of the fit, recorded in `training_metadata.json`.
        """
        subsample = self.dataset.subsample
        return None if subsample is None else subsample.seed

    def split(self, name: str) -> SplitPeriod | None:
        """Return the named fold's period, or None if it was not recorded."""
        return next((split for split in self.splits if split.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_version": self.metadata_version,
            "run_name": self.run_name,
            "created_at_utc": self.created_at_utc,
            "dataset": self.dataset.to_dict(),
            "featureset_version": self.featureset_version,
            "seed": self.seed,
            "code_version": self.code_version,
            "library_versions": dict(self.library_versions),
            "splits": [split.to_dict() for split in self.splits],
            "test_fold_identity": (
                None if self.test_fold_identity is None else self.test_fold_identity.to_dict()
            ),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunMetadata:
        """Inverse of `to_dict`. `seed` is derived, so it is ignored.

        That also reads version 1 records correctly: where one stored the model
        random state as its seed, the dataset had no subsample record, and the
        seed reads back as None. Records before version 3 carry no test fold
        identity, and read back without one.
        """
        identity = payload.get("test_fold_identity")
        return cls(
            run_name=str(payload["run_name"]),
            dataset=DatasetProvenance.from_dict(payload["dataset"]),
            featureset_version=str(payload["featureset_version"]),
            created_at_utc=str(payload["created_at_utc"]),
            code_version=_opt_str(payload.get("code_version")),
            library_versions=dict(payload.get("library_versions") or {}),
            splits=tuple(SplitPeriod.from_dict(s) for s in payload.get("splits") or ()),
            test_fold_identity=None if identity is None else FoldIdentity.from_dict(identity),
            notes=str(payload.get("notes", "")),
            metadata_version=str(payload.get("metadata_version", RUN_METADATA_VERSION)),
        )


def save_run_metadata(metadata: RunMetadata, *, runs_root: Path = RUNS_ROOT) -> Path:
    """Write `run.json` into the run's directory, creating it if needed."""
    target = ensure_dir(run_dir(metadata.run_name, runs_root=runs_root)) / RUN_METADATA_FILENAME
    with target.open("w", encoding="utf-8") as handle:
        json.dump(metadata.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target


def load_run_metadata(run_name: str, *, runs_root: Path = RUNS_ROOT) -> RunMetadata:
    """Read back a run's `run.json`."""
    source = run_dir(run_name, runs_root=runs_root) / RUN_METADATA_FILENAME
    if not source.exists():
        raise FileNotFoundError(f"No {RUN_METADATA_FILENAME} for run {run_name!r} at {source}.")
    with source.open(encoding="utf-8") as handle:
        return RunMetadata.from_dict(json.load(handle))


def current_git_commit() -> str | None:
    """Best-effort commit SHA of the working tree, or None outside a repo.

    Best-effort on purpose: a missing SHA should degrade the record, never
    fail a training run that is otherwise fine.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)
