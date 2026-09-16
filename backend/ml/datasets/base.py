"""Canonical dataset contracts for benchmark datasets.

Phase 5A establishes one representation that every dataset — the in-house
synthetic generator, an external synthetic benchmark, a real-world benchmark —
is adapted *into*. The invariant, stated once here and enforced by
`CanonicalDataset.validate()`:

    The canonical schema is the ORM model. External datasets are adapted
    into it; nothing is adapted out of it.

Concretely, a `CanonicalDataset` holds transient `Customer` / `Merchant` /
`Transaction` instances — the exact types the production `FeatureExtractor`
and rules engine already consume. That choice deletes a whole layer: there is
no "offline schema" to map to a "serving schema", so there is nothing to drift.
The instances are never added to a Session; `validate()` checks that.

Labels live outside the transaction objects, in a separate `id -> 0|1` map.
An adapter physically cannot leak the target into a feature this way, because
no attribute on the object carries it.

This module is pure Python: no Session, no FastAPI, no network. It describes
data; it does not fetch it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from sqlalchemy.orm import object_session

from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction

# Bumped only when the canonical representation itself changes shape — i.e.
# when the ORM models gain or lose a field an adapter is expected to fill.
CANONICAL_SCHEMA_VERSION = "1"

_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")


class DatasetContractError(ValueError):
    """A dataset violates the canonical contract.

    Raised eagerly at load time rather than surfacing later as a confusing
    feature-extraction failure or, worse, as a silently wrong model.
    """


class DataOrigin(StrEnum):
    """Whether the rows describe real card activity or generated activity.

    Kept explicit because it changes how a metric may be read: a strong score
    on generated data says the generator is learnable, not that fraud is
    detectable.
    """

    SYNTHETIC = "synthetic"
    REAL = "real"


@dataclass(frozen=True)
class Subsample:
    """How a dataset was reduced, if it was reduced.

    Sampling is recorded rather than implied because *how* you subsample a
    transaction stream decides whether history features survive it. Sampling
    rows at random deletes a card's past and quietly corrupts every velocity
    feature; sampling whole entities keeps each kept card's history complete.
    """

    strategy: str
    seed: int
    max_entities: int | None = None
    selected_entities: int | None = None

    def __post_init__(self) -> None:
        if not self.strategy:
            raise DatasetContractError("Subsample.strategy must be non-empty.")
        for name in ("max_entities", "selected_entities"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise DatasetContractError(f"Subsample.{name} must be positive, got {value}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "max_entities": self.max_entities,
            "selected_entities": self.selected_entities,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Subsample:
        return cls(
            strategy=str(payload["strategy"]),
            seed=int(payload["seed"]),
            max_entities=_opt_int(payload.get("max_entities")),
            selected_entities=_opt_int(payload.get("selected_entities")),
        )


@dataclass(frozen=True)
class DatasetProvenance:
    """Everything needed to trace a benchmark number back to its inputs.

    A metric without provenance is an anecdote. These fields are what lets a
    reader answer "which bytes, which licence, which slice, which label?"
    without trusting the person who ran it.

    `files` maps filename to SHA-256, so a model can always be tied to the
    exact input bytes — raw data is never committed, but its hash is.
    """

    name: str
    version: str
    origin: DataOrigin
    source_url: str
    citation: str
    license: str
    label_field: str
    label_definition: str
    schema_version: str = CANONICAL_SCHEMA_VERSION
    retrieved_at: datetime | None = None
    files: Mapping[str, str] = field(default_factory=dict)
    row_count: int | None = None
    fraud_count: int | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None
    subsample: Subsample | None = None
    preprocessing: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("name", "version", "license", "label_field", "label_definition"):
            if not str(getattr(self, name)).strip():
                raise DatasetContractError(f"DatasetProvenance.{name} must be non-empty.")

        for name in ("row_count", "fraud_count"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise DatasetContractError(f"DatasetProvenance.{name} must be >= 0, got {value}.")

        if (
            self.row_count is not None
            and self.fraud_count is not None
            and self.fraud_count > self.row_count
        ):
            raise DatasetContractError(
                f"fraud_count ({self.fraud_count}) exceeds row_count ({self.row_count})."
            )

        for name in ("retrieved_at", "period_start", "period_end"):
            _require_tz_aware(getattr(self, name), f"DatasetProvenance.{name}")

        if (
            self.period_start is not None
            and self.period_end is not None
            and self.period_start > self.period_end
        ):
            raise DatasetContractError(
                f"period_start ({self.period_start.isoformat()}) is after "
                f"period_end ({self.period_end.isoformat()})."
            )

        for filename, digest in self.files.items():
            _require_sha256(filename, digest)

        # Freeze the mapping: a frozen dataclass still hands out a mutable dict.
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))
        object.__setattr__(self, "preprocessing", tuple(self.preprocessing))

    @property
    def fraud_rate(self) -> float | None:
        """Positive-class prevalence, or None when the counts are unknown."""
        if self.row_count is None or self.fraud_count is None or self.row_count == 0:
            return None
        return self.fraud_count / self.row_count

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready payload. Datetimes become ISO-8601 strings."""
        return {
            "name": self.name,
            "version": self.version,
            "origin": self.origin.value,
            "source_url": self.source_url,
            "citation": self.citation,
            "license": self.license,
            "label_field": self.label_field,
            "label_definition": self.label_definition,
            "schema_version": self.schema_version,
            "retrieved_at": _iso_or_none(self.retrieved_at),
            "files": dict(self.files),
            "row_count": self.row_count,
            "fraud_count": self.fraud_count,
            "fraud_rate": self.fraud_rate,
            "period_start": _iso_or_none(self.period_start),
            "period_end": _iso_or_none(self.period_end),
            "subsample": self.subsample.to_dict() if self.subsample else None,
            "preprocessing": list(self.preprocessing),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasetProvenance:
        """Inverse of `to_dict`. `fraud_rate` is derived, so it is ignored."""
        subsample = payload.get("subsample")
        return cls(
            name=str(payload["name"]),
            version=str(payload["version"]),
            origin=DataOrigin(payload["origin"]),
            source_url=str(payload["source_url"]),
            citation=str(payload["citation"]),
            license=str(payload["license"]),
            label_field=str(payload["label_field"]),
            label_definition=str(payload["label_definition"]),
            schema_version=str(payload.get("schema_version", CANONICAL_SCHEMA_VERSION)),
            retrieved_at=_parse_or_none(payload.get("retrieved_at")),
            files=dict(payload.get("files") or {}),
            row_count=_opt_int(payload.get("row_count")),
            fraud_count=_opt_int(payload.get("fraud_count")),
            period_start=_parse_or_none(payload.get("period_start")),
            period_end=_parse_or_none(payload.get("period_end")),
            subsample=Subsample.from_dict(subsample) if subsample else None,
            preprocessing=tuple(payload.get("preprocessing") or ()),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class CanonicalDataset:
    """A dataset expressed in the production schema, plus its labels.

    `transactions` is sorted by `created_at` ascending — every downstream
    consumer (chronological splits, rolling history windows) depends on that,
    so it is a contract rather than a convention.
    """

    customers: Mapping[str, Customer]
    merchants: Mapping[str, Merchant]
    transactions: Sequence[Transaction]
    labels: Mapping[str, int]
    provenance: DatasetProvenance

    @property
    def n_rows(self) -> int:
        return len(self.transactions)

    @property
    def fraud_count(self) -> int:
        return sum(self.labels.values())

    @property
    def fraud_rate(self) -> float:
        return self.fraud_count / self.n_rows if self.n_rows else 0.0

    @property
    def period(self) -> tuple[datetime, datetime] | None:
        """(first, last) transaction timestamp, or None when empty."""
        if not self.transactions:
            return None
        return self.transactions[0].created_at, self.transactions[-1].created_at

    def validate(self) -> CanonicalDataset:
        """Enforce every canonical invariant. Returns self so it can be chained."""
        self._validate_identity()
        self._validate_ordering()
        self._validate_labels()
        self._validate_references()
        self._validate_transient()
        return self

    def _validate_identity(self) -> None:
        seen: set[str] = set()
        for tx in self.transactions:
            if tx.id in seen:
                raise DatasetContractError(f"Duplicate transaction id {tx.id!r}.")
            seen.add(tx.id)

    def _validate_ordering(self) -> None:
        previous: datetime | None = None
        for tx in self.transactions:
            _require_tz_aware(tx.created_at, f"transaction {tx.id} created_at")
            if previous is not None and tx.created_at < previous:
                raise DatasetContractError(
                    "Transactions must be sorted by created_at ascending; "
                    f"{tx.id!r} at {tx.created_at.isoformat()} follows {previous.isoformat()}."
                )
            previous = tx.created_at

    def _validate_labels(self) -> None:
        tx_ids = {tx.id for tx in self.transactions}
        missing = tx_ids - set(self.labels)
        if missing:
            raise DatasetContractError(
                f"{len(missing)} transaction(s) have no label, e.g. {sorted(missing)[0]!r}."
            )
        extra = set(self.labels) - tx_ids
        if extra:
            raise DatasetContractError(
                f"{len(extra)} label(s) reference unknown transactions, "
                f"e.g. {sorted(extra)[0]!r}."
            )
        for tx_id, label in self.labels.items():
            if label not in (0, 1):
                raise DatasetContractError(
                    f"Label for {tx_id!r} is {label!r}; labels must be 0 or 1."
                )

    def _validate_references(self) -> None:
        for tx in self.transactions:
            if tx.customer_id not in self.customers:
                raise DatasetContractError(
                    f"Transaction {tx.id!r} references unknown customer {tx.customer_id!r}."
                )
            if tx.merchant_id not in self.merchants:
                raise DatasetContractError(
                    f"Transaction {tx.id!r} references unknown merchant {tx.merchant_id!r}."
                )

    def _validate_transient(self) -> None:
        """Canonical objects must not be attached to a Session.

        A dataset that quietly joined a session would make the offline path
        issue queries — the exact train/serve coupling this layer removes.
        """
        for tx in self.transactions:
            if object_session(tx) is not None:
                raise DatasetContractError(
                    f"Transaction {tx.id!r} is attached to a Session; "
                    "canonical datasets hold transient objects only."
                )


@runtime_checkable
class DatasetAdapter(Protocol):
    """Turns one source dataset into a `CanonicalDataset`.

    `max_entities` is deliberately generic: an adapter subsamples by whatever
    entity owns a history in its source (cards, customers, accounts) and says
    which in the `Subsample` record. Callers stay dataset-agnostic.
    """

    name: str

    def load(
        self,
        root: Path,
        *,
        max_entities: int | None = None,
        seed: int = 42,
    ) -> CanonicalDataset: ...


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _iso_or_none(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _parse_or_none(value: Any) -> datetime | None:
    if value is None:
        return None
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


def _require_tz_aware(value: datetime | None, label: str) -> None:
    """Naive timestamps are the classic source of off-by-one-timezone splits."""
    if value is not None and value.tzinfo is None:
        raise DatasetContractError(f"{label} must be timezone-aware.")


def _require_sha256(filename: str, digest: str) -> None:
    lowered = digest.lower()
    if len(lowered) != _SHA256_LENGTH or not set(lowered) <= _HEX_DIGITS:
        raise DatasetContractError(
            f"files[{filename!r}] must be a 64-character hex SHA-256 digest, got {digest!r}."
        )
