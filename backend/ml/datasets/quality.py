"""Quality reporting for a canonical dataset.

The report answers the questions a reader should ask before trusting any
metric computed downstream: how much of the source survived, what was thrown
away and why, how the label is distributed, and — the part that usually goes
unsaid — which canonical fields are constant on this dataset, because a
constant field is a feature that cannot contribute anything.

Deliberately dataset-agnostic: it reads a `CanonicalDataset` plus the
accounting an adapter produced, so every future benchmark gets the same
report without this module learning anything about that source.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from ml.datasets.base import CanonicalDataset

QUALITY_REPORT_FILENAME = "quality_report.json"


class FieldStatus(StrEnum):
    """What became of a source field, or how a canonical field was filled."""

    MAPPED = "mapped"
    DERIVED = "derived"
    DROPPED = "dropped"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class FieldNote:
    """One line of the field inventory."""

    field: str
    status: FieldStatus
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "status": self.status.value, "detail": self.detail}


@dataclass(frozen=True)
class ExclusionRecord:
    """Rows an adapter refused to convert, and why.

    `count` is reported even when zero: "0 duplicates" is a finding, and a
    reader should not have to infer that the check ran.
    """

    reason: str
    count: int
    examples: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "count": self.count, "examples": list(self.examples)}


@dataclass(frozen=True)
class QualityReport:
    """Everything worth knowing about a loaded dataset before training on it."""

    dataset: str
    version: str
    origin: str
    generated_at_utc: str
    raw_row_count: int
    kept_row_count: int
    excluded_row_count: int
    fraud_count: int
    fraud_rate: float
    period_start: str | None
    period_end: str | None
    customer_count: int
    merchant_count: int
    amount_percentiles: Mapping[str, float]
    transactions_per_customer: Mapping[str, float]
    fraud_rate_by_category: Mapping[str, float]
    rows_by_category: Mapping[str, int]
    card_present_share: float
    constant_fields: Mapping[str, str]
    exclusions: tuple[ExclusionRecord, ...] = ()
    field_notes: tuple[FieldNote, ...] = ()
    subsample: Mapping[str, Any] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "origin": self.origin,
            "generated_at_utc": self.generated_at_utc,
            "rows": {
                "raw": self.raw_row_count,
                "kept": self.kept_row_count,
                "excluded": self.excluded_row_count,
            },
            "label": {
                "fraud_count": self.fraud_count,
                "fraud_rate": self.fraud_rate,
            },
            "period": {"start": self.period_start, "end": self.period_end},
            "entities": {
                "customers": self.customer_count,
                "merchants": self.merchant_count,
                "transactions_per_customer": dict(self.transactions_per_customer),
            },
            "amounts": dict(self.amount_percentiles),
            "categories": {
                "rows": dict(self.rows_by_category),
                "fraud_rate": dict(self.fraud_rate_by_category),
            },
            "card_present_share": self.card_present_share,
            "constant_fields": dict(self.constant_fields),
            "exclusions": [record.to_dict() for record in self.exclusions],
            "field_notes": [note.to_dict() for note in self.field_notes],
            "subsample": dict(self.subsample) if self.subsample else None,
            "extra": dict(self.extra),
            "warnings": list(self.warnings),
        }

    def write(self, directory: Path, *, filename: str = QUALITY_REPORT_FILENAME) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / filename
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return target


def build_quality_report(
    dataset: CanonicalDataset,
    *,
    raw_row_count: int,
    exclusions: Sequence[ExclusionRecord] = (),
    field_notes: Sequence[FieldNote] = (),
    extra: Mapping[str, Any] | None = None,
) -> QualityReport:
    """Summarise a canonical dataset and the accounting behind it."""
    provenance = dataset.provenance
    amounts = np.array([float(tx.amount) for tx in dataset.transactions], dtype=float)
    per_customer = Counter(tx.customer_id for tx in dataset.transactions)

    rows_by_category, fraud_by_category = _category_breakdown(dataset)
    excluded = sum(record.count for record in exclusions)

    return QualityReport(
        dataset=provenance.name,
        version=provenance.version,
        origin=provenance.origin.value,
        generated_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        raw_row_count=raw_row_count,
        kept_row_count=dataset.n_rows,
        excluded_row_count=excluded,
        fraud_count=dataset.fraud_count,
        fraud_rate=dataset.fraud_rate,
        period_start=dataset.period[0].isoformat() if dataset.period else None,
        period_end=dataset.period[1].isoformat() if dataset.period else None,
        customer_count=len(dataset.customers),
        merchant_count=len(dataset.merchants),
        amount_percentiles=_percentiles(amounts, (1, 50, 99)),
        transactions_per_customer=_percentiles(
            np.array(list(per_customer.values()), dtype=float), (50, 99)
        ),
        fraud_rate_by_category={
            category: fraud_by_category[category] / rows_by_category[category]
            for category in sorted(rows_by_category)
        },
        rows_by_category=dict(sorted(rows_by_category.items())),
        card_present_share=_card_present_share(dataset),
        constant_fields=_constant_fields(dataset),
        exclusions=tuple(exclusions),
        field_notes=tuple(field_notes),
        subsample=provenance.subsample.to_dict() if provenance.subsample else None,
        extra=dict(extra or {}),
        warnings=_warnings(dataset, raw_row_count=raw_row_count, excluded=excluded),
    )


def _category_breakdown(dataset: CanonicalDataset) -> tuple[dict[str, int], dict[str, int]]:
    rows: dict[str, int] = defaultdict(int)
    frauds: dict[str, int] = defaultdict(int)
    for tx in dataset.transactions:
        merchant = dataset.merchants.get(tx.merchant_id)
        category = merchant.category if merchant else "UNKNOWN"
        rows[category] += 1
        frauds[category] += dataset.labels.get(tx.id, 0)
    return dict(rows), dict(frauds)


def _card_present_share(dataset: CanonicalDataset) -> float:
    if not dataset.transactions:
        return 0.0
    present = sum(1 for tx in dataset.transactions if tx.is_card_present)
    return present / dataset.n_rows


def _constant_fields(dataset: CanonicalDataset) -> dict[str, str]:
    """Canonical fields that take exactly one value across the dataset.

    A constant field is a dead feature. Naming them here is what turns
    "the model scored X" into "the model scored X with these inputs inert".
    """
    candidates: dict[str, set[str]] = {
        "customer.country": {c.country for c in dataset.customers.values()},
        "customer.risk_tier": {str(c.risk_tier) for c in dataset.customers.values()},
        "customer.account_age_days": {
            str(c.account_age_days) for c in dataset.customers.values()
        },
        "merchant.country": {m.country for m in dataset.merchants.values()},
        "merchant.risk_rating": {str(m.risk_rating) for m in dataset.merchants.values()},
        "merchant.category": {m.category for m in dataset.merchants.values()},
        "transaction.country": {tx.country for tx in dataset.transactions},
        "transaction.currency": {tx.currency for tx in dataset.transactions},
        "transaction.payment_method": {tx.payment_method for tx in dataset.transactions},
        "transaction.is_card_present": {str(tx.is_card_present) for tx in dataset.transactions},
    }
    return {
        name: next(iter(values))
        for name, values in sorted(candidates.items())
        if len(values) == 1
    }


def _percentiles(values: np.ndarray, points: Sequence[int]) -> dict[str, float]:
    if values.size == 0:
        return {f"p{point}": 0.0 for point in points} | {"max": 0.0, "mean": 0.0}
    result = {f"p{point}": float(np.percentile(values, point)) for point in points}
    result["max"] = float(values.max())
    result["mean"] = float(values.mean())
    return result


def _warnings(dataset: CanonicalDataset, *, raw_row_count: int, excluded: int) -> tuple[str, ...]:
    """Conditions a reader should see without having to compute them."""
    warnings: list[str] = []
    if dataset.n_rows == 0:
        warnings.append("Dataset is empty after exclusions.")
    if dataset.fraud_count == 0 and dataset.n_rows:
        warnings.append("No positive labels: every downstream metric will be undefined.")
    if raw_row_count and excluded / raw_row_count > 0.01:
        warnings.append(
            f"{excluded / raw_row_count:.2%} of source rows were excluded; "
            "check the exclusion reasons before trusting the metrics."
        )
    constant = _constant_fields(dataset)
    if constant:
        warnings.append(
            "Constant canonical fields on this dataset: "
            + ", ".join(f"{name}={value}" for name, value in constant.items())
        )
    return tuple(warnings)
