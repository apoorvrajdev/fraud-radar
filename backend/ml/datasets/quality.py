"""Quality reporting for a canonical dataset.

The report answers the questions a reader should ask before trusting any
metric computed downstream: how much of the source survived, what was thrown
away and why, how the label is distributed, and — the part that usually goes
unsaid — which canonical fields are constant on this dataset, because a
constant field is a feature that cannot contribute anything.

It also describes how fraud is distributed across cards and over time, and
which cards carry fraud on both sides of a chronological split boundary: the
Phase 5D methodology leaves both to be recorded from the data, because they
decide the effective sample size behind a fold's metrics. The folds are those
`ml.splits.chronological_split` makes of the loaded rows, which are the folds
a run trained on the same rows records.

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
from ml.splits import chronological_split

QUALITY_REPORT_FILENAME = "quality_report.json"

# What each part of `fraud_distribution` counts, written into the report so
# the numbers can be read without this code.
FRAUD_DISTRIBUTION_DEFINITIONS: Mapping[str, str] = {
    "card": (
        "Transaction.customer_id: the entity whose history the features are built from and "
        "that subsampling keeps whole; one per source card on Sparkov"
    ),
    "month": "the calendar month of a transaction's timestamp in UTC",
    "folds": (
        "ml.splits.chronological_split of these rows in canonical chronological order: 70/15/15 "
        "by row count, the folds a run trained on these rows records in run.json"
    ),
    "boundary": (
        "the position between two adjacent folds; rows before it are every row of the earlier "
        "folds, rows after it every row of the later folds"
    ),
    "card_with_fraud_on_both_sides": (
        "a card with at least one fraud before the boundary and at least one fraud after it"
    ),
    "shared_instant": (
        "the earlier fold's last timestamp equals the later fold's first, so transactions at "
        "that instant fall on both sides of the boundary; recorded, not corrected"
    ),
}

# The fold names run.json records, in time order.
_FOLD_NAMES = ("train", "val", "test")


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
    fraud_distribution: Mapping[str, Any] = field(default_factory=dict)

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
            "fraud_distribution": dict(self.fraud_distribution),
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
        fraud_distribution=describe_fraud_distribution(dataset),
    )


def describe_fraud_distribution(dataset: CanonicalDataset) -> dict[str, Any]:
    """How fraud is spread across cards, over months, and across the chronological folds.

    Descriptive only: nothing here selects, filters or changes a row, a fold
    or a feature.
    """
    labels = [int(dataset.labels[tx.id]) for tx in dataset.transactions]
    cards = [tx.customer_id for tx in dataset.transactions]
    frauds_per_card = Counter(card for card, label in zip(cards, labels, strict=True) if label)
    return {
        "definitions": dict(FRAUD_DISTRIBUTION_DEFINITIONS),
        "cards": {
            "with_transactions": len(set(cards)),
            "with_fraud": len(frauds_per_card),
            "frauds_per_card_with_fraud": (
                _percentiles(np.array(list(frauds_per_card.values()), dtype=float), (50, 99))
                if frauds_per_card
                else None
            ),
        },
        "by_month": _fraud_by_month(dataset, labels),
        "chronological_split": _fraud_across_folds(dataset, cards, labels),
    }


def _fraud_by_month(dataset: CanonicalDataset, labels: Sequence[int]) -> list[dict[str, Any]]:
    """Rows and frauds per calendar month, every month from the first to the last included."""
    if not dataset.transactions:
        return []
    counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for tx, label in zip(dataset.transactions, labels, strict=True):
        moment = tx.created_at.astimezone(UTC)
        counts[(moment.year, moment.month)][0] += 1
        counts[(moment.year, moment.month)][1] += label

    first = dataset.transactions[0].created_at.astimezone(UTC)
    last = dataset.transactions[-1].created_at.astimezone(UTC)
    months = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        rows, frauds = counts.get((year, month), [0, 0])
        months.append({"month": f"{year:04d}-{month:02d}", "rows": rows, "frauds": frauds})
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def _fraud_across_folds(
    dataset: CanonicalDataset, cards: Sequence[str], labels: Sequence[int]
) -> dict[str, Any]:
    """Fraud per fold, and the cards whose fraud falls on both sides of each boundary."""
    timestamps = np.asarray([tx.created_at for tx in dataset.transactions], dtype=object)
    if len(timestamps) < 3:
        return {"available": False, "reason": "Fewer than 3 rows cannot be split into folds."}
    splits = chronological_split(timestamps)
    folds = list(zip(_FOLD_NAMES, (splits.train, splits.val, splits.test), strict=True))
    if any(len(indices) == 0 for _, indices in folds):
        return {
            "available": False,
            "reason": "Too few rows for every fold of the 70/15/15 split to hold one.",
        }

    def fraud_cards(indices: np.ndarray) -> set[str]:
        return {cards[index] for index in indices if labels[index]}

    def rows_at(instant: Any, indices: np.ndarray) -> int:
        return sum(1 for moment in timestamps[indices] if moment == instant)

    fold_summaries = [
        {
            "name": name,
            "rows": len(indices),
            "frauds": sum(labels[index] for index in indices),
            "cards_with_fraud": len(fraud_cards(indices)),
            "first_timestamp": min(timestamps[indices]).isoformat(),
            "last_timestamp": max(timestamps[indices]).isoformat(),
        }
        for name, indices in folds
    ]

    boundaries = []
    for position in range(1, len(folds)):
        earlier_name, earlier = folds[position - 1]
        later_name, later = folds[position]
        before = np.concatenate([indices for _, indices in folds[:position]])
        after = np.concatenate([indices for _, indices in folds[position:]])
        spanning = fraud_cards(before) & fraud_cards(after)
        earlier_last = max(timestamps[earlier])
        later_first = min(timestamps[later])
        shared = bool(earlier_last == later_first)
        boundaries.append(
            {
                "between": [earlier_name, later_name],
                "earlier_fold_last_timestamp": earlier_last.isoformat(),
                "later_fold_first_timestamp": later_first.isoformat(),
                "instant_shared": shared,
                "rows_at_shared_instant": {
                    "earlier_fold": rows_at(earlier_last, earlier) if shared else 0,
                    "later_fold": rows_at(later_first, later) if shared else 0,
                },
                "cards_with_fraud_on_both_sides": len(spanning),
                "frauds_before_on_those_cards": sum(
                    labels[index] for index in before if cards[index] in spanning
                ),
                "frauds_after_on_those_cards": sum(
                    labels[index] for index in after if cards[index] in spanning
                ),
            }
        )
    return {"available": True, "folds": fold_summaries, "boundaries": boundaries}


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
