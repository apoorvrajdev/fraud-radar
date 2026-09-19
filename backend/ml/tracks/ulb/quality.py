"""The ULB quality report: aggregate counts only, written as `ulb_quality_report.json`.

It records what a reader should know about the ULB rows before trusting a ULB
metric, and what Phase 5E leaves to be read from the data rather than assumed:
row, fraud, exclusion, zero-amount and duplicate counts; `Time`'s coverage
and ties; and, for the chronological split a run makes of these rows, each
fold's size, frauds and elapsed-time span, what happens at each boundary, and
which of decision 15's stops the rows trigger.

The report is aggregate-only by construction. ULB is published under the ODbL
for the database and the DbCL for its contents, and a committed report is a
Produced Work. It holds counts, rates and the few `Time` values that bound
the rows and their folds, never a row, a component value, an example
transaction or a matrix.
It carries the licence notice and the method offer with it
(docs/DATA_LICENSES.md).

The shared `ml.datasets.quality` report describes a `CanonicalDataset`, with
cards, merchants and categories ULB does not have. This report reuses its
exclusion and field records and the same split, and fills nothing it cannot
observe.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ml.datasets.quality import ExclusionRecord, FieldNote
from ml.runs import TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT
from ml.splits import assert_no_temporal_leakage, chronological_split
from ml.tracks.ulb.load import (
    LICENCE_NOTICE,
    LICENCE_TERMS,
    METHOD_OFFER,
    UlbSource,
    field_inventory,
)

ULB_QUALITY_REPORT_FILENAME = "ulb_quality_report.json"

# What each count means, written into the report so it can be read without
# this code.
QUALITY_DEFINITIONS: Mapping[str, str] = {
    "elapsed_seconds": (
        "Time as published: seconds from the first transaction in the file; not a calendar "
        "date or a time of day"
    ),
    "exact_duplicate_group": (
        "rows equal in every column, Class included; every row of a group is kept"
    ),
    "repeated_rows": "rows of exact-duplicate groups beyond the first row of each group",
    "label_conflict_group": "rows equal in every column but Class that carry both labels",
    "rows_sharing_a_time": "rows whose Time equals at least one other row's",
    "folds": (
        "ml.splits.chronological_split of the rows in load order (Time, then position in the "
        "file): 70/15/15 by row count, the folds a run trained on these rows records in run.json"
    ),
    "boundary": (
        "the position between two adjacent folds; rows before it are every row of the earlier "
        "folds, rows after it every row of the later folds"
    ),
    "shared_instant": (
        "the earlier fold's last Time equals the later fold's first, so rows at that Time fall "
        "on both sides of the boundary; recorded, not corrected"
    ),
    "straddling_duplicate_pair": (
        "two rows of one exact-duplicate group, one before the boundary and one after it"
    ),
    "stops": (
        "the Phase 5E decision 15 conditions these rows trigger; any one stops the work before "
        "a model is trained"
    ),
}

# The fold names run.json records, in time order.
_FOLD_NAMES = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

# What an empty positive class leaves undefined in each fold decision 15 guards.
_FOLDS_THAT_NEED_FRAUD: Mapping[str, str] = {
    VAL_SPLIT: "early stopping and threshold selection",
    TEST_SPLIT: "the test metrics",
}


@dataclass(frozen=True)
class DuplicateSummary:
    """Exact duplicates, which are kept, and rows whose labels conflict (decision 7)."""

    groups: int
    rows_in_groups: int
    largest_group: int
    fraud_groups: int
    label_conflict_groups: int
    label_conflict_rows: int

    @property
    def repeated_rows(self) -> int:
        return self.rows_in_groups - self.groups

    def to_dict(self) -> dict[str, int]:
        return {
            "exact_duplicate_groups": self.groups,
            "rows_in_exact_duplicate_groups": self.rows_in_groups,
            "repeated_rows": self.repeated_rows,
            "largest_exact_duplicate_group": self.largest_group,
            "exact_duplicate_groups_of_fraud_rows": self.fraud_groups,
            "label_conflict_groups": self.label_conflict_groups,
            "rows_in_label_conflict_groups": self.label_conflict_rows,
        }


@dataclass(frozen=True)
class TimeCoverage:
    """What `Time` spans, and how often rows share one value."""

    first_seconds: float | None
    last_seconds: float | None
    whole_seconds: bool
    file_in_time_order: bool
    distinct_values: int
    rows_sharing_a_time: int
    largest_shared_group: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_elapsed_seconds": _json_seconds(self.first_seconds),
            "last_elapsed_seconds": _json_seconds(self.last_seconds),
            "whole_seconds": self.whole_seconds,
            "file_in_time_order": self.file_in_time_order,
            "distinct_values": self.distinct_values,
            "rows_sharing_a_time": self.rows_sharing_a_time,
            "largest_group_sharing_a_time": self.largest_shared_group,
        }


@dataclass(frozen=True)
class FoldSummary:
    """One chronological fold: its size, its frauds and the elapsed time it spans."""

    name: str
    rows: int
    frauds: int
    first_seconds: float
    last_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "frauds": self.frauds,
            "first_elapsed_seconds": _json_seconds(self.first_seconds),
            "last_elapsed_seconds": _json_seconds(self.last_seconds),
        }


@dataclass(frozen=True)
class BoundarySummary:
    """What sits on both sides of the boundary between two adjacent folds."""

    earlier: str
    later: str
    earlier_last_seconds: float
    later_first_seconds: float
    rows_at_shared_instant_earlier: int
    rows_at_shared_instant_later: int
    duplicate_groups_straddling: int
    duplicate_pairs_straddling: int

    @property
    def instant_shared(self) -> bool:
        return self.earlier_last_seconds == self.later_first_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "between": [self.earlier, self.later],
            "earlier_fold_last_elapsed_seconds": _json_seconds(self.earlier_last_seconds),
            "later_fold_first_elapsed_seconds": _json_seconds(self.later_first_seconds),
            "instant_shared": self.instant_shared,
            "rows_at_shared_instant": {
                "earlier_fold": self.rows_at_shared_instant_earlier,
                "later_fold": self.rows_at_shared_instant_later,
            },
            "exact_duplicate_groups_straddling": self.duplicate_groups_straddling,
            "exact_duplicate_pairs_straddling": self.duplicate_pairs_straddling,
        }


@dataclass(frozen=True)
class SplitSummary:
    """The chronological folds of the rows, or why there are none."""

    folds: tuple[FoldSummary, ...] = ()
    boundaries: tuple[BoundarySummary, ...] = ()
    unavailable_reason: str = ""

    @property
    def available(self) -> bool:
        return not self.unavailable_reason

    def fold(self, name: str) -> FoldSummary | None:
        return next((fold for fold in self.folds if fold.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "reason": self.unavailable_reason}
        return {
            "available": True,
            "folds": [fold.to_dict() for fold in self.folds],
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
        }


@dataclass(frozen=True)
class UlbQualityReport:
    """Everything worth knowing about the ULB rows before training on them."""

    dataset: str
    version: str
    origin: str
    generated_at_utc: str
    files: Mapping[str, str]
    licence: str
    citation: str
    raw_row_count: int
    kept_row_count: int
    exclusions: tuple[ExclusionRecord, ...]
    fraud_count: int
    zero_amount_rows: int
    duplicates: DuplicateSummary
    time: TimeCoverage
    split: SplitSummary
    field_notes: tuple[FieldNote, ...]

    @property
    def excluded_row_count(self) -> int:
        return sum(record.count for record in self.exclusions)

    @property
    def fraud_rate(self) -> float:
        return self.fraud_count / self.kept_row_count if self.kept_row_count else 0.0

    @property
    def stops(self) -> tuple[str, ...]:
        """The decision 15 stops these rows trigger; empty when the work may go on.

        Only the stops the rows themselves can answer. A header or digest that
        does not match acquisition is refused before any report exists, and
        the threshold and verification stops come after training.
        """
        stops: list[str] = []
        if self.excluded_row_count:
            stops.append(
                f"{self.excluded_row_count} row(s) were excluded; no row may be excluded "
                "before a model is trained."
            )
        if not self.split.available:
            stops.append(f"The chronological split is unavailable: {self.split.unavailable_reason}")
        for name, needs in _FOLDS_THAT_NEED_FRAUD.items():
            fold = self.split.fold(name)
            if fold is not None and fold.frauds == 0:
                stops.append(f"The {name} fold holds no fraud, so {needs} would be undefined.")
        return tuple(stops)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "origin": self.origin,
            "generated_at_utc": self.generated_at_utc,
            "source": {"files": dict(self.files)},
            "licence": {
                "licence": self.licence,
                "notice": LICENCE_NOTICE,
                "method_offer": METHOD_OFFER,
                "citation": self.citation,
                "terms": LICENCE_TERMS,
            },
            "rows": {
                "read": self.raw_row_count,
                "kept": self.kept_row_count,
                "excluded": self.excluded_row_count,
            },
            # Counts only: an exclusion never carries an example of the row it excluded.
            "exclusions": [
                {"reason": record.reason, "count": record.count} for record in self.exclusions
            ],
            "label": {"fraud_count": self.fraud_count, "fraud_rate": self.fraud_rate},
            "amounts": {"zero_amount_rows": self.zero_amount_rows},
            "duplicates": self.duplicates.to_dict(),
            "time": self.time.to_dict(),
            "chronological_split": self.split.to_dict(),
            "field_notes": [note.to_dict() for note in self.field_notes],
            "stops": list(self.stops),
            "definitions": dict(QUALITY_DEFINITIONS),
        }

    def write(self, directory: Path, *, filename: str = ULB_QUALITY_REPORT_FILENAME) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / filename
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return target


def build_quality_report(source: UlbSource) -> UlbQualityReport:
    """Summarise the loaded ULB rows. Descriptive only: no row, fold or value is changed."""
    provenance = source.provenance
    group_ids, group_sizes = _exact_duplicate_groups(source)
    return UlbQualityReport(
        dataset=provenance.name,
        version=provenance.version,
        origin=provenance.origin.value,
        generated_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        files=dict(provenance.files),
        licence=provenance.license,
        citation=provenance.citation,
        raw_row_count=source.raw_row_count,
        kept_row_count=source.n_rows,
        exclusions=tuple(
            ExclusionRecord(reason=record.reason, count=record.count)
            for record in source.exclusions
        ),
        fraud_count=source.fraud_count,
        zero_amount_rows=int((source.amounts == 0).sum()),
        duplicates=summarise_duplicates(source, group_ids=group_ids, group_sizes=group_sizes),
        time=summarise_time(source),
        split=summarise_split(source, group_ids=group_ids, group_sizes=group_sizes),
        field_notes=field_inventory(),
    )


def summarise_duplicates(
    source: UlbSource,
    *,
    group_ids: np.ndarray | None = None,
    group_sizes: np.ndarray | None = None,
) -> DuplicateSummary:
    """Count exact-duplicate groups and label conflicts. Every row stays in the data."""
    if group_ids is None or group_sizes is None:
        group_ids, group_sizes = _exact_duplicate_groups(source)
    repeated = group_sizes > 1
    fraud_per_group = np.bincount(group_ids, weights=source.labels, minlength=len(group_sizes))

    # Rows equal in everything but the label, grouped without the label column.
    unlabelled_ids, unlabelled_sizes = _row_groups(_values_without_label(source))
    fraud_per_unlabelled = np.bincount(
        unlabelled_ids, weights=source.labels, minlength=len(unlabelled_sizes)
    )
    conflicting = (fraud_per_unlabelled > 0) & (fraud_per_unlabelled < unlabelled_sizes)

    return DuplicateSummary(
        groups=int(repeated.sum()),
        rows_in_groups=int(group_sizes[repeated].sum()),
        largest_group=int(group_sizes[repeated].max()) if repeated.any() else 0,
        fraud_groups=int((repeated & (fraud_per_group > 0)).sum()),
        label_conflict_groups=int(conflicting.sum()),
        label_conflict_rows=int(unlabelled_sizes[conflicting].sum()),
    )


def summarise_time(source: UlbSource) -> TimeCoverage:
    """`Time`'s range, whether it is whole seconds and in file order, and its ties."""
    seconds = source.time_seconds
    if seconds.size == 0:
        return TimeCoverage(
            first_seconds=None,
            last_seconds=None,
            whole_seconds=True,
            file_in_time_order=source.file_in_time_order,
            distinct_values=0,
            rows_sharing_a_time=0,
            largest_shared_group=0,
        )
    _, counts = np.unique(seconds, return_counts=True)
    shared = counts > 1
    return TimeCoverage(
        first_seconds=float(seconds.min()),
        last_seconds=float(seconds.max()),
        whole_seconds=bool(np.all(seconds == np.floor(seconds))),
        file_in_time_order=source.file_in_time_order,
        distinct_values=len(counts),
        rows_sharing_a_time=int(counts[shared].sum()),
        largest_shared_group=int(counts.max()),
    )


def summarise_split(
    source: UlbSource,
    *,
    group_ids: np.ndarray | None = None,
    group_sizes: np.ndarray | None = None,
) -> SplitSummary:
    """The folds a run makes of these rows, and what straddles each boundary.

    Splits `source.timestamps` with the same `chronological_split` and the
    same leakage check a run applies to the same timestamps, so the folds
    described are the folds a run records.
    """
    if source.n_rows < 3:
        return SplitSummary(unavailable_reason="fewer than 3 rows cannot be split into folds.")
    splits = chronological_split(source.timestamps)
    folds = tuple(zip(_FOLD_NAMES, (splits.train, splits.val, splits.test), strict=True))
    if any(len(indices) == 0 for _, indices in folds):
        return SplitSummary(
            unavailable_reason="too few rows for every fold of the 70/15/15 split to hold one."
        )
    assert_no_temporal_leakage(source.timestamps, splits)
    if group_ids is None or group_sizes is None:
        group_ids, group_sizes = _exact_duplicate_groups(source)

    seconds = source.time_seconds
    fold_summaries = tuple(
        FoldSummary(
            name=name,
            rows=len(indices),
            frauds=int(source.labels[indices].sum()),
            first_seconds=float(seconds[indices].min()),
            last_seconds=float(seconds[indices].max()),
        )
        for name, indices in folds
    )
    boundaries = tuple(
        _boundary(folds, position, seconds=seconds, group_ids=group_ids, group_sizes=group_sizes)
        for position in range(1, len(folds))
    )
    return SplitSummary(folds=fold_summaries, boundaries=boundaries)


def _boundary(
    folds: Sequence[tuple[str, np.ndarray]],
    position: int,
    *,
    seconds: np.ndarray,
    group_ids: np.ndarray,
    group_sizes: np.ndarray,
) -> BoundarySummary:
    earlier_name, earlier = folds[position - 1]
    later_name, later = folds[position]
    earlier_last = float(seconds[earlier].max())
    later_first = float(seconds[later].min())
    shared = earlier_last == later_first

    before = np.concatenate([indices for _, indices in folds[:position]])
    after = np.concatenate([indices for _, indices in folds[position:]])
    rows_before = np.bincount(group_ids[before], minlength=len(group_sizes))
    rows_after = np.bincount(group_ids[after], minlength=len(group_sizes))
    repeated = group_sizes > 1
    straddling = repeated & (rows_before > 0) & (rows_after > 0)
    at_instant_earlier = int((seconds[earlier] == earlier_last).sum()) if shared else 0
    at_instant_later = int((seconds[later] == later_first).sum()) if shared else 0

    return BoundarySummary(
        earlier=earlier_name,
        later=later_name,
        earlier_last_seconds=earlier_last,
        later_first_seconds=later_first,
        rows_at_shared_instant_earlier=at_instant_earlier,
        rows_at_shared_instant_later=at_instant_later,
        duplicate_groups_straddling=int(straddling.sum()),
        duplicate_pairs_straddling=int((rows_before * rows_after)[repeated].sum()),
    )


def _exact_duplicate_groups(source: UlbSource) -> tuple[np.ndarray, np.ndarray]:
    """Each row's exact-duplicate group, and each group's size."""
    return _row_groups(np.column_stack((_values_without_label(source), source.labels)))


def _values_without_label(source: UlbSource) -> np.ndarray:
    """Every column but `Class`, one row per kept row."""
    return np.column_stack((source.time_seconds, source.components, source.amounts))


def _row_groups(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group equal rows: each row's group id, and each group's size."""
    if len(values) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    _, group_ids, sizes = np.unique(values, axis=0, return_inverse=True, return_counts=True)
    return group_ids.reshape(-1), sizes


def _json_seconds(value: float | None) -> int | float | None:
    """Elapsed seconds as an exact JSON number: integral values as int."""
    if value is None:
        return None
    return int(value) if float(value).is_integer() else float(value)
