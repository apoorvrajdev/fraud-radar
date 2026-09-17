"""Temporal drift: a model that never saw 2020, measured month by month through it.

The Phase 5D methodology's drift experiment (decision 3 and the frozen
protocol). Periods are half-open intervals on the wall clock, interpreted as
UTC:

    train   [2019-01-01, 2019-11-01)
    val     [2019-11-01, 2020-01-01)
    months  [2020-01-01, 2020-02-01) … [2020-12-01, 2021-01-01)

The drift model goes through exactly the procedure every benchmark model goes
through (`ml.train.fit_operating_model`), on these calendar folds instead of
the chronological 70/15/15 split: hyperparameters are searched on the train
period alone, the refit stops early against the val period, and the operating
threshold is selected once, on the val period. No hyperparameter of any other
run is reused, because the main runs' tuning data reaches into 2020.

Each month of 2020 is then scored once, at that fixed threshold, and reports
PR-AUC, recall, precision, realised FPR, fraud rate and volume. A metric that
is undefined for a month — PR-AUC and recall without frauds, realised FPR
without legitimate rows, precision when nothing is flagged — is `null`, never
a stand-in number.

Drift results never inform the main runs: their evaluation months overlap the
main test period. They are written to `drift_metrics.json` in a run directory
of their own, and a directory that already holds a trained run is refused.

Usage:
    cd backend
    uv run python -m ml.experiments.temporal_drift --dataset sparkov \\
        --full-corpus --run-name <drift-run-name>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

import ml.datasets.sparkov  # noqa: F401  (registers the adapter)
from app.fraud.feature_spec import DEFAULT_FEATURESET, FEATURESETS
from ml import train
from ml.artifacts import collect_library_versions, utc_now_iso
from ml.data import LabelledDataset
from ml.datasets.base import DatasetContractError
from ml.datasets.registry import available_datasets
from ml.evaluation import confusion_at_threshold, pr_auc
from ml.loading import DEFAULT_SUBSAMPLE_SEED, DataRequest, load_run_data
from ml.paths import (
    FEATURE_CACHE_DIR,
    RUNS_ROOT,
    InvalidRunNameError,
    ensure_dir,
    run_dir,
    validate_run_name,
)
from ml.reporting import LABEL_DELAY_NOTE
from ml.runs import RUN_METADATA_FILENAME, current_git_commit

log = logging.getLogger("ml.experiments.temporal_drift")

DRIFT_METRICS_FILENAME = "drift_metrics.json"

# Bumped when the shape of drift_metrics.json changes.
DRIFT_METRICS_VERSION = "1"

EVALUATION_YEAR = 2020

DRIFT_NOTE = (
    "Drift results never inform the main runs: their evaluation months overlap the main "
    "test period."
)

UNDEFINED_METRICS_NOTE = (
    "A month without frauds has no PR-AUC and no recall; a month without legitimate rows has "
    "no realised FPR; a month in which no row is flagged has no precision. Each is null."
)

TUNING_NOTE = (
    "Searched on the train period alone. The cross-validated PR-AUC comes from folds shuffled "
    "within the train period; it is a diagnostic, not a held-out estimate."
)

THRESHOLD_NOTE = "Selected once, on the val period, and applied unchanged to every month."

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECOND = timedelta(microseconds=1)


class DriftError(DatasetContractError):
    """The drift experiment cannot be run as the methodology defines it."""


@dataclass(frozen=True)
class Period:
    """A half-open interval `[start, end)` on the wall clock, in UTC."""

    name: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        for attribute in ("start", "end"):
            value: datetime = getattr(self, attribute)
            if value.tzinfo is None:
                raise DriftError(f"Period {self.name!r}: {attribute} must be timezone-aware.")
        if self.start >= self.end:
            raise DriftError(f"Period {self.name!r} starts at or after its end.")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "interval": "half-open: start <= timestamp < end",
        }


TRAIN_PERIOD = Period("train", datetime(2019, 1, 1, tzinfo=UTC), datetime(2019, 11, 1, tzinfo=UTC))
VAL_PERIOD = Period("val", datetime(2019, 11, 1, tzinfo=UTC), datetime(2020, 1, 1, tzinfo=UTC))


def evaluation_months(year: int = EVALUATION_YEAR) -> tuple[Period, ...]:
    """One half-open period per calendar month of `year`."""
    starts = [datetime(year, month, 1, tzinfo=UTC) for month in range(1, 13)]
    ends = [*starts[1:], datetime(year + 1, 1, 1, tzinfo=UTC)]
    return tuple(
        Period(f"{start.year:04d}-{start.month:02d}", start, end)
        for start, end in zip(starts, ends, strict=True)
    )


@dataclass(frozen=True)
class PeriodRows:
    """Which matrix rows fall in each period, in matrix order."""

    train: np.ndarray
    val: np.ndarray
    months: tuple[tuple[Period, np.ndarray], ...]
    outside: int


def assign_periods(timestamps: np.ndarray) -> PeriodRows:
    """Place every row in the period its timestamp falls in, or in none.

    Compared as exact integer microseconds since the epoch, so a row at a
    boundary instant belongs to the period that starts there.
    """
    instants = _as_microseconds(timestamps)

    def rows_in(period: Period) -> np.ndarray:
        start, end = _microseconds(period.start), _microseconds(period.end)
        return np.flatnonzero((instants >= start) & (instants < end))

    train_rows = rows_in(TRAIN_PERIOD)
    val_rows = rows_in(VAL_PERIOD)
    months = tuple((month, rows_in(month)) for month in evaluation_months())
    placed = len(train_rows) + len(val_rows) + sum(len(rows) for _, rows in months)
    return PeriodRows(
        train=train_rows, val=val_rows, months=months, outside=len(instants) - placed
    )


def month_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """One month's results at the fixed threshold, with `null` wherever a metric is undefined.

    Flagging is the existing definition (`ml.evaluation.confusion_at_threshold`:
    a score at or above the threshold); the rates are read from its counts.
    """
    volume = len(labels)
    frauds = int(np.sum(labels))
    legitimate = volume - frauds
    if volume:
        confusion = confusion_at_threshold(labels, scores, threshold)
        tp, fp = confusion.true_positives, confusion.false_positives
        tn, fn = confusion.true_negatives, confusion.false_negatives
    else:
        tp = fp = tn = fn = 0
    return {
        "volume": volume,
        "fraud_count": frauds,
        "fraud_rate": frauds / volume if volume else None,
        "pr_auc": pr_auc(labels, scores) if frauds else None,
        "recall": tp / frauds if frauds else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "realised_fpr": fp / legitimate if legitimate else None,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


@dataclass(frozen=True)
class DriftMeasurement:
    """The drift model, the rows each period held, and every month's results."""

    rows: PeriodRows
    selected: train.OperatingModel
    months: tuple[dict[str, Any], ...]


def measure_drift(
    ds: LabelledDataset,
    *,
    n_iter: int,
    cv_splits: int,
    target_fpr: float,
) -> DriftMeasurement:
    """Fit on the train period, choose the threshold on val, then score each 2020 month once."""
    rows = assign_periods(ds.timestamps)
    for period, indices in ((TRAIN_PERIOD, rows.train), (VAL_PERIOD, rows.val)):
        labels = ds.y[indices]
        if not 0 < int(labels.sum()) < len(labels):
            raise DriftError(
                f"The {period.name} period [{period.start.isoformat()}, {period.end.isoformat()}) "
                f"holds {len(labels)} rows with {int(labels.sum())} frauds; it needs both frauds "
                "and legitimate rows for the drift model to be fitted and its threshold chosen."
            )

    selected = train.fit_operating_model(
        ds.X[rows.train],
        ds.y[rows.train],
        ds.X[rows.val],
        ds.y[rows.val],
        n_iter=n_iter,
        cv_splits=cv_splits,
        target_fpr=target_fpr,
    )
    threshold = selected.threshold.value

    months = []
    for month, indices in rows.months:
        labels = ds.y[indices]
        scores = (
            selected.model.predict_proba(ds.X[indices])[:, 1]
            if len(indices)
            else np.empty(0, dtype=np.float64)
        )
        months.append({**month.to_dict(), **month_metrics(labels, scores, threshold)})
    return DriftMeasurement(rows=rows, selected=selected, months=tuple(months))


def drift_payload(
    measurement: DriftMeasurement,
    ds: LabelledDataset,
    *,
    run_name: str,
    dataset: dict[str, Any],
    featureset: str,
    cache_fingerprint: str | None,
    generated_at: str,
    code_version: str | None,
    library_versions: dict[str, str],
) -> dict[str, Any]:
    """The contents of `drift_metrics.json`."""
    selected = measurement.selected
    threshold = selected.threshold
    rows = measurement.rows
    return {
        "drift_metrics_version": DRIFT_METRICS_VERSION,
        "run_name": run_name,
        "note": DRIFT_NOTE,
        "dataset": dataset,
        "featureset_version": featureset,
        "feature_cache_fingerprint": cache_fingerprint,
        "periods": {
            "train": _period_summary(TRAIN_PERIOD, ds, rows.train),
            "val": _period_summary(VAL_PERIOD, ds, rows.val),
        },
        "rows_outside_periods": rows.outside,
        "tuning": {
            "iterations": selected.fit.tuning_iterations,
            "cv_folds": selected.fit.tuning_cv_folds,
            "random_state": selected.fit.random_state,
            "best_hyperparameters": selected.tuning.best_params,
            "best_cv_pr_auc": selected.tuning.best_score,
            "note": TUNING_NOTE,
        },
        "fit": {
            "scale_pos_weight": selected.fit.scale_pos_weight,
            "early_stopping_rounds": selected.fit.early_stopping_rounds,
            "best_iteration": selected.fit.best_iteration,
            "early_stopping_against": "val period",
        },
        "threshold": {
            "value": threshold.value,
            "target_fpr": threshold.target_fpr,
            "realised_fpr_on_val": threshold.realised_fpr_on_val,
            "fallback_used": threshold.fallback_used,
            "selected_on": "val period",
            "note": THRESHOLD_NOTE,
        },
        "months": list(measurement.months),
        "undefined_metrics_note": UNDEFINED_METRICS_NOTE,
        "label_delay_note": LABEL_DELAY_NOTE,
        "provenance": {
            "generated_at_utc": generated_at,
            "code_version": code_version,
            "library_versions": dict(library_versions),
        },
    }


def run_drift(
    request: DataRequest,
    run_name: str,
    *,
    n_iter: int,
    cv_splits: int,
    target_fpr: float,
    runs_root: Path = RUNS_ROOT,
) -> tuple[DriftMeasurement, dict[str, Any], Path]:
    """Load the dataset, measure drift, and write `drift_metrics.json` into its own run directory."""
    directory = run_dir(run_name, runs_root=runs_root)
    _refuse_a_trained_run(directory, run_name)

    data = load_run_data(request, with_provenance=True)
    if data.provenance is None:  # pragma: no cover - registered datasets always supply one
        raise DriftError(f"Dataset {request.dataset!r} was loaded without provenance.")
    measurement = measure_drift(
        data.ds, n_iter=n_iter, cv_splits=cv_splits, target_fpr=target_fpr
    )
    payload = drift_payload(
        measurement,
        data.ds,
        run_name=run_name,
        dataset=data.provenance.to_dict(),
        featureset=request.featureset,
        cache_fingerprint=data.cache_fingerprint,
        generated_at=utc_now_iso(),
        code_version=current_git_commit(),
        library_versions=collect_library_versions(),
    )
    path = ensure_dir(directory) / DRIFT_METRICS_FILENAME
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return measurement, payload, path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _refuse_a_trained_run(directory: Path, run_name: str) -> None:
    if (directory / RUN_METADATA_FILENAME).exists():
        raise DriftError(
            f"Run directory {run_name!r} holds a trained run's {RUN_METADATA_FILENAME}. Drift "
            "results never inform the main runs, so they are written to a directory of their own."
        )


def _period_summary(period: Period, ds: LabelledDataset, indices: np.ndarray) -> dict[str, Any]:
    timestamps = ds.timestamps[indices]
    return {
        **period.to_dict(),
        "rows": len(indices),
        "fraud_count": int(ds.y[indices].sum()),
        "first_timestamp": min(timestamps).isoformat() if len(indices) else None,
        "last_timestamp": max(timestamps).isoformat() if len(indices) else None,
    }


def _microseconds(instant: datetime) -> int:
    return (instant - _EPOCH) // _MICROSECOND


def _as_microseconds(timestamps: Sequence[datetime] | np.ndarray) -> np.ndarray:
    values = []
    for timestamp in timestamps:
        if timestamp.tzinfo is None:
            raise DriftError("Drift periods are UTC instants; the dataset has naive timestamps.")
        values.append(_microseconds(timestamp))
    return np.asarray(values, dtype=np.int64)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure a model tuned on 2019 against each month of 2020"
    )
    parser.add_argument(
        "--dataset",
        default="sparkov",
        choices=available_datasets(),
        help="Registered dataset (default: sparkov)",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run directory of its own for drift_metrics.json; never a trained run's directory",
    )
    parser.add_argument("--root", type=Path, default=None, help="Directory holding the source files")
    parser.add_argument(
        "--max-cards",
        type=int,
        default=None,
        help="Keep only N entities with their full histories (the final result uses none)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SUBSAMPLE_SEED, help="Seed for entity subsampling"
    )
    parser.add_argument(
        "--featureset", default=DEFAULT_FEATURESET, choices=sorted(FEATURESETS)
    )
    parser.add_argument("--cache-root", type=Path, default=FEATURE_CACHE_DIR)
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Allow an unsubsampled load where the adapter guards against one",
    )
    parser.add_argument("--n-iter", type=int, default=25, help="RandomizedSearchCV iterations")
    parser.add_argument(
        "--cv-splits", type=int, default=4, help="StratifiedKFold splits inside the search"
    )
    parser.add_argument(
        "--target-fpr", type=float, default=0.01, help="FPR ceiling on the val period"
    )
    args = parser.parse_args(argv)
    try:
        validate_run_name(args.run_name)
    except InvalidRunNameError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    args = parse_args(argv)
    request = DataRequest(
        dataset=args.dataset,
        featureset=args.featureset,
        root=args.root,
        max_entities=args.max_cards,
        seed=args.seed,
        cache_root=args.cache_root,
        full_corpus=args.full_corpus,
    )
    try:
        measurement, _, path = run_drift(
            request,
            args.run_name,
            n_iter=args.n_iter,
            cv_splits=args.cv_splits,
            target_fpr=args.target_fpr,
            runs_root=RUNS_ROOT,
        )
    except DatasetContractError as exc:
        log.error("Drift run %s was not measured: %s", args.run_name, exc)
        sys.exit(1)

    threshold = measurement.selected.threshold
    if threshold.fallback_used:
        log.warning(
            "No val-period threshold met FPR ≤ %.4f; every month is measured at the fallback %.2f.",
            threshold.target_fpr,
            threshold.value,
        )
    for month in measurement.months:
        log.info(
            "  %s  volume=%d  fraud_rate=%s  pr_auc=%s  recall=%s  precision=%s  realised_fpr=%s",
            month["name"],
            month["volume"],
            month["fraud_rate"],
            month["pr_auc"],
            month["recall"],
            month["precision"],
            month["realised_fpr"],
        )
    log.info("Wrote %s", path)


if __name__ == "__main__":
    main()
