"""Train one pre-registered ULB benchmark run (Phase 5E).

Usage:
    cd backend
    uv run python -m ml.tracks.ulb.train --seed 42

The runs are `ulb_pca_v1_seed42`, `_seed43` and `_seed44` (decisions 9 and
14), trained in that order, each verified with `ml.tracks.ulb.verify` before
its records are committed (decision 13). No other seed can be trained.

The procedure is the one every Phase 5D run went through, unchanged
(decision 8). The ULB rows are loaded from the pinned file, the quality
report is built, and any decision 15 stop it finds ends the work before a
model is trained. Then the `ulb_pca_v1` matrix goes through
`ml.train.train_and_evaluate`: the existing chronological 70/15/15 split,
the existing search on the training fold only with 25 iterations and 4
stratified folds, the refit with early stopping against the val fold, the
threshold chosen on the val fold at FPR <= 1%, and one evaluation of the test
fold. Only the random state differs between the three runs.

A run writes its quality report, its calibration record and the artifacts
`ml.train.write_run` writes for every run; `run.json` is written last, so a
directory without it is not a run. `model.json` and the plot stay gitignored.
A threshold that falls back is recorded in `threshold.json` and then stops
the work. Nothing here promotes a run (decision 12).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.datasets.base import DatasetContractError
from ml.datasets.manifest import DEFAULT_MANIFEST_PATH
from ml.paths import RUNS_ROOT, ensure_dir, run_dir
from ml.run_analysis import CALIBRATION_FILENAME, describe_calibration
from ml.tracks.ulb.load import DEFAULT_ROOT, LICENCE_NOTICE, METHOD_OFFER, load_ulb
from ml.tracks.ulb.matrix import ULB_PCA_V1, build_matrix
from ml.tracks.ulb.quality import UlbQualityReport, build_quality_report
from ml.train import (
    ExistingRunError,
    TrainingOutcome,
    refuse_an_existing_run,
    train_and_evaluate,
    write_run,
)

log = logging.getLogger("ml.tracks.ulb.train")

# The pre-registered random states, in the order the runs are made (decision 9).
ULB_SEEDS: tuple[int, ...] = (42, 43, 44)

# The frozen protocol: the existing search budget, folds and operating point.
TUNING_ITERATIONS = 25
TUNING_CV_FOLDS = 4
TARGET_FPR = 0.01

# Stated with the calibration values themselves, as the frozen protocol requires.
ULB_CALIBRATION_NOTE = (
    "Measured on the test fold; no calibrator fitted. The positive-class values are hard to "
    "read for two reasons: scale_pos_weight inflates scores by design, and few frauds leave "
    "the positive-class bins sparse."
)


class UlbStopError(RuntimeError):
    """A Phase 5E decision 15 stop: the work stops and the finding is reported."""

    def __init__(self, run_name: str, reasons: Sequence[str]) -> None:
        self.run_name = run_name
        self.reasons = tuple(reasons)
        super().__init__(
            f"Run {run_name!r} stopped under Phase 5E decision 15: " + " ".join(self.reasons)
        )


@dataclass(frozen=True)
class UlbRun:
    """A trained, written ULB run. Nothing in it has been verified yet."""

    run_name: str
    directory: Path
    outcome: TrainingOutcome
    report: UlbQualityReport


def run_name_for(seed: int) -> str:
    """The run a pre-registered random state is recorded under (decision 14)."""
    return f"{ULB_PCA_V1}_seed{seed}"


def run_notes(seed: int) -> str:
    """What `run.json` says about the run beyond its fields, with the licence terms it follows."""
    return (
        f"Phase 5E ULB benchmark run under pre-registered random state {seed}. No subsample "
        "applies, so seed is null; the model random_state is recorded with the fit, in "
        "training_metadata.json. Fold periods are elapsed time from the first transaction, "
        "encoded from 1970-01-01T00:00:00+00:00, and are not calendar dates. Never promoted. "
        f"{LICENCE_NOTICE} {METHOD_OFFER}"
    )


def describe_ulb_calibration(test_labels: np.ndarray, test_scores: np.ndarray) -> dict[str, Any]:
    """The calibration record: the Phase 5D measurement, with ULB's reading caveats."""
    payload = describe_calibration(test_labels, test_scores)
    payload["note"] = ULB_CALIBRATION_NOTE
    return payload


def train_ulb_run(
    seed: int,
    *,
    root: Path = DEFAULT_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    runs_root: Path = RUNS_ROOT,
) -> UlbRun:
    """Train and write the run for pre-registered random state `seed`.

    Raises `ExistingRunError` before loading anything if the run is already
    recorded, and `UlbStopError` on a decision 15 stop: before training for a
    stop the rows trigger, after writing the records for a fallback threshold.
    """
    if seed not in ULB_SEEDS:
        raise ValueError(
            f"Random state {seed} is not pre-registered; the ULB runs use "
            f"{', '.join(str(value) for value in ULB_SEEDS)} only."
        )
    run_name = run_name_for(seed)
    refuse_an_existing_run(run_name, runs_root=runs_root)

    source = load_ulb(root, manifest_path=manifest_path)
    report = build_quality_report(source)
    if report.stops:
        raise UlbStopError(run_name, report.stops)
    matrix = build_matrix(source)
    ds = matrix.ds
    log.info(
        "Training %s: %d rows, %d features, random_state=%d",
        run_name,
        ds.n_rows,
        len(ds.feature_names),
        seed,
    )

    outcome = train_and_evaluate(
        ds,
        n_iter=TUNING_ITERATIONS,
        cv_splits=TUNING_CV_FOLDS,
        target_fpr=TARGET_FPR,
        random_state=seed,
    )

    directory = ensure_dir(run_dir(run_name, runs_root=runs_root))
    report.write(directory)
    _write_json(
        directory / CALIBRATION_FILENAME,
        describe_ulb_calibration(ds.y[outcome.splits.test], outcome.test_scores),
    )
    write_run(
        run_name,
        ds,
        outcome,
        dataset=matrix.provenance,
        featureset=ULB_PCA_V1,
        metrics=dict(outcome.metrics),
        notes=run_notes(seed),
        runs_root=runs_root,
    )
    if outcome.threshold.fallback_used:
        raise UlbStopError(
            run_name,
            (
                "No val-fold threshold meets the FPR target, so the fallback threshold was "
                "used; it is recorded in threshold.json.",
            ),
        )
    return UlbRun(run_name=run_name, directory=directory, outcome=outcome, report=report)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one pre-registered ULB benchmark run")
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=ULB_SEEDS,
        help="The run's pre-registered random state",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory holding creditcard.csv (default: ml/data/raw/ulb)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    args = parse_args(argv)
    try:
        run = train_ulb_run(args.seed, root=args.root)
    except (ExistingRunError, UlbStopError, DatasetContractError) as exc:
        log.error("Run %s was not completed: %s", run_name_for(args.seed), exc)
        sys.exit(1)
    live = run.outcome.live_features
    log.info(
        "Run %s written to %s; live features %d of %d in the training fold. Verify it with "
        "`python -m ml.tracks.ulb.verify %s` before committing its records.",
        run.run_name,
        run.directory.resolve(),
        live.live_count,
        live.live_count + len(live.constant),
        run.run_name,
    )


if __name__ == "__main__":
    main()
