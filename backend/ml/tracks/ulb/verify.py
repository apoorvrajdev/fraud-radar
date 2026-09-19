"""Verify a recorded ULB run before its records are committed (Phase 5E decision 13).

Usage:
    cd backend
    uv run python -m ml.tracks.ulb.verify ulb_pca_v1_seed42 [ulb_pca_v1_seed43 ...]

The Phase 5D verifier refuses a featureset the production registry does not
hold, and that guard stays as it is. This verifier checks the featureset
against `ulb_pca_v1`'s own definition instead, and otherwise applies the
Phase 5D checks unchanged: the run's name, the library versions it was
recorded with, its fit record and its saved model. It then rebuilds the matrix
from the pinned file, re-splits it, and requires the provenance, fold periods,
fold sizes, fold fraud counts and test-fold transactions the run recorded.
Finally it scores the test fold, and only the test fold, with the saved model,
and requires `metrics.json` and `calibration_metrics.json` to be reproduced
exactly.

It is read-only: nothing is written, and a run that fails any check is
refused with the reason.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from ml.artifacts import MODEL_FILENAME
from ml.data import LabelledDataset
from ml.datasets.base import DatasetContractError
from ml.datasets.manifest import DEFAULT_MANIFEST_PATH
from ml.loading import RunData
from ml.paths import RUNS_ROOT
from ml.run_analysis import CALIBRATION_FILENAME
from ml.run_verification import (
    FEATURE_LIST_FILENAME,
    RecordedRun,
    RunVerificationError,
    check_loaded_folds,
    check_metrics_reproduced,
    check_record_apart_from_featureset,
    read_recorded_run,
)
from ml.splits import SplitIndices
from ml.tracks.ulb.load import DEFAULT_ROOT, load_ulb
from ml.tracks.ulb.matrix import ULB_PCA_V1, ULB_PCA_V1_FEATURES, build_matrix
from ml.tracks.ulb.train import describe_ulb_calibration

log = logging.getLogger("ml.tracks.ulb.verify")


@dataclass(frozen=True)
class VerifiedUlbRun:
    """A ULB run whose data, folds, model, metrics and calibration all match its records.

    `test_scores` are the saved model's scores for `splits.test`, in that order.
    """

    recorded: RecordedRun
    ds: LabelledDataset
    splits: SplitIndices
    test_scores: np.ndarray


def verify_ulb_run(
    run_name: str,
    *,
    root: Path = DEFAULT_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    runs_root: Path = RUNS_ROOT,
) -> VerifiedUlbRun:
    """Verify `run_name` against the pinned ULB file, or refuse it with the reason."""
    recorded = read_recorded_run(run_name, runs_root=runs_root)
    # On the records alone, before the corpus is read.
    check_record_apart_from_featureset(recorded)
    _check_featureset(recorded)
    calibration = _read_calibration(recorded)

    try:
        matrix = build_matrix(load_ulb(root, manifest_path=manifest_path))
    except DatasetContractError as exc:
        raise RunVerificationError(
            f"Run {run_name!r}: the ULB file cannot be read as the run read it: {exc}"
        ) from exc
    ds = matrix.ds
    if ds.feature_names != recorded.feature_list:
        raise RunVerificationError(
            f"Run {run_name!r}: the rebuilt matrix has features {ds.feature_names}, but the "
            f"run was trained on {recorded.feature_list}."
        )
    splits = check_loaded_folds(recorded, RunData(ds=ds, provenance=matrix.provenance))

    test_scores = score_test_fold(recorded, ds, splits)
    check_metrics_reproduced(recorded, ds, splits, test_scores)
    _check_calibration_reproduced(recorded, calibration, ds.y[splits.test], test_scores)
    return VerifiedUlbRun(recorded=recorded, ds=ds, splits=splits, test_scores=test_scores)


def score_test_fold(recorded: RecordedRun, ds: LabelledDataset, splits: SplitIndices) -> np.ndarray:
    """The saved model's scores for the test fold: the booster call serving scoring makes."""
    booster = xgb.Booster()
    booster.load_model(str(recorded.directory / MODEL_FILENAME))
    test_features = np.asarray(ds.X[splits.test], dtype=np.float64)
    return np.asarray(
        booster.predict(xgb.DMatrix(test_features, feature_names=list(ds.feature_names)))
    )


def _check_featureset(recorded: RecordedRun) -> None:
    version = recorded.record.featureset_version
    if version != ULB_PCA_V1:
        raise RunVerificationError(
            f"Run {recorded.name!r} records featureset {version!r}; a ULB run is trained on "
            f"{ULB_PCA_V1!r} only."
        )
    if recorded.feature_list != list(ULB_PCA_V1_FEATURES):
        raise RunVerificationError(
            f"Run {recorded.name!r}: {FEATURE_LIST_FILENAME} does not list {ULB_PCA_V1!r}'s "
            "features in their defined order."
        )


def _read_calibration(recorded: RecordedRun) -> dict[str, Any]:
    path = recorded.directory / CALIBRATION_FILENAME
    if not path.exists():
        raise RunVerificationError(
            f"Run {recorded.name!r} is incomplete: {CALIBRATION_FILENAME} missing from "
            f"{recorded.directory}."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunVerificationError(
            f"Run {recorded.name!r}: {CALIBRATION_FILENAME} could not be read: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RunVerificationError(
            f"Run {recorded.name!r}: {CALIBRATION_FILENAME} does not hold a JSON object."
        )
    return payload


def _check_calibration_reproduced(
    recorded: RecordedRun,
    written: dict[str, Any],
    test_labels: np.ndarray,
    test_scores: np.ndarray,
) -> None:
    # Compared as JSON, the form the recorded values were written in.
    recomputed = json.loads(json.dumps(describe_ulb_calibration(test_labels, test_scores)))
    if recomputed != written:
        differing = sorted(
            key
            for key in recomputed.keys() | written.keys()
            if recomputed.get(key) != written.get(key)
        )
        raise RunVerificationError(
            f"Run {recorded.name!r}: its saved model and test fold do not reproduce "
            f"{CALIBRATION_FILENAME} exactly (differing: {', '.join(differing)})."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify recorded ULB benchmark runs")
    parser.add_argument("run_names", nargs="+", help="Runs under ml/artifacts/runs/ to verify")
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
    refused = 0
    for run_name in args.run_names:
        try:
            verified = verify_ulb_run(run_name, root=args.root)
        except RunVerificationError as exc:
            log.error("Run %s did not verify: %s", run_name, exc)
            refused += 1
            continue
        log.info(
            "Run %s verified: %d test transactions, metrics.json and %s reproduced exactly.",
            run_name,
            len(verified.splits.test),
            CALIBRATION_FILENAME,
        )
    if refused:
        sys.exit(1)


if __name__ == "__main__":
    main()
