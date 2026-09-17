"""Transfer: one run's frozen model, measured on another run's test fold.

The Phase 5D methodology's cross-generator transfer (decision 8). The source
run's saved model and its threshold — `synthetic_v1` in the benchmark — score
the test fold of a target run trained on another dataset — a Sparkov run. The
rows scored are exactly the rows the target run was itself evaluated on.

What makes the number a transfer measurement rather than an adaptation is what
never happens here, on any row of the target dataset: no threshold is
selected, no score is calibrated, no model is fitted or tuned, and no feature
is dropped, remapped or selected. The source model receives every column of
the featureset it was trained on, in registered order, including columns that
never vary on the target. Which columns those are is recorded as context; it
changes nothing about the model.

Both runs are verified before anything is scored, each through the same
verification a run's analysis uses: rebuilt from its own `run.json`, its
saved model reproducing its own `metrics.json` exactly. Beyond that, the
source run must record the digest of its `model.json`, so the model scoring
the target is provably the one the source run saved, and the target run must
identify its test fold's transactions, so the rows scored are provably the
rows it was evaluated on. The source model then scores the target's test rows
and nothing else; no target train or val row reaches either model.

The result is written to `transfer_metrics.json` in the target run's
directory, beside the target's own `metrics.json`, which is never touched.
Every result names both runs.

Usage:
    cd backend
    uv run python -m ml.experiments.transfer \\
        --source-run synthetic_v1 --target-run sparkov_v1_200cards
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

from app.fraud.explainer import FraudExplainer, load_explainer
from ml.artifacts import collect_library_versions, utc_now_iso
from ml.datasets.base import DatasetProvenance
from ml.holdout import HoldoutEvaluation, evaluate_test_fold
from ml.loading import DEFAULT_SYNTHETIC_CSV, DataRequest, load_run_data
from ml.paths import FEATURE_CACHE_DIR, RUNS_ROOT, InvalidRunNameError, validate_run_name
from ml.reporting import LABEL_DELAY_NOTE, constant_features
from ml.run_verification import (
    RecordedRun,
    RunVerificationError,
    VerifiedRun,
    check_recorded_run,
    read_recorded_run,
    verify_run,
)
from ml.runs import TRANSACTION_IDS_DIGEST_DEFINITION, current_git_commit

log = logging.getLogger("ml.experiments.transfer")

TRANSFER_METRICS_FILENAME = "transfer_metrics.json"

# Bumped when the shape of transfer_metrics.json changes.
TRANSFER_METRICS_VERSION = "1"

# Where the threshold behind each transfer result came from.
#
# SOURCE_RUN_THRESHOLD_SOURCE: the source run's threshold.json, selected on the
# source run's val fold before its test fold was scored, and applied unchanged.
#
# TARGET_TEST_ROC_CURVE_SOURCE: the highest threshold whose FPR on the target
# test fold stays within a ceiling. It is read off the target test fold itself,
# so the result is a point on that fold's ROC curve. The threshold is never
# applied to anything or reported as an operating point.
SOURCE_RUN_THRESHOLD_SOURCE = "source_run_threshold"
TARGET_TEST_ROC_CURVE_SOURCE = "target_test_roc_curve"

SELECTED_THRESHOLD_METHOD = (
    "the highest score threshold whose FPR on the source run's val fold is at most the "
    "source run's FPR ceiling, chosen before the source run's test fold was scored"
)
FALLBACK_THRESHOLD_METHOD = (
    "the fixed fallback threshold, because no threshold on the source run's val fold met "
    "the source run's FPR ceiling"
)

METHOD_NOTE = (
    "The source run's saved model and threshold are applied unchanged to the target run's "
    "test fold. Nothing is selected, fitted, calibrated or tuned on any row of the target "
    "dataset, and every feature of the source featureset is given to the model."
)

THRESHOLD_FREE_NOTE = (
    "PR-AUC and ROC-AUC use no threshold; read PR-AUC against the target test prevalence. "
    "Each recall at a fixed FPR is read at a threshold found on the target test fold itself, "
    "so it is a point on the target test ROC curve, not a result at an operating threshold."
)

AT_SOURCE_THRESHOLD_NOTE = (
    "Measured at the source run's own threshold, unchanged. The FPR ceiling it was selected "
    "for applied to the source run's val fold; the FPR it realises on the target test fold is "
    "measured here, not guaranteed by that ceiling."
)

FEATURE_NOTE = (
    "Context only. The source model receives every column listed, in this order, whatever "
    "values they take on the target; no column is dropped or remapped because it is "
    "constant or outside the source training range."
)


class TransferError(RunVerificationError):
    """The two runs cannot be paired as a transfer measurement."""


@dataclass(frozen=True)
class DataLocation:
    """Where one run's data files are. Everything else comes from the run's own record.

    `root`, `cache_root` and `full_corpus` locate a registered dataset;
    `csv_path` and `limit` the synthetic one.
    """

    root: Path | None = None
    cache_root: Path = FEATURE_CACHE_DIR
    csv_path: Path = DEFAULT_SYNTHETIC_CSV
    limit: int | None = None
    full_corpus: bool = False


@dataclass(frozen=True)
class TransferMeasurement:
    """One transfer measurement and the verified runs it was measured between.

    `scores` are the source model's scores for `target.splits.test`, in that
    order. `evaluation` holds them measured against the target test labels at
    the source threshold; its context's train and val fraud counts are the
    source run's. `payload` is what was written to `written`.
    """

    source: VerifiedRun
    target: VerifiedRun
    scores: np.ndarray
    evaluation: HoldoutEvaluation
    payload: dict[str, Any]
    written: Path


def measure_transfer(
    source_run: str,
    target_run: str,
    *,
    runs_root: Path = RUNS_ROOT,
    source_data: DataLocation | None = None,
    target_data: DataLocation | None = None,
) -> TransferMeasurement:
    """Verify both runs, score the target test fold with the source model, and write the result.

    Raises `RunVerificationError` — `TransferError` where the runs cannot be
    paired — before anything is written if either run does not verify.
    """
    if source_run == target_run:
        raise TransferError(
            f"Run {source_run!r} cannot be both the source and the target of a transfer."
        )
    source = read_recorded_run(source_run, runs_root=runs_root)
    target = read_recorded_run(target_run, runs_root=runs_root)

    # Everything that can be refused from the records is refused before any data is loaded.
    check_pairing(source, target)
    check_recorded_run(source)
    check_recorded_run(target)
    _require_source_model_evidence(source)
    _require_target_fold_evidence(target)

    verified_source, source_model = _load_and_verify(source, source_data or DataLocation())
    verified_target, _ = _load_and_verify(target, target_data or DataLocation())

    test_rows = verified_target.splits.test
    target_features = verified_target.ds.X[test_rows]
    if verified_target.ds.feature_names != source_model.feature_names:
        raise TransferError(
            f"Target run {target.name!r} loads features {verified_target.ds.feature_names}, but "
            f"the source model takes {source_model.feature_names}."
        )
    scores = source_model.predict_proba_batch(target_features)

    evaluation = evaluate_test_fold(
        # The source run's folds: its model was fitted on the one and its
        # threshold chosen on the other. The payload labels them as source counts.
        train_labels=verified_source.ds.y[verified_source.splits.train],
        val_labels=verified_source.ds.y[verified_source.splits.val],
        test_labels=verified_target.ds.y[test_rows],
        test_scores=scores,
        operating_threshold=source.threshold.value,
    )
    payload = transfer_payload(
        verified_source,
        verified_target,
        evaluation,
        features=describe_features(verified_source, verified_target),
        generated_at=utc_now_iso(),
        code_version=current_git_commit(),
        library_versions=collect_library_versions(),
    )
    written = _write_json(target.directory / TRANSFER_METRICS_FILENAME, payload)
    return TransferMeasurement(
        source=verified_source,
        target=verified_target,
        scores=scores,
        evaluation=evaluation,
        payload=payload,
        written=written,
    )


def check_pairing(source: RecordedRun, target: RecordedRun) -> None:
    """Refuse two runs that cannot form a transfer, reading their records only."""
    source_dataset = source.record.dataset.name
    if source_dataset == target.record.dataset.name:
        raise TransferError(
            f"Runs {source.name!r} and {target.name!r} were both trained on dataset "
            f"{source_dataset!r}. A transfer measures a model on a dataset it was not "
            "trained on."
        )
    source_featureset = source.record.featureset_version
    target_featureset = target.record.featureset_version
    if source_featureset != target_featureset:
        raise TransferError(
            f"Source run {source.name!r} uses featureset {source_featureset!r}, but target run "
            f"{target.name!r} uses {target_featureset!r}. The source model scores only the "
            "columns it was trained on, and no mapping between featuresets is made."
        )
    if source.feature_list != target.feature_list:
        raise TransferError(
            f"Runs {source.name!r} and {target.name!r} list different features for featureset "
            f"{source_featureset!r}."
        )


def describe_features(source: VerifiedRun, target: VerifiedRun) -> dict[str, Any]:
    """How each scored column behaves in the source training fold and the target test fold.

    Read from the feature matrices only, never from labels or scores, and
    used for nothing but the record.
    """
    names = source.recorded.feature_list
    source_training = source.ds.X[source.splits.train]
    target_test = target.ds.X[target.splits.test]
    constant_in_source = set(constant_features(source_training, names))
    constant_in_target = set(constant_features(target_test, names))

    by_feature = []
    for column, name in enumerate(names):
        low = float(source_training[:, column].min())
        high = float(source_training[:, column].max())
        values = target_test[:, column]
        by_feature.append(
            {
                "feature": name,
                "constant_in_source_training_fold": name in constant_in_source,
                "constant_in_target_test_fold": name in constant_in_target,
                "source_training_fold_min": low,
                "source_training_fold_max": high,
                "target_test_fold_min": float(values.min()),
                "target_test_fold_max": float(values.max()),
                "target_test_rows_outside_source_training_range": int(
                    ((values < low) | (values > high)).sum()
                ),
            }
        )
    return {
        "featureset_version": source.recorded.record.featureset_version,
        "scored_columns": list(names),
        "constant_in_source_training_fold": [n for n in names if n in constant_in_source],
        "constant_in_target_test_fold": [n for n in names if n in constant_in_target],
        "by_feature": by_feature,
        "note": FEATURE_NOTE,
    }


def transfer_payload(
    source: VerifiedRun,
    target: VerifiedRun,
    evaluation: HoldoutEvaluation,
    *,
    features: dict[str, Any],
    generated_at: str,
    code_version: str | None,
    library_versions: dict[str, str],
) -> dict[str, Any]:
    """The contents of `transfer_metrics.json`.

    Source and target facts are named as such; no count or rate appears under
    a bare train, val or test name.
    """
    source_record = source.recorded
    target_record = target.recorded
    threshold = source_record.threshold
    identity = target_record.record.test_fold_identity
    if identity is None:  # pragma: no cover - refused before any data is loaded
        raise TransferError(f"Target run {target_record.name!r} does not identify its test fold.")
    context = evaluation.context
    realised_fpr = context.realised_fpr_on_test_at_operating_threshold

    return {
        "transfer_metrics_version": TRANSFER_METRICS_VERSION,
        "source_run": source_record.name,
        "target_run": target_record.name,
        "method_note": METHOD_NOTE,
        "source_dataset": _dataset(source_record.record.dataset),
        "target_dataset": _dataset(target_record.record.dataset),
        "source_featureset": source_record.record.featureset_version,
        "target_featureset": target_record.record.featureset_version,
        "source_model_sha256": source_record.model_sha256,
        "source_threshold": {
            "value": threshold.value,
            "recorded_in": f"threshold.json of source run {source_record.name!r}",
            "selected_on": f"the val fold of source run {source_record.name!r}",
            "selection_method": (
                FALLBACK_THRESHOLD_METHOD if threshold.fallback_used else SELECTED_THRESHOLD_METHOD
            ),
            "fpr_ceiling_on_source_val": threshold.target_fpr,
            "realised_fpr_on_source_val": threshold.realised_fpr_on_val,
            "fallback_used": threshold.fallback_used,
        },
        "threshold_free": {
            "pr_auc": evaluation.pr_auc,
            "roc_auc": evaluation.roc_auc,
            "recall_at_1pct_fpr": evaluation.recall_at_1pct_fpr,
            "recall_at_5pct_fpr": evaluation.recall_at_5pct_fpr,
            "threshold_source": {
                "recall_at_1pct_fpr": TARGET_TEST_ROC_CURVE_SOURCE,
                "recall_at_5pct_fpr": TARGET_TEST_ROC_CURVE_SOURCE,
            },
            "note": THRESHOLD_FREE_NOTE,
        },
        "at_source_threshold": {
            **evaluation.at_operating_threshold.as_dict(),
            "realised_fpr_on_target_test": realised_fpr,
            "threshold_source": SOURCE_RUN_THRESHOLD_SOURCE,
            "note": AT_SOURCE_THRESHOLD_NOTE,
        },
        # The conditions the results were measured under, not further results.
        "context": {
            "target_test_rows": len(target.splits.test),
            "target_test_prevalence": context.test_prevalence,
            "target_test_fraud_count": context.test_fraud_count,
            "source_train_fraud_count": context.train_fraud_count,
            "source_val_fraud_count": context.val_fraud_count,
            "realised_fpr_on_target_test_at_source_threshold": realised_fpr,
            "source_threshold_fallback_used": threshold.fallback_used,
            "label_delay_note": LABEL_DELAY_NOTE,
            "target_fold_periods": [period.to_dict() for period in target_record.record.splits],
            "target_test_transaction_count": identity.transaction_count,
            "target_test_transaction_ids_sha256": identity.transaction_ids_sha256,
            "target_test_transaction_ids_digest_definition": TRANSACTION_IDS_DIGEST_DEFINITION,
        },
        "features": features,
        "verification": {
            "source": _verification(source),
            "target": _verification(target),
        },
        "provenance": {
            "generated_at_utc": generated_at,
            "code_version": code_version,
            "library_versions": dict(library_versions),
            "source_code_version": source_record.record.code_version,
            "target_code_version": target_record.record.code_version,
        },
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _require_source_model_evidence(source: RecordedRun) -> None:
    if source.model_sha256 is None:
        raise TransferError(
            f"Source run {source.name!r} records no digest of its model.json, so the model that "
            "would score the target cannot be shown to be the one the run saved. Retrain the run."
        )
    if source.threshold.fallback_used is None:
        raise TransferError(
            f"Source run {source.name!r}: threshold.json does not record whether its threshold "
            "fell back, so the threshold cannot be reported as selected or as the fallback. "
            "Retrain the run."
        )


def _require_target_fold_evidence(target: RecordedRun) -> None:
    if target.record.test_fold_identity is None:
        raise TransferError(
            f"Target run {target.name!r} does not identify its test fold's transactions "
            f"(run.json version {target.record.metadata_version}), so the rows scored cannot be "
            "shown to be the rows it was evaluated on. Retrain the run."
        )


def _load_and_verify(
    recorded: RecordedRun, location: DataLocation
) -> tuple[VerifiedRun, FraudExplainer]:
    """Rebuild a run's data from its record and verify it with its own saved model."""
    try:
        request = DataRequest.from_run_record(
            recorded.record,
            root=location.root,
            cache_root=location.cache_root,
            csv_path=location.csv_path,
            limit=location.limit,
            full_corpus=location.full_corpus,
        )
    except ValueError as exc:
        raise RunVerificationError(f"Run {recorded.name!r} cannot be reloaded: {exc}") from exc

    data = load_run_data(request, with_provenance=True)
    explainer = load_explainer(recorded.directory)
    return verify_run(recorded, data, explainer), explainer


def _dataset(provenance: DatasetProvenance) -> dict[str, Any]:
    return {
        "name": provenance.name,
        "version": provenance.version,
        "origin": provenance.origin.value,
        "files": dict(provenance.files),
        "subsample": None if provenance.subsample is None else provenance.subsample.to_dict(),
    }


def _verification(verified: VerifiedRun) -> dict[str, Any]:
    return {
        "run": verified.recorded.name,
        "metrics_reproduced": True,
        "model_digest_verified": verified.model_digest_verified,
        "test_fold_identity_verified": verified.test_fold_identity_verified,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure a run's frozen model on another run's test fold"
    )
    parser.add_argument(
        "--source-run",
        required=True,
        help="Run whose saved model and threshold are applied unchanged",
    )
    parser.add_argument(
        "--target-run",
        required=True,
        help="Run whose test fold is scored; transfer_metrics.json is written into it",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=FEATURE_CACHE_DIR,
        help="Feature cache directory for registered datasets",
    )
    for side in ("source", "target"):
        group = parser.add_argument_group(f"{side} run data")
        group.add_argument(
            f"--{side}-root",
            type=Path,
            default=None,
            help="Directory holding a registered dataset's files (default: ml/data/raw/<dataset>)",
        )
        group.add_argument(
            f"--{side}-csv-path",
            type=Path,
            default=DEFAULT_SYNTHETIC_CSV,
            help="Synthetic label CSV",
        )
        group.add_argument(
            f"--{side}-limit",
            type=int,
            default=None,
            help="For a synthetic run, the row limit it was trained with",
        )
        group.add_argument(
            f"--{side}-full-corpus",
            action="store_true",
            help="Allow an unsubsampled load where the adapter guards against one",
        )

    args = parser.parse_args(argv)
    for flag, name in (("--source-run", args.source_run), ("--target-run", args.target_run)):
        try:
            validate_run_name(name)
        except InvalidRunNameError as exc:
            parser.error(f"{flag}: {exc}")
    if args.source_run == args.target_run:
        parser.error("--source-run and --target-run name the same run.")
    return args


def _location(args: argparse.Namespace, side: str) -> DataLocation:
    return DataLocation(
        root=getattr(args, f"{side}_root"),
        cache_root=args.cache_root,
        csv_path=getattr(args, f"{side}_csv_path"),
        limit=getattr(args, f"{side}_limit"),
        full_corpus=getattr(args, f"{side}_full_corpus"),
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    args = parse_args(argv)
    try:
        measurement = measure_transfer(
            args.source_run,
            args.target_run,
            runs_root=RUNS_ROOT,
            source_data=_location(args, "source"),
            target_data=_location(args, "target"),
        )
    except RunVerificationError as exc:
        log.error(
            "Transfer from %s to %s was not measured: %s", args.source_run, args.target_run, exc
        )
        sys.exit(1)
    _log_summary(measurement)


def _log_summary(measurement: TransferMeasurement) -> None:
    payload = measurement.payload
    free = payload["threshold_free"]
    at_threshold = payload["at_source_threshold"]
    context = payload["context"]
    log.info(
        "=== Transfer: model of %s on the test fold of %s ===",
        payload["source_run"],
        payload["target_run"],
    )
    log.info(
        "  PR-AUC   : %.4f  (target test prevalence %.4f)",
        free["pr_auc"],
        context["target_test_prevalence"],
    )
    log.info("  ROC-AUC  : %.4f", free["roc_auc"])
    log.info(
        "  Recall @ 1%% FPR : %.4f  (point on the target test ROC curve)",
        free["recall_at_1pct_fpr"],
    )
    log.info(
        "  Recall @ 5%% FPR : %.4f  (point on the target test ROC curve)",
        free["recall_at_5pct_fpr"],
    )
    log.info(
        "  At the source threshold %.4f%s: precision=%.4f  recall=%.4f  f1=%.4f",
        at_threshold["threshold"],
        " (the source run's fallback)" if context["source_threshold_fallback_used"] else "",
        at_threshold["precision"],
        at_threshold["recall"],
        at_threshold["f1"],
    )
    log.info(
        "  Realised FPR on the target test fold at the source threshold: %s",
        context["realised_fpr_on_target_test_at_source_threshold"],
    )
    log.info("Wrote %s", measurement.written)


if __name__ == "__main__":
    main()
