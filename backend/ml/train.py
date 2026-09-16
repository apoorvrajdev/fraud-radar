"""End-to-end training orchestrator for the Fraud Radar XGBoost classifier.

Usage:
    cd backend
    uv run python -m ml.train
    uv run python -m ml.train --limit 5000   # quick smoke-test run

Steps:
    1. Load features + labels (DB rows + synthetic CSV labels)
    2. Chronological 70/15/15 split
    3. RandomizedSearchCV on train only (PR-AUC scoring)
    4. Refit best params on train with early stopping against val
    5. Pick threshold on val at FPR ≤ 0.01
    6. Evaluate on the held-out test set
    7. Save all artifacts and the PR-curve PNG

Steps 2–6 are `train_and_evaluate`, which takes a `LabelledDataset` and knows
nothing about where it came from. Loading and writing stay in `main`, so any
dataset that can produce the matrix goes through exactly the same procedure.
Loading itself is `ml.loading.load_run_data`, which analysis also uses to
reconstruct a run from its `run.json`.

A registered benchmark dataset is loaded through its adapter and the feature
cache instead, and always trains into its own run directory:

    uv run python -m ml.train --dataset sparkov --max-cards 200 \
        --run-name sparkov_v1_200cards
    uv run python -m ml.train --run-name synthetic_v1

A named run writes the artifacts plus `run.json` to
`ml/artifacts/runs/<run-name>/`. Without a name, the synthetic model is written
to `--artifact-dir` as before. Training never writes a benchmark model over the
served artifacts; that is promotion's job.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb

import ml.datasets.sparkov  # noqa: F401  (registers the adapter)
from app.fraud.feature_spec import DEFAULT_FEATURESET, FEATURESETS
from ml.artifacts import (
    ThresholdRecord,
    TrainingMetadata,
    collect_library_versions,
    save_artifacts,
    utc_now_iso,
)
from ml.data import SYNTHETIC_DATASET_NAME, LabelledDataset
from ml.datasets.base import DatasetProvenance
from ml.datasets.registry import available_datasets
from ml.evaluation import (
    confusion_at_threshold,
    find_threshold_at_fpr,
    pr_auc,
    recall_at_fpr,
    roc_auc,
    save_pr_curve_png,
)
from ml.loading import (
    DEFAULT_SUBSAMPLE_SEED,
    DEFAULT_SYNTHETIC_CSV,
    DataRequest,
    RunData,
    load_run_data,
)
from ml.paths import (
    FEATURE_CACHE_DIR,
    RUNS_ROOT,
    InvalidRunNameError,
    run_dir,
    validate_run_name,
)
from ml.reporting import (
    OPERATING_THRESHOLD_SOURCE,
    TEST_ROC_CURVE_SOURCE,
    LiveFeatures,
    live_features,
    result_context,
)
from ml.runs import RunMetadata, current_git_commit, save_run_metadata, split_periods
from ml.splits import SplitIndices, assert_no_temporal_leakage, chronological_split
from ml.tuning import TuningResult, compute_scale_pos_weight, tune_hyperparameters

log = logging.getLogger("train")

RANDOM_STATE = 42

# Boosting stops once this many rounds pass without improving the val fold's PR-AUC.
EARLY_STOPPING_ROUNDS = 50

# The operating threshold when no val-fold threshold meets the target FPR. A
# threshold that falls back is recorded as one in threshold.json.
FALLBACK_THRESHOLD = 0.5

# Goals set against the in-house generator. They are written into the
# synthetic model's metrics.json only; the shared pipeline reports what it
# measured, because a target inherited from another dataset describes nothing
# about a benchmark run.
SYNTHETIC_TARGETS: dict[str, float] = {
    "target_pr_auc": 0.75,
    "target_recall_at_1pct_fpr": 0.60,
}


@dataclass(frozen=True)
class FitRecord:
    """The settings the fit ran with, and the round early stopping chose.

    Taken from the tuner's arguments and the fitted model's own parameters,
    so it describes the fit that happened rather than the one intended.
    """

    random_state: int
    tuning_iterations: int
    tuning_cv_folds: int
    scale_pos_weight: float
    early_stopping_rounds: int
    best_iteration: int


@dataclass(frozen=True)
class TrainingOutcome:
    """What one split → tune → fit → threshold → evaluate pass produced.

    Nothing in it has been written anywhere. `metrics` holds observed results
    only; `test_scores` are the model's scores for the test fold, in the order
    of `splits.test`. `live_features` is counted on the training fold.
    """

    splits: SplitIndices
    live_features: LiveFeatures
    tuning: TuningResult
    model: xgb.XGBClassifier
    fit: FitRecord
    threshold: ThresholdRecord
    test_scores: np.ndarray
    metrics: dict[str, object]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Fraud Radar XGBoost model")
    parser.add_argument(
        "--dataset",
        default=SYNTHETIC_DATASET_NAME,
        choices=[SYNTHETIC_DATASET_NAME, *available_datasets()],
        help="'synthetic' (database + label CSV, the default) or a registered benchmark dataset",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_SYNTHETIC_CSV,
        help="Path to the synthetic CSV with ground-truth labels",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("ml/artifacts"),
        help="Where to write model.json, metrics.json, etc.",
    )
    output.add_argument(
        "--run-name",
        default=None,
        help=(
            "Write a complete run, with its run.json, to ml/artifacts/runs/<run-name>/ "
            "(required for registered datasets)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: cap dataset size for a fast smoke test (synthetic only)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=25,
        help="RandomizedSearchCV iterations",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=4,
        help="StratifiedKFold splits inside RandomizedSearchCV",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="Operating-point FPR ceiling (e.g. 0.01 = 1% false-positive rate)",
    )

    benchmark = parser.add_argument_group("registered datasets")
    benchmark.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Directory holding the source files (default: ml/data/raw/<dataset>)",
    )
    benchmark.add_argument(
        "--max-cards",
        type=int,
        default=None,
        help="Keep only N entities, with their full histories (default: all)",
    )
    benchmark.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SUBSAMPLE_SEED,
        help="Seed for entity subsampling",
    )
    benchmark.add_argument(
        "--featureset",
        default=DEFAULT_FEATURESET,
        choices=sorted(FEATURESETS),
        help="Featureset version to extract",
    )
    benchmark.add_argument(
        "--cache-root",
        type=Path,
        default=FEATURE_CACHE_DIR,
        help="Feature cache directory",
    )
    benchmark.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild the feature matrix even if a cache entry exists",
    )
    benchmark.add_argument(
        "--full-corpus",
        action="store_true",
        help="Allow an unsubsampled load where the adapter guards against one",
    )

    args = parser.parse_args(argv)
    _check_arguments(parser, args)
    return args


def _check_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse combinations that would silently mean something other than they say."""
    if args.run_name is not None:
        try:
            validate_run_name(args.run_name)
        except InvalidRunNameError as exc:
            parser.error(str(exc))

    if args.dataset == SYNTHETIC_DATASET_NAME:
        benchmark_only = [
            flag
            for flag, given in (
                ("--root", args.root is not None),
                ("--max-cards", args.max_cards is not None),
                ("--seed", args.seed != DEFAULT_SUBSAMPLE_SEED),
                ("--featureset", args.featureset != DEFAULT_FEATURESET),
                ("--cache-root", args.cache_root != FEATURE_CACHE_DIR),
                ("--refresh", args.refresh),
                ("--full-corpus", args.full_corpus),
            )
            if given
        ]
        if benchmark_only:
            parser.error(
                f"{', '.join(benchmark_only)} apply to registered datasets only; the "
                "synthetic dataset is read from the database and its label CSV."
            )
        return

    if args.run_name is None:
        parser.error(
            f"--dataset {args.dataset} requires --run-name: a benchmark run is written "
            "to its own run directory, never over the served artifacts."
        )
    if args.limit is not None:
        parser.error(
            "--limit applies to the synthetic dataset only: truncating rows would cut "
            "entity histories short. Use --max-cards to subsample whole entities."
        )


def _slice(ds: LabelledDataset, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for an index slice."""
    return ds.X[idx], ds.y[idx]


def _fraud_rate(y: np.ndarray) -> float:
    return float(y.mean()) if len(y) else 0.0


def _final_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: dict[str, object],
) -> xgb.XGBClassifier:
    """Refit best params on train with early stopping against val.

    XGBoost recent versions accept `early_stopping_rounds` as a constructor
    argument (preferred) rather than a `fit()` keyword.
    """
    scale_pos_weight = compute_scale_pos_weight(y_train)
    log.info("Final fit with early_stopping_rounds=%d against val set...", EARLY_STOPPING_ROUNDS)

    model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_STATE,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None:
        log.info("Best iteration: %d", best_iter)
    return model


def train_and_evaluate(
    ds: LabelledDataset,
    *,
    n_iter: int,
    cv_splits: int,
    target_fpr: float,
) -> TrainingOutcome:
    """Split, tune, fit, choose the operating threshold, then evaluate.

    Each fold has one job. Hyperparameters are searched on train only; early
    stopping and the threshold use val only; test is scored once, after every
    choice has been made.
    """
    # ---- Split ----------------------------------------------------------
    splits = chronological_split(ds.timestamps)
    assert_no_temporal_leakage(ds.timestamps, splits)
    log.info("Split sizes  train=%d  val=%d  test=%d", *splits.sizes)
    X_train, y_train = _slice(ds, splits.train)
    X_val, y_val = _slice(ds, splits.val)
    X_test, y_test = _slice(ds, splits.test)
    log.info(
        "Fraud rates  train=%.3f%%  val=%.3f%%  test=%.3f%%",
        _fraud_rate(y_train) * 100,
        _fraud_rate(y_val) * 100,
        _fraud_rate(y_test) * 100,
    )
    live = live_features(
        training_fold=X_train,
        whole_matrix=ds.X,
        feature_names=ds.feature_names,
    )
    log.info(
        "Live features  %d of %d in the training fold; constant there: %s",
        live.live_count,
        len(ds.feature_names),
        ", ".join(live.constant) or "none",
    )
    if live.live_count_whole_matrix != live.live_count:
        log.info(
            "Live features  %d of %d over the whole matrix",
            live.live_count_whole_matrix,
            len(ds.feature_names),
        )

    # ---- Tune -----------------------------------------------------------
    tuning = tune_hyperparameters(
        X_train,
        y_train,
        n_iter=n_iter,
        n_splits=cv_splits,
        random_state=RANDOM_STATE,
    )

    # ---- Final fit ------------------------------------------------------
    model = _final_fit(X_train, y_train, X_val, y_val, tuning.best_params)
    fitted_with = model.get_params()
    fit = FitRecord(
        random_state=RANDOM_STATE,
        tuning_iterations=n_iter,
        tuning_cv_folds=cv_splits,
        scale_pos_weight=float(fitted_with["scale_pos_weight"]),
        early_stopping_rounds=int(fitted_with["early_stopping_rounds"]),
        best_iteration=int(model.best_iteration),
    )

    # ---- Threshold selection on val -------------------------------------
    val_scores = model.predict_proba(X_val)[:, 1]
    threshold_value = find_threshold_at_fpr(y_val, val_scores, target_fpr)
    fallback_used = not np.isfinite(threshold_value)
    if fallback_used:
        log.warning(
            "No threshold satisfies target FPR ≤ %.4f on val; falling back to %.2f",
            target_fpr,
            FALLBACK_THRESHOLD,
        )
        threshold_value = FALLBACK_THRESHOLD
    realised_val_fpr = float(
        ((val_scores >= threshold_value) & (y_val == 0)).sum() / max((y_val == 0).sum(), 1)
    )
    log.info(
        "Operating threshold = %.4f (target FPR %.4f, realised on val %.4f)",
        threshold_value,
        target_fpr,
        realised_val_fpr,
    )

    # ---- Evaluate on test -----------------------------------------------
    test_scores = model.predict_proba(X_test)[:, 1]
    test_pr_auc = pr_auc(y_test, test_scores)
    test_roc_auc = roc_auc(y_test, test_scores)
    recall_at_1pct, _ = recall_at_fpr(y_test, test_scores, target_fpr=0.01)
    recall_at_5pct, _ = recall_at_fpr(y_test, test_scores, target_fpr=0.05)
    confusion = confusion_at_threshold(y_test, test_scores, threshold_value)
    # The conditions the numbers above were measured under, not further metrics.
    context = result_context(
        train_labels=y_train,
        val_labels=y_val,
        test_labels=y_test,
        at_operating_threshold=confusion,
    )

    metrics: dict[str, object] = {
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
        "recall_at_1pct_fpr": recall_at_1pct,
        "recall_at_5pct_fpr": recall_at_5pct,
        "at_operating_threshold": confusion.as_dict(),
        "best_cv_pr_auc": tuning.best_score,
        "threshold_source": {
            "at_operating_threshold": OPERATING_THRESHOLD_SOURCE,
            "recall_at_1pct_fpr": TEST_ROC_CURVE_SOURCE,
            "recall_at_5pct_fpr": TEST_ROC_CURVE_SOURCE,
        },
        "context": context.to_dict(),
    }
    log.info("=== Test-set evaluation ===")
    log.info("  PR-AUC              : %.4f", test_pr_auc)
    log.info("  ROC-AUC             : %.4f", test_roc_auc)
    log.info("  Recall @ 1%% FPR     : %.4f  (point on the test ROC curve)", recall_at_1pct)
    log.info("  Recall @ 5%% FPR     : %.4f  (point on the test ROC curve)", recall_at_5pct)
    log.info(
        "  @ threshold %.4f  : precision=%.4f  recall=%.4f  f1=%.4f",
        threshold_value,
        confusion.precision,
        confusion.recall,
        confusion.f1,
    )
    log.info(
        "  measured at test prevalence %.4f; frauds train=%d  val=%d  test=%d",
        context.test_prevalence,
        context.train_fraud_count,
        context.val_fraud_count,
        context.test_fraud_count,
    )
    log.info(
        "  realised test FPR at the operating threshold: %s",
        context.realised_fpr_on_test_at_operating_threshold,
    )

    return TrainingOutcome(
        splits=splits,
        live_features=live,
        tuning=tuning,
        model=model,
        fit=fit,
        threshold=ThresholdRecord(
            value=float(threshold_value),
            target_fpr=float(target_fpr),
            realised_fpr_on_val=realised_val_fpr,
            fallback_used=fallback_used,
        ),
        test_scores=test_scores,
        metrics=metrics,
    )


def training_metadata(ds: LabelledDataset, outcome: TrainingOutcome) -> TrainingMetadata:
    """The fit record for `outcome`.

    Fold sizes, fold fraud rates and chosen hyperparameters, plus the settings
    the fit ran with, the round early stopping chose, and the live features.
    """
    splits = outcome.splits
    fit = outcome.fit
    live = outcome.live_features
    return TrainingMetadata(
        trained_at_utc=utc_now_iso(),
        dataset_size=ds.n_rows,
        train_size=int(splits.sizes[0]),
        val_size=int(splits.sizes[1]),
        test_size=int(splits.sizes[2]),
        train_fraud_rate=_fraud_rate(ds.y[splits.train]),
        val_fraud_rate=_fraud_rate(ds.y[splits.val]),
        test_fraud_rate=_fraud_rate(ds.y[splits.test]),
        best_hyperparameters=outcome.tuning.best_params,
        library_versions=collect_library_versions(),
        random_state=fit.random_state,
        tuning_iterations=fit.tuning_iterations,
        tuning_cv_folds=fit.tuning_cv_folds,
        scale_pos_weight=fit.scale_pos_weight,
        early_stopping_rounds=fit.early_stopping_rounds,
        best_iteration=fit.best_iteration,
        live_feature_count=live.live_count,
        constant_features=list(live.constant),
        live_feature_count_whole_matrix=live.live_count_whole_matrix,
        constant_features_whole_matrix=list(live.constant_whole_matrix),
    )


def write_run(
    run_name: str,
    ds: LabelledDataset,
    outcome: TrainingOutcome,
    *,
    dataset: DatasetProvenance,
    featureset: str,
    metrics: dict[str, object],
    notes: str = "",
    runs_root: Path = RUNS_ROOT,
) -> Path:
    """Write a run's artifacts and its `run.json` into its own run directory.

    The run record is assembled before anything is written, so an invalid
    name, provenance or fold period refuses the run instead of leaving a
    partial directory behind. Its seed comes from the dataset's subsample
    record; the model's random state is written with the fit.
    """
    record = RunMetadata(
        run_name=run_name,
        dataset=dataset,
        featureset_version=featureset,
        code_version=current_git_commit(),
        library_versions=collect_library_versions(),
        splits=split_periods(ds.timestamps, outcome.splits),
        notes=notes,
    )
    directory = run_dir(run_name, runs_root=runs_root)
    _write_artifacts(directory, ds, outcome, metrics)
    save_run_metadata(record, runs_root=runs_root)
    return directory


def _write_artifacts(
    directory: Path,
    ds: LabelledDataset,
    outcome: TrainingOutcome,
    metrics: dict[str, object],
) -> None:
    save_artifacts(
        directory,
        model=outcome.model,
        feature_names=ds.feature_names,
        threshold=outcome.threshold,
        metrics=metrics,
        metadata=training_metadata(ds, outcome),
    )
    save_pr_curve_png(
        ds.y[outcome.splits.test],
        outcome.test_scores,
        directory / "pr_curve.png",
        title="Fraud Radar — test-set PR curve",
    )


def data_request(args: argparse.Namespace) -> DataRequest:
    """The load the command-line options describe."""
    return DataRequest(
        dataset=args.dataset,
        featureset=args.featureset,
        csv_path=args.csv_path,
        limit=args.limit,
        root=args.root,
        max_entities=args.max_cards,
        seed=args.seed,
        cache_root=args.cache_root,
        refresh=args.refresh,
        full_corpus=args.full_corpus,
    )


def _run_notes(data: RunData) -> str:
    if data.cache_fingerprint is None:
        return (
            "No subsampling applies to the synthetic dataset, so seed is null. The model "
            "random_state is recorded with the fit, in training_metadata.json."
        )
    return (
        f"Feature matrix {'read from' if data.cache_hit else 'written to'} cache "
        f"{data.cache_fingerprint}. seed is the entity-subsampling seed; the model "
        "random_state is recorded with the fit, in training_metadata.json."
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    args = parse_args(argv)
    log.info("Fraud Radar training run started — random_state=%d", RANDOM_STATE)

    # ---- Load -----------------------------------------------------------
    request = data_request(args)
    # Only a named run writes a run record, so only it needs the provenance.
    data = load_run_data(request, with_provenance=args.run_name is not None)
    # A benchmark records what it measured, never the synthetic-era targets.
    targets = SYNTHETIC_TARGETS if request.is_synthetic else {}
    ds = data.ds
    log.info(
        "Dataset %s loaded: %d rows, %d features, overall fraud rate %.3f%%",
        args.dataset,
        ds.n_rows,
        ds.X.shape[1],
        ds.fraud_rate * 100,
    )

    outcome = train_and_evaluate(
        ds,
        n_iter=args.n_iter,
        cv_splits=args.cv_splits,
        target_fpr=args.target_fpr,
    )
    metrics = {**outcome.metrics, **targets}

    # ---- Save -----------------------------------------------------------
    if args.run_name is None:
        _write_artifacts(args.artifact_dir, ds, outcome, metrics)
        log.info("Artifacts written to %s", args.artifact_dir.resolve())
    else:
        if data.provenance is None:  # pragma: no cover - every loader supplies one for a named run
            raise RuntimeError(f"Run {args.run_name!r} has no provenance record to write.")
        directory = write_run(
            args.run_name,
            ds,
            outcome,
            dataset=data.provenance,
            featureset=args.featureset,
            metrics=metrics,
            notes=_run_notes(data),
            runs_root=RUNS_ROOT,
        )
        log.info("Run %s written to %s", args.run_name, directory.resolve())
    log.info("Training run complete.")


if __name__ == "__main__":
    main()
