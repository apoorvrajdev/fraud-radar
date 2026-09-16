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
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb

from app.db import SessionLocal
from ml.artifacts import (
    ThresholdRecord,
    TrainingMetadata,
    collect_library_versions,
    save_artifacts,
    utc_now_iso,
)
from ml.data import LabelledDataset, load_dataset_with_csv_labels
from ml.evaluation import (
    confusion_at_threshold,
    find_threshold_at_fpr,
    pr_auc,
    recall_at_fpr,
    roc_auc,
    save_pr_curve_png,
)
from ml.splits import SplitIndices, assert_no_temporal_leakage, chronological_split
from ml.tuning import TuningResult, compute_scale_pos_weight, tune_hyperparameters

log = logging.getLogger("train")

RANDOM_STATE = 42

# Goals set against the in-house generator. They are written into the
# synthetic model's metrics.json only; the shared pipeline reports what it
# measured, because a target inherited from another dataset describes nothing
# about a benchmark run.
SYNTHETIC_TARGETS: dict[str, float] = {
    "target_pr_auc": 0.75,
    "target_recall_at_1pct_fpr": 0.60,
}


@dataclass(frozen=True)
class TrainingOutcome:
    """What one split → tune → fit → threshold → evaluate pass produced.

    Nothing in it has been written anywhere. `metrics` holds observed results
    only; `test_scores` are the model's scores for the test fold, in the order
    of `splits.test`.
    """

    splits: SplitIndices
    tuning: TuningResult
    model: xgb.XGBClassifier
    threshold: ThresholdRecord
    test_scores: np.ndarray
    metrics: dict[str, object]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Fraud Radar XGBoost model")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("ml/data/synthetic_transactions.csv"),
        help="Path to the synthetic CSV with ground-truth labels",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("ml/artifacts"),
        help="Where to write model.json, metrics.json, etc.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: cap dataset size for a fast smoke test",
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
    return parser.parse_args(argv)


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
    log.info("Final fit with early_stopping_rounds=50 against val set...")

    model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_STATE,
        early_stopping_rounds=50,
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

    # ---- Threshold selection on val -------------------------------------
    val_scores = model.predict_proba(X_val)[:, 1]
    threshold_value = find_threshold_at_fpr(y_val, val_scores, target_fpr)
    if not np.isfinite(threshold_value):
        log.warning("No threshold satisfies target FPR ≤ %.4f on val", target_fpr)
        threshold_value = 0.5
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

    metrics: dict[str, object] = {
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
        "recall_at_1pct_fpr": recall_at_1pct,
        "recall_at_5pct_fpr": recall_at_5pct,
        "at_operating_threshold": confusion.as_dict(),
        "best_cv_pr_auc": tuning.best_score,
    }
    log.info("=== Test-set evaluation ===")
    log.info("  PR-AUC              : %.4f", test_pr_auc)
    log.info("  ROC-AUC             : %.4f", test_roc_auc)
    log.info("  Recall @ 1%% FPR     : %.4f", recall_at_1pct)
    log.info("  Recall @ 5%% FPR     : %.4f", recall_at_5pct)
    log.info(
        "  @ threshold %.4f  : precision=%.4f  recall=%.4f  f1=%.4f",
        threshold_value,
        confusion.precision,
        confusion.recall,
        confusion.f1,
    )

    return TrainingOutcome(
        splits=splits,
        tuning=tuning,
        model=model,
        threshold=ThresholdRecord(
            value=float(threshold_value),
            target_fpr=float(target_fpr),
            realised_fpr_on_val=realised_val_fpr,
        ),
        test_scores=test_scores,
        metrics=metrics,
    )


def training_metadata(ds: LabelledDataset, outcome: TrainingOutcome) -> TrainingMetadata:
    """The fit record for `outcome`: fold sizes, fold fraud rates, chosen hyperparameters."""
    splits = outcome.splits
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
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    args = parse_args(argv)
    log.info("Fraud Radar training run started — random_state=%d", RANDOM_STATE)

    # ---- Load -----------------------------------------------------------
    with SessionLocal() as db:
        ds = load_dataset_with_csv_labels(
            db,
            csv_path=str(args.csv_path),
            limit=args.limit,
        )
    log.info(
        "Dataset loaded: %d rows, %d features, overall fraud rate %.3f%%",
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

    # ---- Save artifacts -------------------------------------------------
    save_artifacts(
        args.artifact_dir,
        model=outcome.model,
        feature_names=ds.feature_names,
        threshold=outcome.threshold,
        metrics={**outcome.metrics, **SYNTHETIC_TARGETS},
        metadata=training_metadata(ds, outcome),
    )
    save_pr_curve_png(
        ds.y[outcome.splits.test],
        outcome.test_scores,
        args.artifact_dir / "pr_curve.png",
        title="Fraud Radar — test-set PR curve",
    )
    log.info("Artifacts written to %s", args.artifact_dir.resolve())
    log.info("Training run complete.")


if __name__ == "__main__":
    main()
