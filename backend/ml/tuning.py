"""Randomised hyperparameter search for XGBoost on the fraud task.

`RandomizedSearchCV` is the right tool here because the search space is
medium-sized (eight axes) and only a handful of regions actually matter —
sampling 25 points covers the basin of attraction without exhausting compute.
Scoring is `average_precision` (PR-AUC) because the positive class is rare
(~1.5%) and PR-AUC is the metric that survives extreme class imbalance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

log = logging.getLogger(__name__)


# Search space defined inline so it is version-controlled with the trainer.
# Each axis is informed by the XGBoost docs and standard fraud-detection practice.
SEARCH_SPACE: dict[str, list[Any]] = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [200, 400, 600, 800],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [1.0, 3.0, 10.0],
}


@dataclass(frozen=True)
class TuningResult:
    """The best estimator from a hyperparameter sweep, plus its score."""

    best_params: dict[str, Any]
    best_score: float
    cv_results_summary: dict[str, list[float]]


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Return neg_count / pos_count from the training-fold labels only.

    Class-imbalance handling for tree boosters. This is preferred over SMOTE
    because it keeps every real row real — no synthetic positives that could
    mislead the model into picking up interpolation artefacts.
    """
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0:
        raise ValueError("Cannot compute scale_pos_weight: training set has no positives.")
    return float(neg) / float(pos)


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_iter: int = 25,
    n_splits: int = 4,
    random_state: int = 42,
    n_jobs: int = 1,
    verbose: int = 1,
) -> TuningResult:
    """Run RandomizedSearchCV over SEARCH_SPACE and return the best params.

    The search uses StratifiedKFold so each fold keeps the ~1.5% fraud share,
    and scoring is average_precision (PR-AUC). Refit is left enabled (default)
    so the final estimator is also returned, though `train.py` does its own
    early-stopped fit afterwards on the train+val split.
    """
    scale_pos_weight = compute_scale_pos_weight(y_train)
    log.info("scale_pos_weight (train fold): %.2f", scale_pos_weight)

    base_estimator = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=SEARCH_SPACE,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )

    log.info(
        "Starting RandomizedSearchCV: n_iter=%d, n_splits=%d, scoring=average_precision",
        n_iter,
        n_splits,
    )
    search.fit(X_train, y_train)

    log.info("Best CV score (PR-AUC): %.4f", search.best_score_)
    log.info("Best params: %s", search.best_params_)

    return TuningResult(
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        cv_results_summary={
            "mean_test_score": list(map(float, search.cv_results_["mean_test_score"])),
            "std_test_score": list(map(float, search.cv_results_["std_test_score"])),
        },
    )
