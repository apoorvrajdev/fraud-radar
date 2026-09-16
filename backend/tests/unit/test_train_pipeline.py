"""Phase 5D — the train-and-evaluate pipeline, separated from its data source.

`train_and_evaluate` is the procedure the synthetic model and every benchmark
run go through. These tests pin which fold each step may see: hyperparameters
are searched on train only, early stopping and the operating threshold use val
only, and test is scored once, at the end.

The dataset is a small generated matrix whose first column is each row's own
index, so any slice a step receives identifies exactly which rows it saw. Its
timestamps are shuffled against row order, so slicing by position instead of
by time would put the wrong rows in a fold and fail here. The tuner is stubbed:
the search itself belongs to `ml.tuning`, and nothing here depends on its result.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml import train
from ml.data import LabelledDataset
from ml.evaluation import (
    confusion_at_threshold,
    find_threshold_at_fpr,
    pr_auc,
    recall_at_fpr,
    roc_auc,
)
from ml.splits import chronological_split
from ml.tuning import TuningResult

N_ROWS = 400
N_FEATURES = 17
START = datetime(2019, 1, 1, tzinfo=UTC)

STUB_PARAMS: dict[str, Any] = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.3}
STUB_CV_SCORE = 0.42

OBSERVED_METRICS = {
    "test_pr_auc",
    "test_roc_auc",
    "recall_at_1pct_fpr",
    "recall_at_5pct_fpr",
    "at_operating_threshold",
    "best_cv_pr_auc",
    "context",
}


def _dataset(seed: int = 7) -> LabelledDataset:
    rng = np.random.default_rng(seed)
    y = (rng.random(N_ROWS) < 0.2).astype(np.int64)
    X = rng.normal(size=(N_ROWS, N_FEATURES))
    X[:, 1] += 2.0 * y
    X[:, 0] = np.arange(N_ROWS, dtype=np.float64)
    hours = rng.permutation(N_ROWS)
    timestamps = np.asarray([START + timedelta(hours=int(hour)) for hour in hours], dtype=object)

    ds = LabelledDataset(
        X=X,
        y=y,
        timestamps=timestamps,
        transaction_ids=[f"tx{index}" for index in range(N_ROWS)],
        feature_names=[f"f{index}" for index in range(N_FEATURES)],
    )
    splits = chronological_split(ds.timestamps)
    for fold in (splits.train, splits.val, splits.test):
        assert 0 < ds.y[fold].sum() < len(fold), "every fold needs both classes"
    return ds


def _rows(features: np.ndarray) -> np.ndarray:
    """The dataset row indices a feature slice was taken from."""
    return features[:, 0].astype(np.int64)


@pytest.fixture
def tuner_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    calls: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []

    def stub_tune(features: np.ndarray, labels: np.ndarray, **options: Any) -> TuningResult:
        calls.append((features, labels, options))
        return TuningResult(
            best_params=dict(STUB_PARAMS),
            best_score=STUB_CV_SCORE,
            cv_results_summary={"mean_test_score": [STUB_CV_SCORE], "std_test_score": [0.0]},
        )

    monkeypatch.setattr(train, "tune_hyperparameters", stub_tune)
    return calls


def _run(ds: LabelledDataset, *, target_fpr: float = 0.05) -> train.TrainingOutcome:
    return train.train_and_evaluate(ds, n_iter=3, cv_splits=2, target_fpr=target_fpr)


# ---------------------------------------------------------------------------
# Which fold each step sees
# ---------------------------------------------------------------------------


def test_the_folds_are_the_existing_chronological_split(tuner_calls: list[Any]) -> None:
    ds = _dataset()
    expected = chronological_split(ds.timestamps)

    splits = _run(ds).splits

    np.testing.assert_array_equal(splits.train, expected.train)
    np.testing.assert_array_equal(splits.val, expected.val)
    np.testing.assert_array_equal(splits.test, expected.test)
    assert max(ds.timestamps[splits.train]) <= min(ds.timestamps[splits.val])
    assert max(ds.timestamps[splits.val]) <= min(ds.timestamps[splits.test])


def test_hyperparameters_are_searched_on_the_training_fold_only(
    tuner_calls: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]],
) -> None:
    ds = _dataset()
    outcome = _run(ds)

    assert len(tuner_calls) == 1
    features, labels, options = tuner_calls[0]
    np.testing.assert_array_equal(_rows(features), outcome.splits.train)
    np.testing.assert_array_equal(labels, ds.y[outcome.splits.train])
    assert options == {"n_iter": 3, "n_splits": 2, "random_state": train.RANDOM_STATE}


def test_the_final_fit_trains_on_train_and_stops_early_against_val(
    monkeypatch: pytest.MonkeyPatch, tuner_calls: list[Any]
) -> None:
    ds = _dataset()
    seen: dict[str, Any] = {}
    real_fit = train._final_fit

    def spy(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        best_params: dict[str, object],
    ) -> Any:
        seen.update(x_train=x_train, x_val=x_val, best_params=best_params)
        return real_fit(x_train, y_train, x_val, y_val, best_params)

    monkeypatch.setattr(train, "_final_fit", spy)
    outcome = _run(ds)

    np.testing.assert_array_equal(_rows(seen["x_train"]), outcome.splits.train)
    np.testing.assert_array_equal(_rows(seen["x_val"]), outcome.splits.val)
    assert seen["best_params"] == STUB_PARAMS


def test_the_operating_threshold_is_chosen_on_validation_scores_only(
    monkeypatch: pytest.MonkeyPatch, tuner_calls: list[Any]
) -> None:
    ds = _dataset()
    seen: dict[str, Any] = {}
    real_find = train.find_threshold_at_fpr

    def spy(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
        value = real_find(y_true, y_score, target_fpr)
        seen.update(y_true=y_true, y_score=y_score, target_fpr=target_fpr, value=value)
        return value

    monkeypatch.setattr(train, "find_threshold_at_fpr", spy)
    outcome = _run(ds, target_fpr=0.05)

    val = outcome.splits.val
    np.testing.assert_array_equal(seen["y_true"], ds.y[val])
    np.testing.assert_allclose(seen["y_score"], outcome.model.predict_proba(ds.X[val])[:, 1])
    assert seen["target_fpr"] == 0.05
    assert outcome.threshold.value == seen["value"]
    assert outcome.threshold.target_fpr == 0.05


def test_realised_validation_fpr_is_measured_at_the_chosen_threshold(
    tuner_calls: list[Any],
) -> None:
    ds = _dataset()
    outcome = _run(ds)

    val = outcome.splits.val
    scores = outcome.model.predict_proba(ds.X[val])[:, 1]
    negatives = ds.y[val] == 0
    expected = ((scores >= outcome.threshold.value) & negatives).sum() / negatives.sum()
    assert outcome.threshold.realised_fpr_on_val == pytest.approx(expected)


def test_test_metrics_are_computed_on_the_test_fold(tuner_calls: list[Any]) -> None:
    ds = _dataset()
    outcome = _run(ds)

    test = outcome.splits.test
    y_test = ds.y[test]
    scores = outcome.model.predict_proba(ds.X[test])[:, 1]
    np.testing.assert_allclose(outcome.test_scores, scores)

    metrics = outcome.metrics
    assert metrics["test_pr_auc"] == pytest.approx(pr_auc(y_test, scores))
    assert metrics["test_roc_auc"] == pytest.approx(roc_auc(y_test, scores))
    assert metrics["recall_at_1pct_fpr"] == pytest.approx(recall_at_fpr(y_test, scores, 0.01)[0])
    assert metrics["recall_at_5pct_fpr"] == pytest.approx(recall_at_fpr(y_test, scores, 0.05)[0])
    assert metrics["at_operating_threshold"] == (
        confusion_at_threshold(y_test, scores, outcome.threshold.value).as_dict()
    )
    assert metrics["best_cv_pr_auc"] == STUB_CV_SCORE


# ---------------------------------------------------------------------------
# What the pipeline reports
# ---------------------------------------------------------------------------


def test_the_pipeline_reports_observed_results_without_synthetic_targets(
    tuner_calls: list[Any],
) -> None:
    assert set(_run(_dataset()).metrics) == OBSERVED_METRICS


def test_the_context_counts_the_frauds_of_each_fold_it_names(tuner_calls: list[Any]) -> None:
    ds = _dataset()
    outcome = _run(ds)
    splits = outcome.splits
    counts = {
        name: int(ds.y[getattr(splits, name)].sum()) for name in ("train", "val", "test")
    }
    assert len(set(counts.values())) == 3, "a fold read in place of another would go unseen"

    context = outcome.metrics["context"]

    assert context["fraud_counts"] == counts
    assert context["test_prevalence"] == pytest.approx(ds.y[splits.test].mean())
    metadata = train.training_metadata(ds, outcome)
    assert context["test_prevalence"] == pytest.approx(metadata.test_fraud_rate)


def test_the_realised_test_fpr_is_measured_at_the_threshold_chosen_on_val(
    tuner_calls: list[Any],
) -> None:
    ds = _dataset()
    outcome = _run(ds)

    test = outcome.splits.test
    negatives = ds.y[test] == 0
    flagged = outcome.test_scores >= outcome.threshold.value
    expected = (flagged & negatives).sum() / negatives.sum()
    context = outcome.metrics["context"]
    assert context["realised_fpr_on_test_at_operating_threshold"] == pytest.approx(expected)

    # The same threshold and rows as the confusion matrix reported beside it.
    at_threshold = outcome.metrics["at_operating_threshold"]
    assert at_threshold["threshold"] == outcome.threshold.value
    assert context["realised_fpr_on_test_at_operating_threshold"] == pytest.approx(
        at_threshold["false_positives"]
        / (at_threshold["false_positives"] + at_threshold["true_negatives"])
    )


def test_the_context_is_computed_after_the_threshold_from_the_scored_test_fold(
    monkeypatch: pytest.MonkeyPatch, tuner_calls: list[Any]
) -> None:
    """Test labels reach the context only once the threshold has been fixed on val."""
    ds = _dataset()
    events: list[str] = []
    seen: dict[str, Any] = {}
    real_find = train.find_threshold_at_fpr
    real_context = train.result_context

    def find_spy(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
        events.append("threshold")
        return real_find(y_true, y_score, target_fpr)

    def context_spy(**folds: Any) -> Any:
        events.append("context")
        seen.update(folds)
        return real_context(**folds)

    monkeypatch.setattr(train, "find_threshold_at_fpr", find_spy)
    monkeypatch.setattr(train, "result_context", context_spy)
    outcome = _run(ds)

    assert events == ["threshold", "context"]
    splits = outcome.splits
    np.testing.assert_array_equal(seen["train_labels"], ds.y[splits.train])
    np.testing.assert_array_equal(seen["val_labels"], ds.y[splits.val])
    np.testing.assert_array_equal(seen["test_labels"], ds.y[splits.test])
    assert seen["at_operating_threshold"] == confusion_at_threshold(
        ds.y[splits.test], outcome.test_scores, outcome.threshold.value
    )


def test_without_a_threshold_under_the_fpr_ceiling_the_existing_fallback_applies(
    monkeypatch: pytest.MonkeyPatch, tuner_calls: list[Any]
) -> None:
    monkeypatch.setattr(train, "find_threshold_at_fpr", lambda *_: float("inf"))

    assert _run(_dataset()).threshold.value == 0.5


def test_a_fallback_threshold_is_recorded_as_a_fallback(
    monkeypatch: pytest.MonkeyPatch, tuner_calls: list[Any]
) -> None:
    monkeypatch.setattr(train, "find_threshold_at_fpr", lambda *_: float("inf"))

    threshold = _run(_dataset()).threshold

    assert threshold.fallback_used is True
    assert threshold.value == train.FALLBACK_THRESHOLD == 0.5


def test_a_val_fold_where_no_threshold_meets_the_target_records_the_fallback(
    tuner_calls: list[Any],
) -> None:
    """The val fold's negatives carry the training fold's fraud signal.

    Its highest scores then all belong to legitimate rows, so even the
    strictest threshold exceeds a 1% FPR and selection has nothing to return.
    """
    ds = _dataset()
    val = chronological_split(ds.timestamps).val
    ds.X[val, 1] = np.where(ds.y[val] == 1, -6.0, 6.0)

    outcome = _run(ds, target_fpr=0.01)

    val_scores = outcome.model.predict_proba(ds.X[val])[:, 1]
    assert not np.isfinite(find_threshold_at_fpr(ds.y[val], val_scores, 0.01))
    threshold = outcome.threshold
    assert threshold.fallback_used is True
    assert threshold.value == 0.5
    negatives = ds.y[val] == 0
    realised = ((val_scores >= 0.5) & negatives).sum() / negatives.sum()
    assert threshold.realised_fpr_on_val == pytest.approx(realised)
    assert threshold.realised_fpr_on_val > threshold.target_fpr


def test_a_threshold_selected_under_the_fpr_ceiling_is_not_a_fallback(
    tuner_calls: list[Any],
) -> None:
    threshold = _run(_dataset(), target_fpr=0.05).threshold

    assert threshold.fallback_used is False
    assert threshold.realised_fpr_on_val <= threshold.target_fpr


def test_training_metadata_describes_the_folds_that_were_used(tuner_calls: list[Any]) -> None:
    ds = _dataset()
    outcome = _run(ds)

    metadata = train.training_metadata(ds, outcome)

    splits = outcome.splits
    assert metadata.dataset_size == N_ROWS
    assert (metadata.train_size, metadata.val_size, metadata.test_size) == splits.sizes
    assert metadata.train_fraud_rate == pytest.approx(ds.y[splits.train].mean())
    assert metadata.val_fraud_rate == pytest.approx(ds.y[splits.val].mean())
    assert metadata.test_fraud_rate == pytest.approx(ds.y[splits.test].mean())
    assert metadata.best_hyperparameters == STUB_PARAMS


def test_training_metadata_records_the_settings_the_fit_ran_with(
    tuner_calls: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]],
) -> None:
    ds = _dataset()
    outcome = _run(ds)

    metadata = train.training_metadata(ds, outcome)

    # One random state seeds the search, its fold shuffle and the final model.
    assert tuner_calls[0][2] == {
        "n_iter": metadata.tuning_iterations,
        "n_splits": metadata.tuning_cv_folds,
        "random_state": metadata.random_state,
    }
    assert (metadata.tuning_iterations, metadata.tuning_cv_folds) == (3, 2)
    fitted_with = outcome.model.get_params()
    assert fitted_with["random_state"] == metadata.random_state == train.RANDOM_STATE
    assert fitted_with["early_stopping_rounds"] == metadata.early_stopping_rounds
    assert fitted_with["scale_pos_weight"] == metadata.scale_pos_weight

    y_train = ds.y[outcome.splits.train]
    assert metadata.scale_pos_weight == pytest.approx((y_train == 0).sum() / (y_train == 1).sum())


def test_training_metadata_records_the_round_early_stopping_chose(tuner_calls: list[Any]) -> None:
    ds = _dataset()
    outcome = _run(ds)

    metadata = train.training_metadata(ds, outcome)

    # Early stopping keeps the first round with the best val-fold PR-AUC.
    val_pr_auc = outcome.model.evals_result()["validation_0"]["aucpr"]
    assert metadata.best_iteration == int(np.argmax(val_pr_auc))
    assert metadata.best_iteration == outcome.model.best_iteration
    assert 0 <= metadata.best_iteration < outcome.model.get_booster().num_boosted_rounds()


def test_the_fit_settings_are_the_frozen_protocol() -> None:
    """Phase 5D froze random state 42, 25 iterations, 4 folds and 50 early-stopping rounds."""
    args = train.parse_args([])

    assert (train.RANDOM_STATE, args.n_iter, args.cv_splits, train.EARLY_STOPPING_ROUNDS) == (
        42,
        25,
        4,
        50,
    )


# ---------------------------------------------------------------------------
# The synthetic CLI keeps its behaviour
# ---------------------------------------------------------------------------


def test_the_cli_defaults_are_unchanged() -> None:
    args = train.parse_args([])

    assert args.csv_path == Path("ml/data/synthetic_transactions.csv")
    assert args.artifact_dir == Path("ml/artifacts")
    assert args.limit is None
    assert args.n_iter == 25
    assert args.cv_splits == 4
    assert args.target_fpr == 0.01


def test_the_synthetic_cli_writes_its_artifacts_with_the_synthetic_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tuner_calls: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]],
) -> None:
    ds = _dataset()
    loads: list[tuple[str | None, int | None]] = []

    @contextmanager
    def stub_session() -> Iterator[object]:
        yield object()

    def stub_load(db: object, csv_path: str | None = None, *, limit: int | None = None) -> LabelledDataset:
        loads.append((csv_path, limit))
        return ds

    monkeypatch.setattr(train, "SessionLocal", stub_session)
    monkeypatch.setattr(train, "load_dataset_with_csv_labels", stub_load)

    train.main(
        [
            "--artifact-dir", str(tmp_path),
            "--csv-path", "labels.csv",
            "--limit", "123",
            "--n-iter", "3",
            "--cv-splits", "2",
            "--target-fpr", "0.05",
        ]
    )

    assert loads == [("labels.csv", 123)]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "feature_list.json",
        "metrics.json",
        "model.json",
        "pr_curve.png",
        "threshold.json",
        "training_metadata.json",
    ]
    assert tuner_calls[0][2] == {"n_iter": 3, "n_splits": 2, "random_state": train.RANDOM_STATE}

    def read(name: str) -> Any:
        return json.loads((tmp_path / name).read_text(encoding="utf-8"))

    metrics = read("metrics.json")
    assert set(metrics) == OBSERVED_METRICS | {"target_pr_auc", "target_recall_at_1pct_fpr"}
    assert metrics["target_pr_auc"] == 0.75
    assert metrics["target_recall_at_1pct_fpr"] == 0.60
    assert read("feature_list.json") == {"features": ds.feature_names}
    assert read("threshold.json")["target_fpr"] == 0.05
    assert read("threshold.json")["fallback_used"] is False
    assert read("training_metadata.json")["dataset_size"] == N_ROWS
