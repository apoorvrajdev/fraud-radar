"""Phase 5D — one scoring function from training to the served explanation.

Early stopping keeps the rounds boosted after its best one. Training scores the
val and test folds with the rounds up to the best, so the threshold and metrics
describe that model. The saved artifact has to hold exactly the same function,
or the served score, its SHAP explanation, the recorded threshold and the
recorded metrics each describe a different model.

The fixture runs the real pipeline on the deterministic matrix the pipeline
tests use, with production feature names so the real explainer accepts it. The
search is stubbed with parameters that let early stopping stop long before the
last round. Artifacts are written by training's own write path and read back
every way the repository reads them.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

from app.fraud.explainer import FraudExplainer, initialize_explainer, reset_explainer_for_tests
from app.fraud.feature_spec import FEATURE_NAMES
from ml import train
from ml.artifacts import load_model, load_threshold, predict_proba
from ml.data import LabelledDataset
from ml.evaluation import (
    confusion_at_threshold,
    find_threshold_at_fpr,
    pr_auc,
    recall_at_fpr,
    roc_auc,
)
from ml.tuning import TuningResult
from tests.unit.test_train_pipeline import _dataset as pipeline_matrix

# Early stopping picks round 10 of the 61 boosted before it gives up.
SEARCH_RESULT: dict[str, Any] = {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.3}
TARGET_FPR = 0.05


@dataclass(frozen=True)
class TrainedRun:
    ds: LabelledDataset
    outcome: train.TrainingOutcome
    directory: Path

    def fold(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        indices = getattr(self.outcome.splits, name)
        return self.ds.X[indices], self.ds.y[indices]

    def read(self, name: str) -> Any:
        return json.loads((self.directory / name).read_text(encoding="utf-8"))

    def saved_booster(self) -> xgb.Booster:
        booster = xgb.Booster()
        booster.load_model(str(self.directory / "model.json"))
        return booster


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> TrainedRun:
    matrix = pipeline_matrix()
    ds = LabelledDataset(
        X=matrix.X,
        y=matrix.y,
        timestamps=matrix.timestamps,
        transaction_ids=matrix.transaction_ids,
        feature_names=list(FEATURE_NAMES),
    )

    def stub_tune(*_: object, **__: object) -> TuningResult:
        return TuningResult(
            best_params=dict(SEARCH_RESULT),
            best_score=0.5,
            cv_results_summary={"mean_test_score": [0.5], "std_test_score": [0.0]},
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "tune_hyperparameters", stub_tune)
        outcome = train.train_and_evaluate(ds, n_iter=1, cv_splits=2, target_fpr=TARGET_FPR)

    # Every check below is vacuous unless rounds were boosted past the best one.
    rounds = outcome.model.get_booster().num_boosted_rounds()
    assert outcome.fit.best_iteration + 1 < rounds, "the fixture did not stop early"

    directory = tmp_path_factory.mktemp("run")
    train._write_artifacts(directory, ds, outcome, outcome.metrics)
    return TrainedRun(ds=ds, outcome=outcome, directory=directory)


@pytest.fixture
def served(run: TrainedRun) -> Iterator[FraudExplainer]:
    reset_explainer_for_tests()
    yield initialize_explainer(run.directory)
    reset_explainer_for_tests()


def test_the_rounds_after_the_best_one_change_the_scores(run: TrainedRun) -> None:
    """Saving the fitted booster whole would score the test fold differently."""
    X_test, _ = run.fold("test")

    every_round = run.outcome.model.get_booster().predict(xgb.DMatrix(X_test))

    assert not np.array_equal(every_round, run.outcome.test_scores)


def test_the_saved_model_holds_the_best_round_plus_one_and_no_early_stopping_state(
    run: TrainedRun,
) -> None:
    best_iteration = run.read("training_metadata.json")["best_iteration"]
    assert best_iteration == run.outcome.fit.best_iteration

    learner = run.read("model.json")["learner"]
    assert len(learner["gradient_booster"]["model"]["trees"]) == best_iteration + 1
    booster = run.saved_booster()
    assert booster.num_boosted_rounds() == best_iteration + 1
    assert booster.attr("best_iteration") is None


def test_the_saved_model_scores_the_test_fold_exactly_as_training_did(run: TrainedRun) -> None:
    X_test, _ = run.fold("test")
    training = run.outcome.test_scores

    np.testing.assert_array_equal(run.saved_booster().predict(xgb.DMatrix(X_test)), training)
    np.testing.assert_array_equal(predict_proba(load_model(run.directory), X_test), training)
    classifier = xgb.XGBClassifier()
    classifier.load_model(str(run.directory / "model.json"))
    np.testing.assert_array_equal(classifier.predict_proba(X_test)[:, 1], training)


def test_the_explainer_serves_the_training_score_for_every_test_row(
    run: TrainedRun, served: FraudExplainer
) -> None:
    X_test, _ = run.fold("test")

    served_scores = np.array([served.predict_proba(row) for row in X_test], dtype=np.float32)

    np.testing.assert_array_equal(served_scores, run.outcome.test_scores)


def test_the_saved_model_reproduces_the_operating_threshold_on_the_val_fold(
    run: TrainedRun,
) -> None:
    X_val, y_val = run.fold("val")
    threshold = load_threshold(run.directory)
    assert threshold.fallback_used is False, "a fallback threshold is not selected from scores"

    scores = run.saved_booster().predict(xgb.DMatrix(X_val))

    assert find_threshold_at_fpr(y_val, scores, TARGET_FPR) == threshold.value
    negatives = y_val == 0
    realised = float(((scores >= threshold.value) & negatives).sum() / max(negatives.sum(), 1))
    assert realised == threshold.realised_fpr_on_val


def test_the_saved_model_reproduces_the_test_metrics(run: TrainedRun) -> None:
    X_test, y_test = run.fold("test")
    metrics = run.read("metrics.json")
    threshold = run.read("threshold.json")["value"]

    scores = predict_proba(load_model(run.directory), X_test)

    assert metrics["test_pr_auc"] == pr_auc(y_test, scores)
    assert metrics["test_roc_auc"] == roc_auc(y_test, scores)
    assert metrics["recall_at_1pct_fpr"] == recall_at_fpr(y_test, scores, 0.01)[0]
    assert metrics["recall_at_5pct_fpr"] == recall_at_fpr(y_test, scores, 0.05)[0]
    assert metrics["at_operating_threshold"] == (
        confusion_at_threshold(y_test, scores, threshold).as_dict()
    )
    negatives = y_test == 0
    assert metrics["context"]["realised_fpr_on_test_at_operating_threshold"] == float(
        ((scores >= threshold) & negatives).sum() / negatives.sum()
    )


def test_each_served_explanation_adds_up_to_the_score_served_beside_it(
    run: TrainedRun, served: FraudExplainer
) -> None:
    """Base value plus SHAP contributions, through the sigmoid, is the served score.

    Checked against the score the explainer serves, not the booster's margin: an
    explanation of one set of rounds beside a score from another is exactly the
    mismatch this guards against.
    """
    X_test, _ = run.fold("test")

    for row, training_score in zip(X_test, run.outcome.test_scores, strict=True):
        explanation = served.explain_local(row)
        margin = explanation.base_value + float(explanation.shap_values.sum())
        explained_score = 1.0 / (1.0 + np.exp(-margin))

        assert explained_score == pytest.approx(explanation.fraud_score, abs=1e-5)
        assert explanation.fraud_score == float(training_score)
