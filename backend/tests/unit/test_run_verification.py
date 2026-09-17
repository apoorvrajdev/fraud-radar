"""Phase 5D — a named run is verified before anything is said about it.

The fixture is a real run: the pipeline tests' matrix, whose first column is
each row's own index, trained under the production feature names and written
by training's own `write_run`. The search is stubbed with parameters that let
early stopping stop well before the last round, so the saved model is the
truncated one the scoring-parity fix saves.

Each refusal test changes one thing a run recorded, or one thing loaded
beside it, and requires verification to refuse and say why.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

from app.fraud.explainer import FraudExplainer, load_explainer
from app.fraud.feature_spec import FEATURE_NAMES
from ml import train
from ml.data import LabelledDataset, synthetic_provenance
from ml.datasets.manifest import sha256_file
from ml.loading import RunData
from ml.paths import run_dir
from ml.run_verification import (
    RunVerificationError,
    VerifiedRun,
    check_recorded_run,
    read_recorded_run,
    verify_run,
)
from ml.runs import transaction_ids_digest
from ml.tuning import TuningResult
from tests.unit.test_train_pipeline import _dataset as pipeline_matrix

RUN = "synthetic-fixture"
# Early stopping keeps round 10 of the 61 boosted, as in the parity suite.
SEARCH_RESULT: dict[str, Any] = {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.3}
TARGET_FPR = 0.05
COUNTRIES = ("US", "GB", "BR")


@dataclass(frozen=True)
class TrainedRun:
    ds: LabelledDataset
    outcome: train.TrainingOutcome
    runs_root: Path
    labels_csv: Path

    @property
    def directory(self) -> Path:
        return run_dir(RUN, runs_root=self.runs_root)

    def data(self) -> RunData:
        """The data as the synthetic loader would return it."""
        provenance = synthetic_provenance(self.ds, csv_path=self.labels_csv)
        return RunData(ds=self.ds, provenance=provenance)

    def edit(self, filename: str, change: Callable[[dict[str, Any]], None]) -> None:
        path = self.directory / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        change(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")


def train_fixture_run(root: Path) -> TrainedRun:
    """Train and write the fixture run under `root`."""
    matrix = pipeline_matrix()
    ds = LabelledDataset(
        X=matrix.X,
        y=matrix.y,
        timestamps=matrix.timestamps,
        transaction_ids=matrix.transaction_ids,
        feature_names=list(FEATURE_NAMES),
    )
    labels = root / "synthetic_transactions.csv"
    lines = ["id,is_fraud,country"] + [
        f"{tx_id},{bool(label)},{COUNTRIES[index % len(COUNTRIES)]}"
        for index, (tx_id, label) in enumerate(zip(ds.transaction_ids, ds.y, strict=True))
    ]
    labels.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def stub_tune(*_: object, **__: object) -> TuningResult:
        return TuningResult(
            best_params=dict(SEARCH_RESULT),
            best_score=0.5,
            cv_results_summary={"mean_test_score": [0.5], "std_test_score": [0.0]},
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "tune_hyperparameters", stub_tune)
        outcome = train.train_and_evaluate(ds, n_iter=1, cv_splits=2, target_fpr=TARGET_FPR)
    assert outcome.fit.best_iteration + 1 < outcome.model.get_booster().num_boosted_rounds()
    assert outcome.threshold.fallback_used is False

    runs_root = root / "runs"
    train.write_run(
        RUN,
        ds,
        outcome,
        dataset=synthetic_provenance(ds, csv_path=labels),
        featureset="v1",
        metrics={**outcome.metrics, **train.SYNTHETIC_TARGETS},
        runs_root=runs_root,
    )
    return TrainedRun(ds=ds, outcome=outcome, runs_root=runs_root, labels_csv=labels)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> TrainedRun:
    return train_fixture_run(tmp_path_factory.mktemp("trained"))


@pytest.fixture
def run(trained: TrainedRun, tmp_path: Path) -> TrainedRun:
    """A private copy of the fixture run, safe to edit."""
    shutil.copytree(trained.runs_root, tmp_path / "runs")
    return replace(trained, runs_root=tmp_path / "runs")


def _verify(
    run: TrainedRun,
    *,
    data: RunData | None = None,
    explainer: FraudExplainer | None = None,
) -> VerifiedRun:
    recorded = read_recorded_run(RUN, runs_root=run.runs_root)
    return verify_run(
        recorded,
        run.data() if data is None else data,
        load_explainer(recorded.directory) if explainer is None else explainer,
    )


def _refused(run: TrainedRun, match: str, **options: Any) -> None:
    with pytest.raises(RunVerificationError, match=match):
        _verify(run, **options)


# ---------------------------------------------------------------------------
# A run that matches its record
# ---------------------------------------------------------------------------


def test_a_consistent_run_verifies_and_reproduces_training_scores(run: TrainedRun) -> None:
    verified = _verify(run)

    splits = run.outcome.splits
    np.testing.assert_array_equal(verified.splits.train, splits.train)
    np.testing.assert_array_equal(verified.splits.val, splits.val)
    np.testing.assert_array_equal(verified.splits.test, splits.test)
    np.testing.assert_array_equal(verified.test_scores, run.outcome.test_scores)
    assert verified.recorded.name == RUN


def test_verification_gives_the_model_test_rows_only(
    run: TrainedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every matrix handed to XGBoost holds test rows, and every test row is scored once."""
    scored: list[np.ndarray] = []
    real_init = xgb.DMatrix.__init__

    def spy(self: xgb.DMatrix, data: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(data, np.ndarray):
            scored.append(data[:, 0].astype(np.int64))
        real_init(self, data, *args, **kwargs)

    monkeypatch.setattr(xgb.DMatrix, "__init__", spy)
    verified = _verify(run)

    rows = np.concatenate(scored)
    np.testing.assert_array_equal(rows, verified.splits.test)
    assert not set(rows) & (set(verified.splits.train) | set(verified.splits.val))


def test_when_the_files_were_read_does_not_matter(run: TrainedRun) -> None:
    data = run.data()
    assert data.provenance is not None
    reread = replace(data, provenance=replace(data.provenance, retrieved_at=datetime.now(UTC)))

    _verify(run, data=reread)


def test_the_same_instants_in_another_timezone_verify(run: TrainedRun) -> None:
    india = timezone(timedelta(hours=5, minutes=30))
    shifted = replace(
        run.ds, timestamps=np.asarray([ts.astimezone(india) for ts in run.ds.timestamps])
    )

    verified = _verify(
        run,
        data=RunData(ds=shifted, provenance=synthetic_provenance(shifted, csv_path=run.labels_csv)),
    )

    np.testing.assert_array_equal(verified.test_scores, run.outcome.test_scores)


# ---------------------------------------------------------------------------
# The run's records
# ---------------------------------------------------------------------------


def test_a_missing_run_is_refused(run: TrainedRun) -> None:
    with pytest.raises(RunVerificationError, match="does not exist"):
        read_recorded_run("no-such-run", runs_root=run.runs_root)


def test_a_run_record_naming_another_run_is_refused(run: TrainedRun) -> None:
    run.edit("run.json", lambda record: record.update(run_name="another-run"))

    _refused(run, "names run 'another-run', so it is the record of another run")


def test_a_run_without_a_run_record_is_refused(run: TrainedRun) -> None:
    (run.directory / "run.json").unlink()

    _refused(run, "has no run.json")


def test_an_incomplete_run_is_refused(run: TrainedRun) -> None:
    (run.directory / "model.json").unlink()

    with pytest.raises(RunVerificationError, match=r"incomplete: model\.json missing"):
        read_recorded_run(RUN, runs_root=run.runs_root)


def test_a_different_library_version_is_refused(run: TrainedRun) -> None:
    run.edit("run.json", lambda record: record["library_versions"].update(xgboost="0.0.1"))

    _refused(run, r"recorded with xgboost 0\.0\.1, but this environment has")


def test_an_unrecorded_library_version_is_refused(run: TrainedRun) -> None:
    run.edit("run.json", lambda record: record["library_versions"].pop("numpy"))

    _refused(run, "records no numpy version")


def test_an_unregistered_featureset_is_refused(run: TrainedRun) -> None:
    run.edit("run.json", lambda record: record.update(featureset_version="v9"))

    _refused(run, "featureset 'v9', which is not registered")


def test_a_feature_list_out_of_registered_order_is_refused(
    run: TrainedRun, trained: TrainedRun
) -> None:
    run.edit("feature_list.json", lambda payload: payload["features"].reverse())

    _refused(run, "registered order", explainer=load_explainer(trained.directory))


def test_a_fit_record_without_the_best_round_is_refused(run: TrainedRun) -> None:
    run.edit("training_metadata.json", lambda metadata: metadata.pop("best_iteration"))

    _refused(run, "has no best_iteration")


def test_metrics_without_their_context_are_refused(run: TrainedRun) -> None:
    run.edit("metrics.json", lambda metrics: metrics.pop("context"))

    _refused(run, "no context with fold fraud counts")


def test_metrics_measured_at_another_threshold_are_refused(run: TrainedRun) -> None:
    run.edit("threshold.json", lambda threshold: threshold.update(value=0.123))

    _refused(run, "measured at threshold")


# ---------------------------------------------------------------------------
# The saved model
# ---------------------------------------------------------------------------


def test_a_model_with_rounds_after_the_best_one_is_refused(run: TrainedRun) -> None:
    run.outcome.model.get_booster().save_model(str(run.directory / "model.json"))

    _refused(run, r"holds 61 boosting rounds, but the fit's best round was 10")


def test_a_model_carrying_early_stopping_state_is_refused(run: TrainedRun) -> None:
    best = run.outcome.fit.best_iteration
    booster = run.outcome.model.get_booster()[: best + 1]
    booster.set_attr(best_iteration=str(best))
    booster.save_model(str(run.directory / "model.json"))

    _refused(run, "carries early-stopping state")


def test_another_model_with_the_same_rounds_is_refused(run: TrainedRun) -> None:
    """Right shape, wrong function: only the exact metric reproduction can tell.

    The fit record's model digest is rewritten to match the swapped file, so
    the digest check passes and the scores are the only remaining evidence.
    """
    splits = run.outcome.splits
    other = xgb.XGBClassifier(
        n_estimators=run.outcome.fit.best_iteration + 1, max_depth=2, random_state=0
    )
    other.fit(run.ds.X[splits.train], run.ds.y[splits.train])
    other.get_booster().save_model(str(run.directory / "model.json"))
    swapped = sha256_file(run.directory / "model.json")
    run.edit("training_metadata.json", lambda metadata: metadata.update(model_sha256=swapped))

    _refused(run, "do not reproduce metrics.json exactly")


def test_a_run_verifies_the_digest_of_its_saved_model(run: TrainedRun) -> None:
    metadata = json.loads((run.directory / "training_metadata.json").read_text(encoding="utf-8"))
    on_disk = hashlib.sha256((run.directory / "model.json").read_bytes()).hexdigest()

    verified = _verify(run)

    assert metadata["model_sha256"] == on_disk
    assert verified.model_digest_verified is True


def test_a_model_file_other_than_the_recorded_one_is_refused_before_loading_data(
    run: TrainedRun,
) -> None:
    """Refused on the record alone: no data is needed to see the bytes differ."""
    splits = run.outcome.splits
    other = xgb.XGBClassifier(
        n_estimators=run.outcome.fit.best_iteration + 1, max_depth=2, random_state=0
    )
    other.fit(run.ds.X[splits.train], run.ds.y[splits.train])
    other.get_booster().save_model(str(run.directory / "model.json"))

    with pytest.raises(RunVerificationError, match="It is not the model file this run saved"):
        check_recorded_run(read_recorded_run(RUN, runs_root=run.runs_root))


def test_the_same_model_saved_as_other_bytes_is_refused(run: TrainedRun) -> None:
    """The digest pins the persisted artifact itself, not merely an equivalent model."""
    path = run.directory / "model.json"
    path.write_text(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=1), "utf-8")

    _refused(run, r"model\.json hashes to .* but training_metadata\.json records")


def test_a_recorded_model_digest_other_than_the_file_is_refused(run: TrainedRun) -> None:
    run.edit("training_metadata.json", lambda metadata: metadata.update(model_sha256="0" * 64))

    _refused(run, f"training_metadata.json records {'0' * 64}")


def test_a_fit_record_without_a_model_digest_verifies_without_claiming_it(
    run: TrainedRun,
) -> None:
    run.edit("training_metadata.json", lambda metadata: metadata.pop("model_sha256"))

    verified = _verify(run)

    assert verified.model_digest_verified is False
    np.testing.assert_array_equal(verified.test_scores, run.outcome.test_scores)


def test_an_explainer_from_another_run_is_refused(run: TrainedRun, tmp_path: Path) -> None:
    other = tmp_path / "other"
    shutil.copytree(run.directory, other)
    (other / "threshold.json").write_text(json.dumps({"value": 0.9}), encoding="utf-8")

    _refused(run, "was not loaded from this run", explainer=load_explainer(other))


# ---------------------------------------------------------------------------
# The loaded data and its folds
# ---------------------------------------------------------------------------


def test_data_loaded_without_provenance_is_refused(run: TrainedRun) -> None:
    _refused(run, "without a provenance record", data=RunData(ds=run.ds, provenance=None))


def test_a_dataset_with_other_source_bytes_is_refused(run: TrainedRun, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere" / run.labels_csv.name
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(run.labels_csv.read_bytes() + b"\n")

    _refused(
        run,
        r"not the one run\.json records \(files:",
        data=RunData(ds=run.ds, provenance=synthetic_provenance(run.ds, csv_path=elsewhere)),
    )


def test_a_matrix_with_other_features_is_refused(run: TrainedRun) -> None:
    renamed = replace(run.ds, feature_names=[f"f{index}" for index in range(17)])

    _refused(run, "the loaded matrix has features", data=replace(run.data(), ds=renamed))


def test_naive_timestamps_are_refused(run: TrainedRun) -> None:
    naive = replace(
        run.ds, timestamps=np.asarray([ts.replace(tzinfo=None) for ts in run.ds.timestamps])
    )

    _refused(run, "no valid chronological periods", data=replace(run.data(), ds=naive))


def test_a_run_record_without_fold_periods_is_refused(run: TrainedRun) -> None:
    run.edit("run.json", lambda record: record.update(splits=[]))

    _refused(run, "records no fold periods")


def test_fold_periods_other_than_the_recorded_ones_are_refused(run: TrainedRun) -> None:
    def later_test_end(record: dict[str, Any]) -> None:
        test = record["splits"][2]
        test["end"] = (datetime.fromisoformat(test["end"]) + timedelta(seconds=1)).isoformat()

    run.edit("run.json", later_test_end)

    _refused(run, r"the loaded folds cover .* but run\.json records")


def test_fold_sizes_other_than_the_recorded_ones_are_refused(run: TrainedRun) -> None:
    run.edit("training_metadata.json", lambda metadata: metadata.update(test_size=61, val_size=59))

    _refused(run, r"hold \(280, 60, 60\) rows .* recorded \(280, 59, 61\)")


def test_a_test_fraud_count_other_than_the_recorded_one_is_refused(run: TrainedRun) -> None:
    recorded_test_frauds = int(run.ds.y[run.outcome.splits.test].sum())
    run.edit(
        "metrics.json",
        lambda metrics: metrics["context"]["fraud_counts"].update(test=recorded_test_frauds + 1),
    )

    _refused(run, f"test fold holds {recorded_test_frauds} frauds")


# ---------------------------------------------------------------------------
# The test fold's transaction identity
# ---------------------------------------------------------------------------


def _test_ids(run: TrainedRun) -> list[str]:
    return [run.ds.transaction_ids[row] for row in run.outcome.splits.test]


def _with_ids(run: TrainedRun, transaction_ids: list[str]) -> RunData:
    """The run's data with other transaction ids, and nothing else changed."""
    changed = replace(run.ds, transaction_ids=transaction_ids)
    return RunData(ds=changed, provenance=synthetic_provenance(changed, csv_path=run.labels_csv))


def test_a_run_verifies_the_transaction_ids_of_its_test_fold(run: TrainedRun) -> None:
    record = json.loads((run.directory / "run.json").read_text(encoding="utf-8"))

    verified = _verify(run)

    assert record["metadata_version"] == "3"
    assert record["test_fold_identity"]["transaction_count"] == len(_test_ids(run)) == 60
    assert record["test_fold_identity"]["transaction_ids_sha256"] == transaction_ids_digest(
        _test_ids(run)
    )
    assert verified.test_fold_identity_verified is True


def test_a_changed_test_transaction_id_is_refused(run: TrainedRun) -> None:
    """Same periods, sizes, labels, features and scores: only the id tells the rows apart."""
    ids = list(run.ds.transaction_ids)
    ids[int(run.outcome.splits.test[7])] = "tx-from-elsewhere"

    _refused(run, "not the rows, or not the order", data=_with_ids(run, ids))


def test_a_reordered_test_fold_is_refused(run: TrainedRun) -> None:
    ids = list(run.ds.transaction_ids)
    first, second = (int(row) for row in run.outcome.splits.test[:2])
    ids[first], ids[second] = ids[second], ids[first]

    _refused(run, "not the rows, or not the order", data=_with_ids(run, ids))


def test_a_test_fold_count_other_than_the_identified_one_is_refused(run: TrainedRun) -> None:
    run.edit(
        "run.json",
        lambda record: record["test_fold_identity"].update(transaction_count=61),
    )

    _refused(run, r"test fold holds 60 transactions, but run\.json identifies 61")


def test_a_recorded_digest_other_than_the_loaded_one_is_refused(run: TrainedRun) -> None:
    other = transaction_ids_digest([*_test_ids(run)[1:], "tx-extra"])
    run.edit(
        "run.json",
        lambda record: record["test_fold_identity"].update(transaction_ids_sha256=other),
    )

    _refused(run, rf"but run\.json records {other}")


def test_a_version_3_record_without_a_test_fold_identity_is_refused(run: TrainedRun) -> None:
    run.edit("run.json", lambda record: record.pop("test_fold_identity"))

    _refused(run, "malformed record: .*must identify the fold's transactions")


def test_a_version_2_run_verifies_without_claiming_its_test_fold_was_identified(
    run: TrainedRun,
) -> None:
    """An older record is still verified by period, size, fraud count and metrics."""

    def as_version_2(record: dict[str, Any]) -> None:
        record.pop("test_fold_identity")
        record["metadata_version"] = "2"

    run.edit("run.json", as_version_2)

    verified = _verify(run)

    assert verified.recorded.record.metadata_version == "2"
    assert verified.test_fold_identity_verified is False
    np.testing.assert_array_equal(verified.test_scores, run.outcome.test_scores)


def test_a_version_2_run_still_refuses_folds_other_than_the_recorded_ones(
    run: TrainedRun,
) -> None:
    def as_version_2(record: dict[str, Any]) -> None:
        record.pop("test_fold_identity")
        record["metadata_version"] = "2"

    run.edit("run.json", as_version_2)
    run.edit("training_metadata.json", lambda metadata: metadata.update(test_size=61, val_size=59))

    _refused(run, r"hold \(280, 60, 60\) rows")


# ---------------------------------------------------------------------------
# Exact reproduction of metrics.json
# ---------------------------------------------------------------------------


def test_a_metric_one_float_step_away_is_refused(run: TrainedRun) -> None:
    def nudge(metrics: dict[str, Any]) -> None:
        metrics["test_pr_auc"] = float(np.nextafter(metrics["test_pr_auc"], 2.0))

    run.edit("metrics.json", nudge)

    _refused(run, "test_pr_auc: recorded")


def test_a_confusion_count_other_than_the_reproduced_one_is_refused(run: TrainedRun) -> None:
    run.edit(
        "metrics.json",
        lambda metrics: metrics["at_operating_threshold"].update(
            true_positives=metrics["at_operating_threshold"]["true_positives"] + 1
        ),
    )

    _refused(run, "at_operating_threshold: recorded")


def test_an_entry_that_cannot_be_verified_is_refused(run: TrainedRun) -> None:
    run.edit("metrics.json", lambda metrics: metrics.update(surprise=1.0))

    _refused(run, "cannot verify: surprise")


def test_the_search_score_and_synthetic_targets_are_not_test_results(run: TrainedRun) -> None:
    """They are recorded, not recomputed, and never refuse a run."""
    metrics = json.loads((run.directory / "metrics.json").read_text(encoding="utf-8"))
    assert {"best_cv_pr_auc", *train.SYNTHETIC_TARGETS} <= set(metrics)

    _verify(run)
