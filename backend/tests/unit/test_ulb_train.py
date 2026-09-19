"""Phase 5E — the ULB trainer and verifier, on fixture files.

The fixture is 400 rows in the published layout, a fraud every ninth row and
separated on V1, so each fold holds fraud and a threshold under the 1% FPR
ceiling exists. The search is stubbed, as in the Phase 5D pipeline tests;
everything after it — the fit with early stopping, the threshold, the test
evaluation, the records — is the real procedure.

Each verifier refusal test changes one thing a run recorded, or the file
beside it, and requires verification to refuse and say why.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml import train
from ml.datasets.manifest import sha256_file
from ml.promote import PromotionError, promote_run
from ml.run_verification import RunVerificationError
from ml.runs import current_git_commit, load_run_metadata
from ml.tracks.ulb import train as ulb_train
from ml.tracks.ulb.load import (
    ELAPSED_TIME_ORIGIN,
    LICENCE_NOTICE,
    METHOD_OFFER,
    SOURCE_FILENAME,
)
from ml.tracks.ulb.matrix import MATRIX_PREPROCESSING, ULB_PCA_V1, ULB_PCA_V1_FEATURES
from ml.tracks.ulb.quality import ULB_QUALITY_REPORT_FILENAME
from ml.tracks.ulb.train import (
    TARGET_FPR,
    TUNING_CV_FOLDS,
    TUNING_ITERATIONS,
    ULB_CALIBRATION_NOTE,
    ULB_SEEDS,
    UlbRun,
    UlbStopError,
    parse_args,
    run_name_for,
    train_ulb_run,
)
from ml.tracks.ulb.verify import verify_ulb_run
from ml.tuning import TuningResult
from tests.unit.test_ulb_load import FIXTURE_LICENCE, ulb_row, write_ulb_csv, write_ulb_manifest

RUN = run_name_for(42)
STUB_PARAMS: dict[str, Any] = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.3}
RUN_FILES = {
    "run.json",
    "model.json",
    "feature_list.json",
    "threshold.json",
    "metrics.json",
    "training_metadata.json",
    "calibration_metrics.json",
    "ulb_quality_report.json",
    "pr_curve.png",
}


def benchmark_rows(
    n_rows: int = 400, *, is_fraud: Callable[[int], bool] = lambda index: index % 9 == 0
) -> list[list[str]]:
    """Rows at Times 0..n-1; with 400, train is 0-279, val 280-339 and test 340-399."""
    rng = np.random.default_rng(5)
    rows = []
    for index in range(n_rows):
        fraud = is_fraud(index)
        components = rng.normal(size=len(ULB_PCA_V1_FEATURES) - 1)
        if fraud:
            components[0] += 6.0
        rows.append(
            ulb_row(
                str(index),
                marker=index,
                amount=f"{rng.uniform(1, 500):.2f}",
                label="1" if fraud else "0",
                components=[f"{value:.6f}" for value in components],
            )
        )
    return rows


@dataclass(frozen=True)
class Trained:
    root: Path
    manifest: Path
    runs_root: Path
    run: UlbRun
    tuner_options: tuple[dict[str, Any], ...]

    @property
    def directory(self) -> Path:
        return self.runs_root / self.run.run_name

    def edit(self, filename: str, change: Callable[[dict[str, Any]], None]) -> None:
        path = self.directory / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        change(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def verify(self, **options: Any) -> Any:
        return verify_ulb_run(
            self.run.run_name,
            root=options.get("root", self.root),
            manifest_path=self.manifest,
            runs_root=self.runs_root,
        )


def write_fixture(directory: Path, rows: Sequence[Sequence[str]]) -> tuple[Path, Path]:
    root = directory / "raw" / "ulb"
    csv_path = write_ulb_csv(root / SOURCE_FILENAME, rows)
    return root, write_ulb_manifest(directory, csv_path)


def train_fixture(
    directory: Path,
    rows: Sequence[Sequence[str]] | None = None,
    *,
    seed: int = 42,
) -> Trained:
    """Train the fixture run for `seed` under `directory`, with the search stubbed."""
    root, manifest = write_fixture(directory, benchmark_rows() if rows is None else rows)
    options_seen: list[dict[str, Any]] = []

    def stub_tune(features: np.ndarray, labels: np.ndarray, **options: Any) -> TuningResult:
        options_seen.append({**options, "rows": len(features)})
        return TuningResult(
            best_params=dict(STUB_PARAMS),
            best_score=0.5,
            cv_results_summary={"mean_test_score": [0.5], "std_test_score": [0.0]},
        )

    runs_root = directory / "runs"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "tune_hyperparameters", stub_tune)
        run = train_ulb_run(seed, root=root, manifest_path=manifest, runs_root=runs_root)
    return Trained(root, manifest, runs_root, run, tuple(options_seen))


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    return train_fixture(tmp_path_factory.mktemp("ulb-train"))


@pytest.fixture
def run(trained: Trained, tmp_path: Path) -> Trained:
    """A private copy of the trained run, safe to edit."""
    shutil.copytree(trained.runs_root, tmp_path / "runs")
    return replace(trained, runs_root=tmp_path / "runs")


def _record(trained: Trained, filename: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((trained.directory / filename).read_text("utf-8"))
    return payload


# ---------------------------------------------------------------------------
# The frozen protocol
# ---------------------------------------------------------------------------


def test_only_the_pre_registered_random_states_exist() -> None:
    assert ULB_SEEDS == (42, 43, 44)
    assert [run_name_for(seed) for seed in ULB_SEEDS] == [
        "ulb_pca_v1_seed42",
        "ulb_pca_v1_seed43",
        "ulb_pca_v1_seed44",
    ]
    assert (TUNING_ITERATIONS, TUNING_CV_FOLDS, TARGET_FPR) == (25, 4, 0.01)


def test_any_other_random_state_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not pre-registered"):
        train_ulb_run(45, runs_root=tmp_path)
    with pytest.raises(SystemExit):
        parse_args(["--seed", "45"])


def test_the_search_runs_on_the_training_fold_with_the_frozen_budget(trained: Trained) -> None:
    assert trained.tuner_options == (
        {"n_iter": 25, "n_splits": 4, "random_state": 42, "rows": 280},
    )


def test_the_fit_and_threshold_follow_the_frozen_protocol(trained: Trained) -> None:
    outcome = trained.run.outcome

    assert outcome.splits.sizes == (280, 60, 60)
    assert outcome.fit.random_state == 42
    assert outcome.fit.early_stopping_rounds == 50
    assert outcome.threshold.target_fpr == 0.01
    assert outcome.threshold.fallback_used is False
    assert outcome.threshold.realised_fpr_on_val <= 0.01
    assert outcome.live_features.live_count == 29


def test_another_random_state_changes_only_the_random_state(
    trained: Trained, tmp_path: Path
) -> None:
    other = train_fixture(tmp_path, seed=43)

    assert other.run.run_name == "ulb_pca_v1_seed43"
    assert other.tuner_options[0]["random_state"] == 43
    assert other.run.outcome.fit.random_state == 43
    assert load_run_metadata(other.run.run_name, runs_root=other.runs_root).test_fold_identity == (
        load_run_metadata(RUN, runs_root=trained.runs_root).test_fold_identity
    )
    assert other.run.outcome.splits.sizes == trained.run.outcome.splits.sizes


# ---------------------------------------------------------------------------
# What a run records
# ---------------------------------------------------------------------------


def test_a_run_writes_exactly_the_records_the_method_lists(trained: Trained) -> None:
    assert {path.name for path in trained.directory.iterdir()} == RUN_FILES


def test_run_json_records_the_code_featureset_data_and_test_fold(trained: Trained) -> None:
    record = load_run_metadata(RUN, runs_root=trained.runs_root)
    dataset = record.dataset

    assert record.metadata_version == "3"
    assert record.featureset_version == ULB_PCA_V1
    assert record.code_version == current_git_commit()
    assert {"xgboost", "scikit-learn", "numpy"} <= set(record.library_versions)
    assert record.seed is None  # no subsample
    assert dataset.name == "ulb"
    assert dict(dataset.files) == {SOURCE_FILENAME: sha256_file(trained.root / SOURCE_FILENAME)}
    assert dataset.license == FIXTURE_LICENCE
    assert dataset.preprocessing[-1] == MATRIX_PREPROCESSING
    assert [(split.name, split.start) for split in record.splits] == [
        ("train", ELAPSED_TIME_ORIGIN),
        ("val", datetime.fromisoformat("1970-01-01T00:04:40+00:00")),
        ("test", datetime.fromisoformat("1970-01-01T00:05:40+00:00")),
    ]
    identity = record.test_fold_identity
    assert identity is not None
    assert identity.transaction_count == 60
    assert LICENCE_NOTICE in record.notes
    assert METHOD_OFFER in record.notes


def test_metrics_hold_observed_results_only(trained: Trained) -> None:
    metrics = _record(trained, "metrics.json")

    assert not any(key.startswith("target_") for key in metrics)
    assert {"test_pr_auc", "test_roc_auc", "recall_at_1pct_fpr", "recall_at_5pct_fpr"} <= set(
        metrics
    )
    assert metrics["context"]["fraud_counts"] == {"train": 32, "val": 6, "test": 7}
    assert metrics["at_operating_threshold"]["threshold"] == _record(trained, "threshold.json")[
        "value"
    ]


def test_the_fit_record_holds_the_random_state_live_features_and_model_digest(
    trained: Trained,
) -> None:
    metadata = _record(trained, "training_metadata.json")

    assert metadata["random_state"] == 42
    assert (metadata["tuning_iterations"], metadata["tuning_cv_folds"]) == (25, 4)
    assert metadata["live_feature_count"] == 29
    assert metadata["model_sha256"] == hashlib.sha256(
        (trained.directory / "model.json").read_bytes()
    ).hexdigest()
    assert _record(trained, "feature_list.json")["features"] == list(ULB_PCA_V1_FEATURES)


def test_calibration_is_measured_on_the_test_fold_with_its_caveats(trained: Trained) -> None:
    calibration = _record(trained, "calibration_metrics.json")

    assert calibration["note"] == ULB_CALIBRATION_NOTE
    assert calibration["n_test_samples"] == 60
    assert {"brier_score", "expected_calibration_error", "positive_class_ece"} <= set(calibration)


def test_the_quality_report_beside_the_run_describes_its_folds_and_carries_the_notice(
    trained: Trained,
) -> None:
    report = _record(trained, ULB_QUALITY_REPORT_FILENAME)

    assert [fold["rows"] for fold in report["chronological_split"]["folds"]] == [280, 60, 60]
    assert report["stops"] == []
    assert report["licence"]["notice"] == LICENCE_NOTICE
    assert report["licence"]["method_offer"] == METHOD_OFFER


# ---------------------------------------------------------------------------
# Refusals and decision 15 stops
# ---------------------------------------------------------------------------


def test_a_recorded_run_is_never_trained_again(trained: Trained) -> None:
    def must_not_load(*_: object, **__: object) -> None:
        raise AssertionError("the corpus was loaded for a run that is already recorded")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ulb_train, "load_ulb", must_not_load)
        with pytest.raises(train.ExistingRunError):
            train_ulb_run(42, root=trained.root, manifest_path=trained.manifest,
                          runs_root=trained.runs_root)  # fmt: skip


def test_an_excluded_row_stops_the_work_before_training(tmp_path: Path) -> None:
    rows = [*benchmark_rows(), ulb_row("400", marker=400, amount="-1.00")]

    with pytest.raises(UlbStopError, match="1 row\\(s\\) were excluded") as stop:
        train_fixture(tmp_path, rows)
    assert stop.value.run_name == RUN
    assert not (tmp_path / "runs").exists()


def test_a_val_or_test_fold_without_fraud_stops_the_work_before_training(tmp_path: Path) -> None:
    rows = benchmark_rows(is_fraud=lambda index: index % 9 == 0 and index < 280)

    with pytest.raises(UlbStopError, match="val fold holds no fraud") as stop:
        train_fixture(tmp_path, rows)
    assert len(stop.value.reasons) == 2
    assert not (tmp_path / "runs").exists()


def test_a_fallback_threshold_is_recorded_and_then_stops_the_work(tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "find_threshold_at_fpr", lambda *_: float("inf"))
        with pytest.raises(UlbStopError, match="fallback threshold was used"):
            train_fixture(tmp_path)

    threshold = json.loads((tmp_path / "runs" / RUN / "threshold.json").read_text("utf-8"))
    assert threshold["fallback_used"] is True
    assert (tmp_path / "runs" / RUN / "run.json").exists()


def test_a_ulb_run_is_refused_by_promotion(run: Trained, tmp_path: Path) -> None:
    """Decision 12, on a run the ULB trainer itself wrote."""
    served = tmp_path / "served"
    served.mkdir()

    with pytest.raises(PromotionError, match="featureset 'ulb_pca_v1', which is not registered"):
        promote_run(RUN, runs_root=run.runs_root, artifacts_dir=served)
    assert not any(served.iterdir())


# ---------------------------------------------------------------------------
# Verification (decision 13)
# ---------------------------------------------------------------------------


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*")
            if path.is_file()}  # fmt: skip


def test_a_trained_run_verifies_and_reproduces_its_test_scores_exactly(run: Trained) -> None:
    verified = run.verify()

    np.testing.assert_array_equal(verified.test_scores, run.run.outcome.test_scores)
    np.testing.assert_array_equal(verified.splits.test, run.run.outcome.splits.test)


def test_verification_writes_nothing(run: Trained) -> None:
    before = _snapshot(run.runs_root), _snapshot(run.root)

    run.verify()

    assert (_snapshot(run.runs_root), _snapshot(run.root)) == before


def _refused(run: Trained, match: str, **options: Any) -> None:
    with pytest.raises(RunVerificationError, match=match):
        run.verify(**options)


def test_metrics_the_model_does_not_reproduce_are_refused(run: Trained) -> None:
    run.edit("metrics.json", lambda metrics: metrics.update(test_pr_auc=0.123))

    _refused(run, r"do not reproduce metrics\.json exactly")


def test_calibration_the_model_does_not_reproduce_is_refused(run: Trained) -> None:
    run.edit("calibration_metrics.json", lambda payload: payload.update(brier_score=0.5))

    _refused(run, r"calibration_metrics\.json exactly \(differing: brier_score\)")


def test_a_run_without_its_calibration_record_is_refused(run: Trained) -> None:
    (run.directory / "calibration_metrics.json").unlink()

    _refused(run, "is incomplete")


def test_a_run_recorded_on_another_featureset_is_refused(run: Trained) -> None:
    run.edit("run.json", lambda record: record.update(featureset_version="v1"))

    _refused(run, "a ULB run is trained on 'ulb_pca_v1' only")


def test_a_feature_list_out_of_its_defined_order_is_refused(run: Trained) -> None:
    features = list(ULB_PCA_V1_FEATURES)
    features[0], features[1] = features[1], features[0]
    run.edit("feature_list.json", lambda record: record.update(features=features))

    _refused(run, "in their defined order")


def test_a_model_file_other_than_the_saved_one_is_refused(run: Trained) -> None:
    run.edit("training_metadata.json", lambda metadata: metadata.update(model_sha256="0" * 64))

    _refused(run, r"model\.json hashes to")


def test_another_test_fold_identity_is_refused(run: Trained) -> None:
    run.edit(
        "run.json",
        lambda record: record["test_fold_identity"].update(transaction_ids_sha256="f" * 64),
    )

    _refused(run, "These are not the rows")


def test_another_recorded_provenance_is_refused(run: Trained) -> None:
    run.edit("run.json", lambda record: record["dataset"].update(license="another licence"))

    _refused(run, "the loaded dataset is not the one run.json records")


def test_a_different_library_version_is_refused(run: Trained) -> None:
    run.edit("run.json", lambda record: record["library_versions"].update(xgboost="0.0.1"))

    _refused(run, "was recorded with xgboost 0.0.1")


def test_a_source_file_other_than_the_pinned_one_is_refused(run: Trained, tmp_path: Path) -> None:
    changed = tmp_path / "changed" / "ulb"
    changed.mkdir(parents=True)
    original = (run.root / SOURCE_FILENAME).read_bytes()
    (changed / SOURCE_FILENAME).write_bytes(original.replace(b"\n1,", b"\n2,", 1))

    _refused(run, "cannot be read as the run read it", root=changed)
