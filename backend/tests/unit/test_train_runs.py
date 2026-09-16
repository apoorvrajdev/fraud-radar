"""Phase 5D — training into a run directory, for any dataset.

A named run, and every run on a registered benchmark dataset, is written to
`ml/artifacts/runs/<run-name>/` with a `run.json` recording the dataset's
provenance, the featureset, the seed and the period each fold covers. The
unnamed synthetic default still writes the served artifact directory, which is
what CI's bootstrap relies on.

The Sparkov path runs end to end on a generated fixture corpus through the real
adapter, batch builder and feature cache. Only the hyperparameter search is
stubbed: it belongs to `ml.tuning`, and nothing here depends on its result.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.fraud.feature_spec import FEATURESETS
from ml import loading, train
from ml.data import LabelledDataset
from ml.features.cache import load_feature_cache
from ml.reporting import LABEL_DELAY_NOTE
from ml.runs import RUN_METADATA_FILENAME, load_run_metadata, split_periods
from ml.splits import chronological_split
from ml.tuning import TuningResult
from tests.unit.test_dataset_sparkov import _row, _write_csv, fixture_adapter
from tests.unit.test_train_pipeline import OBSERVED_METRICS
from tests.unit.test_train_pipeline import _dataset as synthetic_matrix

N_ROWS = 100
CORPUS_START = datetime(2019, 1, 1, 8, 0)
STEP = timedelta(hours=7)

ARTIFACT_FILES = [
    "feature_list.json",
    "metrics.json",
    "model.json",
    "pr_curve.png",
    "threshold.json",
    "training_metadata.json",
]
RUN_FILES = sorted([*ARTIFACT_FILES, RUN_METADATA_FILENAME])


def _benchmark_corpus(tmp_path: Path) -> Path:
    """One hundred transactions seven hours apart, every fifth one fraudulent.

    The chronological 70/15/15 folds then hold 14, 3 and 3 frauds, so every
    fold has both classes and the threshold and test metrics are defined.
    """
    root = tmp_path / "raw" / "sparkov"
    root.mkdir(parents=True)
    categories = ("grocery_pos", "shopping_net", "travel", "gas_transport")
    rows = [
        _row(
            index,
            trans_num=f"t{index}",
            cc_num=f"411100000000000{index % 4}",
            category=categories[index % len(categories)],
            amt=f"{20 + (index * 37) % 400}.00",
            timestamp=(CORPUS_START + STEP * index).strftime("%Y-%m-%d %H:%M:%S"),
            is_fraud="1" if index % 5 == 0 else "0",
        )
        for index in range(N_ROWS)
    ]
    _write_csv(root / "fraudTrain.csv", rows)
    return root


def _at(index: int) -> datetime:
    return (CORPUS_START + STEP * index).replace(tzinfo=UTC)


@pytest.fixture
def tuner_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def stub_tune(features: Any, labels: Any, **options: Any) -> TuningResult:
        calls.append(options)
        return TuningResult(
            best_params={"n_estimators": 30, "max_depth": 3, "learning_rate": 0.3},
            best_score=0.5,
            cv_results_summary={"mean_test_score": [0.5], "std_test_score": [0.0]},
        )

    monkeypatch.setattr(train, "tune_hyperparameters", stub_tune)
    return calls


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch working directory, with the runs root redirected into it.

    The default artifact directory is relative, so anything written there would
    land under `workspace/ml/artifacts`, where a test can see it.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "RUNS_ROOT", tmp_path / "runs")
    return tmp_path


@pytest.fixture
def sparkov_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = _benchmark_corpus(tmp_path)
    adapter = fixture_adapter(tmp_path)
    monkeypatch.setattr(loading, "get_adapter", lambda name: adapter)
    return root


def _train_sparkov(workspace: Path, sparkov_root: Path, *extra: str) -> Path:
    train.main(
        [
            "--dataset", "sparkov",
            "--root", str(sparkov_root),
            "--cache-root", str(workspace / "cache"),
            "--run-name", "sparkov-fixture",
            "--n-iter", "3",
            "--cv-splits", "2",
            "--target-fpr", "0.05",
            *extra,
        ]
    )
    return workspace / "runs" / "sparkov-fixture"


def _read(directory: Path, name: str) -> Any:
    return json.loads((directory / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# A registered benchmark dataset
# ---------------------------------------------------------------------------


def test_a_benchmark_run_writes_its_artifacts_and_run_record_to_its_run_directory(
    workspace: Path, sparkov_root: Path, tuner_calls: list[dict[str, Any]]
) -> None:
    run = _train_sparkov(workspace, sparkov_root)

    assert sorted(path.name for path in run.iterdir()) == RUN_FILES
    assert len(list((workspace / "cache").glob("sparkov_v1_*.npz"))) == 1
    assert not (workspace / "ml" / "artifacts").exists(), "served artifacts were touched"
    assert tuner_calls == [{"n_iter": 3, "n_splits": 2, "random_state": train.RANDOM_STATE}]


def test_the_run_record_carries_provenance_featureset_seed_and_fold_periods(
    workspace: Path, sparkov_root: Path, tuner_calls: list[dict[str, Any]]
) -> None:
    _train_sparkov(workspace, sparkov_root, "--seed", "11")

    record = load_run_metadata("sparkov-fixture", runs_root=workspace / "runs")

    assert record.dataset.name == "sparkov"
    assert record.dataset.row_count == N_ROWS
    assert record.dataset.subsample is not None
    assert record.dataset.subsample.seed == 11
    assert record.featureset_version == "v1"
    assert record.seed == 11
    assert "entity-subsampling seed" in record.notes
    assert record.code_version == train.current_git_commit()
    assert "xgboost" in record.library_versions
    # 100 rows split 70 / 15 / 15 in time order.
    assert [(split.name, split.start, split.end) for split in record.splits] == [
        ("train", _at(0), _at(69)),
        ("val", _at(70), _at(84)),
        ("test", _at(85), _at(99)),
    ]


def test_the_subsample_seed_and_the_model_random_state_are_recorded_apart(
    workspace: Path, sparkov_root: Path, tuner_calls: list[dict[str, Any]]
) -> None:
    """--seed draws the subsample; it never becomes the model's random state."""
    run = _train_sparkov(workspace, sparkov_root, "--max-cards", "3", "--seed", "11")

    run_record = _read(run, RUN_METADATA_FILENAME)
    assert run_record["seed"] == run_record["dataset"]["subsample"]["seed"] == 11
    assert run_record["dataset"]["subsample"]["strategy"] == "cards"
    assert _read(run, "training_metadata.json")["random_state"] == train.RANDOM_STATE == 42
    assert tuner_calls[0]["random_state"] == train.RANDOM_STATE


def test_benchmark_metrics_record_observed_results_without_synthetic_targets(
    workspace: Path, sparkov_root: Path, tuner_calls: list[dict[str, Any]]
) -> None:
    metrics = _read(_train_sparkov(workspace, sparkov_root), "metrics.json")

    assert set(metrics) == OBSERVED_METRICS
    assert not any(key.startswith("target_") for key in metrics)


def test_benchmark_metrics_report_the_folds_recorded_in_the_run_record(
    workspace: Path, sparkov_root: Path, tuner_calls: list[dict[str, Any]]
) -> None:
    """The context's fraud counts are those of the periods run.json records.

    Each corpus row is placed in a fold by comparing its timestamp with the
    recorded, timezone-aware fold periods; every fifth row is a fraud.
    """
    run = _train_sparkov(workspace, sparkov_root)
    record = load_run_metadata("sparkov-fixture", runs_root=workspace / "runs")
    frauds_by_period = {split.name: 0 for split in record.splits}
    rows_by_period = dict(frauds_by_period)
    for index in range(N_ROWS):
        (name,) = [s.name for s in record.splits if s.start <= _at(index) <= s.end]
        rows_by_period[name] += 1
        frauds_by_period[name] += int(index % 5 == 0)

    context = _read(run, "metrics.json")["context"]

    assert all(split.start.utcoffset() == timedelta(0) for split in record.splits)
    assert context["fraud_counts"] == frauds_by_period == {"train": 14, "val": 3, "test": 3}
    assert context["test_prevalence"] == frauds_by_period["test"] / rows_by_period["test"]
    assert context["label_delay_note"] == LABEL_DELAY_NOTE


def test_a_benchmark_run_reports_the_realised_test_fpr_at_its_recorded_threshold(
    workspace: Path, sparkov_root: Path, tuner_calls: list[dict[str, Any]]
) -> None:
    """Measured at the threshold in threshold.json — here the fallback, as recorded."""
    run = _train_sparkov(workspace, sparkov_root)
    metrics = _read(run, "metrics.json")
    threshold = _read(run, "threshold.json")

    at_threshold = metrics["at_operating_threshold"]
    assert threshold["fallback_used"] is True
    assert at_threshold["threshold"] == threshold["value"]
    legitimate = at_threshold["false_positives"] + at_threshold["true_negatives"]
    assert legitimate == 12
    assert metrics["context"]["realised_fpr_on_test_at_operating_threshold"] == (
        at_threshold["false_positives"] / legitimate
    )


def test_a_benchmark_run_trains_on_the_frozen_featureset(
    workspace: Path, sparkov_root: Path, tuner_calls: list[dict[str, Any]]
) -> None:
    run = _train_sparkov(workspace, sparkov_root)

    assert _read(run, "feature_list.json") == {"features": FEATURESETS["v1"]}
    metadata = _read(run, "training_metadata.json")
    assert metadata["dataset_size"] == N_ROWS
    assert (metadata["train_size"], metadata["val_size"], metadata["test_size"]) == (70, 15, 15)


def test_a_benchmark_run_records_the_facts_of_its_fit(
    workspace: Path, sparkov_root: Path, tuner_calls: list[dict[str, Any]]
) -> None:
    metadata = _read(_train_sparkov(workspace, sparkov_root), "training_metadata.json")

    assert metadata["random_state"] == train.RANDOM_STATE
    assert (metadata["tuning_iterations"], metadata["tuning_cv_folds"]) == (3, 2)
    assert metadata["early_stopping_rounds"] == train.EARLY_STOPPING_ROUNDS
    # The training fold holds 70 rows, 14 of them fraudulent: 56 / 14.
    assert metadata["scale_pos_weight"] == 4.0
    assert isinstance(metadata["best_iteration"], int)
    assert metadata["best_iteration"] >= 0


def test_a_benchmark_run_reloads_exactly_from_its_run_record(
    workspace: Path, sparkov_root: Path, tuner_calls: list[dict[str, Any]]
) -> None:
    """The request built from run.json reads back the matrix training used.

    Only where the files are is supplied; the subsample and featureset come
    from the record.
    """
    _train_sparkov(workspace, sparkov_root, "--max-cards", "3", "--seed", "11")
    record = load_run_metadata("sparkov-fixture", runs_root=workspace / "runs")
    (cache_file,) = (workspace / "cache").glob("sparkov_v1_*.npz")
    trained_on, _ = load_feature_cache(cache_file)

    request = loading.DataRequest.from_run_record(
        record, root=sparkov_root, cache_root=workspace / "cache"
    )
    data = loading.load_run_data(request)

    assert (request.dataset, request.featureset, request.max_entities, request.seed) == (
        "sparkov",
        "v1",
        3,
        11,
    )
    assert data.cache_hit is True
    assert data.cache_fingerprint is not None
    assert data.cache_fingerprint in record.notes
    np.testing.assert_array_equal(data.ds.X, trained_on.X)
    np.testing.assert_array_equal(data.ds.y, trained_on.y)
    assert data.ds.transaction_ids == trained_on.transaction_ids
    assert list(data.ds.timestamps) == list(trained_on.timestamps)
    assert data.provenance is not None
    loaded, recorded = data.provenance.to_dict(), record.dataset.to_dict()
    # retrieved_at is when the files were read, not which bytes they hold.
    loaded.pop("retrieved_at")
    recorded.pop("retrieved_at")
    assert loaded == recorded
    assert record.splits == split_periods(
        data.ds.timestamps, chronological_split(data.ds.timestamps)
    )


def test_a_benchmark_run_counts_its_live_features_from_its_own_matrix(
    workspace: Path, sparkov_root: Path, tuner_calls: list[dict[str, Any]]
) -> None:
    """The recorded constants are those of the cached matrix's training rows.

    The fixture corpus is US-only, has no account-open date or risk tier, and
    no high-risk category, so the five columns the methodology expects to be
    constant on Sparkov are constant here too, among whatever else the small
    corpus leaves constant.
    """
    run = _train_sparkov(workspace, sparkov_root)
    (cache_file,) = (workspace / "cache").glob("sparkov_v1_*.npz")
    matrix, _ = load_feature_cache(cache_file)
    training = matrix.X[chronological_split(matrix.timestamps).train]
    expected_constant = [
        name
        for column, name in enumerate(matrix.feature_names)
        if (training[:, column] == training[0, column]).all()
    ]
    expected_constant_whole = [
        name
        for column, name in enumerate(matrix.feature_names)
        if (matrix.X[:, column] == matrix.X[0, column]).all()
    ]

    metadata = _read(run, "training_metadata.json")

    assert metadata["constant_features"] == expected_constant
    assert metadata["live_feature_count"] == len(FEATURESETS["v1"]) - len(expected_constant)
    assert metadata["constant_features_whole_matrix"] == expected_constant_whole
    assert metadata["live_feature_count_whole_matrix"] == (
        len(FEATURESETS["v1"]) - len(expected_constant_whole)
    )
    assert {
        "country_mismatch_customer",
        "country_mismatch_merchant",
        "customer_account_age_days",
        "customer_risk_tier_encoded",
        "is_high_risk_category",
    } <= set(metadata["constant_features"])


def test_a_benchmark_run_records_that_its_threshold_fell_back(
    workspace: Path, sparkov_root: Path, tuner_calls: list[dict[str, Any]]
) -> None:
    """A val fold too small for its FPR ceiling falls back, and says so.

    The fixture's val fold holds 12 legitimate rows, so a 5% ceiling allows
    none of them at or above the threshold. The fixture model scores some above
    every fraud, no threshold qualifies, and the existing fallback applies.
    """
    threshold = _read(_train_sparkov(workspace, sparkov_root), "threshold.json")

    assert threshold["fallback_used"] is True
    assert threshold["value"] == train.FALLBACK_THRESHOLD
    assert threshold["realised_fpr_on_val"] > threshold["target_fpr"] == 0.05


# ---------------------------------------------------------------------------
# The synthetic dataset
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_loader(monkeypatch: pytest.MonkeyPatch) -> LabelledDataset:
    ds = synthetic_matrix()

    @contextmanager
    def stub_session() -> Iterator[object]:
        yield object()

    monkeypatch.setattr(loading, "SessionLocal", stub_session)
    monkeypatch.setattr(loading, "load_dataset_with_csv_labels", lambda *args, **kwargs: ds)
    return ds


def test_a_named_synthetic_run_records_the_label_csv_and_keeps_its_targets(
    workspace: Path, synthetic_loader: LabelledDataset, tuner_calls: list[dict[str, Any]]
) -> None:
    labels = workspace / "synthetic_transactions.csv"
    labels.write_text("id,is_fraud\ntx0,False\n", encoding="utf-8")

    train.main(
        [
            "--run-name", "synthetic-fixture",
            "--csv-path", str(labels),
            "--n-iter", "3",
            "--cv-splits", "2",
            "--target-fpr", "0.05",
        ]
    )

    run = workspace / "runs" / "synthetic-fixture"
    assert sorted(path.name for path in run.iterdir()) == RUN_FILES
    assert not (workspace / "ml" / "artifacts").exists(), "served artifacts were touched"

    record = load_run_metadata("synthetic-fixture", runs_root=workspace / "runs")
    assert record.dataset.name == "synthetic"
    assert dict(record.dataset.files) == {
        "synthetic_transactions.csv": hashlib.sha256(labels.read_bytes()).hexdigest()
    }
    assert "no digest" in record.dataset.notes
    # No subsample, so no seed in run.json; the model's random state is a fit fact.
    assert record.dataset.subsample is None
    assert record.seed is None
    assert _read(run, RUN_METADATA_FILENAME)["seed"] is None
    assert _read(run, "training_metadata.json")["random_state"] == train.RANDOM_STATE
    assert "random_state" in record.notes
    ds = synthetic_loader
    assert record.splits == split_periods(ds.timestamps, chronological_split(ds.timestamps))

    metrics = _read(run, "metrics.json")
    assert set(metrics) == OBSERVED_METRICS | set(train.SYNTHETIC_TARGETS)


def test_the_unnamed_synthetic_default_writes_no_run_record(
    workspace: Path, synthetic_loader: LabelledDataset, tuner_calls: list[dict[str, Any]]
) -> None:
    served = workspace / "served"

    train.main(["--artifact-dir", str(served), "--n-iter", "3", "--cv-splits", "2"])

    assert sorted(path.name for path in served.iterdir()) == ARTIFACT_FILES
    assert not (workspace / "runs").exists()


# ---------------------------------------------------------------------------
# Refused combinations — refused before anything is loaded or trained
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--dataset", "sparkov"], id="benchmark-without-run-name"),
        pytest.param(
            ["--dataset", "sparkov", "--run-name", "r", "--limit", "10"],
            id="benchmark-with-row-limit",
        ),
        pytest.param(["--max-cards", "2"], id="synthetic-with-max-cards"),
        pytest.param(["--root", "raw"], id="synthetic-with-root"),
        pytest.param(["--seed", "7"], id="synthetic-with-seed"),
        pytest.param(["--cache-root", "cache"], id="synthetic-with-cache-root"),
        pytest.param(["--refresh"], id="synthetic-with-refresh"),
        pytest.param(["--full-corpus"], id="synthetic-with-full-corpus"),
        pytest.param(["--artifact-dir", "out", "--run-name", "r"], id="both-output-locations"),
        pytest.param(["--run-name", "../escape"], id="unsafe-run-name"),
        pytest.param(["--dataset", "ghost", "--run-name", "r"], id="unregistered-dataset"),
    ],
)
def test_misleading_combinations_are_refused_before_any_work(
    argv: list[str],
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    tuner_calls: list[dict[str, Any]],
) -> None:
    loads: list[str] = []
    monkeypatch.setattr(
        train, "load_run_data", lambda request, **_: loads.append(request.dataset)
    )

    with pytest.raises(SystemExit) as refused:
        train.main(argv)

    assert refused.value.code == 2
    assert loads == []
    assert tuner_calls == []
    assert list(workspace.iterdir()) == []
