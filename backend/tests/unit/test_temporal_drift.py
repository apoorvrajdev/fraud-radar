"""Phase 5D — temporal drift: a model tuned on 2019, measured on each month of 2020.

The fixture is a generated Sparkov-format corpus spanning 2019 and 2020, read
through the real adapter, batch builder and feature cache from the test's
temporary directory. It is built to exercise the frozen protocol's edges:

- rows exactly at the train/val and val/2020 boundary instants, and one row
  either side of the whole experiment, which belongs to no period;
- March 2020 holds no frauds, June 2020 holds nothing but frauds, and
  September 2020 holds no rows at all.

The search is stubbed; the early-stopped refit and the threshold selection
are real. Every matrix given to the tuner, the refit, the threshold selection
and the model's scoring is recorded.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

from app.fraud.feature_spec import FEATURE_NAMES
from ml import loading, train
from ml.data import LabelledDataset
from ml.datasets.sparkov import SparkovAdapter
from ml.evaluation import confusion_at_threshold, pr_auc
from ml.experiments import temporal_drift
from ml.experiments.temporal_drift import (
    DRIFT_METRICS_FILENAME,
    TRAIN_PERIOD,
    VAL_PERIOD,
    DriftError,
    assign_periods,
    evaluation_months,
    month_metrics,
)
from ml.features.cache import load_feature_cache
from ml.loading import DataRequest
from ml.paths import RAW_DATA_DIR
from ml.reporting import LABEL_DELAY_NOTE
from ml.tuning import TuningResult
from tests.unit.test_dataset_sparkov import _row, _write_csv, fixture_adapter

STUB_PARAMS: dict[str, Any] = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.3}
TARGET_FPR = 0.05
RUN = "drift-fixture"

# Rows that fall in no period: just before the train period, and exactly at the
# end of 2020, which the half-open December interval excludes.
BEFORE_ALL = datetime(2018, 12, 31, 23, 0)
AT_THE_END = datetime(2021, 1, 1, 0, 0)


# ---------------------------------------------------------------------------
# The fixture corpus
# ---------------------------------------------------------------------------


def _corpus_rows() -> list[tuple[datetime, bool]]:
    """Every fixture transaction as (naive wall-clock time, is_fraud)."""
    rows: list[tuple[datetime, bool]] = [(BEFORE_ALL, False), (AT_THE_END, False)]

    moment = datetime(2019, 1, 1, 12, 0)
    index = 0
    while moment < datetime(2019, 10, 31, 12, 0):
        rows.append((moment, index % 6 == 0))
        moment += timedelta(hours=36)
        index += 1
    rows.append((datetime(2019, 10, 31, 23, 59, 59), True))  # last train instant
    rows.append((datetime(2019, 11, 1, 0, 0), False))  # first val instant

    moment = datetime(2019, 11, 1, 9, 0)
    index = 0
    while moment < datetime(2019, 12, 31, 12, 0):
        rows.append((moment, index % 5 == 0))
        moment += timedelta(hours=30)
        index += 1

    rows.append((datetime(2020, 1, 1, 0, 0), True))  # first January instant
    rows.append((datetime(2020, 12, 31, 23, 59, 59), False))  # last December instant
    for month in range(1, 13):
        if month == 9:
            continue  # September 2020 holds no rows
        for day, hour in ((3, 8), (8, 14), (13, 20), (18, 2), (23, 11), (27, 17)):
            if month == 3:
                fraud = False  # March 2020 holds no frauds
            elif month == 6:
                fraud = True  # June 2020 holds nothing but frauds
            else:
                fraud = day in (8, 23)
            rows.append((datetime(2020, month, day, hour, 0), fraud))
    return rows


def _write_corpus(root: Path) -> Path:
    corpus = root / "raw" / "sparkov"
    corpus.mkdir(parents=True)
    categories = ("grocery_pos", "shopping_net", "travel", "gas_transport", "misc_net")
    rows = [
        _row(
            index,
            trans_num=f"d{index}",
            cc_num=f"422200000000000{index % 5}",
            category=categories[index % len(categories)],
            amt=f"{15 + (index * 53) % 900}.{index % 100:02d}",
            timestamp=moment.strftime("%Y-%m-%d %H:%M:%S"),
            is_fraud="1" if fraud else "0",
        )
        for index, (moment, fraud) in enumerate(_corpus_rows())
    ]
    _write_csv(corpus / "fraudTrain.csv", rows)
    return corpus


def _in(period: temporal_drift.Period, timestamps: np.ndarray) -> np.ndarray:
    """Rows in `period`, found without the module's own assignment."""
    return np.asarray(
        [index for index, ts in enumerate(timestamps) if period.start <= ts < period.end],
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# One observed drift run
# ---------------------------------------------------------------------------


@dataclass
class Observed:
    root: Path
    corpus: Path
    runs_root: Path
    cache_root: Path
    measurement: temporal_drift.DriftMeasurement
    payload: dict[str, Any]
    written: Path
    matrix: LabelledDataset
    tuner_calls: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = field(default_factory=list)
    fit_calls: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    threshold_calls: list[tuple[np.ndarray, float]] = field(default_factory=list)
    scored: list[np.ndarray] = field(default_factory=list)
    adapter_roots: list[Path] = field(default_factory=list)


def _observe(root: Path) -> Observed:
    corpus = _write_corpus(root)
    adapter = fixture_adapter(root)
    tuner_calls: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    fit_calls: list[tuple[np.ndarray, np.ndarray]] = []
    threshold_calls: list[tuple[np.ndarray, float]] = []
    scored: list[np.ndarray] = []
    adapter_roots: list[Path] = []

    def stub_tune(features: np.ndarray, labels: np.ndarray, **options: Any) -> TuningResult:
        tuner_calls.append((np.array(features), np.array(labels), options))
        return TuningResult(
            best_params=dict(STUB_PARAMS),
            best_score=0.5,
            cv_results_summary={"mean_test_score": [0.5], "std_test_score": [0.0]},
        )

    real_fit = train._final_fit

    def fit_spy(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        best_params: dict[str, object],
    ) -> Any:
        fit_calls.append((np.array(x_train), np.array(x_val)))
        return real_fit(x_train, y_train, x_val, y_val, best_params)

    real_threshold = train.find_threshold_at_fpr

    def threshold_spy(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
        threshold_calls.append((np.array(labels), target))
        return real_threshold(labels, scores, target)

    real_predict = xgb.XGBClassifier.predict_proba

    def predict_spy(self: xgb.XGBClassifier, features: Any, *args: Any, **kwargs: Any) -> Any:
        scored.append(np.array(features))
        return real_predict(self, features, *args, **kwargs)

    real_load = SparkovAdapter.load_detailed

    def load_spy(self: SparkovAdapter, source: Path, **options: Any) -> Any:
        adapter_roots.append(Path(source))
        return real_load(self, source, **options)

    request = DataRequest(dataset="sparkov", root=corpus, cache_root=root / "cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loading, "get_adapter", lambda name: adapter)
        patch.setattr(SparkovAdapter, "load_detailed", load_spy)
        patch.setattr(train, "tune_hyperparameters", stub_tune)
        patch.setattr(train, "_final_fit", fit_spy)
        patch.setattr(train, "find_threshold_at_fpr", threshold_spy)
        patch.setattr(xgb.XGBClassifier, "predict_proba", predict_spy)
        measurement, payload, written = temporal_drift.run_drift(
            request,
            RUN,
            n_iter=3,
            cv_splits=2,
            target_fpr=TARGET_FPR,
            runs_root=root / "runs",
        )

    (cache_file,) = (root / "cache").glob("sparkov_v1_*.npz")
    matrix, _ = load_feature_cache(cache_file)
    return Observed(
        root=root,
        corpus=corpus,
        runs_root=root / "runs",
        cache_root=root / "cache",
        measurement=measurement,
        payload=payload,
        written=written,
        matrix=matrix,
        tuner_calls=tuner_calls,
        fit_calls=fit_calls,
        threshold_calls=threshold_calls,
        scored=scored,
        adapter_roots=adapter_roots,
    )


@pytest.fixture(scope="module")
def observed(tmp_path_factory: pytest.TempPathFactory) -> Observed:
    return _observe(tmp_path_factory.mktemp("drift"))


def _month(observed: Observed, name: str) -> dict[str, Any]:
    (entry,) = [month for month in observed.payload["months"] if month["name"] == name]
    return entry


# ---------------------------------------------------------------------------
# The frozen periods
# ---------------------------------------------------------------------------


def test_the_periods_are_the_frozen_half_open_intervals() -> None:
    months = evaluation_months()

    assert (TRAIN_PERIOD.start, TRAIN_PERIOD.end) == (
        datetime(2019, 1, 1, tzinfo=UTC),
        datetime(2019, 11, 1, tzinfo=UTC),
    )
    assert (VAL_PERIOD.start, VAL_PERIOD.end) == (
        datetime(2019, 11, 1, tzinfo=UTC),
        datetime(2020, 1, 1, tzinfo=UTC),
    )
    assert [month.name for month in months] == [f"2020-{m:02d}" for m in range(1, 13)]
    assert months[0].start == VAL_PERIOD.end
    assert months[-1].end == datetime(2021, 1, 1, tzinfo=UTC)
    for earlier, later in pairwise(months):
        assert earlier.end == later.start


def test_every_evaluation_month_lies_strictly_after_the_train_and_val_periods() -> None:
    for month in evaluation_months():
        assert month.start >= VAL_PERIOD.end > TRAIN_PERIOD.end
        assert month.start > TRAIN_PERIOD.end


def test_a_boundary_instant_belongs_to_the_period_that_starts_there() -> None:
    instants = [
        datetime(2018, 12, 31, 23, 59, 59, tzinfo=UTC),  # none
        datetime(2019, 1, 1, tzinfo=UTC),  # train
        datetime(2019, 10, 31, 23, 59, 59, 999999, tzinfo=UTC),  # train
        datetime(2019, 11, 1, tzinfo=UTC),  # val
        datetime(2020, 1, 1, tzinfo=UTC),  # January
        datetime(2020, 2, 1, tzinfo=UTC),  # February
        datetime(2021, 1, 1, tzinfo=UTC),  # none
    ]

    rows = assign_periods(np.asarray(instants, dtype=object))

    assert rows.train.tolist() == [1, 2]
    assert rows.val.tolist() == [3]
    by_month = {month.name: indices.tolist() for month, indices in rows.months}
    assert by_month["2020-01"] == [4]
    assert by_month["2020-02"] == [5]
    assert rows.outside == 2


def test_the_same_instant_in_another_timezone_falls_in_the_same_period() -> None:
    """Midnight on 1 January 2020 UTC is still 31 December 2019 at UTC-5."""
    utc_minus_five = datetime(2019, 12, 31, 19, 0, tzinfo=timezone(timedelta(hours=-5)))

    rows = assign_periods(np.asarray([utc_minus_five], dtype=object))

    assert rows.val.tolist() == []
    by_month = {month.name: indices.tolist() for month, indices in rows.months}
    assert by_month["2020-01"] == [0]


def test_naive_timestamps_are_refused() -> None:
    with pytest.raises(DriftError, match="naive timestamps"):
        assign_periods(np.asarray([datetime(2020, 1, 1)], dtype=object))


# ---------------------------------------------------------------------------
# One month's metrics
# ---------------------------------------------------------------------------


def test_a_month_with_both_classes_reports_every_metric_from_the_existing_definitions() -> None:
    labels = np.array([1, 0, 0, 1, 0, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.1, 0.3, 0.6, 0.2, 0.05, 0.7])
    confusion = confusion_at_threshold(labels, scores, 0.5)

    metrics = month_metrics(labels, scores, 0.5)

    assert metrics["volume"] == 8
    assert metrics["fraud_count"] == 3
    assert metrics["fraud_rate"] == 3 / 8
    assert metrics["pr_auc"] == pr_auc(labels, scores)
    assert metrics["recall"] == confusion.recall == 2 / 3
    assert metrics["precision"] == confusion.precision == 2 / 4
    assert metrics["realised_fpr"] == 2 / 5
    assert (
        metrics["true_positives"],
        metrics["false_positives"],
        metrics["true_negatives"],
        metrics["false_negatives"],
    ) == (2, 2, 3, 1)


def test_a_month_without_frauds_has_no_pr_auc_or_recall() -> None:
    metrics = month_metrics(np.array([0, 0, 0]), np.array([0.9, 0.2, 0.1]), 0.5)

    assert (metrics["pr_auc"], metrics["recall"]) == (None, None)
    assert metrics["realised_fpr"] == 1 / 3
    assert metrics["precision"] == 0.0
    assert metrics["fraud_rate"] == 0.0


def test_a_month_without_legitimate_rows_has_no_realised_fpr() -> None:
    metrics = month_metrics(np.array([1, 1]), np.array([0.9, 0.1]), 0.5)

    assert metrics["realised_fpr"] is None
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["pr_auc"] is not None


def test_a_month_in_which_nothing_is_flagged_has_no_precision() -> None:
    metrics = month_metrics(np.array([1, 0, 0]), np.array([0.2, 0.3, 0.1]), 0.5)

    assert metrics["precision"] is None
    assert metrics["recall"] == 0.0
    assert metrics["realised_fpr"] == 0.0


def test_an_empty_month_reports_its_volume_and_nothing_else() -> None:
    metrics = month_metrics(np.array([], dtype=np.int64), np.array([]), 0.5)

    assert metrics["volume"] == 0
    assert metrics["fraud_count"] == 0
    for name in ("fraud_rate", "pr_auc", "recall", "precision", "realised_fpr"):
        assert metrics[name] is None, name


# ---------------------------------------------------------------------------
# Which rows each step sees
# ---------------------------------------------------------------------------


def test_hyperparameters_are_searched_on_the_train_period_alone(observed: Observed) -> None:
    train_rows = _in(TRAIN_PERIOD, observed.matrix.timestamps)

    (features, labels, options) = observed.tuner_calls[0]

    assert len(observed.tuner_calls) == 1
    np.testing.assert_array_equal(features, observed.matrix.X[train_rows])
    np.testing.assert_array_equal(labels, observed.matrix.y[train_rows])
    assert options == {"n_iter": 3, "n_splits": 2, "random_state": train.RANDOM_STATE}


def test_the_refit_trains_on_the_train_period_and_stops_early_against_val(
    observed: Observed,
) -> None:
    timestamps = observed.matrix.timestamps

    (x_train, x_val) = observed.fit_calls[0]

    assert len(observed.fit_calls) == 1
    np.testing.assert_array_equal(x_train, observed.matrix.X[_in(TRAIN_PERIOD, timestamps)])
    np.testing.assert_array_equal(x_val, observed.matrix.X[_in(VAL_PERIOD, timestamps)])


def test_the_threshold_is_selected_once_on_the_val_period(observed: Observed) -> None:
    val_rows = _in(VAL_PERIOD, observed.matrix.timestamps)

    (labels, target) = observed.threshold_calls[0]

    assert len(observed.threshold_calls) == 1
    np.testing.assert_array_equal(labels, observed.matrix.y[val_rows])
    assert target == TARGET_FPR
    threshold = observed.measurement.selected.threshold
    assert observed.payload["threshold"]["value"] == threshold.value
    assert observed.payload["threshold"]["selected_on"] == "val period"


def test_the_model_scores_val_once_then_each_2020_month_once_and_no_train_row(
    observed: Observed,
) -> None:
    timestamps = observed.matrix.timestamps
    non_empty = [month for month in evaluation_months() if len(_in(month, timestamps))]

    assert len(observed.scored) == 1 + len(non_empty) == 12
    np.testing.assert_array_equal(
        observed.scored[0], observed.matrix.X[_in(VAL_PERIOD, timestamps)]
    )
    for scored, month in zip(observed.scored[1:], non_empty, strict=True):
        np.testing.assert_array_equal(scored, observed.matrix.X[_in(month, timestamps)])


# ---------------------------------------------------------------------------
# What each month reports
# ---------------------------------------------------------------------------


def test_each_month_is_measured_at_the_single_val_threshold(observed: Observed) -> None:
    timestamps = observed.matrix.timestamps
    model = observed.measurement.selected.model
    threshold = observed.measurement.selected.threshold.value

    for month in evaluation_months():
        rows = _in(month, timestamps)
        labels = observed.matrix.y[rows]
        scores = (
            model.predict_proba(observed.matrix.X[rows])[:, 1] if len(rows) else np.empty(0)
        )
        expected = {**month.to_dict(), **month_metrics(labels, scores, threshold)}
        assert _month(observed, month.name) == expected


def test_months_without_frauds_legitimate_rows_or_any_rows_report_nulls(
    observed: Observed,
) -> None:
    march, june, september = (_month(observed, name) for name in ("2020-03", "2020-06", "2020-09"))

    assert march["fraud_count"] == 0
    assert (march["pr_auc"], march["recall"]) == (None, None)
    assert march["realised_fpr"] is not None

    assert june["fraud_count"] == june["volume"] > 0
    assert june["realised_fpr"] is None
    assert june["recall"] is not None

    assert september["volume"] == 0
    assert [september[key] for key in ("fraud_rate", "pr_auc", "recall", "precision")] == [
        None
    ] * 4
    assert september["realised_fpr"] is None


def test_the_boundary_rows_fall_in_the_periods_that_start_at_them(observed: Observed) -> None:
    periods = observed.payload["periods"]

    assert periods["train"]["last_timestamp"] == "2019-10-31T23:59:59+00:00"
    assert periods["val"]["first_timestamp"] == "2019-11-01T00:00:00+00:00"
    assert _month(observed, "2020-01")["volume"] == 6 + 1
    assert _month(observed, "2020-12")["volume"] == 6 + 1
    assert observed.payload["rows_outside_periods"] == 2


# ---------------------------------------------------------------------------
# The written result
# ---------------------------------------------------------------------------


def test_the_result_records_its_procedure_and_provenance(observed: Observed) -> None:
    payload = observed.payload
    selected = observed.measurement.selected
    timestamps = observed.matrix.timestamps

    assert json.loads(observed.written.read_text(encoding="utf-8")) == payload
    assert payload["drift_metrics_version"] == "1"
    assert payload["run_name"] == RUN
    assert payload["dataset"]["name"] == "sparkov"
    assert payload["featureset_version"] == "v1"
    (cache_file,) = observed.cache_root.glob("sparkov_v1_*.npz")
    assert payload["feature_cache_fingerprint"] is not None
    assert payload["feature_cache_fingerprint"] in cache_file.name
    assert payload["tuning"]["best_hyperparameters"] == STUB_PARAMS
    assert (payload["tuning"]["iterations"], payload["tuning"]["cv_folds"]) == (3, 2)
    assert payload["fit"]["best_iteration"] == selected.fit.best_iteration
    assert payload["threshold"]["fallback_used"] == selected.threshold.fallback_used
    assert payload["threshold"]["target_fpr"] == TARGET_FPR
    for name, period in (("train", TRAIN_PERIOD), ("val", VAL_PERIOD)):
        rows = _in(period, timestamps)
        assert payload["periods"][name]["rows"] == len(rows)
        assert payload["periods"][name]["fraud_count"] == int(observed.matrix.y[rows].sum())
        assert payload["periods"][name]["start"] == period.start.isoformat()
    assert payload["label_delay_note"] == LABEL_DELAY_NOTE
    assert "never inform the main runs" in payload["note"]
    assert "NaN" not in observed.written.read_text(encoding="utf-8")


def test_only_the_drift_result_is_written_and_only_into_its_own_directory(
    observed: Observed,
) -> None:
    assert sorted(path.name for path in observed.runs_root.rglob("*") if path.is_file()) == [
        DRIFT_METRICS_FILENAME
    ]
    assert observed.written == observed.runs_root / RUN / DRIFT_METRICS_FILENAME


def test_the_real_sparkov_corpus_is_never_read(observed: Observed) -> None:
    assert observed.adapter_roots
    for root in observed.adapter_roots:
        assert observed.root in root.parents
        assert RAW_DATA_DIR not in root.parents


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_trained_runs_directory_is_refused_before_loading_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trained = tmp_path / "runs" / "sparkov_v1_200cards"
    trained.mkdir(parents=True)
    (trained / "run.json").write_text("{}", encoding="utf-8")
    loads: list[object] = []
    monkeypatch.setattr(temporal_drift, "load_run_data", lambda *a, **k: loads.append(a))

    with pytest.raises(DriftError, match="never inform the main runs"):
        temporal_drift.run_drift(
            DataRequest(dataset="sparkov"),
            "sparkov_v1_200cards",
            n_iter=1,
            cv_splits=2,
            target_fpr=0.01,
            runs_root=tmp_path / "runs",
        )

    assert loads == []
    assert sorted(path.name for path in trained.iterdir()) == ["run.json"]


@pytest.mark.parametrize(
    ("period", "fraud"), [("train", False), ("val", True)], ids=["train-without-frauds", "val-all-fraud"]
)
def test_a_train_or_val_period_without_both_classes_is_refused_before_tuning(
    period: str, fraud: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps = [datetime(2019, 3, 1, hour, tzinfo=UTC) for hour in range(6)]
    timestamps += [datetime(2019, 12, 1, hour, tzinfo=UTC) for hour in range(6)]
    labels = [0, 1] * 6
    target = slice(0, 6) if period == "train" else slice(6, 12)
    labels[target] = [int(fraud)] * 6
    ds = LabelledDataset(
        X=np.zeros((12, len(FEATURE_NAMES))),
        y=np.asarray(labels, dtype=np.int64),
        timestamps=np.asarray(timestamps, dtype=object),
        transaction_ids=[f"tx{index}" for index in range(12)],
        feature_names=list(FEATURE_NAMES),
    )
    tuned: list[object] = []
    monkeypatch.setattr(train, "tune_hyperparameters", lambda *a, **k: tuned.append(a))

    with pytest.raises(DriftError, match=f"The {period} period"):
        temporal_drift.measure_drift(ds, n_iter=1, cv_splits=2, target_fpr=0.01)

    assert tuned == []


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    corpus = _write_corpus(tmp_path)
    adapter = fixture_adapter(tmp_path)

    def stub_tune(*_: object, **__: object) -> TuningResult:
        return TuningResult(
            best_params=dict(STUB_PARAMS),
            best_score=0.5,
            cv_results_summary={"mean_test_score": [0.5], "std_test_score": [0.0]},
        )

    monkeypatch.setattr(loading, "get_adapter", lambda name: adapter)
    monkeypatch.setattr(train, "tune_hyperparameters", stub_tune)
    monkeypatch.setattr(temporal_drift, "RUNS_ROOT", tmp_path / "runs")
    yield corpus


def test_the_cli_writes_the_drift_result_into_its_run_directory(
    tmp_path: Path, cli_corpus: Path
) -> None:
    temporal_drift.main(
        [
            "--run-name", RUN,
            "--root", str(cli_corpus),
            "--cache-root", str(tmp_path / "cache"),
            "--n-iter", "2",
            "--cv-splits", "2",
            "--target-fpr", str(TARGET_FPR),
        ]
    )

    payload = json.loads((tmp_path / "runs" / RUN / DRIFT_METRICS_FILENAME).read_text("utf-8"))
    assert [month["name"] for month in payload["months"]] == [f"2020-{m:02d}" for m in range(1, 13)]


def test_the_cli_exits_with_an_error_for_a_trained_runs_directory(
    tmp_path: Path, cli_corpus: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trained = tmp_path / "runs" / "trained-run"
    trained.mkdir(parents=True)
    (trained / "run.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as refused:
        temporal_drift.main(["--run-name", "trained-run", "--root", str(cli_corpus)])

    assert refused.value.code == 1
    assert "was not measured" in caplog.text
    assert not (trained / DRIFT_METRICS_FILENAME).exists()


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--run-name", "../escape"], id="unsafe-name"),
        pytest.param([], id="no-run-name"),
        pytest.param(["--run-name", "r", "--dataset", "synthetic"], id="synthetic-dataset"),
    ],
)
def test_the_cli_refuses_arguments_before_any_work(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(temporal_drift, "run_drift", lambda *a, **k: calls.append(a))

    with pytest.raises(SystemExit) as refused:
        temporal_drift.main(argv)

    assert refused.value.code == 2
    assert calls == []
