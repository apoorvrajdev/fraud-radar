"""Phase 5D — a named run's test fold, analysed into its own directory.

The synthetic fixture is the verification suite's run: rows identify
themselves in their first feature, and three countries alternate, so country
segments are reported. It is analysed once, with every matrix handed to
XGBoost, every calibration input and every SHAP input recorded.

The benchmark fixture is a Sparkov run trained end to end on the generated
corpus. Every row is US, so its segments are not reported.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

from app.fraud.explainer import FraudExplainer, load_explainer
from ml import analyze, loading, run_analysis, train
from ml.analysis import (
    bucket_for_country,
    compute_calibration_metrics,
    compute_feature_importance,
    compute_segment_metrics,
)
from ml.data import LabelledDataset
from ml.datasets.quality import QUALITY_REPORT_FILENAME
from ml.datasets.sparkov import build_report
from ml.paths import ARTIFACTS_DIR, ML_ROOT
from ml.reporting import SEGMENT_TEST_ROC_CURVE_SOURCE
from ml.run_verification import RunVerificationError
from ml.tuning import TuningResult
from tests.unit.test_dataset_sparkov import fixture_adapter
from tests.unit.test_run_verification import COUNTRIES, RUN, TrainedRun, train_fixture_run
from tests.unit.test_train_runs import _benchmark_corpus

ANALYSIS_FILES = {
    "calibration_metrics.json",
    "feature_importance.json",
    "segment_metrics.json",
    "calibration_curve.png",
    "global_shap_beeswarm.png",
    "global_shap_bar.png",
    "MODEL_CARD.md",
}


def _snapshot(*roots: Path) -> dict[str, str]:
    """SHA-256 of every file under `roots`, by path."""
    digests: dict[str, str] = {}
    for root in roots:
        paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in paths:
            digests[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _served_outputs() -> dict[str, str]:
    """What the API serves and the legacy analysis writes, in this checkout."""
    card = ML_ROOT / "MODEL_CARD.md"
    return _snapshot(*(path for path in (ARTIFACTS_DIR, card) if path.exists()))


def _stub_synthetic_database(patch: pytest.MonkeyPatch, ds: LabelledDataset) -> None:
    @contextmanager
    def stub_session() -> Iterator[object]:
        yield object()

    patch.setattr(loading, "SessionLocal", stub_session)
    patch.setattr(loading, "load_dataset_with_csv_labels", lambda *_, **__: ds)


def _read(directory: Path, name: str) -> Any:
    return json.loads((directory / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The synthetic fixture, analysed once under observation
# ---------------------------------------------------------------------------


@dataclass
class Observed:
    run: TrainedRun
    analysis: run_analysis.RunAnalysis
    served_before: dict[str, str]
    served_after: dict[str, str]
    files_before: dict[str, str]
    files_after: dict[str, str]
    scored_rows: list[np.ndarray] = field(default_factory=list)
    calibration_inputs: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    shap_inputs: list[np.ndarray] = field(default_factory=list)


def observe_analysis(root: Path) -> Observed:
    """Train the synthetic fixture run under `root` and analyse it under observation."""
    run = train_fixture_run(root)
    scored_rows: list[np.ndarray] = []
    calibration_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    shap_inputs: list[np.ndarray] = []
    real_init = xgb.DMatrix.__init__
    real_shap = FraudExplainer.compute_global_shap

    def dmatrix_spy(self: xgb.DMatrix, data: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(data, np.ndarray) and data.ndim == 2:
            scored_rows.append(data[:, 0].astype(np.int64))
        real_init(self, data, *args, **kwargs)

    def calibration_spy(labels: np.ndarray, scores: np.ndarray, **options: Any) -> Any:
        calibration_inputs.append((np.array(labels), np.array(scores)))
        return compute_calibration_metrics(labels, scores, **options)

    def shap_spy(self: FraudExplainer, features: np.ndarray) -> np.ndarray:
        shap_inputs.append(np.array(features))
        return real_shap(self, features)

    served_before = _served_outputs()
    files_before = _snapshot(run.runs_root)
    with pytest.MonkeyPatch.context() as patch:
        _stub_synthetic_database(patch, run.ds)
        patch.setattr(xgb.DMatrix, "__init__", dmatrix_spy)
        patch.setattr(run_analysis, "compute_calibration_metrics", calibration_spy)
        patch.setattr(FraudExplainer, "compute_global_shap", shap_spy)
        analysis = run_analysis.analyze_run(RUN, runs_root=run.runs_root, csv_path=run.labels_csv)

    return Observed(
        run=run,
        analysis=analysis,
        served_before=served_before,
        served_after=_served_outputs(),
        files_before=files_before,
        files_after=_snapshot(run.runs_root),
        scored_rows=scored_rows,
        calibration_inputs=calibration_inputs,
        shap_inputs=shap_inputs,
    )


@pytest.fixture(scope="module")
def observed(tmp_path_factory: pytest.TempPathFactory) -> Observed:
    return observe_analysis(tmp_path_factory.mktemp("analysed"))


def _test_rows(observed: Observed) -> np.ndarray:
    return observed.run.outcome.splits.test


def test_the_analysed_scores_are_the_scores_training_evaluated(observed: Observed) -> None:
    np.testing.assert_array_equal(
        observed.analysis.verified.test_scores, observed.run.outcome.test_scores
    )
    np.testing.assert_array_equal(observed.analysis.verified.splits.test, _test_rows(observed))


def test_no_train_or_val_row_is_ever_given_to_the_model(observed: Observed) -> None:
    splits = observed.run.outcome.splits
    assert observed.scored_rows, "nothing was scored"
    scored = set(np.concatenate(observed.scored_rows).tolist())

    assert scored == set(splits.test.tolist())
    assert not scored & set(splits.train.tolist())
    assert not scored & set(splits.val.tolist())


def test_calibration_receives_exactly_the_test_labels_and_scores(observed: Observed) -> None:
    ((labels, scores),) = observed.calibration_inputs

    np.testing.assert_array_equal(labels, observed.run.ds.y[_test_rows(observed)])
    np.testing.assert_array_equal(scores, observed.run.outcome.test_scores)


def test_shap_receives_exactly_the_test_feature_matrix(observed: Observed) -> None:
    (features,) = observed.shap_inputs

    np.testing.assert_array_equal(features, observed.run.ds.X[_test_rows(observed)])


def test_batch_shap_adds_up_to_the_persisted_score_for_every_test_row(observed: Observed) -> None:
    """Base value plus each row's SHAP values, through the sigmoid, is its served score."""
    explainer = load_explainer(observed.run.directory)
    test_features = observed.run.ds.X[_test_rows(observed)]
    served = explainer.predict_proba_batch(test_features)
    np.testing.assert_array_equal(served, observed.run.outcome.test_scores)

    shap_values = observed.analysis.shap_values
    assert shap_values.shape == test_features.shape
    # The base value is read after this explainer has computed SHAP values:
    # TreeExplainer reports an expected value of 0.0 until its first call.
    np.testing.assert_array_equal(explainer.compute_global_shap(test_features), shap_values)
    margins = explainer.base_value + shap_values.sum(axis=1)
    np.testing.assert_allclose(1.0 / (1.0 + np.exp(-margins)), served, rtol=0, atol=1e-5)


def test_calibration_is_the_unchanged_calculation_with_a_neutral_note(observed: Observed) -> None:
    labels = observed.run.ds.y[_test_rows(observed)]
    expected = compute_calibration_metrics(labels, observed.run.outcome.test_scores)
    calibration = observed.analysis.calibration

    assert calibration["note"] == "Measured on the test fold; no calibrator fitted."
    assert {k: v for k, v in calibration.items() if k != "note"} == {
        k: v for k, v in expected.items() if k != "note"
    }
    assert calibration["n_test_samples"] == len(labels)
    assert _read(observed.run.directory, "calibration_metrics.json") == calibration


def test_feature_importance_ranks_the_test_fold_shap_in_log_odds(observed: Observed) -> None:
    importance = observed.analysis.feature_importance
    expected = compute_feature_importance(
        observed.analysis.shap_values, observed.run.ds.feature_names
    )

    assert importance == {**expected, "units": "log-odds"}
    assert importance["n_test_samples"] == len(_test_rows(observed))
    assert _read(observed.run.directory, "feature_importance.json") == importance


def test_several_country_groups_are_reported_with_labelled_recall(observed: Observed) -> None:
    test = _test_rows(observed)
    countries = [COUNTRIES[row % len(COUNTRIES)] for row in test]
    expected = compute_segment_metrics(
        observed.run.ds.y[test], observed.run.outcome.test_scores, countries
    )
    segments = observed.analysis.segments

    assert segments["reported"] is True
    assert segments["threshold_source"] == {"recall_at_1pct_fpr": SEGMENT_TEST_ROC_CURVE_SOURCE}
    assert segments["segments"] == expected["segments"]
    assert segments["global"] == expected["global"]
    groups = Counter(bucket_for_country(country) for country in countries)
    assert segments["country_groups"] == dict(groups)
    assert set(groups) == {"US", "Developed", "Other"}
    assert "synthetic" not in segments["notes"]
    assert _read(observed.run.directory, "segment_metrics.json") == segments


def test_outputs_are_written_into_the_run_directory_and_nowhere_else(observed: Observed) -> None:
    assert observed.served_after == observed.served_before
    new_files = {
        Path(path) for path in observed.files_after.keys() - observed.files_before.keys()
    }
    assert {path.name for path in new_files} == ANALYSIS_FILES
    assert all(path.parent == observed.run.directory for path in new_files)
    assert {path.name for path in observed.analysis.written} == ANALYSIS_FILES
    # The run's own records are read, never rewritten.
    assert {
        path: digest
        for path, digest in observed.files_after.items()
        if path in observed.files_before
    } == observed.files_before


# ---------------------------------------------------------------------------
# Segments decided from the rows
# ---------------------------------------------------------------------------


def test_a_single_country_group_is_not_reported() -> None:
    labels = np.array([0, 1, 0, 1, 0, 0])
    scores = np.linspace(0.1, 0.9, 6)

    segments = run_analysis.describe_segments(labels, scores, ["US"] * 6)

    assert segments == {
        "reported": False,
        "reason": (
            "Every test-fold row falls in the US country group, so a segment breakdown "
            "would only repeat the overall result."
        ),
        "country_groups": {"US": 6},
    }


def test_different_countries_in_one_group_are_still_one_group() -> None:
    segments = run_analysis.describe_segments(
        np.array([0, 1, 0, 1]), np.linspace(0.1, 0.9, 4), ["BR", "IN", "MX", "BR"]
    )

    assert segments["reported"] is False
    assert segments["country_groups"] == {"Other": 4}


# ---------------------------------------------------------------------------
# A registered benchmark run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Benchmark:
    runs_root: Path
    corpus: Path
    cache_root: Path
    analysis: run_analysis.RunAnalysis
    cache_files: list[Path]


BENCHMARK_RUN = "sparkov-fixture"


def analyze_benchmark(root: Path) -> Benchmark:
    """Train a Sparkov fixture run under `root`, report its data quality, and analyse it.

    The quality report is written into the run directory, where the adapter's
    CLI writes a benchmark run's report.
    """
    corpus = _benchmark_corpus(root)
    adapter = fixture_adapter(root)

    def stub_tune(*_: object, **__: object) -> TuningResult:
        return TuningResult(
            best_params={"n_estimators": 30, "max_depth": 3, "learning_rate": 0.3},
            best_score=0.5,
            cv_results_summary={"mean_test_score": [0.5], "std_test_score": [0.0]},
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loading, "get_adapter", lambda name: adapter)
        patch.setattr(train, "tune_hyperparameters", stub_tune)
        patch.setattr(train, "RUNS_ROOT", root / "runs")
        train.main(
            [
                "--dataset", "sparkov",
                "--root", str(corpus),
                "--cache-root", str(root / "cache"),
                "--run-name", BENCHMARK_RUN,
                "--max-cards", "3",
                "--seed", "11",
                "--n-iter", "1",
                "--cv-splits", "2",
                "--target-fpr", "0.05",
            ]
        )
        cache_files = sorted((root / "cache").glob("*.npz"))
        build_report(adapter.load_detailed(corpus, max_entities=3, seed=11)).write(
            root / "runs" / BENCHMARK_RUN
        )
        analysis = run_analysis.analyze_run(
            BENCHMARK_RUN,
            runs_root=root / "runs",
            root=corpus,
            cache_root=root / "cache",
        )
    return Benchmark(
        runs_root=root / "runs",
        corpus=corpus,
        cache_root=root / "cache",
        analysis=analysis,
        cache_files=cache_files,
    )


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory: pytest.TempPathFactory) -> Benchmark:
    return analyze_benchmark(tmp_path_factory.mktemp("benchmark"))


def test_a_benchmark_run_is_analysed_from_its_own_record(benchmark: Benchmark) -> None:
    verified = benchmark.analysis.verified
    record = verified.recorded.record

    assert record.dataset.subsample is not None
    assert (record.dataset.subsample.max_entities, record.dataset.subsample.seed) == (3, 11)
    assert sorted(benchmark.cache_root.glob("*.npz")) == benchmark.cache_files
    assert benchmark.analysis.calibration["n_test_samples"] == len(verified.splits.test)
    assert {path.name for path in benchmark.analysis.written} == ANALYSIS_FILES
    assert (verified.recorded.directory / QUALITY_REPORT_FILENAME).exists()


def test_a_benchmark_run_with_one_country_reports_no_segments(benchmark: Benchmark) -> None:
    segments = benchmark.analysis.segments
    directory = benchmark.analysis.verified.recorded.directory

    assert segments["reported"] is False
    assert segments["country_groups"] == {"US": len(benchmark.analysis.verified.splits.test)}
    assert _read(directory, "segment_metrics.json") == segments


def test_a_row_limit_is_refused_for_a_benchmark_run(benchmark: Benchmark) -> None:
    with pytest.raises(RunVerificationError, match="cannot be reloaded"):
        run_analysis.analyze_run(
            BENCHMARK_RUN,
            runs_root=benchmark.runs_root,
            root=benchmark.corpus,
            cache_root=benchmark.cache_root,
            limit=10,
        )


# ---------------------------------------------------------------------------
# Refusals write nothing
# ---------------------------------------------------------------------------


@pytest.fixture
def run(observed: Observed, tmp_path: Path) -> TrainedRun:
    """A private copy of the synthetic run's training outputs, before any analysis."""
    source = observed.run.directory
    target = tmp_path / "runs" / RUN
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(*ANALYSIS_FILES))
    return replace(observed.run, runs_root=tmp_path / "runs")


def test_a_run_refused_on_its_record_loads_no_data_and_writes_nothing(
    run: TrainedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    run.edit("run.json", lambda record: record["library_versions"].update(numpy="0.0.1"))
    before = _snapshot(run.runs_root)
    loads: list[object] = []
    monkeypatch.setattr(run_analysis, "load_run_data", lambda *a, **k: loads.append(a))

    with pytest.raises(RunVerificationError, match=r"numpy 0\.0\.1"):
        run_analysis.analyze_run(RUN, runs_root=run.runs_root, csv_path=run.labels_csv)

    assert loads == []
    assert _snapshot(run.runs_root) == before


def test_a_run_whose_metrics_do_not_reproduce_writes_nothing(
    run: TrainedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    def nudge(metrics: dict[str, Any]) -> None:
        metrics["test_roc_auc"] = float(np.nextafter(metrics["test_roc_auc"], 0.0))

    run.edit("metrics.json", nudge)
    before = _snapshot(run.runs_root)
    _stub_synthetic_database(monkeypatch, run.ds)

    with pytest.raises(RunVerificationError, match="do not reproduce"):
        run_analysis.analyze_run(RUN, runs_root=run.runs_root, csv_path=run.labels_csv)

    assert _snapshot(run.runs_root) == before


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_the_cli_analyses_a_named_run_into_its_directory(
    run: TrainedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_synthetic_database(monkeypatch, run.ds)
    monkeypatch.setattr(analyze, "RUNS_ROOT", run.runs_root)
    served = _served_outputs()

    analyze.main(["--run-name", RUN, "--csv-path", str(run.labels_csv)])

    assert {path.name for path in run.directory.iterdir()} >= ANALYSIS_FILES
    assert _served_outputs() == served


def test_the_cli_exits_with_an_error_when_a_run_does_not_verify(
    run: TrainedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    run.edit("training_metadata.json", lambda metadata: metadata.pop("best_iteration"))
    monkeypatch.setattr(analyze, "RUNS_ROOT", run.runs_root)

    with pytest.raises(SystemExit) as refused:
        analyze.main(["--run-name", RUN, "--csv-path", str(run.labels_csv)])

    assert refused.value.code == 1
    assert not ANALYSIS_FILES & {path.name for path in run.directory.iterdir()}


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--run-name", RUN, "--artifact-dir", "served"], id="run-with-artifact-dir"),
        pytest.param(["--run-name", RUN, "--card-path", "card.md"], id="run-with-card-path"),
        pytest.param(["--root", "raw"], id="root-without-run"),
        pytest.param(["--cache-root", "cache"], id="cache-root-without-run"),
        pytest.param(["--full-corpus"], id="full-corpus-without-run"),
    ],
)
def test_the_cli_refuses_options_that_do_not_apply(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(analyze, "analyze_run", lambda *a, **k: calls.append("named"))
    monkeypatch.setattr(analyze, "_analyze_served", lambda args: calls.append("served"))

    with pytest.raises(SystemExit) as refused:
        analyze.main(argv)

    assert refused.value.code == 2
    assert calls == []


def test_without_a_run_name_the_cli_runs_the_served_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: list[Any] = []
    monkeypatch.setattr(analyze, "analyze_run", lambda *a, **k: pytest.fail("named analysis ran"))
    monkeypatch.setattr(analyze, "_analyze_served", served.append)

    analyze.main(["--limit", "500"])

    (args,) = served
    assert (args.artifact_dir, args.card_path, args.limit) == (None, None, 500)
    assert Path("ml/artifacts") == analyze.DEFAULT_ARTIFACT_DIR
    assert Path("ml/MODEL_CARD.md") == analyze.DEFAULT_CARD_PATH
