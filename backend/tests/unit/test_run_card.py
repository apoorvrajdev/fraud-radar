"""Phase 5D — each analysed run gets a model card built from its own records.

Both fixture runs are analysed for real: the synthetic run, whose test fold
spans several country groups, and the Sparkov run, whose threshold falls back
and whose run directory holds a quality report. Every assertion reads the
value it expects from the run's records, not from a copy written here.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from app.fraud.feature_spec import FEATURE_NAMES
from ml import run_analysis
from ml.datasets.quality import QUALITY_REPORT_FILENAME
from ml.paths import ML_ROOT
from ml.reporting import LABEL_DELAY_NOTE
from ml.run_card import RUN_CARD_FILENAME, build_run_card
from ml.run_verification import RecordedRun, RunVerificationError, read_recorded_run
from tests.unit.test_run_analysis import (
    ANALYSIS_FILES,
    Benchmark,
    Observed,
    analyze_benchmark,
    observe_analysis,
)
from tests.unit.test_run_verification import RUN


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> Observed:
    return observe_analysis(tmp_path_factory.mktemp("synthetic-card"))


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory: pytest.TempPathFactory) -> Benchmark:
    return analyze_benchmark(tmp_path_factory.mktemp("benchmark-card"))


def _recorded(analysis: run_analysis.RunAnalysis) -> RecordedRun:
    return analysis.verified.recorded


def _line(card: str, prefix: str) -> str:
    (line,) = [line for line in card.splitlines() if line.startswith(prefix)]
    return line


def _section(card: str, number: int) -> str:
    match = re.search(rf"^## {number}\. .*?(?=^## {number + 1}\. |\Z)", card, re.S | re.M)
    assert match is not None, f"section {number} missing"
    return match.group(0)


def _four(value: float) -> str:
    return f"{value:.4f}"


# ---------------------------------------------------------------------------
# Where the card comes from and where it goes
# ---------------------------------------------------------------------------


def test_the_card_is_written_into_the_run_directory_only(synthetic: Observed) -> None:
    directory = _recorded(synthetic.analysis).directory
    card_path = directory / RUN_CARD_FILENAME

    assert card_path in synthetic.analysis.written
    assert card_path.read_text(encoding="utf-8") == synthetic.analysis.card
    assert ML_ROOT / "MODEL_CARD.md" not in synthetic.analysis.written
    assert synthetic.served_after == synthetic.served_before
    assert RUN_CARD_FILENAME in ANALYSIS_FILES


@pytest.mark.parametrize("which", ["synthetic", "benchmark"])
def test_the_card_rebuilds_exactly_from_the_files_in_the_run_directory(
    which: str, request: pytest.FixtureRequest
) -> None:
    fixture = request.getfixturevalue(which)
    analysis: run_analysis.RunAnalysis = fixture.analysis
    directory = _recorded(analysis).directory
    written = (directory / RUN_CARD_FILENAME).read_text(encoding="utf-8")
    generated = re.search(r"Generated at `([^`]+)` by code `([^`]+)`", written)
    assert generated is not None

    def read(name: str) -> Any:
        return json.loads((directory / name).read_text(encoding="utf-8"))

    quality = directory / QUALITY_REPORT_FILENAME
    rebuilt = build_run_card(
        read_recorded_run(directory.name, runs_root=directory.parent),
        calibration=read("calibration_metrics.json"),
        feature_importance=read("feature_importance.json"),
        segments=read("segment_metrics.json"),
        quality_report=(
            json.loads(quality.read_text(encoding="utf-8")) if quality.exists() else None
        ),
        generated_at=generated.group(1),
        code_version=None if generated.group(2) == "not recorded" else generated.group(2),
    )

    assert rebuilt == written


def test_a_malformed_quality_report_refuses_the_run_before_anything_is_written(
    synthetic: Observed, tmp_path: Path
) -> None:
    source = _recorded(synthetic.analysis).directory
    target = tmp_path / "runs" / RUN
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(*ANALYSIS_FILES))
    (target / QUALITY_REPORT_FILENAME).write_text("not json", encoding="utf-8")
    before = sorted(path.name for path in target.iterdir())

    with pytest.raises(RunVerificationError, match=r"quality_report\.json could not be read"):
        run_analysis.analyze_run(
            RUN, runs_root=tmp_path / "runs", csv_path=synthetic.run.labels_csv
        )

    assert sorted(path.name for path in target.iterdir()) == before


# ---------------------------------------------------------------------------
# Provenance and folds
# ---------------------------------------------------------------------------


def test_provenance_comes_from_the_run_record(benchmark: Benchmark) -> None:
    recorded = _recorded(benchmark.analysis)
    dataset = recorded.record.dataset
    section = _section(benchmark.analysis.card, 1)

    assert f"`{dataset.name}` version `{dataset.version}`" in section
    assert dataset.source_url in section
    for name, digest in dataset.files.items():
        assert f"- `{name}`: `{digest}`" in section
    assert "strategy `cards`, seed 11, at most 3 entities, 3 selected" in section
    assert f"| Featureset | `{recorded.record.featureset_version}` |" in section
    assert f"| Training code | `{recorded.record.code_version}` |" in section
    assert f"xgboost {recorded.record.library_versions['xgboost']}" in section


def test_each_fold_is_shown_with_its_utc_period_rows_and_frauds(benchmark: Benchmark) -> None:
    recorded = _recorded(benchmark.analysis)
    section = _section(benchmark.analysis.card, 2)
    counts = recorded.metrics["context"]["fraud_counts"]

    for period in recorded.record.splits:
        size = recorded.training_metadata[f"{period.name}_size"]
        row = _line(section, f"| {period.name} |")
        assert period.start.isoformat().endswith("+00:00")
        assert (
            f"| {period.name} | {period.start.isoformat()} | {period.end.isoformat()} | "
            f"{size} | {counts[period.name]} |"
        ) in row


# ---------------------------------------------------------------------------
# Fit and threshold
# ---------------------------------------------------------------------------


def test_the_search_score_is_labelled_a_diagnostic(synthetic: Observed) -> None:
    recorded = _recorded(synthetic.analysis)
    row = _line(synthetic.analysis.card, "| Cross-validated PR-AUC |")

    assert _four(recorded.metrics["best_cv_pr_auc"]) in row
    assert "a diagnostic" in row and "not a held-out estimate" in row


def test_live_features_are_shown_with_the_constant_columns(benchmark: Benchmark) -> None:
    metadata = _recorded(benchmark.analysis).training_metadata
    section = _section(benchmark.analysis.card, 3)

    assert f"{metadata['live_feature_count']} of {len(FEATURE_NAMES)} take more than one" in section
    for name in metadata["constant_features"]:
        assert f"`{name}`" in section
    assert metadata["live_feature_count_whole_matrix"] == metadata["live_feature_count"]
    assert "Over the whole matrix" not in section


def test_the_whole_matrix_count_is_shown_when_it_differs(benchmark: Benchmark) -> None:
    analysis = benchmark.analysis
    recorded = _recorded(analysis)
    metadata = {
        **recorded.training_metadata,
        "live_feature_count_whole_matrix": recorded.training_metadata["live_feature_count"] + 1,
        "constant_features_whole_matrix": ["is_high_risk_category"],
    }

    card = build_run_card(
        replace(recorded, training_metadata=metadata),
        calibration=analysis.calibration,
        feature_importance=analysis.feature_importance,
        segments=analysis.segments,
        quality_report=None,
        generated_at="2026-01-01T00:00:00+00:00",
        code_version=None,
    )

    assert (
        f"Over the whole matrix {metadata['live_feature_count_whole_matrix']} of 17 are live; "
        "constant there: `is_high_risk_category`."
    ) in card


def test_the_operating_threshold_is_labelled_as_selected_on_validation(
    synthetic: Observed,
) -> None:
    threshold = _recorded(synthetic.analysis).threshold
    section = _section(synthetic.analysis.card, 4)

    assert section.startswith("## 4. Operating threshold — selected on the validation fold")
    assert f"| Threshold | {_four(threshold.value)} |" in section
    assert f"| Realised FPR on the val fold | {_four(threshold.realised_fpr_on_val)} |" in section
    assert "| Fallback used | no |" in section
    assert "fixed fallback" not in section


def test_a_fallback_threshold_is_shown_as_one(benchmark: Benchmark) -> None:
    section = _section(benchmark.analysis.card, 4)

    assert _recorded(benchmark.analysis).threshold.fallback_used is True
    assert "| Fallback used | yes |" in section
    assert "the threshold is the fixed fallback, not one selected for the target" in section


# ---------------------------------------------------------------------------
# Test-fold results
# ---------------------------------------------------------------------------


def test_pr_auc_is_shown_with_the_test_prevalence(synthetic: Observed) -> None:
    metrics = _recorded(synthetic.analysis).metrics
    row = _line(synthetic.analysis.card, "| PR-AUC |")

    assert _four(metrics["test_pr_auc"]) in row
    assert f"test prevalence {_four(metrics['context']['test_prevalence'])}" in row


def test_recall_at_fixed_fpr_is_labelled_a_point_on_the_test_roc_curve(
    synthetic: Observed,
) -> None:
    metrics = _recorded(synthetic.analysis).metrics
    for label, key in (("1%", "recall_at_1pct_fpr"), ("5%", "recall_at_5pct_fpr")):
        row = _line(synthetic.analysis.card, f"| Recall @ {label} FPR |")
        assert _four(metrics[key]) in row
        assert row.endswith("| point on the test ROC curve |")


def test_results_at_the_operating_threshold_are_labelled_and_complete(
    synthetic: Observed,
) -> None:
    metrics = _recorded(synthetic.analysis).metrics
    at = metrics["at_operating_threshold"]
    card = synthetic.analysis.card
    realised = metrics["context"]["realised_fpr_on_test_at_operating_threshold"]

    assert "At the operating threshold selected on the validation fold (`threshold.json`):" in card
    assert (
        f"| {_four(at['threshold'])} | {_four(at['precision'])} | {_four(at['recall'])} | "
        f"{_four(at['f1'])} | {at['true_positives']} | {at['false_positives']} | "
        f"{at['true_negatives']} | {at['false_negatives']} | {_four(realised)} |"
    ) in card


def test_fold_fraud_counts_and_the_label_delay_note_are_shown(synthetic: Observed) -> None:
    counts = _recorded(synthetic.analysis).metrics["context"]["fraud_counts"]
    section = _section(synthetic.analysis.card, 5)

    assert (
        f"**Fraud counts:** train {counts['train']}, val {counts['val']}, test {counts['test']}."
    ) in section
    assert f"**Label delay:** {LABEL_DELAY_NOTE}" in section


# ---------------------------------------------------------------------------
# Calibration, importance, segments, quality, limitations
# ---------------------------------------------------------------------------


def test_calibration_is_shown_as_measured_with_no_calibrator_fitted(synthetic: Observed) -> None:
    calibration = synthetic.analysis.calibration
    section = _section(synthetic.analysis.card, 6)

    assert section.startswith("## 6. Calibration — test fold, no calibrator fitted")
    assert "Measured on the test fold; no calibrator fitted." in section
    assert (
        f"| Brier score | {_four(calibration['brier_score'])} | "
        f"{_four(calibration['positive_class_brier'])} |"
    ) in section
    assert len([line for line in section.splitlines() if re.match(r"\| \d+ \| ", line)]) == 10


def test_the_shap_ranking_lists_every_feature_in_log_odds(synthetic: Observed) -> None:
    importance = synthetic.analysis.feature_importance
    section = _section(synthetic.analysis.card, 7)

    assert "in log-odds" in section
    ranked = re.findall(r"^\| (\d+) \| `([a-z0-9_]+)` \|", section, re.M)
    assert [(int(rank), name) for rank, name in ranked] == [
        (entry["rank"], entry["feature"]) for entry in importance["features"]
    ]
    assert {name for _, name in ranked} == set(FEATURE_NAMES)


def test_reported_segments_label_their_recall_as_segment_roc_points(synthetic: Observed) -> None:
    section = _section(synthetic.analysis.card, 8)

    assert "Recall @ 1% FPR (point on the segment's own test ROC curve)" in section
    for name, block in synthetic.analysis.segments["segments"].items():
        assert f"| {name} | {block['n_transactions']} | {block['n_frauds']} |" in section


def test_unreported_segments_say_why(benchmark: Benchmark) -> None:
    section = _section(benchmark.analysis.card, 8)

    assert f"Not reported. {benchmark.analysis.segments['reason']}" in section
    assert "| Segment |" not in section


def test_the_quality_report_is_summarised_when_the_run_has_one(benchmark: Benchmark) -> None:
    directory = _recorded(benchmark.analysis).directory
    report = json.loads((directory / QUALITY_REPORT_FILENAME).read_text(encoding="utf-8"))
    section = _section(benchmark.analysis.card, 9)

    assert f"| Rows read | {report['rows']['raw']} |" in section
    assert f"| Rows kept | {report['rows']['kept']} |" in section
    for name, value in report["constant_fields"].items():
        assert f"- `{name}` = `{value}`" in section


def test_a_run_without_a_quality_report_says_so(synthetic: Observed) -> None:
    assert "No `quality_report.json` in this run directory." in _section(synthetic.analysis.card, 9)


def test_the_accepted_limitations_are_stated(synthetic: Observed) -> None:
    section = _section(synthetic.analysis.card, 10)

    assert "appears in several folds with different transactions" in section
    assert "realised val FPR is slightly optimistic. The test fold is untouched by both." in section
    assert f"- Label delay: {LABEL_DELAY_NOTE}" in section


@pytest.mark.parametrize("which", ["synthetic", "benchmark"])
def test_the_card_carries_no_targets_or_unmeasured_claims(
    which: str, request: pytest.FixtureRequest
) -> None:
    card: str = request.getfixturevalue(which).analysis.card

    for forbidden in (
        "target_pr_auc",
        "target_recall",
        "Target PR-AUC",
        "Target recall",
        "500 customers",
        "fraud patterns are injected",
        "seed=42",
        "0.9327",
        "Platt",
        "isotonic",
        "over-prediction",
        "drift",
        "transfer",
        "rules audit",
    ):
        assert forbidden not in card, forbidden
