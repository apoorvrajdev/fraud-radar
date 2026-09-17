"""The Phase 5D benchmark card, generated from the recorded benchmark runs.

Phase 5D produced three results that are never merged — a synthetic baseline,
Sparkov in-domain runs and a cross-generator transfer — alongside a temporal
drift experiment and a rules audit (`docs/adr/PHASE_5D_BENCHMARK_METHODOLOGY.md`,
decision 7). Each wrote machine-readable records into a run directory. This
module reads those records, refuses any that do not describe the benchmark as
recorded, and lays them out side by side.

It reads nothing else: no model, no dataset, no feature cache. Every number on
the card is read from a record and formatted, never computed or typed, so the
card changes only when a record does (decision 11).

Records are identified by a digest of their text with CRLF line endings
normalised to LF, the form Git stores. A digest of the raw bytes would differ
between a Windows working copy and a Linux checkout of the same commit.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.fraud.feature_spec import FEATURESETS
from ml.datasets.base import DatasetContractError, DatasetProvenance
from ml.paths import InvalidRunNameError, validate_run_name
from ml.run_card import _ACCEPTED_LIMITATIONS, _number, _or_dash, _table
from ml.runs import RunMetadata

# The benchmark's runs (decision 14), and the drift experiment's own directory,
# named when the drift experiment was run.
SYNTHETIC_RUN = "synthetic_v1"
DEV_RUN = "sparkov_v1_200cards"
FULL_RUN = "sparkov_v1_full"
DRIFT_RUN = "sparkov_v1_drift"

TRANSFER_FILE = "transfer_metrics.json"
RULES_AUDIT_FILE = "rules_audit.json"
DRIFT_FILE = "drift_metrics.json"

RUN_RECORD_FILES: tuple[str, ...] = (
    "run.json",
    "metrics.json",
    "threshold.json",
    "training_metadata.json",
    "feature_list.json",
    "calibration_metrics.json",
    "feature_importance.json",
)
QUALITY_REPORT_FILE = "quality_report.json"

INPUT_DIGEST_DEFINITION = (
    "SHA-256 of the file's text with CRLF line endings normalised to LF, the form Git stores"
)

METHODOLOGY_LINK = "../../docs/adr/PHASE_5D_BENCHMARK_METHODOLOGY.md"

# The threshold sources under which a recall at a fixed FPR is a point on the
# evaluated test fold's ROC curve. The card says so of every such recall, so
# any other recorded source is refused rather than relabelled.
_TEST_ROC_CURVE_SOURCES = frozenset({"test_roc_curve", "target_test_roc_curve"})


class BenchmarkCardError(ValueError):
    """The recorded runs cannot be laid out as the benchmark they claim to be."""


@dataclass(frozen=True)
class RecordFile:
    """One JSON record read for the card, where it came from, and its digest.

    `source` is the record's path relative to the runs root, with forward
    slashes on every platform.
    """

    source: str
    payload: Mapping[str, Any]
    sha256: str

    def get(self, *path: str) -> Any:
        """The value at `path`, refusing a record that does not hold it."""
        value: Any = self.payload
        for depth, key in enumerate(path):
            if not isinstance(value, Mapping) or key not in value:
                raise BenchmarkCardError(
                    f"{self.source} has no {'.'.join(path[: depth + 1])}."
                )
            value = value[key]
        return value


@dataclass(frozen=True)
class RecordedBenchmarkRun:
    """The records one benchmark run wrote, read and checked against each other."""

    name: str
    record: RunMetadata
    run: RecordFile
    metrics: RecordFile
    threshold: RecordFile
    fit: RecordFile
    feature_list: RecordFile
    calibration: RecordFile
    importance: RecordFile
    quality: RecordFile | None

    @property
    def files(self) -> tuple[RecordFile, ...]:
        """Every record read for this run, in a fixed order."""
        files = (
            self.run,
            self.metrics,
            self.threshold,
            self.fit,
            self.feature_list,
            self.calibration,
            self.importance,
        )
        return files if self.quality is None else (*files, self.quality)


@dataclass(frozen=True)
class BenchmarkRecords:
    """Every record the card is built from, checked to describe one benchmark."""

    synthetic: RecordedBenchmarkRun
    dev: RecordedBenchmarkRun
    full: RecordedBenchmarkRun
    transfer: RecordFile
    rules_audit: RecordFile
    drift: RecordFile

    @property
    def files(self) -> tuple[RecordFile, ...]:
        """Every record read, in a fixed order."""
        return (
            *self.synthetic.files,
            *self.dev.files,
            *self.full.files,
            self.transfer,
            self.rules_audit,
            self.drift,
        )


def normalised_sha256(path: Path) -> str:
    """The digest `INPUT_DIGEST_DEFINITION` describes."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_record(runs_root: Path, source: str) -> RecordFile:
    """Read one strict-JSON object from `runs_root / source`."""
    path = runs_root / source
    if not path.is_file():
        raise BenchmarkCardError(f"{source} is missing from {runs_root}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise BenchmarkCardError(f"{source} is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkCardError(f"{source} does not hold a JSON object.")
    return RecordFile(source=source, payload=payload, sha256=normalised_sha256(path))


def load_recorded_run(
    runs_root: Path, name: str, *, dataset: str, with_quality_report: bool
) -> RecordedBenchmarkRun:
    """Read one run's records, refusing records that describe another run or disagree.

    The run record must name this run and dataset `dataset`; the feature list
    must be its featureset's registered columns in order; and the test
    metrics must have been measured at the threshold the run recorded.
    """
    try:
        validate_run_name(name)
    except InvalidRunNameError as exc:
        raise BenchmarkCardError(str(exc)) from exc
    files = {
        filename: read_record(runs_root, f"{name}/{filename}") for filename in RUN_RECORD_FILES
    }
    run = files["run.json"]
    try:
        record = RunMetadata.from_dict(run.payload)
    except (KeyError, TypeError, ValueError, DatasetContractError) as exc:
        raise BenchmarkCardError(f"{run.source} is not a run record: {exc}") from exc

    if record.run_name != name:
        raise BenchmarkCardError(
            f"{run.source} is the record of run {record.run_name!r}, not {name!r}."
        )
    if record.dataset.name != dataset:
        raise BenchmarkCardError(
            f"{run.source} records dataset {record.dataset.name!r}; run {name!r} is the "
            f"{dataset!r} run of the benchmark."
        )
    registered = FEATURESETS.get(record.featureset_version)
    if registered is None:
        raise BenchmarkCardError(
            f"{run.source} records featureset {record.featureset_version!r}, which is not "
            "registered."
        )
    feature_list = files["feature_list.json"]
    if list(feature_list.get("features")) != registered:
        raise BenchmarkCardError(
            f"{feature_list.source} does not list featureset {record.featureset_version!r}'s "
            "features in their registered order."
        )
    metrics, threshold = files["metrics.json"], files["threshold.json"]
    measured_at = metrics.get("at_operating_threshold", "threshold")
    if measured_at != threshold.get("value"):
        raise BenchmarkCardError(
            f"{metrics.source} was measured at threshold {measured_at}, but "
            f"{threshold.source} records {threshold.get('value')}."
        )

    return RecordedBenchmarkRun(
        name=name,
        record=record,
        run=run,
        metrics=metrics,
        threshold=threshold,
        fit=files["training_metadata.json"],
        feature_list=feature_list,
        calibration=files["calibration_metrics.json"],
        importance=files["feature_importance.json"],
        quality=(
            read_record(runs_root, f"{name}/{QUALITY_REPORT_FILE}") if with_quality_report else None
        ),
    )


def load_benchmark(runs_root: Path) -> BenchmarkRecords:
    """Read the benchmark's runs and experiment records, refusing any that do not belong.

    The two Sparkov runs must be the development subsample and the full
    corpus of the same source files. The transfer must measure the synthetic
    run's own saved model and threshold on exactly the full run's test fold,
    the rules audit must cover every row of the full run, and the drift
    experiment must have run on the full run's dataset. Every verification
    an experiment recorded must have passed.
    """
    synthetic = load_recorded_run(
        runs_root, SYNTHETIC_RUN, dataset="synthetic", with_quality_report=False
    )
    dev = load_recorded_run(runs_root, DEV_RUN, dataset="sparkov", with_quality_report=True)
    full = load_recorded_run(runs_root, FULL_RUN, dataset="sparkov", with_quality_report=True)
    _check_development_and_full_runs(dev, full)

    transfer = read_record(runs_root, f"{FULL_RUN}/{TRANSFER_FILE}")
    _check_transfer(transfer, source=synthetic, target=full)
    rules_audit = read_record(runs_root, f"{FULL_RUN}/{RULES_AUDIT_FILE}")
    _check_rules_audit(rules_audit, full)
    drift = read_record(runs_root, f"{DRIFT_RUN}/{DRIFT_FILE}")
    _check_drift(drift, full)

    return BenchmarkRecords(
        synthetic=synthetic,
        dev=dev,
        full=full,
        transfer=transfer,
        rules_audit=rules_audit,
        drift=drift,
    )


def build_benchmark_card(records: BenchmarkRecords) -> str:
    """Compose the card's Markdown from the records, deterministically.

    The card carries no generation time: the same records always give the
    same bytes, so a committed card can be checked against its records.
    """
    sections = [
        _header(records),
        _results_section(records),
        _operating_threshold_section(records),
        _live_features_section(records),
        _calibration_section(records),
        _importance_section(records),
        _drift_section(records),
        _rules_audit_section(records),
        _quality_section(records),
        _provenance_section(records),
        _limitations_section(records),
    ]
    return "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _header(records: BenchmarkRecords) -> str:
    rows = [(f"`{record.source}`", f"`{record.sha256}`") for record in records.files]
    return "\n".join(
        [
            "# Benchmark card — Phase 5D",
            "",
            "> Generated by `python -m ml.benchmark_card` from the benchmark's recorded runs. "
            "Do not hand-edit it; regenerate it instead.",
            "",
            "Every value on this card is read from a record listed below and formatted; none is "
            "computed or typed. The method behind the values is fixed in "
            f"[`docs/adr/PHASE_5D_BENCHMARK_METHODOLOGY.md`]({METHODOLOGY_LINK}).",
            "",
            "<details>",
            f"<summary>Records read ({len(rows)}), relative to <code>backend/ml/artifacts/runs/"
            "</code></summary>",
            "",
            f"Each record is identified by the {INPUT_DIGEST_DEFINITION}.",
            "",
            _table(("Record", "SHA-256"), rows),
            "",
            "</details>",
        ]
    )


def _results_section(records: BenchmarkRecords) -> str:
    _check_recall_points(records)
    rows = [
        _in_domain_result_row("Synthetic baseline", records.synthetic),
        _in_domain_result_row(
            f"Sparkov in-domain, development ({_cards(records.dev)})", records.dev
        ),
        _in_domain_result_row(
            f"Sparkov in-domain, full corpus ({_cards(records.full)})", records.full
        ),
        _transfer_result_row(records),
    ]
    header = (
        "Result",
        "Run",
        "Trained on",
        "Evaluated on",
        "Threshold selected on",
        "Test rows",
        "Test frauds",
        "Test prevalence",
        "PR-AUC",
        "ROC-AUC",
        "Recall @ 1% FPR",
        "Recall @ 5% FPR",
    )
    return "\n".join(
        [
            "## 1. Results — never merged",
            "",
            "Each row is one result, stating what its model was trained on, what it was evaluated "
            "on and where its operating threshold was selected. No figure combines rows. A PR-AUC "
            "is read against the test prevalence beside it: a scorer with no signal scores about "
            "the prevalence.",
            "",
            _table(header, rows),
            "",
            "Each recall at a fixed FPR is read at a threshold found on the evaluated test fold "
            "itself, so it is a point on that fold's ROC curve, not a result at an operating "
            "threshold.",
        ]
    )


def _in_domain_result_row(label: str, run: RecordedBenchmarkRun) -> tuple[str, ...]:
    metrics = run.metrics
    return (
        label,
        f"`{run.name}`",
        f"`{run.name}` train fold",
        f"`{run.name}` test fold",
        f"`{run.name}` val fold",
        _or_dash(run.fit.get("test_size")),
        _or_dash(metrics.get("context", "fraud_counts", "test")),
        _number(metrics.get("context", "test_prevalence")),
        _number(metrics.get("test_pr_auc")),
        _number(metrics.get("test_roc_auc")),
        _number(metrics.get("recall_at_1pct_fpr")),
        _number(metrics.get("recall_at_5pct_fpr")),
    )


def _transfer_result_row(records: BenchmarkRecords) -> tuple[str, ...]:
    transfer = records.transfer
    source, target = records.synthetic.name, records.full.name
    return (
        "Cross-generator transfer",
        f"`{source}` → `{target}`",
        f"`{source}` train fold, not retrained",
        f"`{target}` test fold",
        f"`{source}` val fold; nothing selected on `{target}`",
        _or_dash(transfer.get("context", "target_test_rows")),
        _or_dash(transfer.get("context", "target_test_fraud_count")),
        _number(transfer.get("context", "target_test_prevalence")),
        _number(transfer.get("threshold_free", "pr_auc")),
        _number(transfer.get("threshold_free", "roc_auc")),
        _number(transfer.get("threshold_free", "recall_at_1pct_fpr")),
        _number(transfer.get("threshold_free", "recall_at_5pct_fpr")),
    )


def _check_recall_points(records: BenchmarkRecords) -> None:
    """Refuse a recall at a fixed FPR that its record does not label as a test ROC point."""
    sources: list[tuple[RecordFile, tuple[str, ...]]] = [
        (run.metrics, ("threshold_source", key))
        for run in (records.synthetic, records.dev, records.full)
        for key in ("recall_at_1pct_fpr", "recall_at_5pct_fpr")
    ]
    sources += [
        (records.transfer, ("threshold_free", "threshold_source", key))
        for key in ("recall_at_1pct_fpr", "recall_at_5pct_fpr")
    ]
    for record, path in sources:
        source = record.get(*path)
        if source not in _TEST_ROC_CURVE_SOURCES:
            raise BenchmarkCardError(
                f"{record.source}: {'.'.join(path)} is {source!r}, which is not a recall read "
                "on a test ROC curve."
            )


def _operating_threshold_section(records: BenchmarkRecords) -> str:
    header = (
        "Result",
        "Threshold",
        "Selected on",
        "FPR ceiling there",
        "Realised FPR there",
        "Fallback used",
        "Precision",
        "Recall",
        "F1",
        "TP",
        "FP",
        "TN",
        "FN",
        "Realised test FPR",
    )
    in_domain = [
        ("Synthetic baseline", records.synthetic),
        ("Sparkov in-domain, development", records.dev),
        ("Sparkov in-domain, full corpus", records.full),
    ]
    rows = [_in_domain_threshold_row(label, run) for label, run in in_domain]
    rows.append(_transfer_threshold_row(records.transfer))

    counts = [
        (
            label,
            _or_dash(run.metrics.get("context", "fraud_counts", "train")),
            _or_dash(run.metrics.get("context", "fraud_counts", "val")),
            _or_dash(run.metrics.get("context", "fraud_counts", "test")),
        )
        for label, run in in_domain
    ]
    transfer = records.transfer
    source, target = records.synthetic.name, records.full.name
    counts.append(
        (
            "Cross-generator transfer",
            f"{_or_dash(transfer.get('context', 'source_train_fraud_count'))} (source `{source}`)",
            f"{_or_dash(transfer.get('context', 'source_val_fraud_count'))} (source `{source}`)",
            f"{_or_dash(transfer.get('context', 'target_test_fraud_count'))} (target `{target}`)",
        )
    )

    notes = sorted(
        {run.metrics.get("context", "label_delay_note") for _, run in in_domain}
        | {transfer.get("context", "label_delay_note")}
    )
    return "\n".join(
        [
            "## 2. At each operating threshold",
            "",
            "Each in-domain threshold was selected on its run's own val fold before the test fold "
            "was scored. The transfer applies the synthetic run's threshold unchanged, so its row "
            "is measured at the source run's threshold and nothing is selected on the target's "
            "data.",
            "",
            _table(header, rows),
            "",
            "**Fraud counts per fold.**",
            "",
            _table(("Result", "Train frauds", "Val frauds", "Test frauds"), counts),
            "",
            "**Label delay.** " + " ".join(notes),
        ]
    )


def _in_domain_threshold_row(label: str, run: RecordedBenchmarkRun) -> tuple[str, ...]:
    threshold, confusion = run.threshold, run.metrics
    return (
        label,
        _number(threshold.get("value")),
        f"`{run.name}` val fold",
        _number(threshold.get("target_fpr")),
        _number(threshold.get("realised_fpr_on_val")),
        _yes_no(threshold.get("fallback_used")),
        _number(confusion.get("at_operating_threshold", "precision")),
        _number(confusion.get("at_operating_threshold", "recall")),
        _number(confusion.get("at_operating_threshold", "f1")),
        _or_dash(confusion.get("at_operating_threshold", "true_positives")),
        _or_dash(confusion.get("at_operating_threshold", "false_positives")),
        _or_dash(confusion.get("at_operating_threshold", "true_negatives")),
        _or_dash(confusion.get("at_operating_threshold", "false_negatives")),
        _number(confusion.get("context", "realised_fpr_on_test_at_operating_threshold")),
    )


def _transfer_threshold_row(transfer: RecordFile) -> tuple[str, ...]:
    return (
        "Cross-generator transfer, at the source run's threshold",
        _number(transfer.get("source_threshold", "value")),
        str(transfer.get("source_threshold", "selected_on")),
        _number(transfer.get("source_threshold", "fpr_ceiling_on_source_val")),
        _number(transfer.get("source_threshold", "realised_fpr_on_source_val")),
        _yes_no(transfer.get("source_threshold", "fallback_used")),
        _number(transfer.get("at_source_threshold", "precision")),
        _number(transfer.get("at_source_threshold", "recall")),
        _number(transfer.get("at_source_threshold", "f1")),
        _or_dash(transfer.get("at_source_threshold", "true_positives")),
        _or_dash(transfer.get("at_source_threshold", "false_positives")),
        _or_dash(transfer.get("at_source_threshold", "true_negatives")),
        _or_dash(transfer.get("at_source_threshold", "false_negatives")),
        _number(transfer.get("context", "realised_fpr_on_target_test_at_source_threshold")),
    )


def _live_features_section(records: BenchmarkRecords) -> str:
    rows = [
        (
            f"`{run.name}`",
            _or_dash(len(run.feature_list.get("features"))),
            _or_dash(run.fit.get("live_feature_count")),
            _names_or_none(run.fit.get("constant_features")),
            _or_dash(run.fit.get("live_feature_count_whole_matrix")),
            _names_or_none(run.fit.get("constant_features_whole_matrix")),
        )
        for run in _runs(records)
    ]
    features = records.transfer
    outside = [
        f"`{entry['feature']}` {_or_dash(entry['target_test_rows_outside_source_training_range'])}"
        for entry in features.get("features", "by_feature")
        if entry.get("target_test_rows_outside_source_training_range")
    ]
    return "\n".join(
        [
            "## 3. Live features",
            "",
            "A feature is live in a run when it takes more than one distinct value in that run's "
            "training fold. Each count is taken from the run's own feature matrix.",
            "",
            _table(
                (
                    "Run",
                    "Registered features",
                    "Live in training fold",
                    "Constant in training fold",
                    "Live over the whole matrix",
                    "Constant over the whole matrix",
                ),
                rows,
            ),
            "",
            f"**Transfer.** The `{records.synthetic.name}` model scored every column of featureset "
            f"`{features.get('features', 'featureset_version')}`. Constant in its training fold: "
            f"{_names_or_none(features.get('features', 'constant_in_source_training_fold'))}. "
            f"Constant in the `{records.full.name}` test fold: "
            f"{_names_or_none(features.get('features', 'constant_in_target_test_fold'))}. "
            "Target test rows outside the source training range, by feature: "
            f"{', '.join(outside) if outside else 'none'}.",
        ]
    )


def _calibration_section(records: BenchmarkRecords) -> str:
    rows = []
    bins = set()
    for run in _runs(records):
        calibration = run.calibration
        _require_test_rows(calibration, run)
        bins.add(len(calibration.get("bin_counts")))
        worst = calibration.get("worst_positive_bin")
        rows.append(
            (
                f"`{run.name}`",
                _or_dash(calibration.get("n_test_samples")),
                _number(calibration.get("brier_score")),
                _number(calibration.get("positive_class_brier")),
                _number(calibration.get("expected_calibration_error")),
                _number(calibration.get("positive_class_ece")),
                "—"
                if worst is None
                else (
                    f"bin {worst['bin_index']}: mean score {_number(worst['mean_predicted'])}, "
                    f"fraud rate {_number(worst['mean_observed'])} "
                    f"({_or_dash(worst['n_samples'])} rows, "
                    f"{_or_dash(worst['n_positives'])} frauds)"
                ),
            )
        )
    if len(bins) != 1:
        raise BenchmarkCardError(
            f"The calibration records use different numbers of bins: {sorted(bins)}."
        )
    return "\n".join(
        [
            "## 4. Calibration — test folds, no calibrator fitted",
            "",
            f"Measured on each run's test fold with {bins.pop()} equal-width score bins. The "
            "positive-class Brier score is computed on fraud rows only; the positive-class ECE "
            "averages the calibration gap uniformly over the bins that hold at least one fraud.",
            "",
            _table(
                (
                    "Run",
                    "Test rows",
                    "Brier score",
                    "Positive-class Brier",
                    "Expected calibration error",
                    "Positive-class ECE",
                    "Largest gap in a bin holding frauds",
                ),
                rows,
            ),
        ]
    )


def _importance_section(records: BenchmarkRecords) -> str:
    runs = _runs(records)
    rankings = []
    units = set()
    for run in runs:
        importance = run.importance
        _require_test_rows(importance, run)
        units.add(str(importance.get("units")))
        ranked = sorted(importance.get("features"), key=lambda entry: int(entry["rank"]))
        rankings.append(
            [f"`{entry['feature']}` {_number(entry['mean_abs_shap'])}" for entry in ranked]
        )
    if len(units) != 1:
        raise BenchmarkCardError(
            f"The feature importance records use different units: {', '.join(sorted(units))}."
        )
    depth = max(len(ranking) for ranking in rankings)
    rows = [
        (str(rank + 1), *(ranking[rank] if rank < len(ranking) else "—" for ranking in rankings))
        for rank in range(depth)
    ]
    return "\n".join(
        [
            "## 5. Global feature importance — test folds",
            "",
            f"Mean absolute SHAP value of each feature over each run's test rows, in "
            f"{units.pop()}. Each column ranks one model on its own test fold.",
            "",
            _table(("Rank", *(f"`{run.name}`" for run in runs)), rows),
        ]
    )


def _drift_section(records: BenchmarkRecords) -> str:
    drift = records.drift
    periods = [
        (
            name,
            str(drift.get("periods", name, "start")),
            str(drift.get("periods", name, "end")),
            str(drift.get("periods", name, "interval")),
            _or_dash(drift.get("periods", name, "rows")),
            _or_dash(drift.get("periods", name, "fraud_count")),
        )
        for name in ("train", "val")
    ]
    months = [
        _part(drift, f"months[{index}]", month) for index, month in enumerate(drift.get("months"))
    ]
    names = [str(month.get("name")) for month in months]
    if names != sorted(set(names)):
        raise BenchmarkCardError(f"{drift.source}: the months are not in calendar order.")
    month_rows = [
        (
            f"`{month.get('name')}`",
            str(month.get("interval")),
            _or_dash(month.get("volume")),
            _or_dash(month.get("fraud_count")),
            _number(month.get("fraud_rate")),
            _number(month.get("pr_auc")),
            _number(month.get("recall")),
            _number(month.get("precision")),
            _number(month.get("realised_fpr")),
        )
        for month in months
    ]
    chosen = ", ".join(
        f"`{key}={value}`"
        for key, value in sorted(drift.get("tuning", "best_hyperparameters").items())
    )
    procedure = [
        (
            "Hyperparameter search",
            f"{drift.get('tuning', 'iterations')} iterations over "
            f"{drift.get('tuning', 'cv_folds')} folds of the train period, random state "
            f"{drift.get('tuning', 'random_state')}",
        ),
        (
            "Cross-validated PR-AUC",
            f"{_number(drift.get('tuning', 'best_cv_pr_auc'))} — {drift.get('tuning', 'note')}",
        ),
        ("Chosen hyperparameters", chosen),
        ("scale_pos_weight", _number(drift.get("fit", "scale_pos_weight"))),
        (
            "Early stopping",
            f"after {drift.get('fit', 'early_stopping_rounds')} rounds without improvement on "
            f"the {drift.get('fit', 'early_stopping_against')}; best round "
            f"{drift.get('fit', 'best_iteration')}",
        ),
        (
            "Operating threshold",
            f"{_number(drift.get('threshold', 'value'))}, selected on the "
            f"{drift.get('threshold', 'selected_on')} for a target FPR of "
            f"{_number(drift.get('threshold', 'target_fpr'))}, realising "
            f"{_number(drift.get('threshold', 'realised_fpr_on_val'))} there; fallback used: "
            f"{_yes_no(drift.get('threshold', 'fallback_used'))}. "
            f"{drift.get('threshold', 'note')}",
        ),
        ("Rows outside every period", _or_dash(drift.get("rows_outside_periods"))),
    ]
    stated = {run.metrics.get("context", "label_delay_note") for run in _runs(records)}
    delay = str(drift.get("label_delay_note"))
    delay_line = "As stated in section 2." if stated == {delay} else delay
    return "\n".join(
        [
            "## 6. Temporal drift",
            "",
            f"A model of its own, run as `{drift.get('run_name')}` on the "
            f"`{records.full.name}` dataset, is tuned and fitted on the train period, has its "
            "threshold selected once on the val period, and scores each month after them once at "
            f"that threshold. {drift.get('note')}",
            "",
            _table(("Period", "Start", "End", "Interval", "Rows", "Frauds"), periods),
            "",
            _table(("Setting", "Value"), procedure),
            "",
            _table(
                (
                    "Month",
                    "Interval",
                    "Rows",
                    "Frauds",
                    "Fraud rate",
                    "PR-AUC",
                    "Recall",
                    "Precision",
                    "Realised FPR",
                ),
                month_rows,
            ),
            "",
            f"— marks a metric the month leaves undefined. {drift.get('undefined_metrics_note')}",
            "",
            f"**Label delay.** {delay_line}",
        ]
    )


def _rules_audit_section(records: BenchmarkRecords) -> str:
    audit = records.rules_audit
    rules = [
        _part(audit, f"rules[{index}]", rule) for index, rule in enumerate(audit.get("rules"))
    ]
    rows = []
    for rule in rules:
        if not rule.get("evaluable"):
            not_evaluable = f"no — {rule.get('reason')}"
            rows.append((f"`{rule.get('rule')}`", "—", not_evaluable, "—", "—", "—", "—"))
            continue
        rows.append(
            (
                f"`{rule.get('rule')}`",
                str(rule.get("severity")),
                "yes",
                _or_dash(rule.get("rows_fired")),
                _or_dash(rule.get("fired_on_fraud")),
                _number(rule.get("precision")),
                _number(rule.get("fraud_recall")),
            )
        )

    outcome = _part(audit, "rules_only_outcome", audit.get("rules_only_outcome"))
    by_outcome = outcome.get("by_outcome")
    ordered = [name for name in ("DECLINE", "REVIEW", "APPROVE") if name in by_outcome]
    ordered += sorted(name for name in by_outcome if name not in ordered)
    outcome_rows = [
        (
            name,
            _or_dash(by_outcome[name]["rows"]),
            _or_dash(by_outcome[name]["frauds"]),
            _or_dash(by_outcome[name]["legitimate"]),
        )
        for name in ordered
    ]
    population = _part(audit, "population", audit.get("population"))
    return "\n".join(
        [
            "## 7. Rules audit",
            "",
            f"The production rules, unmodified, on every row of `{population.get('run')}`: "
            f"{_or_dash(population.get('row_count'))} rows holding "
            f"{_or_dash(population.get('fraud_count'))} frauds "
            f"(rate {_number(population.get('fraud_rate'))}), "
            f"{population.get('first_timestamp')} to {population.get('last_timestamp')}. "
            f"{population.get('note')}",
            "",
            f"Each row was evaluated on the context the scoring service builds for it: "
            f"{audit.get('context', 'definition')} The history is drawn from "
            f"{audit.get('context', 'history_drawn_from')}, over "
            f"{audit.get('context', 'history_window_days')} days.",
            "",
            _table(
                (
                    "Rule",
                    "Severity",
                    "Evaluable",
                    "Rows fired",
                    "Fired on fraud",
                    "Precision",
                    "Fraud recall",
                ),
                rows,
            ),
            "",
            f"**Rules-only outcome.** {outcome.get('note')} {outcome.get('decision_rule')}",
            "",
            _table(("Outcome", "Rows", "Frauds", "Legitimate"), outcome_rows),
        ]
    )


def _quality_section(records: BenchmarkRecords) -> str:
    sparkov = [records.dev, records.full]
    loads, fraud = [], []
    for run in sparkov:
        report = run.quality
        if report is None:  # pragma: no cover - the Sparkov runs are loaded with their reports
            raise BenchmarkCardError(f"Run {run.name!r} was read without its quality report.")
        cards = _part(report, "fraud_distribution.cards", report.get("fraud_distribution", "cards"))
        loads.append(
            (
                f"`{run.name}`",
                _or_dash(report.get("rows", "raw")),
                _or_dash(report.get("rows", "kept")),
                _or_dash(report.get("rows", "excluded")),
                _or_dash(report.get("entities", "customers")),
                _or_dash(report.get("entities", "merchants")),
                _or_dash(report.get("label", "fraud_count")),
            )
        )
        folds = {
            str(fold["name"]): fold
            for fold in report.get("fraud_distribution", "chronological_split", "folds")
        }
        boundaries = report.get("fraud_distribution", "chronological_split", "boundaries")
        fraud.append(
            (
                f"`{run.name}`",
                _or_dash(cards.get("with_fraud")),
                _count(
                    _part(
                        report,
                        "fraud_distribution.cards.frauds_per_card_with_fraud",
                        cards.get("frauds_per_card_with_fraud"),
                    ).get("p50")
                ),
                *(_or_dash(folds[name]["cards_with_fraud"]) for name in ("train", "val", "test")),
                "; ".join(
                    f"{'/'.join(boundary['between'])}: "
                    f"{_or_dash(boundary['cards_with_fraud_on_both_sides'])}"
                    for boundary in boundaries
                ),
            )
        )
    return "\n".join(
        [
            "## 8. Data quality — Sparkov runs",
            "",
            "Each Sparkov run's quality report, written from its own load of the source files.",
            "",
            _table(
                (
                    "Run",
                    "Rows read",
                    "Rows kept",
                    "Rows excluded",
                    "Cards",
                    "Merchants",
                    "Frauds kept",
                ),
                loads,
            ),
            "",
            "**How fraud is spread.** A card's frauds decide the effective sample size behind a "
            "fold's result, and fraud on both sides of a fold boundary belongs to one card's "
            "history.",
            "",
            _table(
                (
                    "Run",
                    "Cards with fraud",
                    "Median frauds per card with fraud",
                    "Fraud cards, train",
                    "Fraud cards, val",
                    "Fraud cards, test",
                    "Cards with fraud on both sides of a boundary",
                ),
                fraud,
            ),
        ]
    )


def _provenance_section(records: BenchmarkRecords) -> str:
    runs = [
        (
            f"`{run.name}`",
            f"`{run.record.code_version}`" if run.record.code_version else "not recorded",
            f"`{run.record.featureset_version}`",
            f"`{run.record.dataset.name}` `{run.record.dataset.version}`",
            _subsample(run),
            ", ".join(
                f"{name} {version}" for name, version in sorted(run.record.library_versions.items())
            ),
        )
        for run in _runs(records)
    ]
    datasets = []
    for run in (records.synthetic, records.full):
        dataset = run.record.dataset
        files = ", ".join(f"`{name}` `{digest}`" for name, digest in sorted(dataset.files.items()))
        datasets.append(
            (
                f"`{dataset.name}` `{dataset.version}`",
                dataset.license,
                dataset.source_url or "in this repository",
                _or_dash(dataset.row_count),
                _or_dash(dataset.fraud_count),
                files,
            )
        )
    return "\n".join(
        [
            "## 9. Provenance",
            "",
            _table(
                ("Run", "Code", "Featureset", "Dataset", "Subsample", "Libraries"),
                runs,
            ),
            "",
            "**Source data.** Raw data is never committed, so these digests are the durable link "
            "between a result and the bytes it was computed from.",
            "",
            _table(
                ("Dataset", "Licence", "Source", "Rows", "Frauds", "Files (SHA-256)"),
                datasets,
            ),
            "",
            f"`{records.full.record.dataset.name}`: {records.full.record.dataset.notes}",
        ]
    )


def _limitations_section(records: BenchmarkRecords) -> str:
    limitations = [
        *_ACCEPTED_LIMITATIONS,
        "Label delay, as stated in section 2.",
        "Every number on this card was read from a record; what the numbers mean is written in "
        f"[the decision record]({METHODOLOGY_LINK}), not here.",
    ]
    return "\n".join(
        ["## 10. Accepted limitations", "", *(f"- {limitation}" for limitation in limitations)]
    )


def _subsample(run: RecordedBenchmarkRun) -> str:
    subsample = run.record.dataset.subsample
    if subsample is None:
        return "none"
    if subsample.strategy == "none":
        return f"none (seed {subsample.seed}, {_or_dash(subsample.selected_entities)} entities)"
    return (
        f"{subsample.strategy}, seed {subsample.seed}, "
        f"{_or_dash(subsample.selected_entities)} of at most {_or_dash(subsample.max_entities)}"
    )


def _require_test_rows(record: RecordFile, run: RecordedBenchmarkRun) -> None:
    """Refuse an analysis record measured on another number of rows than the run's test fold."""
    if record.get("n_test_samples") != run.fit.get("test_size"):
        raise BenchmarkCardError(
            f"{record.source} was measured on {record.get('n_test_samples')} rows, but "
            f"{run.fit.source} records a test fold of {run.fit.get('test_size')}."
        )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _part(record: RecordFile, label: str, payload: Any) -> RecordFile:
    """A nested object of `record`, read with the same refusals as the record itself."""
    if not isinstance(payload, Mapping):
        raise BenchmarkCardError(f"{record.source}: {label} is not an object.")
    return RecordFile(source=f"{record.source} {label}", payload=payload, sha256=record.sha256)


def _runs(records: BenchmarkRecords) -> tuple[RecordedBenchmarkRun, ...]:
    return (records.synthetic, records.dev, records.full)


def _count(value: Any) -> str:
    """A recorded count, whether it was written as an integer or a whole float."""
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    return _or_dash(value) if isinstance(value, int) else _number(value)


def _names_or_none(names: Any) -> str:
    return ", ".join(f"`{name}`" for name in names) if names else "none"


def _cards(run: RecordedBenchmarkRun) -> str:
    subsample = run.record.dataset.subsample
    if subsample is None or subsample.selected_entities is None:
        return "cards not recorded"
    return f"{subsample.selected_entities:,} cards"


def _yes_no(value: object) -> str:
    if value is None:
        return "not recorded"
    return "yes" if value is True else "no" if value is False else str(value)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_development_and_full_runs(dev: RecordedBenchmarkRun, full: RecordedBenchmarkRun) -> None:
    dev_data, full_data = dev.record.dataset, full.record.dataset
    if (dev_data.version, dict(dev_data.files)) != (full_data.version, dict(full_data.files)):
        raise BenchmarkCardError(
            f"{dev.run.source} and {full.run.source} record different source data; the "
            "development run is a subsample of the full run's corpus."
        )
    if dev.record.featureset_version != full.record.featureset_version:
        raise BenchmarkCardError(
            f"{dev.run.source} and {full.run.source} record different featuresets."
        )
    if dev_data.subsample is None or dev_data.subsample.strategy != "cards":
        raise BenchmarkCardError(f"{dev.run.source} does not record a card subsample.")
    if full_data.subsample is not None and full_data.subsample.strategy != "none":
        raise BenchmarkCardError(
            f"{full.run.source} records a {full_data.subsample.strategy!r} subsample; the full "
            "run covers the whole corpus."
        )


def _check_transfer(
    transfer: RecordFile, *, source: RecordedBenchmarkRun, target: RecordedBenchmarkRun
) -> None:
    pairing = (transfer.get("source_run"), transfer.get("target_run"))
    if pairing != (source.name, target.name):
        raise BenchmarkCardError(
            f"{transfer.source} measures {pairing[0]!r} on {pairing[1]!r}; the benchmark's "
            f"transfer measures {source.name!r} on {target.name!r}."
        )
    featuresets = (transfer.get("source_featureset"), transfer.get("target_featureset"))
    if featuresets != (source.record.featureset_version, target.record.featureset_version):
        raise BenchmarkCardError(
            f"{transfer.source} records featuresets {featuresets}, not those of its runs."
        )
    if transfer.get("source_model_sha256") != source.fit.get("model_sha256"):
        raise BenchmarkCardError(
            f"{transfer.source} scored with a model other than the one {source.fit.source} "
            "records."
        )
    if transfer.get("source_threshold", "value") != source.threshold.get("value"):
        raise BenchmarkCardError(
            f"{transfer.source} applied a threshold other than the one {source.threshold.source} "
            "records."
        )
    identity = target.record.test_fold_identity
    scored = (
        transfer.get("context", "target_test_transaction_count"),
        transfer.get("context", "target_test_transaction_ids_sha256"),
    )
    if identity is None or scored != (identity.transaction_count, identity.transaction_ids_sha256):
        raise BenchmarkCardError(
            f"{transfer.source} did not score exactly the test fold {target.run.source} "
            "identifies."
        )
    for side, run in (("source", source), ("target", target)):
        if transfer.get("verification", side, "run") != run.name:
            raise BenchmarkCardError(
                f"{transfer.source} records the {side} verification of another run."
            )
        _require_verified(transfer, "verification", side)


def _check_rules_audit(audit: RecordFile, full: RecordedBenchmarkRun) -> None:
    if (audit.get("run"), audit.get("population", "run")) != (full.name, full.name):
        raise BenchmarkCardError(f"{audit.source} audits another run, not {full.name!r}.")
    if audit.get("population", "rows") != "all":
        raise BenchmarkCardError(
            f"{audit.source} audits the {audit.get('population', 'rows')!r} rows; the final "
            "audit covers every row of the full corpus."
        )
    if audit.get("population", "row_count") != full.record.dataset.row_count:
        raise BenchmarkCardError(
            f"{audit.source} counts {audit.get('population', 'row_count')} rows, but "
            f"{full.run.source} records {full.record.dataset.row_count}."
        )
    _require_verified(audit, "verification")


def _check_drift(drift: RecordFile, full: RecordedBenchmarkRun) -> None:
    if drift.get("run_name") != DRIFT_RUN:
        raise BenchmarkCardError(
            f"{drift.source} names run {drift.get('run_name')!r}, not {DRIFT_RUN!r}."
        )
    if drift.get("featureset_version") != full.record.featureset_version:
        raise BenchmarkCardError(
            f"{drift.source} records another featureset than {full.run.source}."
        )
    try:
        dataset = DatasetProvenance.from_dict(drift.get("dataset"))
    except (KeyError, TypeError, ValueError, DatasetContractError) as exc:
        raise BenchmarkCardError(f"{drift.source} has no readable dataset record: {exc}") from exc
    recorded = {k: v for k, v in dataset.to_dict().items() if k != "retrieved_at"}
    expected = {k: v for k, v in full.record.dataset.to_dict().items() if k != "retrieved_at"}
    if recorded != expected:
        differing = sorted(k for k in expected if recorded.get(k) != expected[k])
        raise BenchmarkCardError(
            f"{drift.source} ran on another dataset than {full.run.source} "
            f"(differs in {', '.join(differing)})."
        )


def _require_verified(record: RecordFile, *path: str) -> None:
    flags = record.get(*path)
    if not isinstance(flags, Mapping):
        raise BenchmarkCardError(f"{record.source}: {'.'.join(path)} is not a set of flags.")
    unverified = sorted(key for key, value in flags.items() if key != "run" and value is not True)
    if unverified:
        raise BenchmarkCardError(
            f"{record.source}: {'.'.join(path)} does not record these checks as passed: "
            f"{', '.join(unverified)}."
        )


def _reject_constant(value: str) -> float:
    raise ValueError(f"{value} is not a JSON number")
