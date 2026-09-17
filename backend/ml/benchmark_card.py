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
from ml.run_card import _number, _or_dash, _table
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


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


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
