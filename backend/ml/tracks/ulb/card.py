"""The ULB benchmark card, generated from the three recorded ULB runs (Phase 5E).

Usage:
    cd backend
    uv run python -m ml.tracks.ulb.card

Decision 11 of `docs/adr/PHASE_5E_ULB_BENCHMARK_METHODOLOGY.md` lays the ULB
results out in their own card, built from the ULB run records only and never
beside a v1 result. The three pre-registered runs are shown side by side,
random state 42 as the primary result, and never averaged (decision 9). Every
number is shown with its featureset, its test prevalence and its fold fraud
counts.

The records are read as the Phase 5D card reads its own: strict JSON, each
identified by the digest of its LF-normalised text, every number read from a
record and formatted, never computed or typed (Phase 5D decision 11). Records
that do not describe the three pre-registered runs of one benchmark are
refused. Nothing else is read — no model, no source file, no matrix — so the
card holds no row of the ULB database.

Every published ULB-derived work carries the ODbL notice, and a record
computed through the ULB track carries the method offer
(`docs/DATA_LICENSES.md`). The card prints both, with the code version each
run records.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from ml.benchmark_card import (
    INPUT_DIGEST_DEFINITION,
    BenchmarkCardError,
    RecordFile,
    _yes_no,
    read_record,
    write_benchmark_card,
)
from ml.datasets.base import DatasetContractError
from ml.paths import ML_ROOT, RUNS_ROOT
from ml.reporting import TEST_ROC_CURVE_SOURCE
from ml.run_card import _number, _or_dash, _table
from ml.runs import RUN_METADATA_FILENAME, TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT, RunMetadata
from ml.tracks.ulb.load import DATASET_NAME, ELAPSED_TIME_ORIGIN, LICENCE_NOTICE, METHOD_OFFER
from ml.tracks.ulb.matrix import ULB_PCA_V1, ULB_PCA_V1_FEATURES
from ml.tracks.ulb.quality import ULB_QUALITY_REPORT_FILENAME
from ml.tracks.ulb.train import ULB_SEEDS, run_name_for

log = logging.getLogger("ml.tracks.ulb.card")

ULB_CARD_PATH = ML_ROOT / "ULB_BENCHMARK_CARD.md"

# Decision 9: random state 42 is the primary result, because it is the
# procedure Sparkov was run with; 43 and 44 are its pre-registered repeats.
PRIMARY_SEED = 42

RUN_RECORD_FILES: tuple[str, ...] = (
    RUN_METADATA_FILENAME,
    "metrics.json",
    "threshold.json",
    "training_metadata.json",
    "feature_list.json",
    "calibration_metrics.json",
    ULB_QUALITY_REPORT_FILENAME,
)

# Links as they resolve from the card's own directory, backend/ml/.
METHODOLOGY_LINK = "../../docs/adr/PHASE_5E_ULB_BENCHMARK_METHODOLOGY.md"
LICENCE_TERMS_LINK = "../../docs/DATA_LICENSES.md"
TRACK_LINK = "tracks/ulb/"

# The ODbL section 4.3 notice with its links as Markdown, word for word the
# notice docs/DATA_LICENSES.md records. Written out with its links, it is
# `LICENCE_NOTICE`, the form the run records carry.
LICENCE_NOTICE_MARKDOWN = (
    "Contains information from the "
    "[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) "
    "database of the Machine Learning Group, ULB, which is available under the "
    "[Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/); "
    "its contents are under the "
    "[Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/)."
)

_FOLDS = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

# The fit settings every run must share: only the random state differs (decision 9).
_SHARED_FIT_SETTINGS = ("tuning_iterations", "tuning_cv_folds", "early_stopping_rounds")

# Quality-report fields that describe when it was written, not the rows.
_UNCOMPARED_QUALITY_FIELDS = frozenset({"generated_at_utc"})


@dataclass(frozen=True)
class UlbRecordedRun:
    """The records one ULB run wrote, read and checked against each other."""

    name: str
    seed: int
    record: RunMetadata
    run: RecordFile
    metrics: RecordFile
    threshold: RecordFile
    fit: RecordFile
    feature_list: RecordFile
    calibration: RecordFile
    quality: RecordFile

    @property
    def is_primary(self) -> bool:
        return self.seed == PRIMARY_SEED

    @property
    def files(self) -> tuple[RecordFile, ...]:
        """Every record read for this run, in a fixed order."""
        return (
            self.run,
            self.metrics,
            self.threshold,
            self.fit,
            self.feature_list,
            self.calibration,
            self.quality,
        )


@dataclass(frozen=True)
class UlbCardRecords:
    """The three pre-registered runs, checked to describe one benchmark."""

    runs: tuple[UlbRecordedRun, ...]

    @property
    def primary(self) -> UlbRecordedRun:
        return next(run for run in self.runs if run.is_primary)

    @property
    def files(self) -> tuple[RecordFile, ...]:
        return tuple(record for run in self.runs for record in run.files)


def load_ulb_run(runs_root: Path, seed: int) -> UlbRecordedRun:
    """Read one run's records, refusing records that describe another run or disagree."""
    name = run_name_for(seed)
    files = {filename: read_record(runs_root, f"{name}/{filename}") for filename in RUN_RECORD_FILES}
    run = files[RUN_METADATA_FILENAME]
    try:
        record = RunMetadata.from_dict(run.payload)
    except (KeyError, TypeError, ValueError, DatasetContractError) as exc:
        raise BenchmarkCardError(f"{run.source} is not a run record: {exc}") from exc
    recorded = UlbRecordedRun(
        name=name,
        seed=seed,
        record=record,
        run=run,
        metrics=files["metrics.json"],
        threshold=files["threshold.json"],
        fit=files["training_metadata.json"],
        feature_list=files["feature_list.json"],
        calibration=files["calibration_metrics.json"],
        quality=files[ULB_QUALITY_REPORT_FILENAME],
    )
    _check_run(recorded)
    return recorded


def load_ulb_records(runs_root: Path = RUNS_ROOT) -> UlbCardRecords:
    """Read the three pre-registered runs, refusing any that are not one benchmark.

    The runs must share their source data and its record, their folds, their
    test fold's transactions, their quality report and every fit setting but
    the random state.
    """
    runs = tuple(load_ulb_run(runs_root, seed) for seed in ULB_SEEDS)
    _check_one_benchmark(runs)
    return UlbCardRecords(runs=runs)


def build_ulb_card(records: UlbCardRecords) -> str:
    """Compose the card's Markdown from the records, deterministically.

    The card carries no generation time: the same records always give the
    same bytes, so a committed card can be checked against its records.
    """
    sections = [
        _header(records),
        _protocol_section(records),
        _results_section(records),
        _operating_threshold_section(records),
        _live_features_section(records),
        _calibration_section(records),
        _quality_section(records),
        _provenance_section(records),
        _not_reported_section(),
        _limitations_section(records),
    ]
    return "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _header(records: UlbCardRecords) -> str:
    digests = [(f"`{record.source}`", f"`{record.sha256}`") for record in records.files]
    recorded_versions = {run.record.code_version for run in records.runs}
    code_versions = (
        f"`{recorded_versions.pop()}`, recorded by all three runs"
        if len(recorded_versions) == 1
        else ", ".join(f"`{run.record.code_version}` (`{run.name}`)" for run in records.runs)
    )
    return "\n".join(
        [
            "# ULB benchmark card — Phase 5E",
            "",
            "> Generated by `python -m ml.tracks.ulb.card` from the recorded ULB runs. Do not "
            "hand-edit it; regenerate it instead.",
            "",
            "**Real, anonymised card data, on its own track.** ULB is real card-transaction data, "
            "anonymised by its publisher: apart from `Time`, `Amount` and the label `Class`, every "
            "column is a PCA component of inputs the publisher withholds. It is neither the "
            "in-house synthetic data nor the Sparkov simulation, and nothing on this card is placed "
            "beside, merged with or averaged with their results. The model inputs are featureset "
            f"`{ULB_PCA_V1}`, the published columns {_feature_span()}, kept outside the production "
            "featureset registry, so no ULB model is promoted or served.",
            "",
            "Every value on this card is read from a record listed below and formatted; none is "
            "computed or typed. The method is fixed in "
            f"[`docs/adr/PHASE_5E_ULB_BENCHMARK_METHODOLOGY.md`]({METHODOLOGY_LINK}), and what the "
            "values mean is written there, not here.",
            "",
            "**Licence notice** (ODbL §4.3).",
            "",
            f"> {LICENCE_NOTICE_MARKDOWN}",
            "",
            f"**Method offer.** {METHOD_OFFER} The track is [`backend/ml/tracks/ulb/`]"
            f"({TRACK_LINK}), at code version {code_versions}. Licence terms: "
            f"[`docs/DATA_LICENSES.md`]({LICENCE_TERMS_LINK}).",
            "",
            "<details>",
            f"<summary>Records read ({len(digests)}), relative to <code>backend/ml/artifacts/runs/"
            "</code></summary>",
            "",
            f"Each record is identified by the {INPUT_DIGEST_DEFINITION}.",
            "",
            _table(("Record", "SHA-256"), digests),
            "",
            "</details>",
        ]
    )


def _protocol_section(records: UlbCardRecords) -> str:
    primary = records.primary
    fit, threshold, quality = primary.fit, primary.threshold, primary.quality
    folds = [
        (
            name,
            _or_dash(fold["rows"]),
            _or_dash(fold["frauds"]),
            _number(fit.get(f"{name}_fraud_rate")),
            _elapsed(fold["first_elapsed_seconds"]),
            _elapsed(fold["last_elapsed_seconds"]),
        )
        for name, fold in zip(_FOLDS, _folds(quality), strict=True)
    ]
    boundaries = []
    for boundary in quality.get("chronological_split", "boundaries"):
        shared = boundary["instant_shared"]
        at_instant = boundary["rows_at_shared_instant"]
        boundaries.append(
            (
                " → ".join(boundary["between"]),
                _elapsed(boundary["earlier_fold_last_elapsed_seconds"]),
                _elapsed(boundary["later_fold_first_elapsed_seconds"]),
                _yes_no(shared),
                f"{_or_dash(at_instant['earlier_fold'])} / {_or_dash(at_instant['later_fold'])}"
                if shared
                else "—",
                _or_dash(boundary["exact_duplicate_groups_straddling"]),
            )
        )
    return "\n".join(
        [
            "## 1. Protocol — the same for all three runs",
            "",
            "The rows are split chronologically 70/15/15 by row count, in load order (`Time`, then "
            "position in the source file): train, then val, then test. Hyperparameters are "
            f"searched on the training fold only, with {_or_dash(fit.get('tuning_iterations'))} "
            f"iterations over {_or_dash(fit.get('tuning_cv_folds'))} stratified folds; the final "
            "fit stops early against the val fold after "
            f"{_or_dash(fit.get('early_stopping_rounds'))} rounds without improvement; the "
            "operating threshold is the highest whose FPR on the val fold is at most "
            f"{_number(threshold.get('target_fpr'))}; and the test fold is scored once. Only the "
            "random state differs between the runs.",
            "",
            _table(
                ("Fold", "Rows", "Frauds", "Fraud rate", "First `Time`", "Last `Time`"),
                folds,
            ),
            "",
            "`Time` is the seconds elapsed from the first transaction in the source file. It is "
            "never a calendar date or a time of day.",
            "",
            "**Fold boundaries.** A `Time` shared by two folds puts rows of the same second on both "
            "sides of the boundary. It is recorded, not corrected.",
            "",
            _table(
                (
                    "Boundary",
                    "Earlier fold's last `Time`",
                    "Later fold's first `Time`",
                    "`Time` shared",
                    "Rows at the shared `Time` (earlier / later fold)",
                    "Exact-duplicate groups straddling it",
                ),
                boundaries,
            ),
        ]
    )


def _results_section(records: UlbCardRecords) -> str:
    rows = [
        (
            f"`{run.name}`",
            str(run.seed),
            "primary" if run.is_primary else "pre-registered repeat",
            f"`{run.record.featureset_version}`",
            _or_dash(run.fit.get("test_size")),
            _or_dash(run.metrics.get("context", "fraud_counts", "test")),
            _number(run.metrics.get("context", "test_prevalence")),
            _number(run.metrics.get("test_pr_auc")),
            _number(run.metrics.get("test_roc_auc")),
            _number(run.metrics.get("recall_at_1pct_fpr")),
            _number(run.metrics.get("recall_at_5pct_fpr")),
        )
        for run in records.runs
    ]
    primary = records.primary
    return "\n".join(
        [
            "## 2. Results — three pre-registered random states, never averaged",
            "",
            f"Random state {primary.seed} is the primary result; the others are its "
            "pre-registered repeats. Only the random state differs between the rows, so they show "
            "how much a result moves with the fit's randomness. They are shown side by side and "
            "never averaged.",
            "",
            _table(
                (
                    "Run",
                    "Random state",
                    "Role",
                    "Featureset",
                    "Test rows",
                    "Test frauds",
                    "Test prevalence",
                    "PR-AUC",
                    "ROC-AUC",
                    "Recall @ 1% FPR",
                    "Recall @ 5% FPR",
                ),
                rows,
            ),
            "",
            "A PR-AUC is read against the test prevalence beside it: a scorer with no signal "
            "scores about the prevalence. Each recall at a fixed FPR is read at a threshold found "
            "on the test fold itself, so it is a point on that fold's ROC curve, not a result at "
            "an operating threshold.",
            "",
            f"**Few frauds.** The test fold holds {_test_frauds(primary)} frauds among "
            f"{_or_dash(primary.fit.get('test_size'))} rows, so a handful of rows moves every "
            "figure on this card. The three random states do not measure that uncertainty, and no "
            "resampling-based interval is computed.",
        ]
    )


def _operating_threshold_section(records: UlbCardRecords) -> str:
    rows = [
        (
            f"`{run.name}`",
            _number(run.threshold.get("value")),
            _number(run.threshold.get("target_fpr")),
            _number(run.threshold.get("realised_fpr_on_val")),
            _yes_no(run.threshold.get("fallback_used")),
            _number(run.metrics.get("at_operating_threshold", "precision")),
            _number(run.metrics.get("at_operating_threshold", "recall")),
            _number(run.metrics.get("at_operating_threshold", "f1")),
            _or_dash(run.metrics.get("at_operating_threshold", "true_positives")),
            _or_dash(run.metrics.get("at_operating_threshold", "false_positives")),
            _or_dash(run.metrics.get("at_operating_threshold", "true_negatives")),
            _or_dash(run.metrics.get("at_operating_threshold", "false_negatives")),
            _number(run.metrics.get("context", "realised_fpr_on_test_at_operating_threshold")),
        )
        for run in records.runs
    ]
    notes = sorted({run.metrics.get("context", "label_delay_note") for run in records.runs})
    return "\n".join(
        [
            "## 3. At the operating threshold",
            "",
            "Each run's threshold was selected on its own val fold before its test fold was "
            "scored.",
            "",
            _table(
                (
                    "Run",
                    "Threshold",
                    "FPR ceiling on val",
                    "Realised FPR on val",
                    "Fallback used",
                    "Precision",
                    "Recall",
                    "F1",
                    "TP",
                    "FP",
                    "TN",
                    "FN",
                    "Realised test FPR",
                ),
                rows,
            ),
            "",
            "**Label delay.** " + " ".join(notes),
        ]
    )


def _live_features_section(records: UlbCardRecords) -> str:
    rows = [
        (
            f"`{run.name}`",
            _or_dash(len(run.feature_list.get("features"))),
            _or_dash(run.fit.get("live_feature_count")),
            _names_or_none(run.fit.get("constant_features")),
            _or_dash(run.fit.get("live_feature_count_whole_matrix")),
            _names_or_none(run.fit.get("constant_features_whole_matrix")),
        )
        for run in records.runs
    ]
    return "\n".join(
        [
            "## 4. Live features",
            "",
            "A feature is live when it takes more than one distinct value in the run's training "
            "fold. Each count is taken from the run's own matrix.",
            "",
            _table(
                (
                    "Run",
                    f"`{ULB_PCA_V1}` features",
                    "Live in training fold",
                    "Constant in training fold",
                    "Live over the whole matrix",
                    "Constant over the whole matrix",
                ),
                rows,
            ),
        ]
    )


def _calibration_section(records: UlbCardRecords) -> str:
    rows = []
    for run in records.runs:
        calibration = run.calibration
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
                    f"({_or_dash(worst['n_samples'])} rows, {_or_dash(worst['n_positives'])} "
                    "frauds)"
                ),
            )
        )
    bins = sorted({len(run.calibration.get("bin_counts")) for run in records.runs})
    notes = sorted({run.calibration.get("note") for run in records.runs})
    return "\n".join(
        [
            "## 5. Calibration — test fold, no calibrator fitted",
            "",
            f"Measured on each run's test fold with {', '.join(str(n) for n in bins)} equal-width "
            "score bins. The positive-class Brier score is computed on fraud rows only; the "
            "positive-class ECE averages the calibration gap uniformly over the bins that hold at "
            "least one fraud.",
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
            "",
            "**As recorded with the values.** " + " ".join(notes),
        ]
    )


def _quality_section(records: UlbCardRecords) -> str:
    quality = records.primary.quality
    exclusions = ", ".join(
        f"{entry['reason']} {_or_dash(entry['count'])}" for entry in quality.get("exclusions")
    )
    duplicates, time = quality.get("duplicates"), quality.get("time")
    stops = quality.get("stops")
    return "\n".join(
        [
            "## 6. Data quality — the loaded rows",
            "",
            "Read from each run's aggregate-only quality report, which is the same for all three "
            "runs. It holds counts and the `Time` values that bound the rows and their folds, "
            "never a row.",
            "",
            _table(
                ("Rows read", "Kept", "Excluded", "Frauds", "Fraud rate", "Zero-amount rows"),
                [
                    (
                        _or_dash(quality.get("rows", "read")),
                        _or_dash(quality.get("rows", "kept")),
                        _or_dash(quality.get("rows", "excluded")),
                        _or_dash(quality.get("label", "fraud_count")),
                        _number(quality.get("label", "fraud_rate")),
                        _or_dash(quality.get("amounts", "zero_amount_rows")),
                    )
                ],
            ),
            "",
            f"**Exclusions by reason:** {exclusions}.",
            "",
            "**Exact duplicates** are kept, not removed: "
            f"{_or_dash(duplicates['exact_duplicate_groups'])} groups holding "
            f"{_or_dash(duplicates['rows_in_exact_duplicate_groups'])} rows, "
            f"{_or_dash(duplicates['repeated_rows'])} of them repeating an earlier row; the "
            f"largest group has {_or_dash(duplicates['largest_exact_duplicate_group'])} rows, and "
            f"{_or_dash(duplicates['exact_duplicate_groups_of_fraud_rows'])} groups are fraud "
            f"rows. Groups whose labels conflict: {_or_dash(duplicates['label_conflict_groups'])}.",
            "",
            f"**`Time`** runs from {_elapsed(time['first_elapsed_seconds'])} to "
            f"{_elapsed(time['last_elapsed_seconds'])}; whole seconds: "
            f"{_yes_no(time['whole_seconds'])}; already in file order: "
            f"{_yes_no(time['file_in_time_order'])}. It takes "
            f"{_or_dash(time['distinct_values'])} distinct values; "
            f"{_or_dash(time['rows_sharing_a_time'])} rows share their `Time` with another row, "
            f"and the largest group sharing one has {_or_dash(time['largest_group_sharing_a_time'])} "
            "rows.",
            "",
            "**Stops the rows triggered:** "
            + ("none." if not stops else "; ".join(str(stop) for stop in stops)),
        ]
    )


def _provenance_section(records: UlbCardRecords) -> str:
    runs = [
        (
            f"`{run.name}`",
            f"`{run.record.code_version}`" if run.record.code_version else "not recorded",
            f"`{run.record.featureset_version}`",
            _or_dash(run.fit.get("random_state")),
            "none" if run.record.dataset.subsample is None else "recorded",
            ", ".join(
                f"{name} {version}" for name, version in sorted(run.record.library_versions.items())
            ),
        )
        for run in records.runs
    ]
    primary = records.primary.record
    dataset = primary.dataset
    files = ", ".join(f"`{name}` `{digest}`" for name, digest in sorted(dataset.files.items()))
    identity = primary.test_fold_identity
    return "\n".join(
        [
            "## 7. Provenance",
            "",
            _table(("Run", "Code", "Featureset", "Random state", "Subsample", "Libraries"), runs),
            "",
            "**Source data.** Raw data is never committed, so this digest is the durable link "
            "between a result and the bytes it was computed from.",
            "",
            _table(
                ("Dataset", "Licence", "Source", "Rows", "Frauds", "File (SHA-256)"),
                [
                    (
                        f"`{dataset.name}` `{dataset.version}`",
                        dataset.license,
                        dataset.source_url,
                        _or_dash(dataset.row_count),
                        _or_dash(dataset.fraud_count),
                        files,
                    )
                ],
            ),
            "",
            "**Test fold.** "
            + (
                "not identified."
                if identity is None
                else f"{_or_dash(identity.transaction_count)} transactions, the same in all three "
                f"runs, identified by `{identity.transaction_ids_sha256}`, the "
                f"{identity.to_dict()['digest_definition']}."
            ),
            "",
            "**How the rows were read and made into the matrix**, as each run records it:",
            "",
            *(f"- {step}" for step in dataset.preprocessing),
            "",
            f"`{dataset.name}`: {dataset.notes}",
        ]
    )


def _not_reported_section() -> str:
    return "\n".join(
        [
            "## 8. Not reported for ULB",
            "",
            "- **Featureset v1.** One of its seventeen features is derivable from ULB and sixteen "
            "are not, so v1 is not evaluated on ULB at all (decision 3).",
            "- **SHAP rankings.** The components are anonymised, so a ranking of one over another "
            "says nothing anyone can act on or check.",
            "- **Segment breakdowns.** No column describes a segment.",
            "- **A rules audit.** The rules read customers, merchants, countries and history, none "
            "of which exist in ULB.",
            "- **A transfer measurement**, in either direction. ULB and v1 share no feature space.",
            "- **Promotion.** No ULB run is promoted to the served artifacts (decision 12).",
            "- No resampling, synthetic minority rows, fitted calibrator, resampling-based "
            "interval, hour-of-day proxy or drift experiment.",
        ]
    )


def _limitations_section(records: UlbCardRecords) -> str:
    primary = records.primary
    limitations = [
        "**Chronological split.** The results come from one chronological split and are not "
        "comparable to ULB results from random splits, nor to any synthetic or Sparkov result.",
        "**Few test frauds.** The test fold is one slice at the end of the second day, holding "
        f"{_test_frauds(primary)} frauds among {_or_dash(primary.fit.get('test_size'))} rows; a "
        "handful of rows can move the metrics substantially.",
        "**Unknown PCA fit.** The publisher does not say which rows the PCA was fitted on. If it "
        "was fitted on all of them, the components carry unsupervised information from the test "
        "period; no labels are involved, and nothing downstream can undo it.",
        "**No card identifiers.** Whether a card appears in several folds cannot be checked, and "
        "rows are treated as independent, although frauds on one card probably are not.",
        "**Reused val fold.** The val fold is used both for early stopping and for threshold "
        "selection, so the realised val FPR is slightly optimistic. The test fold is untouched "
        "by both.",
        "**Label delay**, as stated in section 3.",
        "**Age and scope.** The data is from 2013, covers two days, and comes from European "
        "cardholders only. The results say nothing about production performance or present-day "
        "fraud.",
    ]
    return "\n".join(
        ["## 9. Accepted limitations", "", *(f"- {limitation}" for limitation in limitations)]
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _feature_span() -> str:
    components = [name for name in ULB_PCA_V1_FEATURES if name.startswith("V")]
    others = [name for name in ULB_PCA_V1_FEATURES if not name.startswith("V")]
    named_others = ", ".join(f"`{name}`" for name in others)
    return f"`{components[0]}` to `{components[-1]}` and {named_others}"


def _elapsed(seconds: Any) -> str:
    """A recorded `Time` in seconds, with the elapsed hours it spans."""
    value = float(seconds)
    shown = f"{int(value):,}" if value.is_integer() else f"{value:,}"
    return f"{shown} s ({value / 3600:.2f} h)"


def _names_or_none(names: Any) -> str:
    return ", ".join(f"`{name}`" for name in names) if names else "none"


def _test_frauds(run: UlbRecordedRun) -> str:
    return _or_dash(run.metrics.get("context", "fraud_counts", "test"))


def _folds(quality: RecordFile) -> list[Mapping[str, Any]]:
    folds: list[Mapping[str, Any]] = list(quality.get("chronological_split", "folds"))
    return folds


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_run(run: UlbRecordedRun) -> None:
    record, source = run.record, run.run.source
    if record.run_name != run.name:
        raise BenchmarkCardError(f"{source} is the record of run {record.run_name!r}, not {run.name!r}.")
    if record.dataset.name != DATASET_NAME:
        raise BenchmarkCardError(f"{source} records dataset {record.dataset.name!r}, not ULB.")
    if record.featureset_version != ULB_PCA_V1:
        raise BenchmarkCardError(
            f"{source} records featureset {record.featureset_version!r}, not {ULB_PCA_V1!r}."
        )
    if LICENCE_NOTICE not in record.notes or METHOD_OFFER not in record.notes:
        raise BenchmarkCardError(f"{source} does not carry the licence notice and method offer.")
    if list(run.feature_list.get("features")) != list(ULB_PCA_V1_FEATURES):
        raise BenchmarkCardError(
            f"{run.feature_list.source} does not list {ULB_PCA_V1!r}'s features in their order."
        )
    if run.fit.get("random_state") != run.seed:
        raise BenchmarkCardError(
            f"{run.fit.source} records random state {run.fit.get('random_state')}, but run "
            f"{run.name!r} is the run of random state {run.seed}."
        )

    measured_at = run.metrics.get("at_operating_threshold", "threshold")
    if measured_at != run.threshold.get("value"):
        raise BenchmarkCardError(
            f"{run.metrics.source} was measured at threshold {measured_at}, but "
            f"{run.threshold.source} records {run.threshold.get('value')}."
        )
    for key in ("recall_at_1pct_fpr", "recall_at_5pct_fpr"):
        if run.metrics.get("threshold_source", key) != TEST_ROC_CURVE_SOURCE:
            raise BenchmarkCardError(
                f"{run.metrics.source}: threshold_source.{key} is not a recall read on the test "
                "ROC curve."
            )
    if run.calibration.get("n_test_samples") != run.fit.get("test_size"):
        raise BenchmarkCardError(
            f"{run.calibration.source} was measured on {run.calibration.get('n_test_samples')} "
            f"rows, but {run.fit.source} records a test fold of {run.fit.get('test_size')}."
        )
    _check_quality_report(run)


def _check_quality_report(run: UlbRecordedRun) -> None:
    quality = run.quality
    if quality.get("dataset") != DATASET_NAME:
        raise BenchmarkCardError(f"{quality.source} reports on {quality.get('dataset')!r}.")
    if quality.get("licence", "notice") != LICENCE_NOTICE:
        raise BenchmarkCardError(f"{quality.source} does not carry the licence notice.")
    if quality.get("stops"):
        raise BenchmarkCardError(
            f"{quality.source} records stops, so run {run.name!r} should not have been trained: "
            f"{'; '.join(str(stop) for stop in quality.get('stops'))}"
        )
    if quality.get("chronological_split", "available") is not True:
        raise BenchmarkCardError(f"{quality.source} records no chronological split.")
    folds = _folds(quality)
    described = [(fold.get("name"), fold.get("rows"), fold.get("frauds")) for fold in folds]
    recorded = [
        (
            name,
            run.fit.get(f"{name}_size"),
            run.metrics.get("context", "fraud_counts", name),
        )
        for name in _FOLDS
    ]
    if described != recorded:
        raise BenchmarkCardError(
            f"{quality.source} describes folds {described}, but the run recorded {recorded}."
        )
    periods = [(period.name, period.start, period.end) for period in run.record.splits]
    spans = [
        (
            fold["name"],
            ELAPSED_TIME_ORIGIN + timedelta(seconds=float(fold["first_elapsed_seconds"])),
            ELAPSED_TIME_ORIGIN + timedelta(seconds=float(fold["last_elapsed_seconds"])),
        )
        for fold in folds
    ]
    if periods != spans:
        raise BenchmarkCardError(
            f"{quality.source} describes other fold spans than {run.run.source} records."
        )


def _check_one_benchmark(runs: Sequence[UlbRecordedRun]) -> None:
    first = runs[0]
    for other in runs[1:]:
        _check_same(first, other, "source data", _dataset_without_read_time)
        _check_same(first, other, "folds", lambda run: run.record.splits)
        _check_same(first, other, "test fold", lambda run: run.record.test_fold_identity)
        _check_same(first, other, "quality report", _quality_without_write_time)
        _check_same(
            first,
            other,
            "fit settings",
            lambda run: [run.fit.get(name) for name in _SHARED_FIT_SETTINGS]
            + [run.threshold.get("target_fpr")],
        )
    if sum(run.is_primary for run in runs) != 1:
        raise BenchmarkCardError(f"Exactly one run must be the primary run of seed {PRIMARY_SEED}.")


def _check_same(
    first: UlbRecordedRun, other: UlbRecordedRun, what: str, read: Any
) -> None:
    if read(first) != read(other):
        raise BenchmarkCardError(
            f"{first.name!r} and {other.name!r} record different {what}; the three runs are one "
            "benchmark and differ in their random state only."
        )


def _dataset_without_read_time(run: UlbRecordedRun) -> dict[str, Any]:
    return {k: v for k, v in run.record.dataset.to_dict().items() if k != "retrieved_at"}


def _quality_without_write_time(run: UlbRecordedRun) -> dict[str, Any]:
    return {
        k: v for k, v in run.quality.payload.items() if k not in _UNCOMPARED_QUALITY_FIELDS
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the ULB benchmark card from its runs")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=RUNS_ROOT,
        help="Directory holding the ULB run directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ULB_CARD_PATH,
        help=f"Where to write the card (default: {ULB_CARD_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Read the records, build the card, and write it only once all of it is built."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args = parse_args(argv)
    try:
        records = load_ulb_records(args.runs_root)
        card = build_ulb_card(records)
    except BenchmarkCardError as exc:
        log.error("The ULB benchmark card was not written: %s", exc)
        return 1
    written = write_benchmark_card(card, args.output)
    log.info("Wrote %s from %d records", written, len(records.files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
