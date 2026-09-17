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
from ml.datasets.base import DatasetContractError
from ml.paths import InvalidRunNameError, validate_run_name
from ml.runs import RunMetadata

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


def _reject_constant(value: str) -> float:
    raise ValueError(f"{value} is not a JSON number")
