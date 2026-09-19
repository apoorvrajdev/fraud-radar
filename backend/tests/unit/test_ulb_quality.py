"""Phase 5E — the ULB quality report, on fixture files.

Beyond the counts themselves, these tests hold the report to two promises:
its folds are the folds a run makes of the same rows, and it is aggregate-only,
so committing it never publishes a row of the ULB database.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ml.paths import ML_ROOT
from ml.splits import chronological_split
from ml.tracks.ulb.load import LICENCE_NOTICE, METHOD_OFFER, field_inventory
from ml.tracks.ulb.quality import (
    QUALITY_DEFINITIONS,
    ULB_QUALITY_REPORT_FILENAME,
    UlbQualityReport,
    build_quality_report,
)
from tests.unit.test_ulb_load import load_fixture, ulb_row

DATA_LICENSES = ML_ROOT.parents[1] / "docs" / "DATA_LICENSES.md"


def _corpus(
    n_rows: int = 20,
    *,
    fraud_times: Sequence[int] = (2, 15, 18),
    times: Sequence[str] | None = None,
) -> list[list[str]]:
    """`n_rows` distinct rows at Times 0..n-1, fraudulent at `fraud_times`.

    With 20 rows the split puts Times 0-13 in train, 14-16 in val and 17-19 in
    test, so the default frauds fall one in each fold.
    """
    stamps = list(times) if times is not None else [str(index) for index in range(n_rows)]
    return [
        ulb_row(stamp, marker=index, label="1" if index in fraud_times else "0")
        for index, stamp in enumerate(stamps)
    ]


def _report(tmp_path: Path, rows: Sequence[Sequence[str]]) -> UlbQualityReport:
    return build_quality_report(load_fixture(tmp_path, rows))


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def test_counts_rows_frauds_exclusions_and_zero_amounts(tmp_path: Path) -> None:
    rows = [
        *_corpus(),
        ulb_row("20", marker=20, amount="0.00"),
        ulb_row("21", marker=21, amount="-3.00"),
        ulb_row("22", marker=22, label="5"),
    ]
    report = _report(tmp_path, rows)
    payload = report.to_dict()

    assert payload["rows"] == {"read": 23, "kept": 21, "excluded": 2}
    assert payload["exclusions"] == [
        {"reason": "missing_value", "count": 0},
        {"reason": "non_numeric_value", "count": 0},
        {"reason": "label_not_zero_or_one", "count": 1},
        {"reason": "negative_amount", "count": 1},
    ]
    assert payload["label"] == {"fraud_count": 3, "fraud_rate": 3 / 21}
    assert payload["amounts"] == {"zero_amount_rows": 1}


def test_exact_duplicates_are_counted_and_every_one_is_kept(tmp_path: Path) -> None:
    same = ulb_row("1", marker=1)
    fraud = ulb_row("2", marker=2, label="1")
    conflict = ulb_row("3", marker=3)
    rows = [
        same, same, same,
        fraud, fraud,
        conflict, [*conflict[:-1], "1"],
        ulb_row("4", marker=4),
    ]  # fmt: skip
    report = _report(tmp_path, rows)

    assert report.kept_row_count == 8
    assert report.to_dict()["duplicates"] == {
        "exact_duplicate_groups": 2,
        "rows_in_exact_duplicate_groups": 5,
        "repeated_rows": 3,
        "largest_exact_duplicate_group": 3,
        "exact_duplicate_groups_of_fraud_rows": 1,
        "label_conflict_groups": 1,
        "rows_in_label_conflict_groups": 2,
    }


def test_time_coverage_and_ties(tmp_path: Path) -> None:
    times = ["9", "0", "0", "5", "5", "5"]
    report = _report(tmp_path, [ulb_row(time, marker=i) for i, time in enumerate(times)])

    assert report.to_dict()["time"] == {
        "first_elapsed_seconds": 0,
        "last_elapsed_seconds": 9,
        "whole_seconds": True,
        "file_in_time_order": False,
        "distinct_values": 3,
        "rows_sharing_a_time": 5,
        "largest_group_sharing_a_time": 3,
    }


def test_fractional_time_is_reported_not_rounded(tmp_path: Path) -> None:
    report = _report(tmp_path, [ulb_row("0", marker=0), ulb_row("7.5", marker=1)])

    assert report.time.whole_seconds is False
    assert report.to_dict()["time"]["last_elapsed_seconds"] == 7.5


# ---------------------------------------------------------------------------
# The chronological split a run makes of the same rows
# ---------------------------------------------------------------------------


def test_folds_are_the_split_a_run_makes_of_the_same_rows(tmp_path: Path) -> None:
    source = load_fixture(tmp_path, _corpus())
    report = build_quality_report(source)
    splits = chronological_split(source.timestamps)

    folds = report.to_dict()["chronological_split"]["folds"]
    assert tuple(fold["rows"] for fold in folds) == splits.sizes == (14, 3, 3)
    assert folds == [
        {"name": "train", "rows": 14, "frauds": 1,
         "first_elapsed_seconds": 0, "last_elapsed_seconds": 13},
        {"name": "val", "rows": 3, "frauds": 1,
         "first_elapsed_seconds": 14, "last_elapsed_seconds": 16},
        {"name": "test", "rows": 3, "frauds": 1,
         "first_elapsed_seconds": 17, "last_elapsed_seconds": 19},
    ]  # fmt: skip
    boundaries = report.to_dict()["chronological_split"]["boundaries"]
    assert [boundary["between"] for boundary in boundaries] == [["train", "val"], ["val", "test"]]
    assert not any(boundary["instant_shared"] for boundary in boundaries)


def test_a_time_shared_across_a_boundary_is_recorded_not_corrected(tmp_path: Path) -> None:
    """Rows 13 and 14 share Time 13; the split puts one in train and one in val."""
    times = [str(index) for index in range(20)]
    times[14] = "13"
    report = _report(tmp_path, _corpus(times=times))

    first = report.to_dict()["chronological_split"]["boundaries"][0]
    assert first["instant_shared"] is True
    assert first["earlier_fold_last_elapsed_seconds"] == 13
    assert first["later_fold_first_elapsed_seconds"] == 13
    assert first["rows_at_shared_instant"] == {"earlier_fold": 1, "later_fold": 1}
    # Tied, but not identical: no duplicate straddles.
    assert first["exact_duplicate_groups_straddling"] == 0
    train = report.split.fold("train")
    assert train is not None
    assert train.rows == 14


def test_exact_duplicates_straddling_a_boundary_are_counted(tmp_path: Path) -> None:
    rows = _corpus()
    rows[14] = list(rows[13])  # an exact copy of the last train row, first in val
    report = _report(tmp_path, rows)

    first, second = report.to_dict()["chronological_split"]["boundaries"]
    assert first["exact_duplicate_groups_straddling"] == 1
    assert first["exact_duplicate_pairs_straddling"] == 1
    assert second["exact_duplicate_groups_straddling"] == 0
    assert report.duplicates.groups == 1


def test_the_split_is_unavailable_when_a_fold_would_be_empty(tmp_path: Path) -> None:
    report = _report(tmp_path, _corpus(3, fraud_times=(0,)))

    assert report.to_dict()["chronological_split"] == {
        "available": False,
        "reason": "too few rows for every fold of the 70/15/15 split to hold one.",
    }
    assert any("split is unavailable" in stop for stop in report.stops)


def test_a_file_with_no_rows_is_reported_and_stops(tmp_path: Path) -> None:
    report = _report(tmp_path, [])

    assert report.kept_row_count == 0
    assert report.to_dict()["time"]["first_elapsed_seconds"] is None
    assert report.stops == (
        "The chronological split is unavailable: fewer than 3 rows cannot be split into folds.",
    )


def test_the_split_is_unavailable_below_three_rows(tmp_path: Path) -> None:
    report = _report(tmp_path, _corpus(2, fraud_times=(0,)))

    assert report.split.available is False
    assert "fewer than 3 rows" in report.split.unavailable_reason


# ---------------------------------------------------------------------------
# Decision 15 stops the rows can answer
# ---------------------------------------------------------------------------


def test_no_stop_when_nothing_is_excluded_and_every_fold_holds_fraud(tmp_path: Path) -> None:
    report = _report(tmp_path, _corpus())

    assert report.stops == ()
    assert report.to_dict()["stops"] == []


def test_any_excluded_row_is_a_stop(tmp_path: Path) -> None:
    report = _report(tmp_path, [*_corpus(), ulb_row("20", marker=20, amount="-1.00")])

    assert report.stops == (
        "1 row(s) were excluded; no row may be excluded before a model is trained.",
    )


def test_a_val_or_test_fold_without_fraud_is_a_stop(tmp_path: Path) -> None:
    report = _report(tmp_path, _corpus(fraud_times=(2,)))

    assert report.stops == (
        "The val fold holds no fraud, so early stopping and threshold selection would be "
        "undefined.",
        "The test fold holds no fraud, so the test metrics would be undefined.",
    )


# ---------------------------------------------------------------------------
# Licence and schema accounting travel with the report
# ---------------------------------------------------------------------------


def test_the_report_carries_the_licence_notice_and_method_offer(tmp_path: Path) -> None:
    source = load_fixture(tmp_path, _corpus())
    licence = build_quality_report(source).to_dict()["licence"]

    assert licence["notice"] == LICENCE_NOTICE
    assert "https://opendatacommons.org/licenses/odbl/1-0/" in licence["notice"]
    assert "https://opendatacommons.org/licenses/dbcl/1-0/" in licence["notice"]
    assert licence["method_offer"] == METHOD_OFFER
    assert "4.6(b)" in licence["method_offer"]
    assert "backend/ml/tracks/ulb/" in licence["method_offer"]
    assert licence["licence"] == source.provenance.license
    assert licence["citation"] == source.provenance.citation
    assert licence["terms"] == "docs/DATA_LICENSES.md"


def test_the_notice_is_the_one_data_licenses_records() -> None:
    """The code's plain-text notice and the documented markdown one say the same thing."""
    quoted = [
        line[2:]
        for line in DATA_LICENSES.read_text(encoding="utf-8").splitlines()
        if line.startswith("> Contains information from the")
    ]
    assert quoted, "DATA_LICENSES.md no longer holds the ULB notice"

    as_plain_text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", quoted[0])
    assert as_plain_text == LICENCE_NOTICE


def test_the_report_carries_the_schema_accounting(tmp_path: Path) -> None:
    payload = _report(tmp_path, _corpus()).to_dict()

    assert payload["field_notes"] == [note.to_dict() for note in field_inventory()]
    assert payload["definitions"] == dict(QUALITY_DEFINITIONS)


def test_the_report_records_the_pinned_digest(tmp_path: Path) -> None:
    source = load_fixture(tmp_path, _corpus())

    assert build_quality_report(source).to_dict()["source"] == {"files": dict(source.provenance.files)}


# ---------------------------------------------------------------------------
# Aggregate-only
# ---------------------------------------------------------------------------


def _shape(value: Any) -> Any:
    """The report's structure, with every scalar replaced by its type and lists kept by length."""
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return type(value).__name__


def test_the_report_does_not_grow_with_the_rows(tmp_path: Path) -> None:
    """A report of 60 rows has exactly the structure of a report of 20: nothing is per-row."""
    small = _report(tmp_path / "small", _corpus(20, fraud_times=(2, 15, 18)))
    large = _report(tmp_path / "large", _corpus(60, fraud_times=(2, 45, 55)))

    assert _shape(small.to_dict()) == _shape(large.to_dict())


def test_the_report_holds_no_value_of_any_row(tmp_path: Path) -> None:
    """No component, amount or example appears anywhere in the written report."""
    components = [f"7.12345{index:02d}9" for index in range(28)]
    rows = [
        ulb_row(str(index), marker=index, amount=f"4321.{index:02d}",
                label="1" if index in (2, 15, 18) else "0",
                components=[f"{value}{index:02d}" for value in components])
        for index in range(20)
    ]  # fmt: skip
    rows.append(ulb_row("20", marker=20, amount="-98765.43"))  # excluded, never an example
    text = json.dumps(_report(tmp_path, rows).to_dict())

    # Checked as the numbers they are, in the form JSON would write them.
    for row in rows:
        for value in row[1:-1]:
            assert repr(abs(float(value))) not in text, f"{value} leaked into the report"
    assert "examples" not in text


def test_write_produces_the_ulb_quality_report_file(tmp_path: Path) -> None:
    report = _report(tmp_path / "data", _corpus())
    target = report.write(tmp_path / "run")

    assert target == tmp_path / "run" / ULB_QUALITY_REPORT_FILENAME == tmp_path / "run" / (
        "ulb_quality_report.json"
    )
    text = target.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert json.loads(text) == report.to_dict()


@pytest.mark.parametrize("key", ["licence", "stops", "chronological_split", "duplicates"])
def test_written_report_always_has_its_sections(tmp_path: Path, key: str) -> None:
    report = _report(tmp_path / "data", _corpus(2, fraud_times=(0,)))
    written = json.loads(report.write(tmp_path / "run").read_text(encoding="utf-8"))

    assert key in written
