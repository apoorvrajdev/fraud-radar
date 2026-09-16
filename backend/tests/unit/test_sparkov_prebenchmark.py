"""Pre-benchmark gate — the properties that only bite on the real corpus.

Fixtures cannot reproduce 1.85M rows, but they can pin the behaviours that
would go wrong there: one timestamp parser rather than two, a refusal to load
the whole corpus by accident, and the arithmetic behind that refusal.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from ml.datasets.base import Subsample
from ml.datasets.sparkov import (
    ESTIMATED_BYTES_PER_ROW,
    FULL_CORPUS_ROW_LIMIT,
    SPARKOV_TIMESTAMP_FORMAT,
    CorpusTooLargeError,
    SparkovAdapter,
    _guard_projected_size,
    parse_timestamps,
    projected_bytes,
)
from tests.unit.test_dataset_sparkov import _manifest, _row, _write_csv

PUBLISHED_ROW_COUNT = 1_852_394  # fraudTrain.csv + fraudTest.csv, as published


def _adapter(tmp_path: Path, **kwargs: object) -> SparkovAdapter:
    return SparkovAdapter(
        manifest_path=_manifest(tmp_path),
        verify_hashes=False,
        **kwargs,  # type: ignore[arg-type]
    )


def _corpus(tmp_path: Path, rows: list[list[str]], name: str = "gate") -> Path:
    root = tmp_path / "raw" / name
    root.mkdir(parents=True)
    _write_csv(root / "fraudTrain.csv", rows)
    return root


# ---------------------------------------------------------------------------
# Finding A/B — one parser, one interpretation
# ---------------------------------------------------------------------------


def test_the_documented_source_format_is_what_we_parse() -> None:
    assert SPARKOV_TIMESTAMP_FORMAT == "%Y-%m-%d %H:%M:%S"


def test_valid_timestamps_parse_to_the_same_instant_in_both_paths(tmp_path: Path) -> None:
    """The validation gate and the object builder now read one parsed column."""
    stamps = ["2019-01-01 00:00:18", "2019-06-15 13:45:59", "2020-12-31 23:59:59"]
    rows = [
        _row(index, trans_num=f"t{index}", timestamp=stamp)
        for index, stamp in enumerate(stamps)
    ]
    dataset = _adapter(tmp_path).load(_corpus(tmp_path, rows))

    built = [tx.created_at for tx in dataset.transactions]
    expected = [datetime.fromisoformat(stamp).replace(tzinfo=UTC) for stamp in stamps]
    assert built == expected


def test_parse_timestamps_adds_one_reusable_column() -> None:
    frame = pd.DataFrame({"trans_date_trans_time": ["2019-01-01 00:00:18", "nonsense"]})
    parsed = parse_timestamps(frame)

    assert parsed["parsed_at"].iloc[0] == pd.Timestamp("2019-01-01 00:00:18")
    assert pd.isna(parsed["parsed_at"].iloc[1])


@pytest.mark.parametrize(
    "bad_stamp",
    [
        "not-a-date",
        "2019-13-45 00:00:00",  # impossible month and day
        "01/02/2019 10:15:00",  # a format inference would have accepted, ambiguously
        "2019-01-01T00:00:18",  # ISO separator, not the published format
    ],
)
def test_malformed_timestamps_are_excluded_not_crashed_on(
    tmp_path: Path, bad_stamp: str
) -> None:
    """Previously these could pass the gate and then raise mid-build."""
    rows = [
        _row(0, trans_num="good", timestamp="2019-01-01 10:00:00"),
        _row(1, trans_num="bad", timestamp=bad_stamp),
    ]
    result = _adapter(tmp_path).load_detailed(_corpus(tmp_path, rows, name=f"bad{len(bad_stamp)}"))
    reasons = {record.reason: record.count for record in result.exclusions}

    assert reasons["unparsable_timestamp"] == 1
    assert result.dataset.n_rows == 1
    assert result.dataset.transactions[0].idempotency_key == "good"


def test_an_empty_timestamp_is_caught_as_a_missing_field(tmp_path: Path) -> None:
    """Excluded either way; the reason should name what was actually wrong."""
    rows = [
        _row(0, trans_num="good", timestamp="2019-01-01 10:00:00"),
        _row(1, trans_num="blank", timestamp=""),
    ]
    result = _adapter(tmp_path).load_detailed(_corpus(tmp_path, rows, name="blank"))
    reasons = {record.reason: record.count for record in result.exclusions}

    assert reasons["missing_required_field"] == 1
    assert reasons["unparsable_timestamp"] == 0
    assert result.dataset.n_rows == 1


def test_ambiguous_day_month_strings_are_refused_rather_than_guessed(
    tmp_path: Path,
) -> None:
    """01/02/2019 could be January 2nd or February 1st. Neither is assumed."""
    rows = [_row(0, trans_num="ambiguous", timestamp="01/02/2019 10:15:00")]
    result = _adapter(tmp_path).load_detailed(_corpus(tmp_path, rows, name="ambiguous"))

    assert result.dataset.n_rows == 0
    assert {r.reason: r.count for r in result.exclusions}["unparsable_timestamp"] == 1


def test_timestamps_are_utc_aware_explicitly(tmp_path: Path) -> None:
    rows = [_row(0, trans_num="t1", timestamp="2019-07-04 12:00:00")]
    dataset = _adapter(tmp_path).load(_corpus(tmp_path, rows, name="tz"))

    created = dataset.transactions[0].created_at
    assert created.tzinfo is not None
    assert created.utcoffset() == datetime.now(UTC).utcoffset()
    assert type(created) is datetime  # a plain datetime, not a pandas Timestamp


# ---------------------------------------------------------------------------
# Finding C — the full corpus is an explicit decision
# ---------------------------------------------------------------------------


def test_projection_uses_the_measured_per_row_cost() -> None:
    assert projected_bytes(1_000) == 1_000 * ESTIMATED_BYTES_PER_ROW
    # The published corpus needs multiple gigabytes for canonical objects alone.
    assert projected_bytes(PUBLISHED_ROW_COUNT) > 3_000_000_000


def test_full_corpus_above_the_limit_is_refused_by_default() -> None:
    with pytest.raises(CorpusTooLargeError, match="without subsampling"):
        _guard_projected_size(
            PUBLISHED_ROW_COUNT,
            subsample=Subsample(strategy="none", seed=42),
            allow_full_corpus=False,
        )


def test_the_refusal_names_both_ways_forward() -> None:
    with pytest.raises(CorpusTooLargeError) as excinfo:
        _guard_projected_size(
            PUBLISHED_ROW_COUNT,
            subsample=Subsample(strategy="none", seed=42),
            allow_full_corpus=False,
        )
    message = str(excinfo.value)
    assert "--max-cards" in message
    assert "--full-corpus" in message
    assert "GB" in message


def test_explicit_opt_in_allows_the_full_corpus() -> None:
    _guard_projected_size(
        PUBLISHED_ROW_COUNT,
        subsample=Subsample(strategy="none", seed=42),
        allow_full_corpus=True,
    )


def test_a_subsampled_run_is_never_guarded() -> None:
    """Subsampling is the documented path; it must not need a second flag."""
    _guard_projected_size(
        PUBLISHED_ROW_COUNT,
        subsample=Subsample(strategy="cards", seed=42, max_entities=200),
        allow_full_corpus=False,
    )


def test_a_small_unsubsampled_corpus_is_not_guarded() -> None:
    _guard_projected_size(
        FULL_CORPUS_ROW_LIMIT,
        subsample=Subsample(strategy="none", seed=42),
        allow_full_corpus=False,
    )


def test_the_limit_leaves_room_for_the_documented_dev_run() -> None:
    """200 cards is ~370k rows; the guard must not fire on the normal path."""
    assert FULL_CORPUS_ROW_LIMIT > 370_000
    assert FULL_CORPUS_ROW_LIMIT < PUBLISHED_ROW_COUNT


def test_fixture_loads_are_unaffected_by_the_guard(tmp_path: Path) -> None:
    rows = [_row(index, trans_num=f"t{index}") for index in range(5)]
    assert _adapter(tmp_path).load(_corpus(tmp_path, rows, name="small")).n_rows == 5
