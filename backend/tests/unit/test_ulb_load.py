"""Phase 5E — the ULB loader, exercised on fixture files.

No test reads the real corpus. The fixtures reproduce the published file's
layout as acquisition recorded it: every header name quoted, data values
unquoted except `Class`, which arrives as "0" or "1", LF line endings, and a
`Time` written in scientific notation.
"""
from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from app.fraud.feature_spec import FEATURESETS
from ml.datasets.base import CANONICAL_SCHEMA_VERSION, DataOrigin
from ml.datasets.manifest import ManifestError, load_manifest_entry, sha256_file
from ml.datasets.quality import FieldStatus
from ml.datasets.registry import available_datasets
from ml.paths import ML_ROOT, RAW_DATA_DIR
from ml.tracks.ulb.load import (
    COMPONENT_COLUMNS,
    DATASET_NAME,
    DATASET_VERSION,
    DEFAULT_ROOT,
    ELAPSED_TIME_ORIGIN,
    EXCLUSION_REASONS,
    LABEL_NOT_ZERO_OR_ONE,
    MISSING_VALUE,
    NEGATIVE_AMOUNT,
    NON_NUMERIC_VALUE,
    SOURCE_COLUMNS,
    SOURCE_FILENAME,
    SOURCE_SCHEMA_VERSION,
    UlbSchemaError,
    UlbSource,
    elapsed_timestamps,
    field_inventory,
    load_ulb,
)

FIXTURE_LICENCE = (
    "DbCL-1.0 for the contents, ODbL-1.0 for the database (fixture, retrieved 2026-09-18)"
)


def ulb_row(
    time: str,
    *,
    marker: int,
    amount: str = "10.00",
    label: str = "0",
    components: Sequence[str] | None = None,
) -> list[str]:
    """One data row, unquoted. `marker` makes a row's components distinct from other rows'."""
    values = (
        list(components)
        if components is not None
        else [f"{marker + index / 100:.4f}" for index in range(len(COMPONENT_COLUMNS))]
    )
    return [time, *values, amount, label]


def write_ulb_csv(
    path: Path,
    rows: Sequence[Sequence[str]],
    *,
    header: Sequence[str] = SOURCE_COLUMNS,
    quote_class: bool = True,
) -> Path:
    """Write rows in the published layout: quoted header, quoted Class, LF endings."""
    lines = [",".join(f'"{name}"' for name in header)]
    for row in rows:
        fields = list(row)
        if quote_class and fields:
            fields[-1] = f'"{fields[-1]}"'
        lines.append(",".join(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


def write_ulb_manifest(
    directory: Path,
    csv_path: Path,
    *,
    pinned: bool = True,
    filenames: Sequence[str] = (SOURCE_FILENAME,),
) -> Path:
    """A manifest whose `ulb` entry is pinned to the fixture file's size and digest."""
    files = {
        name: {
            "bytes": csv_path.stat().st_size if pinned and csv_path.exists() else None,
            "sha256": sha256_file(csv_path) if pinned and csv_path.exists() else None,
        }
        for name in filenames
    }
    payload = {
        "manifest_version": "1",
        "datasets": {
            "ulb": {
                "display_name": "Fixture ULB",
                "origin": "real",
                "source": "kaggle",
                "reference": "mlg-ulb/creditcardfraud",
                "source_url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
                "license": FIXTURE_LICENCE,
                "citation": "Fixture citation",
                "label_field": "Class",
                "label_definition": "1 = fraud, 0 = otherwise",
                "notes": "Real, anonymised fixture.",
                "files": files,
            }
        },
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_fixture(
    tmp_path: Path,
    rows: Sequence[Sequence[str]],
    *,
    quote_class: bool = True,
) -> UlbSource:
    """Write `rows` as the ULB file, pin a manifest to it, and load it."""
    root = tmp_path / "raw" / "ulb"
    csv_path = write_ulb_csv(root / SOURCE_FILENAME, rows, quote_class=quote_class)
    return load_ulb(root, manifest_path=write_ulb_manifest(tmp_path, csv_path))


def _pinned_fixture(tmp_path: Path, content: bytes) -> tuple[Path, Path]:
    """A raw file with exactly `content`, and a manifest pinned to it."""
    root = tmp_path / "raw" / "ulb"
    root.mkdir(parents=True)
    csv_path = root / SOURCE_FILENAME
    csv_path.write_bytes(content)
    return root, write_ulb_manifest(tmp_path, csv_path)


# ---------------------------------------------------------------------------
# The published layout
# ---------------------------------------------------------------------------


def test_source_columns_are_the_published_header() -> None:
    """Pinned literally, as acquisition observed it."""
    assert SOURCE_COLUMNS == (
        "Time",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Amount",
        "Class",
    )  # fmt: skip


def test_reads_the_published_layout(tmp_path: Path) -> None:
    source = load_fixture(
        tmp_path,
        [
            ulb_row("0", marker=1, amount="149.62", label="0"),
            ulb_row("1", marker=2, amount="2.69", label="1"),
        ],
    )

    assert source.n_rows == 2
    assert source.raw_row_count == 2
    assert source.time_seconds.tolist() == [0.0, 1.0]
    assert source.amounts.tolist() == [149.62, 2.69]
    assert source.labels.dtype == np.int64
    assert source.labels.tolist() == [0, 1]
    assert source.fraud_count == 1
    assert source.components.shape == (2, 28)
    assert source.components[0, 0] == 1.0
    assert source.components[0, 27] == 1.27
    assert source.components[1, 0] == 2.0


def test_values_are_read_exactly_as_published(tmp_path: Path) -> None:
    """No transform and no lossy parse: each value is the float its text denotes."""
    published = [f"-1.35980713367{index:02d}" for index in range(28)]
    source = load_fixture(tmp_path, [ulb_row("0", marker=0, components=published)])

    assert source.components[0].tolist() == [float(value) for value in published]


def test_time_in_scientific_notation_is_read_as_a_number(tmp_path: Path) -> None:
    """The published file writes one Time, 100000, as 1e+05."""
    source = load_fixture(
        tmp_path,
        [
            ulb_row("99999", marker=1),
            ulb_row("1e+05", marker=2),
            ulb_row("100001", marker=3),
            ulb_row("1.5E3", marker=4),
        ],
    )

    assert source.time_seconds.tolist() == [1500.0, 99999.0, 100000.0, 100001.0]
    assert source.positions.tolist() == [4, 1, 2, 3]
    assert source.excluded_row_count == 0


def test_class_is_read_through_its_quotes_or_without_them(tmp_path: Path) -> None:
    rows = [ulb_row("0", marker=1, label="1"), ulb_row("1", marker=2, label="0")]
    quoted = load_fixture(tmp_path / "quoted", rows, quote_class=True)
    unquoted = load_fixture(tmp_path / "unquoted", rows, quote_class=False)

    assert quoted.labels.tolist() == unquoted.labels.tolist() == [1, 0]


def test_zero_amounts_are_kept(tmp_path: Path) -> None:
    source = load_fixture(
        tmp_path,
        [
            ulb_row("0", marker=1, amount="0"),
            ulb_row("1", marker=2, amount="0.00"),
            ulb_row("2", marker=3, amount="5.00"),
        ],
    )

    assert source.n_rows == 3
    assert source.excluded_row_count == 0
    assert source.amounts.tolist() == [0.0, 0.0, 5.0]


# ---------------------------------------------------------------------------
# Refusals: the file is not the published layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        pytest.param(SOURCE_COLUMNS[:-1], id="missing-class"),
        pytest.param((*SOURCE_COLUMNS, "Extra"), id="extra-column"),
        pytest.param((*SOURCE_COLUMNS[:-2], "Class", "Amount"), id="reordered"),
        pytest.param(("time", *SOURCE_COLUMNS[1:]), id="renamed"),
    ],
)
def test_a_header_that_is_not_exactly_the_published_one_is_refused(
    tmp_path: Path, header: Sequence[str]
) -> None:
    root = tmp_path / "raw" / "ulb"
    csv_path = write_ulb_csv(root / SOURCE_FILENAME, [ulb_row("0", marker=1)], header=header)
    manifest = write_ulb_manifest(tmp_path, csv_path)

    with pytest.raises(UlbSchemaError, match="expected exactly"):
        load_ulb(root, manifest_path=manifest)


def test_a_byte_order_mark_is_refused(tmp_path: Path) -> None:
    """Acquisition found no BOM; one would turn the first header name into something else."""
    header = ",".join(f'"{name}"' for name in SOURCE_COLUMNS)
    root, manifest = _pinned_fixture(tmp_path, b"\xef\xbb\xbf" + header.encode() + b"\n")

    with pytest.raises(UlbSchemaError, match="expected exactly"):
        load_ulb(root, manifest_path=manifest)


def test_a_row_with_the_wrong_number_of_fields_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "ulb"
    short = ulb_row("1", marker=2)[:-1]
    csv_path = write_ulb_csv(
        root / SOURCE_FILENAME, [ulb_row("0", marker=1), short], quote_class=False
    )
    manifest = write_ulb_manifest(tmp_path, csv_path)

    with pytest.raises(UlbSchemaError, match="data row 2 has 30 fields, not 31"):
        load_ulb(root, manifest_path=manifest)


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    root, manifest = _pinned_fixture(tmp_path, b"")

    with pytest.raises(UlbSchemaError, match="is empty"):
        load_ulb(root, manifest_path=manifest)


def test_a_file_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    header = ",".join(f'"{name}"' for name in SOURCE_COLUMNS)
    root, manifest = _pinned_fixture(tmp_path, header.encode() + b"\n\xff\xfe\n")

    with pytest.raises(UlbSchemaError, match="not UTF-8"):
        load_ulb(root, manifest_path=manifest)


# ---------------------------------------------------------------------------
# Exclusions: counted by reason, never silent, never with examples
# ---------------------------------------------------------------------------


def test_invalid_rows_are_excluded_and_counted_by_reason(tmp_path: Path) -> None:
    def with_component(time: str, marker: int, index: int, value: str) -> list[str]:
        row = ulb_row(time, marker=marker)
        row[1 + index] = value
        return row

    rows = [
        ulb_row("0", marker=1),  # 1: kept
        ulb_row("", marker=2),  # 2: missing Time
        with_component("2", 3, 4, "abc"),  # 3: non-numeric component
        ulb_row("3", marker=4, amount="nan"),  # 4: not a number
        with_component("4", 5, 0, "inf"),  # 5: not finite
        ulb_row("1e999", marker=6),  # 6: overflows to infinity
        ulb_row("6", marker=7, amount=" 12.00"),  # 7: padded
        ulb_row("7", marker=8, label="2"),  # 8: label out of range
        ulb_row("8", marker=9, label="0.5"),  # 9: label out of range
        ulb_row("9", marker=10, amount="-0.01"),  # 10: negative amount
        ulb_row("10", marker=11, amount="0.00"),  # 11: kept, zero amount
        ulb_row("", marker=12, amount="abc"),  # 12: missing is checked first
    ]
    source = load_fixture(tmp_path, rows)

    counts = {record.reason: record.count for record in source.exclusions}
    assert counts == {
        MISSING_VALUE: 2,
        NON_NUMERIC_VALUE: 5,
        LABEL_NOT_ZERO_OR_ONE: 2,
        NEGATIVE_AMOUNT: 1,
    }
    assert source.raw_row_count == 12
    assert source.excluded_row_count == 10
    assert source.positions.tolist() == [1, 11]
    assert source.provenance.row_count == 2
    assert "excluded 5 row(s): non_numeric_value" in source.provenance.preprocessing


def test_every_exclusion_reason_is_reported_even_when_nothing_is_excluded(
    tmp_path: Path,
) -> None:
    """Zero is a finding: the reader should not have to infer that a check ran."""
    source = load_fixture(tmp_path, [ulb_row("0", marker=1)])

    assert tuple(record.reason for record in source.exclusions) == EXCLUSION_REASONS
    assert all(record.count == 0 for record in source.exclusions)
    assert not any("excluded" in step for step in source.provenance.preprocessing)


def test_exclusions_never_carry_an_example_of_the_excluded_row(tmp_path: Path) -> None:
    source = load_fixture(tmp_path, [ulb_row("0", marker=1), ulb_row("1", marker=2, label="7")])

    assert all(record.examples == () for record in source.exclusions)


# ---------------------------------------------------------------------------
# Order, identity and elapsed time
# ---------------------------------------------------------------------------


def test_rows_are_ordered_by_time_then_file_position(tmp_path: Path) -> None:
    rows = [
        ulb_row("5", marker=1),
        ulb_row("3", marker=2),
        ulb_row("5", marker=3),
        ulb_row("1", marker=4),
        ulb_row("3", marker=5),
    ]
    source = load_fixture(tmp_path, rows)

    assert source.time_seconds.tolist() == [1.0, 3.0, 3.0, 5.0, 5.0]
    assert source.positions.tolist() == [4, 2, 5, 1, 3]
    assert source.transaction_ids == ["4", "2", "5", "1", "3"]
    # Each row's values travel with it.
    assert source.components[:, 0].tolist() == [4.0, 2.0, 5.0, 1.0, 3.0]
    assert source.file_in_time_order is False


def test_a_file_already_in_time_order_is_reported_as_such(tmp_path: Path) -> None:
    source = load_fixture(tmp_path, [ulb_row("0", marker=1), ulb_row("0", marker=2)])

    assert source.file_in_time_order is True
    assert source.positions.tolist() == [1, 2]


def test_transaction_ids_count_excluded_rows_as_positions(tmp_path: Path) -> None:
    """An id is the position in the file, so excluding a row renumbers nothing."""
    source = load_fixture(
        tmp_path,
        [ulb_row("0", marker=1), ulb_row("1", marker=2, label="9"), ulb_row("2", marker=3)],
    )

    assert source.transaction_ids == ["1", "3"]


def test_timestamps_are_elapsed_time_from_the_origin_not_calendar_dates(tmp_path: Path) -> None:
    source = load_fixture(tmp_path, [ulb_row("0", marker=1), ulb_row("90061", marker=2)])

    assert datetime(1970, 1, 1, tzinfo=UTC) == ELAPSED_TIME_ORIGIN
    assert source.timestamps.tolist() == [
        ELAPSED_TIME_ORIGIN,
        ELAPSED_TIME_ORIGIN + timedelta(days=1, hours=1, minutes=1, seconds=1),
    ]
    assert all(moment.tzinfo is UTC for moment in source.timestamps)
    assert elapsed_timestamps(np.array([2.5])).tolist() == [
        ELAPSED_TIME_ORIGIN + timedelta(seconds=2.5)
    ]


# ---------------------------------------------------------------------------
# Provenance: the pinned bytes and the source's own terms
# ---------------------------------------------------------------------------


def test_provenance_records_the_pinned_digest_and_the_manifest_terms(tmp_path: Path) -> None:
    source = load_fixture(
        tmp_path, [ulb_row("0", marker=1), ulb_row("7200", marker=2, label="1")]
    )
    provenance = source.provenance
    csv_path = tmp_path / "raw" / "ulb" / SOURCE_FILENAME

    assert dict(provenance.files) == {SOURCE_FILENAME: sha256_file(csv_path)}
    assert provenance.name == DATASET_NAME
    assert provenance.version == DATASET_VERSION
    assert provenance.origin is DataOrigin.REAL
    assert provenance.license == FIXTURE_LICENCE
    assert provenance.citation == "Fixture citation"
    assert provenance.label_field == "Class"
    assert provenance.row_count == 2
    assert provenance.fraud_count == 1
    assert provenance.subsample is None
    assert provenance.period_start == ELAPSED_TIME_ORIGIN
    assert provenance.period_end == ELAPSED_TIME_ORIGIN + timedelta(hours=2)
    # Not in the canonical schema, so it does not claim the canonical version.
    assert provenance.schema_version == SOURCE_SCHEMA_VERSION != CANONICAL_SCHEMA_VERSION
    assert any("not calendar dates" in step for step in provenance.preprocessing)
    assert any("1-based position" in step for step in provenance.preprocessing)
    assert "not calendar dates" in provenance.notes


def test_a_file_that_differs_from_its_pinned_digest_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "ulb"
    csv_path = write_ulb_csv(root / SOURCE_FILENAME, [ulb_row("0", marker=1, amount="10.00")])
    manifest = write_ulb_manifest(tmp_path, csv_path)
    # Same size, different bytes.
    csv_path.write_bytes(csv_path.read_bytes().replace(b"10.00", b"19.00"))

    with pytest.raises(ManifestError, match="hash_mismatch"):
        load_ulb(root, manifest_path=manifest)


def test_a_missing_file_is_refused_by_verification(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "ulb"
    csv_path = write_ulb_csv(root / SOURCE_FILENAME, [ulb_row("0", marker=1)])
    manifest = write_ulb_manifest(tmp_path, csv_path)
    csv_path.unlink()

    with pytest.raises(ManifestError, match="missing"):
        load_ulb(root, manifest_path=manifest)


def test_an_unpinned_entry_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "ulb"
    csv_path = write_ulb_csv(root / SOURCE_FILENAME, [ulb_row("0", marker=1)])
    manifest = write_ulb_manifest(tmp_path, csv_path, pinned=False)

    with pytest.raises(ManifestError, match="no pinned digest"):
        load_ulb(root, manifest_path=manifest)


def test_an_entry_listing_other_files_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "ulb"
    csv_path = write_ulb_csv(root / SOURCE_FILENAME, [ulb_row("0", marker=1)])
    manifest = write_ulb_manifest(
        tmp_path, csv_path, filenames=(SOURCE_FILENAME, "extra.csv")
    )

    with pytest.raises(UlbSchemaError, match=r"one file creditcard\.csv"):
        load_ulb(root, manifest_path=manifest)


def test_the_loader_reads_the_committed_pinned_entry_by_default() -> None:
    """Without the corpus: the default entry and location are the committed, pinned ones."""
    entry = load_manifest_entry(DATASET_NAME)

    assert entry.filenames == (SOURCE_FILENAME,)
    assert entry.file(SOURCE_FILENAME).is_pinned
    assert DEFAULT_ROOT == RAW_DATA_DIR / "ulb"


# ---------------------------------------------------------------------------
# Canonical mapping: the schema accounting
# ---------------------------------------------------------------------------


def test_field_inventory_is_the_schema_accounting() -> None:
    """Every ULB column's canonical target, and every canonical field ULB cannot supply."""
    statuses = {note.field: note.status for note in field_inventory()}

    assert statuses == {
        "Time": FieldStatus.MAPPED,
        "Amount": FieldStatus.MAPPED,
        "Class": FieldStatus.MAPPED,
        "V1-V28": FieldStatus.MAPPED,
        "row position": FieldStatus.DERIVED,
        "Transaction.customer_id / Customer.*": FieldStatus.UNAVAILABLE,
        "Transaction.merchant_id / Merchant.*": FieldStatus.UNAVAILABLE,
        "Transaction.country / is_card_present / payment_method / status / currency": (
            FieldStatus.UNAVAILABLE
        ),
    }


def test_field_inventory_states_what_each_mapping_does_and_does_not_mean() -> None:
    details = {note.field: note.detail for note in field_inventory()}

    assert "Transaction.created_at" in details["Time"]
    assert "never a calendar date" in details["Time"]
    assert "not a feature" in details["Time"]
    assert "Transaction.amount" in details["Amount"]
    assert "none assumed" in details["Amount"]
    assert "no canonical field" in details["V1-V28"]
    assert "Transaction.id" in details["row position"]
    assert "not a source identifier" in details["row position"]


def test_field_inventory_covers_every_source_column() -> None:
    fields = {note.field for note in field_inventory()}
    first, last = COMPONENT_COLUMNS[0], COMPONENT_COLUMNS[-1]

    assert {"Time", "Amount", "Class", f"{first}-{last}"} <= fields
    assert tuple(f"V{index}" for index in range(1, 29)) == COMPONENT_COLUMNS


# ---------------------------------------------------------------------------
# Kept apart from the production feature path (decisions 2 and 4)
# ---------------------------------------------------------------------------


def test_ulb_is_neither_a_registered_adapter_nor_a_production_featureset() -> None:
    import ml.tracks.ulb.quality  # noqa: F401  (import everything the track defines)

    assert "ulb" not in available_datasets()
    assert not any("ulb" in version for version in FEATURESETS)
    # No ULB column can be joined to a v1 feature by name.
    assert set(SOURCE_COLUMNS).isdisjoint(FEATURESETS["v1"])


def test_the_track_does_not_import_the_production_feature_path() -> None:
    """ULB never passes through the extractor, the feature cache or the adapter registry."""
    forbidden_modules = ("app.fraud", "ml.features", "ml.datasets.registry")
    forbidden_names = {"CanonicalDataset", "DatasetAdapter", "FeatureExtractor", "FEATURESETS"}
    track = ML_ROOT / "tracks" / "ulb"
    modules = sorted(track.glob("*.py"))
    assert modules

    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
                names: set[str] = set()
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
                names = {alias.name for alias in node.names}
            else:
                continue
            for name in imported:
                assert not name.startswith(forbidden_modules), f"{module.name} imports {name}"
            assert not names & forbidden_names, f"{module.name} imports {names & forbidden_names}"
