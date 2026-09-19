"""ULB — the real-world benchmark, read and validated from its pinned source file.

The source is Kaggle `mlg-ulb/creditcardfraud`, published by the Machine
Learning Group of the Université Libre de Bruxelles: card transactions made by
European cardholders over two days in September 2013. It is real, anonymised
data. Apart from `Time`, `Amount` and the label `Class`, every column is a PCA
component of inputs the publisher withholds.

It is deliberately **not** a `DatasetAdapter`, and it never becomes a
`CanonicalDataset` (Phase 5E decision 2). It identifies no customer and no
merchant, and satisfying that contract would mean inventing both. What it does
satisfy is the provenance contract: the manifest's pinned digest, checked
before a byte is parsed, and a `DatasetProvenance` record. How each column
maps onto the canonical schema, and what the schema needs that ULB lacks, is
`field_inventory()`.

How the file is read (decisions 6 and 7):

- the header must be exactly `SOURCE_COLUMNS`, in that order, and every row
  must have one field per column, or the file is refused;
- every value is read as a number in plain or scientific notation, so the one
  `Time` written as `1e+05` is 100000 seconds, and `Class` is read through the
  quotes it arrives in;
- a row with a missing or non-numeric value, a `Class` other than 0 or 1, or a
  negative `Amount` is excluded and counted by reason, never dropped silently;
  zero amounts and exact duplicate rows are kept;
- rows are ordered by `Time`, then by position in the file, and a row's
  transaction id is its 1-based position in the file;
- a row's timestamp is `ELAPSED_TIME_ORIGIN` plus `Time` seconds: an
  elapsed-time position that satisfies the timezone-aware contracts, never a
  calendar date or a time of day.

Nothing here chooses features. Which columns a model sees is the `ulb_pca_v1`
featureset, defined apart from this module and kept out of the production
featureset registry (decision 4). `Amount` is read as a float64 model input
like every other column: it never becomes a money value this system handles,
its currency is unstated, and it is never compared with the amounts of any
other dataset (decision 5).
"""
from __future__ import annotations

import csv
import logging
import math
import re
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from ml.datasets.base import DataOrigin, DatasetContractError, DatasetProvenance
from ml.datasets.manifest import (
    DEFAULT_MANIFEST_PATH,
    DatasetManifestEntry,
    ManifestError,
    load_manifest_entry,
    require_verified,
)
from ml.datasets.quality import ExclusionRecord, FieldNote, FieldStatus
from ml.paths import RAW_DATA_DIR

log = logging.getLogger("ml.tracks.ulb.load")

DATASET_NAME = "ulb"
SOURCE_FILENAME = "creditcard.csv"
DEFAULT_ROOT = RAW_DATA_DIR / DATASET_NAME

# The source's metadata carries no dataset version or last-updated date; the
# Kaggle files API dates the file 2019-09-20. The pinned digest, not this
# string, is what identifies the bytes.
DATASET_VERSION = "kaggle-file-2019-09-20"

# The layout this module reads. ULB is not in the canonical schema, so its
# provenance does not claim the canonical schema version.
SOURCE_SCHEMA_VERSION = "ulb-source-1"

TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
LABEL_COLUMN = "Class"

# The 28 PCA components, under their source names and in source order.
COMPONENT_COLUMNS: tuple[str, ...] = (
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
)  # fmt: skip

# The header the published file carries, exactly and in this order.
SOURCE_COLUMNS: tuple[str, ...] = (
    TIME_COLUMN,
    *COMPONENT_COLUMNS,
    AMOUNT_COLUMN,
    LABEL_COLUMN,
)

_TIME_INDEX = SOURCE_COLUMNS.index(TIME_COLUMN)
_COMPONENTS_START = SOURCE_COLUMNS.index(COMPONENT_COLUMNS[0])
_AMOUNT_INDEX = SOURCE_COLUMNS.index(AMOUNT_COLUMN)
_LABEL_INDEX = SOURCE_COLUMNS.index(LABEL_COLUMN)

# A row's timestamp is this instant plus its `Time` seconds. The origin only
# anchors elapsed time; it is not the date the transactions were made.
ELAPSED_TIME_ORIGIN = datetime(1970, 1, 1, tzinfo=UTC)

# Exclusion reasons, in the fixed order the checks run: a row is counted
# under the first check it fails.
MISSING_VALUE = "missing_value"
NON_NUMERIC_VALUE = "non_numeric_value"
LABEL_NOT_ZERO_OR_ONE = "label_not_zero_or_one"
NEGATIVE_AMOUNT = "negative_amount"
EXCLUSION_REASONS: tuple[str, ...] = (
    MISSING_VALUE,
    NON_NUMERIC_VALUE,
    LABEL_NOT_ZERO_OR_ONE,
    NEGATIVE_AMOUNT,
)

# A number in plain or scientific notation. Stricter than float(), which also
# accepts padding, underscores, "nan" and "inf"; none of those is a value.
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

# The ODbL section 4.3 notice every published ULB-derived record carries, as
# plain text with its links written out. docs/DATA_LICENSES.md holds the same
# notice with the links as markdown.
LICENCE_NOTICE = (
    "Contains information from the Credit Card Fraud Detection "
    "(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) database of the Machine "
    "Learning Group, ULB, which is available under the Open Database License (ODbL) v1.0 "
    "(https://opendatacommons.org/licenses/odbl/1-0/); its contents are under the Database "
    "Contents License (DbCL) v1.0 (https://opendatacommons.org/licenses/dbcl/1-0/)."
)

# The ODbL section 4.6(b) offer that goes with every record computed through
# this module, worded as docs/DATA_LICENSES.md words it.
METHOD_OFFER = (
    "ODbL section 4.6(b): the method of deriving the rows this record is computed from is "
    "offered, free of charge and in machine-readable form, as this repository's code: the "
    "ULB track in backend/ml/tracks/ulb/, at the commit the run records as code_version in "
    "its run.json. It adds no contents from outside the source file."
)

# Where the licence terms these notices follow are recorded.
LICENCE_TERMS = "docs/DATA_LICENSES.md"


class UlbSchemaError(DatasetContractError):
    """The source file does not have the layout of the published ULB file."""


@dataclass(frozen=True)
class UlbSource:
    """The validated ULB rows, in load order, with the accounting behind them.

    Load order is `Time`, then position in the file. Every array holds one
    entry per kept row, in that order: `components` has one column per
    `COMPONENT_COLUMNS` entry, `labels` is 0 or 1, and `positions` are 1-based
    positions in the source file. `timestamps` are the elapsed-time encoding
    of `time_seconds`, and `transaction_ids` are the positions as strings.
    """

    time_seconds: np.ndarray
    components: np.ndarray
    amounts: np.ndarray
    labels: np.ndarray
    positions: np.ndarray
    timestamps: np.ndarray
    transaction_ids: list[str]
    provenance: DatasetProvenance
    raw_row_count: int
    exclusions: tuple[ExclusionRecord, ...]
    file_in_time_order: bool

    @property
    def n_rows(self) -> int:
        return len(self.labels)

    @property
    def fraud_count(self) -> int:
        return int(self.labels.sum())

    @property
    def excluded_row_count(self) -> int:
        return sum(record.count for record in self.exclusions)


@dataclass(frozen=True)
class _ReadRows:
    """What reading the file yields, before rows are put in load order."""

    values: np.ndarray
    positions: np.ndarray
    raw_row_count: int
    excluded: Mapping[str, int]


def load_ulb(
    root: Path = DEFAULT_ROOT, *, manifest_path: Path = DEFAULT_MANIFEST_PATH
) -> UlbSource:
    """Read the ULB file under `root`, after verifying it against its pinned digest.

    Refuses an entry whose digest is not pinned: a ULB result must trace to
    the bytes acquisition recorded, and an unpinned entry cannot say which.
    """
    entry = load_manifest_entry(DATASET_NAME, path=manifest_path)
    if entry.filenames != (SOURCE_FILENAME,):
        raise UlbSchemaError(
            f"The {DATASET_NAME!r} manifest entry lists {', '.join(entry.filenames) or 'no files'}; "
            f"the published dataset is the one file {SOURCE_FILENAME}."
        )
    if not entry.file(SOURCE_FILENAME).is_pinned:
        raise ManifestError(
            f"{DATASET_NAME}: {SOURCE_FILENAME} has no pinned digest. ULB is read only against "
            "the digest acquisition pinned."
        )
    file_digests = require_verified(entry, root)

    read = _read_rows(root / SOURCE_FILENAME)
    values, positions = read.values, read.positions
    time_in_file_order = values[:, _TIME_INDEX]
    file_in_time_order = bool(np.all(np.diff(time_in_file_order) >= 0))

    # Time first, file position to break ties: the order decision 6 fixes.
    order = np.lexsort((positions, time_in_file_order))
    values, positions = values[order], positions[order]

    time_seconds = values[:, _TIME_INDEX].copy()
    labels = values[:, _LABEL_INDEX].astype(np.int64)
    timestamps = elapsed_timestamps(time_seconds)
    exclusions = tuple(
        ExclusionRecord(reason=reason, count=read.excluded[reason]) for reason in EXCLUSION_REASONS
    )
    for record in exclusions:
        if record.count:
            log.warning("Excluded %d row(s): %s", record.count, record.reason)
    log.info("Kept %d of %d rows from %s", len(labels), read.raw_row_count, SOURCE_FILENAME)

    return UlbSource(
        time_seconds=time_seconds,
        components=values[:, _COMPONENTS_START : _COMPONENTS_START + len(COMPONENT_COLUMNS)].copy(),
        amounts=values[:, _AMOUNT_INDEX].copy(),
        labels=labels,
        positions=positions,
        timestamps=timestamps,
        transaction_ids=[str(position) for position in positions],
        provenance=_build_provenance(
            entry=entry,
            file_digests=file_digests,
            timestamps=timestamps,
            labels=labels,
            exclusions=exclusions,
        ),
        raw_row_count=read.raw_row_count,
        exclusions=exclusions,
        file_in_time_order=file_in_time_order,
    )


def elapsed_timestamps(time_seconds: np.ndarray) -> np.ndarray:
    """Each `Time` as `ELAPSED_TIME_ORIGIN` plus that many seconds, timezone-aware.

    These exist to satisfy the timezone-aware contracts of the split and
    provenance records, and to give the chronological split its order. No
    calendar value is ever read from them.
    """
    return np.asarray(
        [ELAPSED_TIME_ORIGIN + timedelta(seconds=float(seconds)) for seconds in time_seconds],
        dtype=object,
    )


def field_inventory() -> tuple[FieldNote, ...]:
    """Every ULB column and what it becomes, and every canonical field ULB cannot supply.

    The Phase 5E schema accounting, in the form quality reports carry. It is
    what keeps "a model scored X on ULB" from being read as a claim about the
    customer, merchant and history features ULB has no data for.
    """
    return (
        FieldNote(
            TIME_COLUMN,
            FieldStatus.MAPPED,
            "Transaction.created_at position: 1970-01-01T00:00:00+00:00 plus Time seconds; "
            "orders and splits rows; elapsed time only, never a calendar date or a time of day; "
            "not a feature",
        ),
        FieldNote(
            AMOUNT_COLUMN,
            FieldStatus.MAPPED,
            "Transaction.amount, as published and untransformed; currency not stated, none assumed",
        ),
        FieldNote(
            LABEL_COLUMN, FieldStatus.MAPPED, "the label, 0 or 1, held apart from the values"
        ),
        FieldNote(
            f"{COMPONENT_COLUMNS[0]}-{COMPONENT_COLUMNS[-1]}",
            FieldStatus.MAPPED,
            "no canonical field: PCA components of undisclosed inputs, carried as published "
            "into ULB-only features",
        ),
        FieldNote(
            "row position",
            FieldStatus.DERIVED,
            "Transaction.id: the 1-based position in the source file, stable because the "
            "digest is pinned; not a source identifier",
        ),
        FieldNote(
            "Transaction.customer_id / Customer.*",
            FieldStatus.UNAVAILABLE,
            "no card or account key: no history, velocity or customer features",
        ),
        FieldNote(
            "Transaction.merchant_id / Merchant.*",
            FieldStatus.UNAVAILABLE,
            "no merchant name, category, MCC, risk rating or country",
        ),
        FieldNote(
            "Transaction.country / is_card_present / payment_method / status / currency",
            FieldStatus.UNAVAILABLE,
            "no geography or channel",
        ),
    )


def _read_rows(path: Path) -> _ReadRows:
    """Parse every data row, keeping valid ones and counting the rest by reason."""
    if not path.exists():
        raise UlbSchemaError(f"Expected source file not found: {path}")

    width = len(SOURCE_COLUMNS)
    values = array("d")
    positions = array("q")
    excluded = dict.fromkeys(EXCLUSION_REASONS, 0)
    raw_row_count = 0
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            _check_header(next(reader, None), path)
            for position, fields in enumerate(reader, start=1):
                raw_row_count = position
                if len(fields) != width:
                    raise UlbSchemaError(
                        f"{path.name}: data row {position} has {len(fields)} fields, not {width}. "
                        "The file does not have the published layout."
                    )
                reason, numbers = _parse_row(fields)
                if reason is not None:
                    excluded[reason] += 1
                    continue
                values.extend(numbers)
                positions.append(position)
    except UnicodeDecodeError as exc:
        raise UlbSchemaError(f"{path.name} is not UTF-8 text: {exc}") from exc
    except csv.Error as exc:
        raise UlbSchemaError(f"{path.name} is not a readable CSV file: {exc}") from exc

    return _ReadRows(
        values=np.frombuffer(values, dtype=np.float64).reshape(-1, width).copy(),
        positions=np.frombuffer(positions, dtype=np.int64).copy(),
        raw_row_count=raw_row_count,
        excluded=excluded,
    )


def _check_header(header: Sequence[str] | None, path: Path) -> None:
    if header is None:
        raise UlbSchemaError(f"{path.name} is empty: no header row.")
    if tuple(header) != SOURCE_COLUMNS:
        raise UlbSchemaError(
            f"{path.name} header is {', '.join(header)}; expected exactly "
            f"{', '.join(SOURCE_COLUMNS)}, in that order. The publisher may have changed the "
            "file; re-check it against acquisition before loading."
        )


def _parse_row(fields: Sequence[str]) -> tuple[str | None, list[float]]:
    """The first exclusion reason a row meets, or None and its values."""
    if not all(fields):
        return MISSING_VALUE, []
    if not all(_NUMBER.fullmatch(field) for field in fields):
        return NON_NUMERIC_VALUE, []
    numbers = [float(field) for field in fields]
    # An exponent too large for a float64 parses to infinity: not a value.
    if not all(math.isfinite(number) for number in numbers):
        return NON_NUMERIC_VALUE, []
    if numbers[_LABEL_INDEX] not in (0.0, 1.0):
        return LABEL_NOT_ZERO_OR_ONE, []
    if numbers[_AMOUNT_INDEX] < 0:
        return NEGATIVE_AMOUNT, []
    return None, numbers


def _build_provenance(
    *,
    entry: DatasetManifestEntry,
    file_digests: Mapping[str, str],
    timestamps: np.ndarray,
    labels: np.ndarray,
    exclusions: Sequence[ExclusionRecord],
) -> DatasetProvenance:
    preprocessing = [
        "header required to be exactly Time, V1-V28, Amount, Class, in that order",
        "every value read as a number in plain or scientific notation; Class read through its "
        "quotes",
        "values used as published: no transform, scaling or rounding",
        "rows ordered by Time, then by position in the source file",
        "transaction id: the row's 1-based position in the source file, not a source identifier",
        "timestamps: 1970-01-01T00:00:00+00:00 plus Time seconds, elapsed-time positions and "
        "not calendar dates",
        "exact duplicate rows and zero amounts kept",
    ]
    preprocessing += [
        f"excluded {record.count} row(s): {record.reason}" for record in exclusions if record.count
    ]
    notes = (
        f"{entry.notes} Timestamps are elapsed time from the first transaction, encoded from "
        "1970-01-01T00:00:00+00:00; they are not calendar dates."
    ).strip()

    return DatasetProvenance(
        name=entry.name,
        version=DATASET_VERSION,
        origin=DataOrigin.REAL,
        source_url=entry.source_url,
        citation=entry.citation,
        license=entry.license,
        label_field=entry.label_field,
        label_definition=entry.label_definition,
        schema_version=SOURCE_SCHEMA_VERSION,
        retrieved_at=datetime.now(UTC).replace(microsecond=0),
        files=dict(file_digests),
        row_count=len(labels),
        fraud_count=int(labels.sum()),
        period_start=timestamps[0] if len(timestamps) else None,
        period_end=timestamps[-1] if len(timestamps) else None,
        preprocessing=tuple(preprocessing),
        notes=notes,
    )
