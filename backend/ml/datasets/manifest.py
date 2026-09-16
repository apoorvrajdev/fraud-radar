"""Expected source files, and whether what is on disk is what we expect.

Raw data is never committed, so `ml/data/manifest.json` is the only thing in
version control that says which bytes a benchmark number was computed from.
It carries, per file, the size the publisher advertises and — once anyone has
actually retrieved the file — its SHA-256.

Digests start unpinned (`null`). We cannot honestly commit a hash for a file
nobody in this repository has downloaded yet, and inventing one would make
verification theatre. The first retrieval pins it with `--pin-hashes`; from
then on a mismatch refuses to proceed rather than warning, because a silently
different file means every metric traced to it is wrong.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ml.datasets.base import DataOrigin, DatasetContractError
from ml.paths import DATA_DIR

DEFAULT_MANIFEST_PATH = DATA_DIR / "manifest.json"

_HASH_CHUNK_BYTES = 1024 * 1024


class ManifestError(DatasetContractError):
    """The manifest is malformed, or disk does not match it."""


class FileStatus(StrEnum):
    """Outcome of checking one expected file against disk."""

    OK = "ok"
    MISSING = "missing"
    SIZE_MISMATCH = "size_mismatch"
    HASH_MISMATCH = "hash_mismatch"
    UNPINNED = "unpinned"


@dataclass(frozen=True)
class FileExpectation:
    """What a source file should look like when it lands."""

    name: str
    expected_bytes: int | None = None
    sha256: str | None = None

    @property
    def is_pinned(self) -> bool:
        return self.sha256 is not None


@dataclass(frozen=True)
class FileCheck:
    """What a source file actually looks like."""

    name: str
    status: FileStatus
    actual_bytes: int | None = None
    actual_sha256: str | None = None
    detail: str = ""

    @property
    def is_blocking(self) -> bool:
        """UNPINNED is informational; everything else stops a load."""
        return self.status not in (FileStatus.OK, FileStatus.UNPINNED)


@dataclass(frozen=True)
class DatasetManifestEntry:
    """One dataset's acquisition and provenance facts."""

    name: str
    display_name: str
    origin: DataOrigin
    source: str
    reference: str
    source_url: str
    license: str
    citation: str
    label_field: str
    label_definition: str
    files: tuple[FileExpectation, ...]
    notes: str = ""

    def file(self, name: str) -> FileExpectation:
        for expectation in self.files:
            if expectation.name == name:
                return expectation
        raise ManifestError(f"{self.name}: no manifest entry for file {name!r}.")

    @property
    def filenames(self) -> tuple[str, ...]:
        return tuple(expectation.name for expectation in self.files)


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, DatasetManifestEntry]:
    """Parse the committed manifest into typed entries."""
    if not path.exists():
        raise ManifestError(f"Manifest not found at {path}.")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ManifestError(f"Manifest at {path} has no 'datasets' object.")

    return {name: _entry_from_dict(name, body) for name, body in datasets.items()}


def load_manifest_entry(
    dataset: str, *, path: Path = DEFAULT_MANIFEST_PATH
) -> DatasetManifestEntry:
    """Parse the manifest and return one dataset's entry."""
    manifest = load_manifest(path)
    try:
        return manifest[dataset]
    except KeyError:
        known = ", ".join(sorted(manifest)) or "none"
        raise ManifestError(f"Unknown dataset {dataset!r} in manifest. Known: {known}.") from None


def sha256_file(path: Path, *, chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    """Stream a file through SHA-256 — these inputs are hundreds of megabytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(expectation: FileExpectation, root: Path) -> FileCheck:
    """Compare one expected file against what is on disk."""
    path = root / expectation.name
    if not path.exists():
        return FileCheck(
            name=expectation.name,
            status=FileStatus.MISSING,
            detail=f"Expected at {path}.",
        )

    actual_bytes = path.stat().st_size
    if expectation.expected_bytes is not None and actual_bytes != expectation.expected_bytes:
        return FileCheck(
            name=expectation.name,
            status=FileStatus.SIZE_MISMATCH,
            actual_bytes=actual_bytes,
            detail=(
                f"Expected {expectation.expected_bytes} bytes, found {actual_bytes}. "
                "The publisher may have republished the file."
            ),
        )

    if not expectation.is_pinned:
        return FileCheck(
            name=expectation.name,
            status=FileStatus.UNPINNED,
            actual_bytes=actual_bytes,
            actual_sha256=sha256_file(path),
            detail="No digest pinned yet; run the download command with --pin-hashes.",
        )

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expectation.sha256:
        return FileCheck(
            name=expectation.name,
            status=FileStatus.HASH_MISMATCH,
            actual_bytes=actual_bytes,
            actual_sha256=actual_sha256,
            detail=f"Expected sha256 {expectation.sha256}, computed {actual_sha256}.",
        )

    return FileCheck(
        name=expectation.name,
        status=FileStatus.OK,
        actual_bytes=actual_bytes,
        actual_sha256=actual_sha256,
    )


def verify_dataset(entry: DatasetManifestEntry, root: Path) -> tuple[FileCheck, ...]:
    """Check every file of a dataset. Returns results; raises nothing."""
    return tuple(check_file(expectation, root) for expectation in entry.files)


def require_verified(entry: DatasetManifestEntry, root: Path) -> dict[str, str]:
    """Verify and refuse to continue on any blocking problem.

    Returns filename -> sha256 for the verified files, which is exactly the
    shape `DatasetProvenance.files` wants.
    """
    checks = verify_dataset(entry, root)
    blocking = [check for check in checks if check.is_blocking]
    if blocking:
        detail = "; ".join(f"{check.name}: {check.status.value} ({check.detail})" for check in blocking)
        raise ManifestError(f"{entry.name}: source files failed verification — {detail}")
    return {check.name: check.actual_sha256 or "" for check in checks}


def pin_hashes(
    entry: DatasetManifestEntry,
    root: Path,
    *,
    path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, str]:
    """Write the digests of the files on disk into the manifest.

    Refuses to overwrite a digest that is already pinned: re-pinning would
    turn a genuine mismatch into a silent acceptance of different bytes.
    """
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    files_payload = payload["datasets"][entry.name]["files"]
    pinned: dict[str, str] = {}
    for expectation in entry.files:
        file_path = root / expectation.name
        if not file_path.exists():
            raise ManifestError(f"Cannot pin {expectation.name}: not found at {file_path}.")
        digest = sha256_file(file_path)
        if expectation.is_pinned and expectation.sha256 != digest:
            raise ManifestError(
                f"{expectation.name} is already pinned to {expectation.sha256} but the file "
                f"on disk hashes to {digest}. Investigate before re-pinning."
            )
        files_payload[expectation.name]["sha256"] = digest
        files_payload[expectation.name]["bytes"] = file_path.stat().st_size
        pinned[expectation.name] = digest

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return pinned


def _entry_from_dict(name: str, body: Mapping[str, Any]) -> DatasetManifestEntry:
    try:
        files = tuple(
            FileExpectation(
                name=filename,
                expected_bytes=_opt_int(spec.get("bytes")),
                sha256=_opt_lower(spec.get("sha256")),
            )
            for filename, spec in body["files"].items()
        )
        return DatasetManifestEntry(
            name=name,
            display_name=str(body["display_name"]),
            origin=DataOrigin(body["origin"]),
            source=str(body["source"]),
            reference=str(body["reference"]),
            source_url=str(body["source_url"]),
            license=str(body["license"]),
            citation=str(body["citation"]),
            label_field=str(body["label_field"]),
            label_definition=str(body["label_definition"]),
            files=files,
            notes=str(body.get("notes", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"Malformed manifest entry for {name!r}: {exc}") from exc


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _opt_lower(value: Any) -> str | None:
    return None if value is None else str(value).lower()
