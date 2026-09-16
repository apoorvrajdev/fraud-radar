"""Phase 5B — manifest verification and the acquisition entry point.

Raw data is never committed, so the manifest is the only thing in version
control tying a benchmark number to specific bytes. These tests hold it to
that: a changed file must stop a load, not warn.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.datasets import download
from ml.datasets.base import DataOrigin
from ml.datasets.manifest import (
    DEFAULT_MANIFEST_PATH,
    FileStatus,
    ManifestError,
    check_file,
    load_manifest,
    load_manifest_entry,
    pin_hashes,
    require_verified,
    sha256_file,
    verify_dataset,
)

CONTENT = b"trans_num,amt,is_fraud\nabc,10.00,0\n"
DIGEST = hashlib.sha256(CONTENT).hexdigest()


def _manifest_payload(*, sha256: str | None = None, size: int | None = None) -> dict[str, object]:
    return {
        "manifest_version": "1",
        "datasets": {
            "fixture": {
                "display_name": "Fixture Dataset",
                "origin": "synthetic",
                "source": "kaggle",
                "reference": "someone/fixture",
                "source_url": "https://example.test/fixture",
                "license": "CC0 (fixture)",
                "citation": "Fixture citation",
                "label_field": "is_fraud",
                "label_definition": "1 = fraud",
                "notes": "fixture",
                "files": {
                    "data.csv": {
                        "bytes": len(CONTENT) if size is None else size,
                        "sha256": sha256,
                    }
                },
            }
        },
    }


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    return path


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw" / "fixture"
    root.mkdir(parents=True)
    (root / "data.csv").write_bytes(CONTENT)
    return root


# ---------------------------------------------------------------------------
# The committed manifest
# ---------------------------------------------------------------------------


def test_committed_manifest_parses() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    assert "sparkov" in manifest


def test_committed_sparkov_entry_states_its_provenance() -> None:
    entry = load_manifest_entry("sparkov")

    assert entry.origin is DataOrigin.SYNTHETIC
    assert entry.source == "kaggle"
    assert entry.reference == "kartik2112/fraud-detection"
    assert entry.license.startswith("CC0")
    assert entry.label_field == "is_fraud"
    assert set(entry.filenames) == {"fraudTrain.csv", "fraudTest.csv"}


def test_committed_sparkov_entry_is_labelled_synthetic_in_prose() -> None:
    """The notes must not let a reader mistake this for real card data."""
    entry = load_manifest_entry("sparkov")
    assert "synthetic" in entry.notes.lower()
    assert "not real" in entry.notes.lower()


def test_unknown_dataset_is_refused() -> None:
    with pytest.raises(ManifestError, match="Unknown dataset 'ghost'"):
        load_manifest_entry("ghost")


def test_missing_manifest_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="Manifest not found"):
        load_manifest(tmp_path / "absent.json")


def test_malformed_entry_is_refused(tmp_path: Path) -> None:
    payload = _manifest_payload()
    del payload["datasets"]["fixture"]["license"]  # type: ignore[index]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="Malformed manifest entry"):
        load_manifest(path)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_sha256_file_matches_hashlib(data_root: Path) -> None:
    assert sha256_file(data_root / "data.csv") == DIGEST


def test_sha256_streams_a_file_larger_than_one_chunk(tmp_path: Path) -> None:
    payload = b"x" * 3_000
    path = tmp_path / "big.csv"
    path.write_bytes(payload)
    assert sha256_file(path, chunk_bytes=1024) == hashlib.sha256(payload).hexdigest()


def test_unpinned_file_reports_its_digest_without_blocking(
    manifest_path: Path, data_root: Path
) -> None:
    entry = load_manifest_entry("fixture", path=manifest_path)
    check = check_file(entry.file("data.csv"), data_root)

    assert check.status is FileStatus.UNPINNED
    assert check.actual_sha256 == DIGEST
    assert not check.is_blocking


def test_pinned_and_matching_file_verifies(tmp_path: Path, data_root: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest_payload(sha256=DIGEST)), encoding="utf-8")
    entry = load_manifest_entry("fixture", path=path)

    assert check_file(entry.file("data.csv"), data_root).status is FileStatus.OK
    assert require_verified(entry, data_root) == {"data.csv": DIGEST}


def test_changed_bytes_are_refused(tmp_path: Path, data_root: Path) -> None:
    """The whole point: different bytes must stop the run."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest_payload(sha256="f" * 64)), encoding="utf-8")
    entry = load_manifest_entry("fixture", path=path)

    assert check_file(entry.file("data.csv"), data_root).status is FileStatus.HASH_MISMATCH
    with pytest.raises(ManifestError, match="hash_mismatch"):
        require_verified(entry, data_root)


def test_size_mismatch_against_a_pinned_digest_is_refused(
    tmp_path: Path, data_root: Path
) -> None:
    """A pinned file that changed size is different data, full stop."""
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(_manifest_payload(sha256=DIGEST, size=999_999)), encoding="utf-8"
    )
    entry = load_manifest_entry("fixture", path=path)

    check = check_file(entry.file("data.csv"), data_root)
    assert check.status is FileStatus.SIZE_MISMATCH
    assert check.is_blocking
    assert check.actual_sha256 is None


def test_stale_advertised_size_does_not_block_a_first_acquisition(
    tmp_path: Path, data_root: Path
) -> None:
    """The size came from the publisher's metadata, not from bytes we held.

    Refusing here would dead-end the only download that can establish the
    truth, since pinning is unreachable while verification fails.
    """
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest_payload(size=999_999)), encoding="utf-8")
    entry = load_manifest_entry("fixture", path=path)

    check = check_file(entry.file("data.csv"), data_root)
    assert check.status is FileStatus.UNPINNED
    assert not check.is_blocking
    assert check.actual_sha256 == DIGEST
    assert "reported rather than refused" in check.detail


def test_pinning_records_the_size_that_actually_arrived(
    tmp_path: Path, data_root: Path
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest_payload(size=999_999)), encoding="utf-8")
    entry = load_manifest_entry("fixture", path=path)

    pin_hashes(entry, data_root, path=path)

    repinned = load_manifest_entry("fixture", path=path)
    assert repinned.file("data.csv").expected_bytes == len(CONTENT)
    assert repinned.file("data.csv").sha256 == DIGEST
    assert check_file(repinned.file("data.csv"), data_root).status is FileStatus.OK


def test_first_acquisition_flow_end_to_end(tmp_path: Path, data_root: Path) -> None:
    """Stale size -> verify passes -> pin -> later mismatch is refused."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest_payload(size=42)), encoding="utf-8")

    assert (
        download.main(
            [
                "--dataset",
                "fixture",
                "--manifest",
                str(path),
                "--raw-root",
                str(data_root.parent),
                "--pin-hashes",
            ]
        )
        == download.EXIT_OK
    )

    pinned = load_manifest_entry("fixture", path=path)
    assert pinned.file("data.csv").sha256 == DIGEST

    (data_root / "data.csv").write_bytes(CONTENT + b"tampered\n")
    assert check_file(
        load_manifest_entry("fixture", path=path).file("data.csv"), data_root
    ).is_blocking


def test_missing_file_is_reported(manifest_path: Path, tmp_path: Path) -> None:
    entry = load_manifest_entry("fixture", path=manifest_path)
    check = check_file(entry.file("data.csv"), tmp_path / "empty")

    assert check.status is FileStatus.MISSING
    assert check.is_blocking


def test_verify_dataset_covers_every_declared_file(manifest_path: Path, data_root: Path) -> None:
    entry = load_manifest_entry("fixture", path=manifest_path)
    assert [check.name for check in verify_dataset(entry, data_root)] == ["data.csv"]


def test_unknown_filename_lookup_is_refused(manifest_path: Path) -> None:
    entry = load_manifest_entry("fixture", path=manifest_path)
    with pytest.raises(ManifestError, match="no manifest entry for file"):
        entry.file("ghost.csv")


# ---------------------------------------------------------------------------
# Pinning
# ---------------------------------------------------------------------------


def test_pinning_writes_the_digest_back_to_the_manifest(
    manifest_path: Path, data_root: Path
) -> None:
    entry = load_manifest_entry("fixture", path=manifest_path)
    assert pin_hashes(entry, data_root, path=manifest_path) == {"data.csv": DIGEST}

    repinned = load_manifest_entry("fixture", path=manifest_path)
    assert repinned.file("data.csv").sha256 == DIGEST
    assert check_file(repinned.file("data.csv"), data_root).status is FileStatus.OK


def test_pinning_refuses_to_overwrite_a_conflicting_digest(
    tmp_path: Path, data_root: Path
) -> None:
    """Re-pinning a mismatch would turn a real problem into a silent accept."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest_payload(sha256="f" * 64)), encoding="utf-8")
    entry = load_manifest_entry("fixture", path=path)

    with pytest.raises(ManifestError, match="already pinned"):
        pin_hashes(entry, data_root, path=path)


def test_pinning_a_missing_file_is_refused(manifest_path: Path, tmp_path: Path) -> None:
    entry = load_manifest_entry("fixture", path=manifest_path)
    with pytest.raises(ManifestError, match="Cannot pin"):
        pin_hashes(entry, tmp_path / "empty", path=manifest_path)


# ---------------------------------------------------------------------------
# Acquisition CLI
# ---------------------------------------------------------------------------


def test_dataset_root_is_namespaced_per_dataset(tmp_path: Path) -> None:
    assert download.dataset_root("sparkov", raw_root=tmp_path) == tmp_path / "sparkov"


def test_verify_only_succeeds_without_touching_the_network(
    manifest_path: Path, data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("verify-only must not download")

    monkeypatch.setattr(download, "download_with_kaggle_cli", fail)
    exit_code = download.main(
        [
            "--dataset",
            "fixture",
            "--manifest",
            str(manifest_path),
            "--raw-root",
            str(data_root.parent),
            "--verify-only",
        ]
    )
    assert exit_code == download.EXIT_OK


def test_verify_only_fails_when_bytes_differ(
    tmp_path: Path, data_root: Path
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest_payload(sha256="f" * 64)), encoding="utf-8")

    exit_code = download.main(
        [
            "--dataset",
            "fixture",
            "--manifest",
            str(path),
            "--raw-root",
            str(data_root.parent),
            "--verify-only",
        ]
    )
    assert exit_code == download.EXIT_VERIFICATION_FAILED


def test_missing_files_without_kaggle_cli_print_manual_instructions(
    manifest_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download, "kaggle_cli_available", lambda: False)
    exit_code = download.main(
        [
            "--dataset",
            "fixture",
            "--manifest",
            str(manifest_path),
            "--raw-root",
            str(tmp_path / "empty"),
        ]
    )
    assert exit_code == download.EXIT_MANUAL_DOWNLOAD_REQUIRED


def test_manual_instructions_name_the_files_licence_and_destination(
    manifest_path: Path, tmp_path: Path
) -> None:
    entry = load_manifest_entry("fixture", path=manifest_path)
    text = download.manual_instructions(entry, tmp_path / "raw" / "fixture")

    assert "data.csv" in text
    assert entry.source_url in text
    assert "CC0 (fixture)" in text
    assert "never be committed" in text


def test_present_files_are_not_redownloaded(
    manifest_path: Path, data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("files are present; download must be skipped")

    monkeypatch.setattr(download, "download_with_kaggle_cli", fail)
    monkeypatch.setattr(download, "kaggle_cli_available", lambda: True)

    exit_code = download.main(
        [
            "--dataset",
            "fixture",
            "--manifest",
            str(manifest_path),
            "--raw-root",
            str(data_root.parent),
        ]
    )
    assert exit_code == download.EXIT_OK


def test_non_kaggle_source_refuses_automatic_download(
    tmp_path: Path, manifest_path: Path
) -> None:
    payload = _manifest_payload()
    payload["datasets"]["fixture"]["source"] = "http"  # type: ignore[index]
    path = tmp_path / "other.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    entry = load_manifest_entry("fixture", path=path)

    with pytest.raises(ManifestError, match="supports kaggle only"):
        download.download_with_kaggle_cli(entry, tmp_path / "dest")
