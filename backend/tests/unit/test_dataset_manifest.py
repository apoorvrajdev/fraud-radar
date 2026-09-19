"""Phase 5B — manifest verification and the acquisition entry point.

Raw data is never committed, so the manifest is the only thing in version
control tying a benchmark number to specific bytes. These tests hold it to
that: a changed file must stop a load, not warn.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
from ml.datasets.registry import available_datasets
from ml.paths import RAW_DATA_DIR
from ml.train import parse_args as parse_train_args

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


def test_committed_ulb_entry_states_its_provenance() -> None:
    """Phase 5E: what the source's own metadata says, and nothing it does not."""
    entry = load_manifest_entry("ulb")

    assert entry.origin is DataOrigin.REAL
    assert entry.source == "kaggle"
    assert entry.reference == "mlg-ulb/creditcardfraud"
    assert entry.source_url == "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    assert entry.license.startswith("DbCL-1.0")
    assert entry.label_field == "Class"
    assert entry.filenames == ("creditcard.csv",)


def test_committed_ulb_licence_names_the_database_licence() -> None:
    """DbCL 1.0 section 2.2 requires compliance with the ODbL, which covers the database.

    The licence field reaches every ULB run's provenance, so it must name
    both licences with their texts' addresses, not only the one Kaggle's API returns.
    """
    licence = load_manifest_entry("ulb").license

    assert "ODbL-1.0" in licence
    assert "https://opendatacommons.org/licenses/odbl/1-0/" in licence
    assert "DbCL-1.0" in licence
    assert "https://opendatacommons.org/licenses/dbcl/1-0/" in licence


# Retrieved and verified on 2026-09-18: the size the Kaggle files API advertised,
# and the digest of the bytes that arrived.
ULB_BYTES = 150_828_752
ULB_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"


def test_committed_ulb_entry_is_pinned_to_the_verified_bytes() -> None:
    expected = load_manifest_entry("ulb").file("creditcard.csv")

    assert expected.is_pinned
    assert expected.sha256 == ULB_SHA256
    assert expected.expected_bytes == ULB_BYTES


def test_committed_ulb_pin_refuses_other_bytes_and_repinning(tmp_path: Path) -> None:
    """Once pinned, a different file stops a load, and cannot be pinned over the digest."""
    manifest_copy = tmp_path / "manifest.json"
    shutil.copyfile(DEFAULT_MANIFEST_PATH, manifest_copy)
    entry = load_manifest_entry("ulb", path=manifest_copy)
    root = tmp_path / "raw" / "ulb"
    root.mkdir(parents=True)
    (root / "creditcard.csv").write_bytes(b'"Time","V1","Amount","Class"\n0,1.0,2.50,"0"\n')

    check = check_file(entry.file("creditcard.csv"), root)
    assert check.status is FileStatus.SIZE_MISMATCH
    assert check.is_blocking
    with pytest.raises(ManifestError, match="size_mismatch"):
        require_verified(entry, root)

    with pytest.raises(ManifestError, match="already pinned"):
        pin_hashes(entry, root, path=manifest_copy)
    assert load_manifest_entry("ulb", path=manifest_copy).file("creditcard.csv").sha256 == (
        ULB_SHA256
    )


def test_committed_ulb_entry_is_labelled_real_and_anonymised_in_prose() -> None:
    """The notes must not let a reader mistake this for generated data."""
    notes = load_manifest_entry("ulb").notes.lower()
    assert "real" in notes
    assert "anonymised" in notes
    assert "synthetic" not in notes


def test_every_committed_licence_states_when_it_was_retrieved() -> None:
    """A licence is recorded as the source displayed it on a given day, never from memory."""
    for name, entry in load_manifest(DEFAULT_MANIFEST_PATH).items():
        assert re.search(r"retrieved \d{4}-\d{2}-\d{2}", entry.license), name


def test_ulb_is_acquired_like_any_source_but_is_not_a_training_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Phase 5E decision 2: ULB names no customer or merchant, so it has no adapter.

    It is in the manifest, so it is downloaded and verified like any source,
    but the v1 training path cannot be pointed at it.
    """
    assert "ulb" in load_manifest(DEFAULT_MANIFEST_PATH)
    assert "ulb" not in available_datasets()

    with pytest.raises(SystemExit):
        parse_train_args(["--dataset", "ulb", "--run-name", "ulb_pca_v1_seed42"])
    assert "invalid choice: 'ulb'" in capsys.readouterr().err


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


LISTING = "name,size,creationDate\ndata.csv,36,2020-08-05 15:21:00.483000\n"

Responder = Callable[[list[str]], subprocess.CompletedProcess[str]]


class _RecordingRun:
    """Stands in for `subprocess.run`: records each call, answers via `respond`."""

    def __init__(self, respond: Responder) -> None:
        self.respond = respond
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(command), kwargs))
        return self.respond(list(command))

    @property
    def subcommands(self) -> list[str]:
        return [command[2] for command, _ in self.calls]


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> Responder:
    def respond(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    return respond


def _raising(exc: BaseException) -> Responder:
    def respond(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise exc

    return respond


def _fail_if_called(command: list[str]) -> subprocess.CompletedProcess[str]:
    raise AssertionError(f"the Kaggle CLI must not be run here: {command}")


def _listing_then_download(content: bytes | None) -> Responder:
    """A CLI that lists the fixture, then 'downloads' `content` (None: writes nothing)."""

    def respond(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[2] == "files":
            return subprocess.CompletedProcess(command, 0, stdout=LISTING, stderr="")
        if content is not None:
            destination = Path(command[command.index("-p") + 1])
            (destination / "data.csv").write_bytes(content)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return respond


def _main_args(manifest_path: Path, raw_root: Path) -> list[str]:
    return [
        "--dataset",
        "fixture",
        "--manifest",
        str(manifest_path),
        "--raw-root",
        str(raw_root),
    ]


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
    manifest_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="ml.datasets.download")
    monkeypatch.setattr(download, "find_kaggle_cli", lambda: None)
    monkeypatch.setattr(subprocess, "run", _RecordingRun(_fail_if_called))
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
    assert "Kaggle CLI not found on PATH" in caplog.text
    assert "Manual download required" in caplog.text


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
    monkeypatch.setattr(download, "find_kaggle_cli", lambda: "/opt/kaggle")
    monkeypatch.setattr(subprocess, "run", _RecordingRun(_fail_if_called))

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
        download.download_with_kaggle_cli(entry, tmp_path / "dest", executable="/opt/kaggle")


# ---------------------------------------------------------------------------
# Kaggle CLI integration (subprocess faked; nothing is downloaded)
# ---------------------------------------------------------------------------

def test_find_kaggle_cli_returns_the_resolved_executable_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full path is executed, so a Windows `.exe`/`.cmd` shim still runs."""
    monkeypatch.setattr(shutil, "which", lambda name: f"C:/tools/{name}.exe")
    assert download.find_kaggle_cli() == "C:/tools/kaggle.exe"


def test_access_check_lists_the_dataset_non_interactively(
    manifest_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _RecordingRun(_completed(stdout=LISTING))
    monkeypatch.setattr(subprocess, "run", fake)
    entry = load_manifest_entry("fixture", path=manifest_path)

    assert download.kaggle_access_problem(entry, "/opt/kaggle") is None

    [(command, kwargs)] = fake.calls
    assert command == ["/opt/kaggle", "datasets", "files", "someone/fixture", "-v"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["capture_output"] is True
    assert kwargs["timeout"] > 0


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param("", id="exits-zero-listing-nothing"),
        pytest.param("name,size,creationDate\nother.csv,1,x\n", id="expected-file-absent"),
    ],
)
def test_access_check_trusts_the_listing_not_the_exit_code(
    manifest_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    """Exit 0 alone shows nothing: a launcher that fails silently also exits 0."""
    monkeypatch.setattr(subprocess, "run", _RecordingRun(_completed(stdout=stdout)))
    entry = load_manifest_entry("fixture", path=manifest_path)

    problem = download.kaggle_access_problem(entry, "/opt/kaggle")

    assert problem is not None
    assert "did not list data.csv" in problem
    assert "kaggle auth login" in problem


def test_access_check_reports_a_failing_cli_without_echoing_its_output(
    manifest_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "key-sentinel-7f3a"
    monkeypatch.setattr(
        subprocess,
        "run",
        _RecordingRun(_completed(returncode=1, stdout=secret, stderr=f"401 token={secret}")),
    )
    entry = load_manifest_entry("fixture", path=manifest_path)

    problem = download.kaggle_access_problem(entry, "/opt/kaggle")

    assert problem is not None
    assert "exited 1" in problem
    assert secret not in problem


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (subprocess.TimeoutExpired(["kaggle"], 120), "did not list someone/fixture within"),
        (FileNotFoundError("gone"), "could not be run (FileNotFoundError)"),
    ],
)
def test_access_check_reports_a_hung_or_unrunnable_cli(
    manifest_path: Path, monkeypatch: pytest.MonkeyPatch, exc: BaseException, expected: str
) -> None:
    monkeypatch.setattr(subprocess, "run", _RecordingRun(_raising(exc)))
    entry = load_manifest_entry("fixture", path=manifest_path)

    problem = download.kaggle_access_problem(entry, "/opt/kaggle")

    assert problem is not None
    assert expected in problem


def test_access_check_refuses_a_non_kaggle_source_without_running_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _manifest_payload()
    payload["datasets"]["fixture"]["source"] = "http"  # type: ignore[index]
    path = tmp_path / "other.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", _RecordingRun(_fail_if_called))

    problem = download.kaggle_access_problem(
        load_manifest_entry("fixture", path=path), "/opt/kaggle"
    )

    assert problem is not None
    assert "not a Kaggle dataset" in problem


def test_missing_files_are_downloaded_by_the_cli_then_verified_without_pinning(
    manifest_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acquisition checks access, downloads, verifies — and never touches the manifest."""
    raw_root = tmp_path / "raw"
    fake = _RecordingRun(_listing_then_download(CONTENT))
    monkeypatch.setattr(download, "find_kaggle_cli", lambda: "/opt/kaggle")
    monkeypatch.setattr(subprocess, "run", fake)
    manifest_before = manifest_path.read_bytes()

    exit_code = download.main(_main_args(manifest_path, raw_root))

    assert exit_code == download.EXIT_OK
    assert fake.subcommands == ["files", "download"]
    download_command, download_kwargs = fake.calls[1]
    assert download_command == [
        "/opt/kaggle",
        "datasets",
        "download",
        "someone/fixture",
        "-p",
        str(raw_root / "fixture"),
        "--unzip",
    ]
    assert download_kwargs["stdin"] is subprocess.DEVNULL
    assert "capture_output" not in download_kwargs
    assert manifest_path.read_bytes() == manifest_before
    assert load_manifest_entry("fixture", path=manifest_path).file("data.csv").sha256 is None


def test_downloaded_bytes_are_still_verified_against_a_pinned_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful CLI download is not verification: different bytes still fail."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest_payload(sha256=DIGEST)), encoding="utf-8")
    monkeypatch.setattr(download, "find_kaggle_cli", lambda: "/opt/kaggle")
    monkeypatch.setattr(
        subprocess, "run", _RecordingRun(_listing_then_download(b"other bytes\n"))
    )

    exit_code = download.main(_main_args(path, tmp_path / "raw"))

    assert exit_code == download.EXIT_VERIFICATION_FAILED


def test_cli_without_access_falls_back_to_manual_instructions(
    manifest_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="ml.datasets.download")
    fake = _RecordingRun(_completed(stdout=""))
    monkeypatch.setattr(download, "find_kaggle_cli", lambda: "/opt/kaggle")
    monkeypatch.setattr(subprocess, "run", fake)

    exit_code = download.main(_main_args(manifest_path, tmp_path / "raw"))

    assert exit_code == download.EXIT_MANUAL_DOWNLOAD_REQUIRED
    assert fake.subcommands == ["files"]
    assert "Automatic download unavailable" in caplog.text
    assert "Manual download required" in caplog.text
    assert not (tmp_path / "raw" / "fixture").exists()


@pytest.mark.parametrize(
    "respond_to_download",
    [
        pytest.param(_completed(returncode=1), id="cli-exits-nonzero"),
        pytest.param(_completed(returncode=0), id="cli-exits-zero-without-files"),
        pytest.param(_raising(subprocess.TimeoutExpired(["kaggle"], 3600)), id="timeout"),
    ],
)
def test_failed_download_falls_back_to_manual_instructions(
    manifest_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    respond_to_download: Responder,
) -> None:
    caplog.set_level(logging.INFO, logger="ml.datasets.download")

    def respond(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[2] == "files":
            return subprocess.CompletedProcess(command, 0, stdout=LISTING, stderr="")
        return respond_to_download(command)

    monkeypatch.setattr(download, "find_kaggle_cli", lambda: "/opt/kaggle")
    monkeypatch.setattr(subprocess, "run", _RecordingRun(respond))

    exit_code = download.main(_main_args(manifest_path, tmp_path / "raw"))

    assert exit_code == download.EXIT_MANUAL_DOWNLOAD_REQUIRED
    assert "Automatic download failed" in caplog.text
    assert "Manual download required" in caplog.text


def test_download_that_exits_zero_without_the_files_is_refused(
    manifest_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", _RecordingRun(_completed()))
    entry = load_manifest_entry("fixture", path=manifest_path)

    with pytest.raises(ManifestError, match=r"exited 0 but these files are not in .*data\.csv"):
        download.download_with_kaggle_cli(entry, tmp_path / "dest", executable="/opt/kaggle")


def test_credentials_never_reach_the_log(
    manifest_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Neither the environment's credentials nor anything the CLI prints is logged."""
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("KAGGLE_USERNAME", "user-sentinel-7f3a")
    monkeypatch.setenv("KAGGLE_KEY", "key-sentinel-7f3a")
    monkeypatch.setattr(download, "find_kaggle_cli", lambda: "/opt/kaggle")
    monkeypatch.setattr(
        subprocess,
        "run",
        _RecordingRun(_completed(returncode=1, stderr="401 key=key-sentinel-7f3a")),
    )

    exit_code = download.main(_main_args(manifest_path, tmp_path / "raw"))

    assert exit_code == download.EXIT_MANUAL_DOWNLOAD_REQUIRED
    assert "sentinel-7f3a" not in caplog.text


ULB_LISTING = "name,size,creationDate\ncreditcard.csv,150828752,2019-09-20 00:04:39.013000\n"


def test_committed_ulb_entry_drives_the_existing_acquisition_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ULB is fetched by the same command as Sparkov, into its own gitignored directory."""
    fake = _RecordingRun(_completed(stdout=ULB_LISTING))
    monkeypatch.setattr(subprocess, "run", fake)
    entry = load_manifest_entry("ulb")
    destination = download.dataset_root("ulb")

    assert download.kaggle_access_problem(entry, "/opt/kaggle") is None
    [(command, _)] = fake.calls
    assert command == ["/opt/kaggle", "datasets", "files", "mlg-ulb/creditcardfraud", "-v"]
    assert destination == RAW_DATA_DIR / "ulb"

    text = download.manual_instructions(entry, destination)
    assert "creditcard.csv" in text
    assert "DbCL-1.0" in text
    assert "never be committed" in text
