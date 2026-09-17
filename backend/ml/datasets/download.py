"""Acquire a source dataset into the raw-data layout, then verify it.

    uv run python -m ml.datasets.download --dataset sparkov
    uv run python -m ml.datasets.download --dataset sparkov --pin-hashes
    uv run python -m ml.datasets.download --dataset sparkov --verify-only

Kaggle requires an account and an API token, so this script cannot be fully
automatic for everyone. It shells out to the Kaggle CLI when that CLI can see
the dataset, and otherwise prints exactly what to download, from where, and
where to put it. Either way the verification step is identical — the manual
path is a first-class route, not a degraded one.

No credential is read, printed, or stored here. Where the CLI keeps its token
depends on its version, so this module does not look for one: it asks the CLI
to list the dataset's files and requires the manifest's files in that listing,
since an exit code of 0 alone does not show the CLI saw anything. A download
that still fails, authorisation included, falls back to the manual steps. The
CLI's output is parsed, never logged.
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from ml.datasets.manifest import (
    DEFAULT_MANIFEST_PATH,
    DatasetManifestEntry,
    FileStatus,
    ManifestError,
    load_manifest_entry,
    pin_hashes,
    verify_dataset,
)
from ml.paths import RAW_DATA_DIR, ensure_dir

log = logging.getLogger("ml.datasets.download")

_KAGGLE_TIMEOUT_SECONDS = 3600
_KAGGLE_PROBE_TIMEOUT_SECONDS = 120

EXIT_OK = 0
EXIT_VERIFICATION_FAILED = 1
EXIT_MANUAL_DOWNLOAD_REQUIRED = 2


def dataset_root(dataset: str, *, raw_root: Path = RAW_DATA_DIR) -> Path:
    """Where one dataset's raw files live: `ml/data/raw/<dataset>/`."""
    return raw_root / dataset


def find_kaggle_cli() -> str | None:
    """Full path to the installed Kaggle CLI, or None.

    The resolved path is what gets executed: on Windows a bare `kaggle` only
    runs if the entry point happens to be an `.exe`.
    """
    return shutil.which("kaggle")


def kaggle_access_problem(entry: DatasetManifestEntry, executable: str) -> str | None:
    """Why the Kaggle CLI cannot fetch `entry`, or None when it can.

    Runs `kaggle datasets files <ref> -v` and requires every manifest file in
    the listing. The listing shows the CLI runs and can see the dataset. It
    does not prove the download will be authorised — that failure is handled
    when the download runs — and the advertised sizes prove nothing about the
    bytes that arrive, which only post-download verification establishes.
    """
    if entry.source != "kaggle":
        return f"{entry.name} is not a Kaggle dataset (source: {entry.source})."
    command = [executable, "datasets", "files", entry.reference, "-v"]
    how_to_diagnose = f"Run `kaggle datasets files {entry.reference}` to see why."
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=_KAGGLE_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return (
            f"Kaggle CLI did not list {entry.reference} within "
            f"{_KAGGLE_PROBE_TIMEOUT_SECONDS}s. {how_to_diagnose}"
        )
    except OSError as exc:
        return f"Kaggle CLI at {executable} could not be run ({type(exc).__name__})."

    if result.returncode != 0:
        return (
            f"Kaggle CLI exited {result.returncode} listing {entry.reference}: it may not "
            f"be authenticated, or cannot access the dataset. {how_to_diagnose}"
        )

    listed = _listed_filenames(result.stdout)
    unlisted = [name for name in entry.filenames if name not in listed]
    if unlisted:
        return (
            f"Kaggle CLI exited 0 but did not list {', '.join(unlisted)} for "
            f"{entry.reference}. Check it is installed correctly and authenticated "
            f"(for example `kaggle auth login`). {how_to_diagnose}"
        )
    return None


def _listed_filenames(csv_listing: str) -> set[str]:
    """File names from `kaggle datasets files -v` output (CSV with a `name` column)."""
    reader = csv.DictReader(io.StringIO(csv_listing))
    return {(row.get("name") or "").strip() for row in reader} - {""}


def download_with_kaggle_cli(
    entry: DatasetManifestEntry, destination: Path, *, executable: str
) -> None:
    """Run `kaggle datasets download <ref> -p <dest> --unzip`.

    Raises `ManifestError` if the CLI fails, times out, or exits 0 without
    producing every manifest file. Output streams straight to the terminal so
    progress on a multi-hundred-megabyte download stays visible.
    """
    if entry.source != "kaggle":
        raise ManifestError(f"{entry.name}: automatic download supports kaggle only.")
    ensure_dir(destination)
    command = [
        executable,
        "datasets",
        "download",
        entry.reference,
        "-p",
        str(destination),
        "--unzip",
    ]
    log.info("Downloading %s from Kaggle into %s", entry.reference, destination)
    try:
        result = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=_KAGGLE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise ManifestError(
            f"Kaggle CLI did not finish downloading within {_KAGGLE_TIMEOUT_SECONDS}s."
        ) from None
    except OSError as exc:
        raise ManifestError(
            f"Kaggle CLI at {executable} could not be run ({type(exc).__name__})."
        ) from None
    if result.returncode != 0:
        raise ManifestError(f"Kaggle CLI exited {result.returncode}.")

    absent = [name for name in entry.filenames if not (destination / name).exists()]
    if absent:
        raise ManifestError(
            f"Kaggle CLI exited 0 but these files are not in {destination}: {', '.join(absent)}."
        )


def manual_instructions(entry: DatasetManifestEntry, destination: Path) -> str:
    """What to do when the CLI route is unavailable."""
    files = "\n".join(f"      - {name}" for name in entry.filenames)
    return (
        f"\nManual download required for {entry.name} ({entry.display_name}).\n\n"
        f"  1. Sign in to Kaggle and open:\n       {entry.source_url}\n"
        f"  2. Download these files:\n{files}\n"
        f"  3. Put them (unzipped) in:\n       {destination}\n"
        f"  4. Verify:\n"
        f"       uv run python -m ml.datasets.download --dataset {entry.name} --verify-only\n\n"
        f"  Licence: {entry.license}\n"
        f"  Raw data is gitignored and must never be committed.\n\n"
        f"  To automate this instead, install the Kaggle CLI and authenticate it, so that\n"
        f"       kaggle datasets files {entry.reference}\n"
        f"  lists the files above.\n"
    )


def report_verification(entry: DatasetManifestEntry, destination: Path) -> int:
    """Print per-file verification results and return a process exit code."""
    checks = verify_dataset(entry, destination)
    for check in checks:
        if check.status is FileStatus.OK:
            log.info("  OK          %s  sha256=%s", check.name, check.actual_sha256)
        elif check.status is FileStatus.UNPINNED:
            log.warning(
                "  UNPINNED    %s  sha256=%s  (%s)",
                check.name,
                check.actual_sha256,
                check.detail,
            )
        else:
            log.error("  %-11s %s  %s", check.status.value.upper(), check.name, check.detail)

    if any(check.is_blocking for check in checks):
        return EXIT_VERIFICATION_FAILED
    return EXIT_OK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify a source dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name as listed in the manifest")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_DATA_DIR,
        help="Root for raw source files (default: ml/data/raw)",
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to manifest.json"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip downloading; only check what is already on disk",
    )
    parser.add_argument(
        "--pin-hashes",
        action="store_true",
        help="Record the digests of the downloaded files in the manifest (first retrieval)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    entry = load_manifest_entry(args.dataset, path=args.manifest)
    destination = dataset_root(args.dataset, raw_root=args.raw_root)
    log.info("%s — %s", entry.name, entry.display_name)
    log.info("Licence: %s", entry.license)
    log.info("Destination: %s", destination)

    if not args.verify_only:
        missing = [name for name in entry.filenames if not (destination / name).exists()]
        if not missing:
            log.info("All expected files already present; skipping download.")
        else:
            executable = find_kaggle_cli()
            problem = (
                "Kaggle CLI not found on PATH."
                if executable is None
                else kaggle_access_problem(entry, executable)
            )
            if executable is None or problem is not None:
                log.warning("Automatic download unavailable: %s", problem)
                log.warning(manual_instructions(entry, destination))
                return EXIT_MANUAL_DOWNLOAD_REQUIRED
            try:
                download_with_kaggle_cli(entry, destination, executable=executable)
            except ManifestError as exc:
                log.error("Automatic download failed: %s", exc)
                log.warning(manual_instructions(entry, destination))
                return EXIT_MANUAL_DOWNLOAD_REQUIRED

    log.info("Verifying against %s", args.manifest)
    exit_code = report_verification(entry, destination)

    if args.pin_hashes and exit_code == EXIT_OK:
        pinned = pin_hashes(entry, destination, path=args.manifest)
        for name, digest in pinned.items():
            log.info("  PINNED      %s  sha256=%s", name, digest)
        log.info("Manifest updated — commit it so others verify the same bytes.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
