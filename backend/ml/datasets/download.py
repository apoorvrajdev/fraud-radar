"""Acquire a source dataset into the raw-data layout, then verify it.

    uv run python -m ml.datasets.download --dataset sparkov
    uv run python -m ml.datasets.download --dataset sparkov --pin-hashes
    uv run python -m ml.datasets.download --dataset sparkov --verify-only

Kaggle requires an account and an API token, so this script cannot be fully
automatic for everyone. It shells out to the Kaggle CLI when that CLI and its
credentials are present, and otherwise prints exactly what to download, from
where, and where to put it. Either way the verification step is identical —
the manual path is a first-class route, not a degraded one.

No credential is read, printed, or stored here: the Kaggle CLI handles its own
token, and this module only checks whether one appears to be configured.
"""
from __future__ import annotations

import argparse
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

EXIT_OK = 0
EXIT_VERIFICATION_FAILED = 1
EXIT_MANUAL_DOWNLOAD_REQUIRED = 2


def dataset_root(dataset: str, *, raw_root: Path = RAW_DATA_DIR) -> Path:
    """Where one dataset's raw files live: `ml/data/raw/<dataset>/`."""
    return raw_root / dataset


def kaggle_cli_available() -> bool:
    """True when the Kaggle CLI is installed and a token appears configured."""
    if shutil.which("kaggle") is None:
        return False
    import os

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    candidates = [Path(config_dir) / "kaggle.json"] if config_dir else []
    candidates.append(Path.home() / ".kaggle" / "kaggle.json")
    return any(candidate.exists() for candidate in candidates)


def download_with_kaggle_cli(entry: DatasetManifestEntry, destination: Path) -> None:
    """Run `kaggle datasets download -d <ref> -p <dest> --unzip`."""
    if entry.source != "kaggle":
        raise ManifestError(f"{entry.name}: automatic download supports kaggle only.")
    ensure_dir(destination)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        entry.reference,
        "-p",
        str(destination),
        "--unzip",
    ]
    log.info("Downloading %s from Kaggle into %s", entry.reference, destination)
    result = subprocess.run(
        command, check=False, timeout=_KAGGLE_TIMEOUT_SECONDS
    )
    if result.returncode != 0:
        raise ManifestError(
            f"Kaggle CLI exited {result.returncode}. Re-run with the manual instructions below."
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
        f"  To automate this instead, install the Kaggle CLI and place an API token at\n"
        f"  ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME / KAGGLE_KEY).\n"
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
        elif kaggle_cli_available():
            download_with_kaggle_cli(entry, destination)
        else:
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
