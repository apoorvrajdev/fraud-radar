"""Where each stage of the offline pipeline keeps its files.

Four stages, four locations, one direction of travel:

    data/raw/         source files exactly as downloaded        (never committed)
    data/cache/       derived caches: canonical datasets,        (never committed)
                      feature matrices keyed by input hash
    artifacts/runs/   per-run evaluation outputs and metadata    (committed, minus model.json)
    artifacts/        the promoted run — what the API loads      (committed, minus model.json)

Keeping them apart is what makes a benchmark re-runnable: raw bytes are
reproducible from a hash, caches are disposable, run outputs are the record,
and exactly one promoted set of artifacts is what production serves. Deleting
`data/cache/` must never lose anything but time.
"""
from __future__ import annotations

import re
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

# Data root. The synthetic generator's CSV lives directly here (legacy path,
# unchanged); external source data goes under raw/.
DATA_DIR = ML_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
FEATURE_CACHE_DIR = CACHE_DIR / "features"

# Artifact root — the directory the serving API reads its model from.
ARTIFACTS_DIR = ML_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_DIR / "runs"

# Run names end up as directory names and appear in metadata, so they are
# restricted rather than sanitised: a name that needs cleaning is a bug in
# the caller, and "../.." should never reach a mkdir.
_RUN_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class InvalidRunNameError(ValueError):
    """A run name is empty, malformed, or would escape the runs directory."""


def validate_run_name(run_name: str) -> str:
    """Return `run_name` if it is a safe single path segment."""
    if not _RUN_NAME_PATTERN.fullmatch(run_name):
        raise InvalidRunNameError(
            f"Invalid run name {run_name!r}: use lowercase letters, digits, "
            "'.', '_' or '-', starting with a letter or digit."
        )
    return run_name


def run_dir(run_name: str, *, runs_root: Path = RUNS_ROOT) -> Path:
    """Path to one run's directory. Does not create it."""
    return runs_root / validate_run_name(run_name)


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
