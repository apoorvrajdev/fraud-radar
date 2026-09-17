"""Promotion: copying one run's model artifacts into the directory the API serves.

Phase 5D decision 2. Promotion is a mechanism, not an outcome: this command is
built and tested in 5D and not run as part of it. Replacing the served model
is a decision made with the benchmark results in hand, and it is taken by
running this command explicitly.

The guard reads the featureset version from the run's `run.json` and refuses
any version absent from the featureset registry: the API extracts registered
features only, so a model trained on anything else would be served columns
that do not mean what it learned. Beyond that, promotion refuses what would
make the served files describe something other than the run: a `run.json`
naming another run, a missing artifact, a feature list out of the registered
order, and a `model.json` that differs from the digest its fit record holds.

Exactly five files are copied — `model.json`, `feature_list.json`,
`threshold.json`, `metrics.json` and `training_metadata.json` — by copy, not
symlink. Every file is staged beside its destination first and only then moved
into place, so a failed copy never leaves a mix of two runs' files. Nothing
else in the served directory is touched.

Usage:
    cd backend
    uv run python -m ml.promote <run-name>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from app.fraud.feature_spec import FEATURESETS, feature_names
from ml.artifacts import MODEL_FILENAME, load_feature_list
from ml.datasets.base import DatasetContractError
from ml.datasets.manifest import sha256_file
from ml.paths import ARTIFACTS_DIR, RUNS_ROOT, InvalidRunNameError, run_dir, validate_run_name
from ml.runs import RUN_METADATA_FILENAME, RunMetadata, load_run_metadata

log = logging.getLogger("ml.promote")

TRAINING_METADATA_FILENAME = "training_metadata.json"
FEATURE_LIST_FILENAME = "feature_list.json"

PROMOTED_FILES: tuple[str, ...] = (
    MODEL_FILENAME,
    FEATURE_LIST_FILENAME,
    "threshold.json",
    "metrics.json",
    TRAINING_METADATA_FILENAME,
)

_STAGING_SUFFIX = ".promoting"


class PromotionError(Exception):
    """A run cannot be promoted to the served artifacts."""


@dataclass(frozen=True)
class Promotion:
    """What a promotion copied, from which run, into where."""

    run_name: str
    featureset_version: str
    source: Path
    destination: Path
    copied: tuple[Path, ...]


def check_promotable(run_name: str, *, runs_root: Path = RUNS_ROOT) -> RunMetadata:
    """Refuse a run the served directory must not take; return its record otherwise."""
    directory = run_dir(run_name, runs_root=runs_root)
    if not directory.is_dir():
        raise PromotionError(f"No run named {run_name!r}: {directory} does not exist.")
    try:
        record = load_run_metadata(run_name, runs_root=runs_root)
    except FileNotFoundError as exc:
        raise PromotionError(
            f"Run {run_name!r} has no {RUN_METADATA_FILENAME}, so its featureset cannot be "
            "checked. Only a run written by `python -m ml.train --run-name` can be promoted."
        ) from exc
    except (KeyError, TypeError, ValueError, DatasetContractError) as exc:
        raise PromotionError(f"Run {run_name!r} has a malformed {RUN_METADATA_FILENAME}: {exc}") from exc

    if record.run_name != run_name:
        raise PromotionError(
            f"Run {run_name!r}: {RUN_METADATA_FILENAME} names run {record.run_name!r}."
        )
    version = record.featureset_version
    if version not in FEATURESETS:
        raise PromotionError(
            f"Run {run_name!r} was trained on featureset {version!r}, which is not registered "
            f"({', '.join(sorted(FEATURESETS))}). The API extracts registered featuresets only."
        )
    missing = [name for name in PROMOTED_FILES if not (directory / name).is_file()]
    if missing:
        raise PromotionError(f"Run {run_name!r} is missing {', '.join(missing)}.")
    if load_feature_list(directory) != feature_names(version):
        raise PromotionError(
            f"Run {run_name!r}: {FEATURE_LIST_FILENAME} does not list featureset {version!r}'s "
            "features in their registered order."
        )
    _check_model_digest(run_name, directory)
    return record


def promote_run(
    run_name: str,
    *,
    runs_root: Path = RUNS_ROOT,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Promotion:
    """Copy the run's five model artifacts into `artifacts_dir`, after the guard passes."""
    record = check_promotable(run_name, runs_root=runs_root)
    source = run_dir(run_name, runs_root=runs_root)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    staged: list[tuple[Path, Path]] = []
    try:
        for name in PROMOTED_FILES:
            target = artifacts_dir / name
            staging = artifacts_dir / f"{name}{_STAGING_SUFFIX}"
            shutil.copyfile(source / name, staging)
            staged.append((staging, target))
    except OSError:
        for staging, _ in staged:
            staging.unlink(missing_ok=True)
        raise
    for staging, target in staged:
        os.replace(staging, target)

    return Promotion(
        run_name=run_name,
        featureset_version=record.featureset_version,
        source=source,
        destination=artifacts_dir,
        copied=tuple(target for _, target in staged),
    )


def _check_model_digest(run_name: str, directory: Path) -> None:
    with (directory / TRAINING_METADATA_FILENAME).open(encoding="utf-8") as handle:
        recorded = json.load(handle).get("model_sha256")
    if recorded is None:
        return
    actual = sha256_file(directory / MODEL_FILENAME)
    if actual != recorded:
        raise PromotionError(
            f"Run {run_name!r}: {MODEL_FILENAME} hashes to {actual}, but "
            f"{TRAINING_METADATA_FILENAME} records {recorded}; it is not the model the run saved."
        )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a run's model artifacts into the directory the API serves"
    )
    parser.add_argument("run_name", help="Run under ml/artifacts/runs/ to promote")
    args = parser.parse_args(argv)
    try:
        validate_run_name(args.run_name)
    except InvalidRunNameError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    args = parse_args(argv)
    try:
        promotion = promote_run(args.run_name, runs_root=RUNS_ROOT, artifacts_dir=ARTIFACTS_DIR)
    except PromotionError as exc:
        log.error("Run %s was not promoted: %s", args.run_name, exc)
        sys.exit(1)
    for path in promotion.copied:
        log.info("Promoted %s", path)
    log.info(
        "Run %s (featureset %s) is now in %s; restart the API to serve it.",
        promotion.run_name,
        promotion.featureset_version,
        promotion.destination,
    )


if __name__ == "__main__":
    main()
