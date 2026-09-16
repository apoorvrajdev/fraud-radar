"""CLI: dataset name in, cached feature matrix and run record out.

    uv run python -m ml.features.build --dataset sparkov --max-cards 200

This is the seam Phase 5D will evaluate from:

    adapter -> CanonicalDataset -> batch builder -> versioned cache -> (5D)

It stops at the cache on purpose. No model is trained, no metric is computed,
and nothing here decides anything about a model — 5C's job is to make the
matrix reproducible, not to draw conclusions from it.

The run record written alongside it is the 5A `run.json`: dataset provenance,
featureset version, seed, commit and library versions. Splits are left empty
because choosing folds is an evaluation decision, not a feature-building one.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import ml.datasets.sparkov  # noqa: F401  (registers the adapter)
from app.fraud.feature_spec import DEFAULT_FEATURESET
from ml.artifacts import collect_library_versions
from ml.datasets.registry import available_datasets, get_adapter
from ml.features.cache import build_or_load
from ml.paths import FEATURE_CACHE_DIR, RAW_DATA_DIR
from ml.runs import RunMetadata, current_git_commit, save_run_metadata

log = logging.getLogger("ml.features.build")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build (or reuse) the cached feature matrix for a benchmark dataset"
    )
    parser.add_argument(
        "--dataset", required=True, help=f"Registered dataset: {', '.join(available_datasets())}"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Directory holding the source files (default: ml/data/raw/<dataset>)",
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=None,
        help="Keep only N entities, with their full histories (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for entity subsampling")
    parser.add_argument(
        "--featureset", default=DEFAULT_FEATURESET, help="Featureset version to extract"
    )
    parser.add_argument(
        "--cache-root", type=Path, default=FEATURE_CACHE_DIR, help="Feature cache directory"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Rebuild even if a cache entry exists"
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Write a run record under ml/artifacts/runs/<run-name>/",
    )
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Allow an unsubsampled load where the adapter guards against one",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args = parse_args(argv)

    adapter = get_adapter(args.dataset)
    root = args.root or RAW_DATA_DIR / args.dataset

    # Adapters that guard against an accidental full-corpus load expose the
    # opt-in as an attribute. Set it only where it exists, so this entry point
    # stays dataset-agnostic.
    if args.full_corpus and hasattr(adapter, "allow_full_corpus"):
        adapter.allow_full_corpus = True

    dataset = adapter.load(root, max_entities=args.max_cards, seed=args.seed)
    matrix, metadata, cache_hit = build_or_load(
        dataset,
        featureset=args.featureset,
        cache_root=args.cache_root,
        refresh=args.refresh,
    )

    log.info(
        "%s | %s | %d rows x %d features | fraud %d (%.3f%%) | cache %s",
        metadata.dataset_name,
        metadata.featureset_version,
        matrix.n_rows,
        matrix.X.shape[1],
        int(matrix.y.sum()),
        matrix.fraud_rate * 100,
        "hit" if cache_hit else "written",
    )

    if args.run_name:
        record = RunMetadata(
            run_name=args.run_name,
            dataset=dataset.provenance,
            featureset_version=args.featureset,
            code_version=current_git_commit(),
            library_versions=collect_library_versions(),
            notes=f"Feature matrix cached at {metadata.fingerprint}.",
        )
        log.info("Run record written to %s", save_run_metadata(record))

    return 0


if __name__ == "__main__":
    sys.exit(main())
