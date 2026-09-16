"""Loading the labelled matrix a run is trained on, or analysed on.

Training and post-training analysis have to read exactly the same rows, so
both go through `load_run_data`. What to load is a `DataRequest`, built either
from command-line options when a run is trained or from the run's own
`run.json` when it is analysed. A run is then reconstructed from what it
recorded, not from what someone remembers passing on the command line.

The synthetic dataset is read from the operational database and its label
CSV. A registered dataset is read through its adapter and the fingerprinted
feature cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ml.datasets.sparkov  # noqa: F401  (registers the adapter)
from app.db import SessionLocal
from app.fraud.feature_spec import DEFAULT_FEATURESET
from ml.data import (
    SYNTHETIC_DATASET_NAME,
    LabelledDataset,
    load_dataset_with_csv_labels,
    synthetic_provenance,
)
from ml.datasets.base import DatasetProvenance
from ml.datasets.registry import get_adapter
from ml.features.cache import build_or_load
from ml.paths import FEATURE_CACHE_DIR, RAW_DATA_DIR
from ml.runs import RunMetadata

DEFAULT_SYNTHETIC_CSV = Path("ml/data/synthetic_transactions.csv")
DEFAULT_SUBSAMPLE_SEED = 42


@dataclass(frozen=True)
class DataRequest:
    """Everything that decides which labelled matrix is loaded.

    `csv_path` and `limit` apply to the synthetic dataset only. `root`,
    `max_entities`, `seed`, `cache_root`, `refresh` and `full_corpus` apply to
    registered datasets only. `root` defaults to `ml/data/raw/<dataset>`.
    """

    dataset: str = SYNTHETIC_DATASET_NAME
    featureset: str = DEFAULT_FEATURESET
    csv_path: Path = DEFAULT_SYNTHETIC_CSV
    limit: int | None = None
    root: Path | None = None
    max_entities: int | None = None
    seed: int = DEFAULT_SUBSAMPLE_SEED
    cache_root: Path = FEATURE_CACHE_DIR
    refresh: bool = False
    full_corpus: bool = False

    @property
    def is_synthetic(self) -> bool:
        return self.dataset == SYNTHETIC_DATASET_NAME

    @classmethod
    def from_run_record(
        cls,
        record: RunMetadata,
        *,
        root: Path | None = None,
        cache_root: Path = FEATURE_CACHE_DIR,
        csv_path: Path = DEFAULT_SYNTHETIC_CSV,
        limit: int | None = None,
        full_corpus: bool = False,
    ) -> DataRequest:
        """The request that reloads the data `record` describes.

        The dataset, featureset, subsample seed and entity limit come from the
        record. Only where the files are, and the full-corpus opt-in, are
        supplied by the caller. A synthetic run records no structured row
        limit, so a limit it was trained with has to be passed again; whether
        it matches is for the caller to check against the recorded provenance.
        """
        dataset = record.dataset
        if dataset.name == SYNTHETIC_DATASET_NAME:
            return cls(
                dataset=dataset.name,
                featureset=record.featureset_version,
                csv_path=csv_path,
                limit=limit,
            )
        if limit is not None:
            raise ValueError(
                "A row limit applies to the synthetic dataset only; run "
                f"{record.run_name!r} was trained on {dataset.name!r}."
            )
        subsample = dataset.subsample
        return cls(
            dataset=dataset.name,
            featureset=record.featureset_version,
            root=root,
            max_entities=None if subsample is None else subsample.max_entities,
            seed=DEFAULT_SUBSAMPLE_SEED if subsample is None else subsample.seed,
            cache_root=cache_root,
            full_corpus=full_corpus,
        )


@dataclass(frozen=True)
class RunData:
    """A loaded matrix and what is known about where it came from.

    `provenance` is None only for a synthetic load that did not ask for it.
    `cache_fingerprint` and `cache_hit` describe the feature cache, which only
    registered datasets use.
    """

    ds: LabelledDataset
    provenance: DatasetProvenance | None
    cache_fingerprint: str | None = None
    cache_hit: bool | None = None


def load_run_data(request: DataRequest, *, with_provenance: bool = True) -> RunData:
    """Load the matrix `request` describes.

    `with_provenance=False` skips the synthetic provenance record, which
    hashes the label CSV; a registered dataset's adapter always supplies one.
    """
    if request.is_synthetic:
        return _load_synthetic(request, with_provenance=with_provenance)
    return _load_registered(request)


def _load_synthetic(request: DataRequest, *, with_provenance: bool) -> RunData:
    if request.featureset != DEFAULT_FEATURESET:
        raise ValueError(
            f"The synthetic dataset is extracted with featureset {DEFAULT_FEATURESET!r} "
            f"only, not {request.featureset!r}."
        )
    with SessionLocal() as db:
        ds = load_dataset_with_csv_labels(
            db,
            csv_path=str(request.csv_path),
            limit=request.limit,
        )
    provenance = (
        synthetic_provenance(ds, csv_path=request.csv_path, limit=request.limit)
        if with_provenance
        else None
    )
    return RunData(ds=ds, provenance=provenance)


def _load_registered(request: DataRequest) -> RunData:
    adapter = get_adapter(request.dataset)
    # Adapters that guard against an accidental full-corpus load expose the
    # opt-in as an attribute. Set it only where it exists, as the feature-build
    # CLI does, so this entry point stays dataset-agnostic.
    if request.full_corpus and hasattr(adapter, "allow_full_corpus"):
        adapter.allow_full_corpus = True

    dataset = adapter.load(
        request.root or RAW_DATA_DIR / request.dataset,
        max_entities=request.max_entities,
        seed=request.seed,
    )
    ds, cache, cache_hit = build_or_load(
        dataset,
        featureset=request.featureset,
        cache_root=request.cache_root,
        refresh=request.refresh,
    )
    return RunData(
        ds=ds,
        provenance=dataset.provenance,
        cache_fingerprint=cache.fingerprint,
        cache_hit=cache_hit,
    )
