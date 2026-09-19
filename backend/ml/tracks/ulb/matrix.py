"""The `ulb_pca_v1` matrix: the ULB rows as the labelled matrix a run trains on.

Phase 5E decision 4 defines `ulb_pca_v1` as the 29 columns `V1`, …, `V28`,
`Amount`, under their source names and in source order. `Time` and `Class`
are not features: `Time` orders the rows and gives them their elapsed-time
timestamps (decision 6), and `Class` is the label.

The featureset lives here and nowhere else. It is deliberately absent from
the production registry in `app/fraud/feature_spec.py`, which means "the
production extractor can build these columns from a transaction, and a model
trained on them can be served". Neither holds for PCA components, and keeping
the name out makes promotion, run verification and analysis, and the serving
explainer refuse a ULB run by construction.

Every value is the published value (decision 5): no log transform, scaling,
imputation, clipping, or any step fitted on the data. A row's features are its
own published values and nothing else, so no row's features depend on another
row's, least of all a later one's, and changing one fold's rows cannot change
another fold's features. The one thing no code can undo is noted in the Phase
5E record: which rows the publisher fitted the PCA on is not stated.

There is no feature cache. The matrix is rebuilt in memory from the pinned
file whenever it is needed and is never written, so no committed file holds a
ULB row or matrix.

The matrix is a `LabelledDataset`, the contract `ml.train.train_and_evaluate`
consumes. Its rows are in load order, `Time` then position in the file, so
`chronological_split` of its timestamps gives exactly the folds the ULB quality
report describes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ml.data import LabelledDataset
from ml.datasets.base import DatasetProvenance
from ml.tracks.ulb.load import AMOUNT_COLUMN, COMPONENT_COLUMNS, UlbSource

ULB_PCA_V1 = "ulb_pca_v1"

# The featureset's columns, in the order they appear in the matrix.
ULB_PCA_V1_FEATURES: tuple[str, ...] = (*COMPONENT_COLUMNS, AMOUNT_COLUMN)

# Added to the loaded provenance, so a run record states how its matrix was
# made from the rows the loader describes.
MATRIX_PREPROCESSING = (
    f"matrix: featureset {ULB_PCA_V1}, the 29 columns V1-V28 and Amount as published, in "
    "source order; Time orders the rows and Class is the label, and neither is a feature; "
    "rows in load order; no transform, scaling, imputation or fitted step; no feature cache"
)


@dataclass(frozen=True)
class UlbMatrix:
    """The `ulb_pca_v1` matrix of the loaded ULB rows, and where it came from.

    `provenance` is the loaded rows' provenance with the matrix step added.
    """

    ds: LabelledDataset
    provenance: DatasetProvenance


def build_matrix(source: UlbSource) -> UlbMatrix:
    """The `ulb_pca_v1` matrix of `source`, one row per loaded row, in load order."""
    features = np.ascontiguousarray(
        np.column_stack((source.components, source.amounts)), dtype=np.float64
    )
    ds = LabelledDataset(
        X=features,
        y=source.labels.copy(),
        timestamps=source.timestamps.copy(),
        transaction_ids=list(source.transaction_ids),
        feature_names=list(ULB_PCA_V1_FEATURES),
    )
    provenance = replace(
        source.provenance,
        preprocessing=(*source.provenance.preprocessing, MATRIX_PREPROCESSING),
    )
    return UlbMatrix(ds=ds, provenance=provenance)
