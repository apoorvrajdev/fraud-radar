"""Global feature importance from SHAP values.

The SHAP matrix itself is computed by `FraudExplainer.compute_global_shap` —
this module only ranks and visualises. Keeping the ranking logic in a pure
function means the unit tests don't need a trained model to verify ordering
and rank assignment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def compute_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Rank features by mean absolute SHAP value over a SHAP matrix.

    Returns the JSON payload directly:
        {
            "method": "mean absolute SHAP value over test set",
            "n_test_samples": int,
            "features": [
                {"feature": "...", "mean_abs_shap": float, "rank": int},
                ...
            ]
        }
    Features are sorted descending by `mean_abs_shap`; ties broken by name.
    """
    shap_arr = np.asarray(shap_values, dtype=np.float64)
    if shap_arr.ndim != 2:
        raise ValueError(f"Expected 2-D SHAP matrix; got shape {shap_arr.shape}")
    if shap_arr.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP matrix has {shap_arr.shape[1]} columns but {len(feature_names)} "
            "feature names were provided."
        )

    mean_abs = np.abs(shap_arr).mean(axis=0)
    # Stable secondary sort by name so ties are deterministic.
    order = sorted(
        range(len(feature_names)),
        key=lambda i: (-float(mean_abs[i]), feature_names[i]),
    )

    ranked: list[dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        ranked.append({
            "feature": feature_names[idx],
            "mean_abs_shap": float(mean_abs[idx]),
            "rank": rank,
        })

    return {
        "method": "mean absolute SHAP value over test set",
        "n_test_samples": int(shap_arr.shape[0]),
        "features": ranked,
    }


def render_beeswarm_plot(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    path: Path,
) -> None:
    """Render shap.summary_plot in 'dot' (beeswarm) mode."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    plt.figure(figsize=(10, 8), dpi=100)
    try:
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(path)
    finally:
        plt.close("all")


def render_bar_plot(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    path: Path,
) -> None:
    """Render shap.summary_plot in 'bar' mode."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    plt.figure(figsize=(8, 6), dpi=100)
    try:
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(path)
    finally:
        plt.close("all")
