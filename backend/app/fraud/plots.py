"""SHAP visualisation renderers — PNG bytes for embedding in dashboard tiles.

Both renderers always close their matplotlib figure on exit. In a long-lived
server, leaking figures means leaking process memory; the `try/finally` in
each function is load-bearing.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402

_FIG_WIDTH_INCHES = 12.0
_FIG_DPI = 100


_FORCE_FIG_WIDTH_INCHES = 14.0


def _short_id(transaction_id: str) -> str:
    """Trim a UUID to 8 chars for plot titles."""
    return transaction_id.replace("-", "")[:8]


def _format_feature_value(value: float) -> str:
    """Format a single feature value for a force-plot bar label.

    Whole-number floats (booleans, small integers) render as plain integers so
    `1.0` becomes `"1"` and `20.0` becomes `"20"`. Everything else rounds to
    two decimal places via `f"{value:.2f}"`, so `3.7812307` becomes `"3.78"`.
    Values near zero (`0.0042`) still render as `"0.00"` rather than being
    dropped — the reader can tell the feature was observed even when its
    value contributed nothing.
    """
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _aggregate_top_k_with_remainder(
    feature_names: list[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    top_k: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Reduce a full SHAP attribution to top-k bars + one aggregated remainder bar.

    Sort by absolute SHAP value, keep the top-k, sum every remaining contribution
    into a single bar named "other (N)" where N is the aggregated feature count.
    When the input has at most `top_k` features, returns the inputs unchanged
    (no aggregation needed).

    SHAP additive invariant — preserved by construction:
        sum(returned shap_values) == sum(input shap_values).
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive; got {top_k}")

    shap_arr = np.asarray(shap_values, dtype=np.float64)
    value_arr = np.asarray(feature_values, dtype=np.float64)
    n = len(shap_arr)
    if n <= top_k:
        return list(feature_names), value_arr, shap_arr

    order = np.argsort(-np.abs(shap_arr))
    keep_idx = order[:top_k]
    rest_idx = order[top_k:]

    kept_names = [feature_names[i] for i in keep_idx]
    kept_values = value_arr[keep_idx]
    kept_shap = shap_arr[keep_idx]

    aggregated_shap = float(shap_arr[rest_idx].sum())
    n_other = int(len(rest_idx))

    out_names = [*kept_names, f"other ({n_other})"]
    # Feature value for the aggregated bar is not meaningful; the count is in the name.
    out_values = np.concatenate([kept_values, np.array([0.0])])
    out_shap = np.concatenate([kept_shap, np.array([aggregated_shap])])
    return out_names, out_values, out_shap


def render_force_plot(
    transaction_id: str,
    feature_names: list[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    base_value: float,
    *,
    top_k: int = 8,
) -> bytes:
    """Render shap.force_plot to a PNG byte string and free the figure.

    The plot shows the top `top_k` features by |shap_value|, with the rest
    collapsed into a single "other N features" bar. This is the industry
    convention (Datadog, Stripe) for keeping the per-segment labels legible
    when a model has more features than the figure width can accommodate.
    The aggregated remainder preserves the SHAP additive property.
    """
    display_names, display_values, display_shap = _aggregate_top_k_with_remainder(
        feature_names, feature_values, shap_values, top_k=top_k
    )

    # Pre-bake the full label for each bar and pass `features=None` to SHAP so
    # it renders our strings verbatim. This keeps us in control of formatting:
    # integers stay short, floats round to 2 decimals, and the aggregated
    # "other (N)" bar carries no meaningless "= value" suffix.
    labels: list[str] = []
    for name, value in zip(display_names, display_values, strict=True):
        if name.startswith("other ("):
            labels.append(name)
        else:
            labels.append(f"{name} = {_format_feature_value(float(value))}")

    plt.close("all")  # defensive: pre-empt any orphaned figure from prior calls
    fig = plt.figure(figsize=(_FORCE_FIG_WIDTH_INCHES, 4.0), dpi=_FIG_DPI)
    try:
        shap.force_plot(
            base_value,
            display_shap,
            features=None,
            feature_names=labels,
            matplotlib=True,
            show=False,
            figsize=(_FORCE_FIG_WIDTH_INCHES, 4.0),
        )
        # shap.force_plot creates its own figure when matplotlib=True; grab it
        current = plt.gcf()
        current.suptitle(
            f"SHAP force plot — transaction {_short_id(transaction_id)}",
            fontsize=11,
        )
        buf = io.BytesIO()
        current.savefig(buf, format="png", bbox_inches="tight")
        return buf.getvalue()
    finally:
        plt.close("all")
        plt.close(fig)


def render_waterfall_plot(
    transaction_id: str,
    feature_names: list[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    base_value: float,
) -> bytes:
    """Render shap.plots.waterfall to a PNG byte string and free the figure."""
    plt.close("all")
    fig = plt.figure(figsize=(_FIG_WIDTH_INCHES, 6.0), dpi=_FIG_DPI)
    try:
        explanation = shap.Explanation(
            values=np.asarray(shap_values, dtype=np.float64),
            base_values=float(base_value),
            data=np.asarray(feature_values, dtype=np.float64),
            feature_names=list(feature_names),
        )
        shap.plots.waterfall(explanation, show=False, max_display=len(feature_names))
        current = plt.gcf()
        current.suptitle(
            f"SHAP waterfall — transaction {_short_id(transaction_id)}",
            fontsize=11,
        )
        buf = io.BytesIO()
        current.savefig(buf, format="png", bbox_inches="tight")
        return buf.getvalue()
    finally:
        plt.close("all")
        plt.close(fig)
