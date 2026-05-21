"""Tests for SHAP visualisation renderers — PNG bytes + no figure leak."""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app.fraud.feature_spec import FEATURE_NAMES, N_FEATURES  # noqa: E402
from app.fraud.plots import (  # noqa: E402
    _aggregate_top_k_with_remainder,
    _format_feature_value,
    render_force_plot,
    render_waterfall_plot,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _example_inputs() -> tuple[str, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(0)
    feature_values = rng.normal(size=N_FEATURES)
    shap_values = rng.normal(scale=0.1, size=N_FEATURES)
    return ("00000000-0000-0000-0000-000000000abc", feature_values, shap_values, 0.05)


def test_force_plot_returns_png_bytes() -> None:
    tx_id, fv, sv, base = _example_inputs()
    png = render_force_plot(tx_id, FEATURE_NAMES, fv, sv, base)
    assert isinstance(png, bytes)
    assert png.startswith(PNG_MAGIC)
    assert len(png) > 1000  # sanity: a real plot, not an empty buffer


def test_waterfall_plot_returns_png_bytes() -> None:
    tx_id, fv, sv, base = _example_inputs()
    png = render_waterfall_plot(tx_id, FEATURE_NAMES, fv, sv, base)
    assert isinstance(png, bytes)
    assert png.startswith(PNG_MAGIC)
    assert len(png) > 1000


@pytest.mark.parametrize("renderer", [render_force_plot, render_waterfall_plot])
def test_plots_close_figures(renderer) -> None:  # type: ignore[no-untyped-def]
    """No matplotlib figures should remain open after rendering."""
    plt.close("all")
    assert len(plt.get_fignums()) == 0  # baseline: clean state
    tx_id, fv, sv, base = _example_inputs()
    renderer(tx_id, FEATURE_NAMES, fv, sv, base)
    assert plt.get_fignums() == [], "Renderer leaked a matplotlib figure"


def test_force_plot_aggregates_features_beyond_top_k() -> None:
    """Top-8 dominate; remaining 9 collapse into one bar; additive sum is preserved."""
    feature_values = np.zeros(N_FEATURES)
    shap_values = np.zeros(N_FEATURES)
    # Top 8 features carry large, varied contributions
    shap_values[:8] = np.linspace(-1.0, 1.0, 8)
    # Remaining 9 features carry small contributions
    shap_values[8:] = np.linspace(-0.05, 0.05, N_FEATURES - 8)

    names, values, shaps = _aggregate_top_k_with_remainder(
        list(FEATURE_NAMES), feature_values, shap_values, top_k=8
    )

    # 8 kept + 1 aggregated remainder bar
    assert len(names) == 9
    assert len(values) == 9
    assert len(shaps) == 9
    assert names[-1] == f"other ({N_FEATURES - 8})"

    # Additive invariant: aggregation must not change the total contribution.
    assert shaps.sum() == pytest.approx(float(shap_values.sum()), abs=1e-9)

    # Top kept entries must be sorted by |shap| descending
    abs_kept = np.abs(shaps[:8])
    assert list(abs_kept) == sorted(abs_kept, reverse=True)

    # And the renderer end-to-end still produces a PNG with the aggregated bar
    tx_id = "00000000-0000-0000-0000-0000000000ab"
    png = render_force_plot(tx_id, list(FEATURE_NAMES), feature_values, shap_values, 0.05)
    assert png.startswith(PNG_MAGIC)
    assert len(png) > 1000


def test_aggregate_passthrough_when_features_fit() -> None:
    """If n_features <= top_k, no aggregation bar is appended."""
    names = ["a", "b", "c"]
    fv = np.array([1.0, 2.0, 3.0])
    sv = np.array([0.1, -0.2, 0.3])
    out_names, out_values, out_shap = _aggregate_top_k_with_remainder(
        names, fv, sv, top_k=5
    )
    assert out_names == names
    np.testing.assert_array_equal(out_values, fv)
    np.testing.assert_array_equal(out_shap, sv)


def test_format_feature_value_whole_number_floats() -> None:
    assert _format_feature_value(1.0) == "1"
    assert _format_feature_value(0.0) == "0"
    assert _format_feature_value(20.0) == "20"
    assert _format_feature_value(-3.0) == "-3"


def test_format_feature_value_fractional() -> None:
    assert _format_feature_value(3.7812307) == "3.78"
    assert _format_feature_value(-0.424) == "-0.42"
    assert _format_feature_value(0.0042) == "0.00"


def test_format_feature_value_large_numbers() -> None:
    assert _format_feature_value(150.391) == "150.39"
    assert _format_feature_value(535.0) == "535"
