"""Post-training analyses: segments, calibration, global SHAP."""
from ml.analysis.calibration import (
    compute_calibration_metrics,
    expected_calibration_error,
    positive_class_brier,
    positive_class_ece,
    render_calibration_plot,
)
from ml.analysis.global_importance import (
    compute_feature_importance,
    render_bar_plot,
    render_beeswarm_plot,
)
from ml.analysis.segments import (
    BUCKET_ORDER,
    bucket_for_country,
    compute_segment_metrics,
)

__all__ = [
    "BUCKET_ORDER",
    "bucket_for_country",
    "compute_calibration_metrics",
    "compute_feature_importance",
    "compute_segment_metrics",
    "expected_calibration_error",
    "positive_class_brier",
    "positive_class_ece",
    "render_bar_plot",
    "render_beeswarm_plot",
    "render_calibration_plot",
]
