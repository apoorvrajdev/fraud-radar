"""Canonical feature names and ordering.

XGBoost is positional — column order at training time must match the order
at inference time. This module is the single source of truth.
"""
from __future__ import annotations

# Feature ordering: keep stable across training and inference. Adding new
# features means appending here AND retraining the model.
FEATURE_NAMES: list[str] = [
    # Transaction-level
    "log_amount",
    "hour_of_day",
    "is_weekend",
    "is_off_hours",
    "is_card_present",
    # Geographic
    "country_mismatch_customer",
    "country_mismatch_merchant",
    # Velocity (recent activity)
    "tx_count_1h",
    "tx_count_24h",
    "log_amount_sum_24h",
    # Customer history
    "customer_account_age_days",
    "customer_risk_tier_encoded",
    "avg_amount_30d",
    "amount_zscore_30d",
    "days_since_last_tx",
    # Merchant context
    "merchant_risk_encoded",
    "is_high_risk_category",
]

# Number of features (used for shape validation)
N_FEATURES: int = len(FEATURE_NAMES)
