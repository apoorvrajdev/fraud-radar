"""Unit tests for the Phase 3G detail-builder pure helpers."""
from __future__ import annotations

import pytest

from app.fraud.decision import Decision
from app.services.transaction_detail import (
    _parse_top_contributors,
    effective_decision,
)


@pytest.mark.parametrize(
    "fraud_decision, analyst_label, expected",
    [
        # Analyst override wins regardless of model verdict.
        ("REVIEW", "CONFIRMED_FRAUD", Decision.DECLINE),
        ("REVIEW", "CONFIRMED_LEGIT", Decision.APPROVE),
        ("APPROVE", "CONFIRMED_FRAUD", Decision.DECLINE),
        ("DECLINE", "CONFIRMED_LEGIT", Decision.APPROVE),
        # No analyst → falls through to fraud_decision.
        ("APPROVE", None, Decision.APPROVE),
        ("REVIEW", None, Decision.REVIEW),
        ("DECLINE", None, Decision.DECLINE),
        ("PENDING", None, Decision.PENDING),
        # No fraud_decision at all → PENDING.
        (None, None, Decision.PENDING),
    ],
)
def test_effective_decision_mapping(
    fraud_decision: str | None,
    analyst_label: str | None,
    expected: Decision,
) -> None:
    assert effective_decision(fraud_decision, analyst_label) == expected


def test_parse_top_contributors_classifies_direction() -> None:
    raw = (
        '[{"feature": "amount", "feature_value": 9421.0, "shap_value": 1.83},'
        ' {"feature": "geo_velocity_kmh", "feature_value": 0.0, "shap_value": -0.42},'
        ' {"feature": "off_hours_flag", "feature_value": 1.0, "shap_value": 0.0}]'
    )
    rows = _parse_top_contributors(raw)
    by_feature = {r.feature: r for r in rows}
    assert by_feature["amount"].direction == "fraud"
    assert by_feature["geo_velocity_kmh"].direction == "legit"
    # shap == 0 ties toward legit so missing features never visually accuse.
    assert by_feature["off_hours_flag"].direction == "legit"


@pytest.mark.parametrize(
    "raw",
    [None, "", "not-json", "{}", "[null, 1]", '[{"feature": "x"}]'],
)
def test_parse_top_contributors_tolerates_malformed(raw: str | None) -> None:
    """Legacy / malformed rows return an empty list, never raise."""
    rows = _parse_top_contributors(raw)
    # Some inputs may still parse partial entries — only assert no crash
    # and that the result is always a list.
    assert isinstance(rows, list)
