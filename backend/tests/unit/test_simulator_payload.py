"""Unit tests for the Phase 3D simulator payload generators.

Pure-function tests: no DB session, no HTTP client, no mocks. Every test
seeds Python's `random` module so payload generation is deterministic.
"""
from __future__ import annotations

import random
from decimal import Decimal

import pytest

from app.schemas.transaction import TransactionCreate
from app.simulator.main import (
    _FRAUD_PATTERNS,
    _HIGH_RISK_COUNTRIES,
    _build_payload,
    _generate_clean_payload,
    _generate_fraud_payload,
)


# Sample IDs matching the 36-char Pydantic constraint on the schema.
_CUSTOMER_IDS = [
    "10000000-0000-4000-8000-00000000000a",
    "10000000-0000-4000-8000-00000000000b",
    "10000000-0000-4000-8000-00000000000c",
]
_MERCHANT_IDS = [
    "20000000-0000-4000-8000-00000000000a",
    "20000000-0000-4000-8000-00000000000b",
]


def test_clean_payload_passes_pydantic_validation() -> None:
    """The clean generator must produce a body the TransactionCreate
    schema accepts — otherwise every clean POST would 422 in real
    production use."""
    random.seed(42)
    payload = _generate_clean_payload(_CUSTOMER_IDS[0], _MERCHANT_IDS[0])
    parsed = TransactionCreate.model_validate(payload)
    assert parsed.customer_id == _CUSTOMER_IDS[0]
    assert parsed.merchant_id == _MERCHANT_IDS[0]


def test_clean_payload_amount_stays_under_5000() -> None:
    """The amount_ceiling rule fires at $5,000. A 'clean' payload that
    crosses that line is a generator bug — it'd mark a clean transaction
    as REVIEW in the log and confuse the operator."""
    random.seed(42)
    for _ in range(100):
        payload = _generate_clean_payload(_CUSTOMER_IDS[0], _MERCHANT_IDS[0])
        assert Decimal(payload["amount"]) < Decimal("5000.00"), payload


def test_clean_payload_country_is_not_high_risk() -> None:
    """The high_risk_country rule fires when the country is in
    {RU, CN, NG, RO, VE, ID} and the amount is over $500. The clean
    generator must avoid every member of that set to keep its label honest."""
    random.seed(42)
    for _ in range(100):
        payload = _generate_clean_payload(_CUSTOMER_IDS[0], _MERCHANT_IDS[0])
        assert payload["country"] not in _HIGH_RISK_COUNTRIES, payload


@pytest.mark.parametrize("pattern", list(_FRAUD_PATTERNS))
def test_fraud_payload_passes_pydantic_validation_for_each_pattern(
    pattern: str,
) -> None:
    """Every fraud pattern must also produce a schema-valid body —
    otherwise that pattern would never reach the scoring service."""
    random.seed(42)
    payload = _generate_fraud_payload(_CUSTOMER_IDS[0], _MERCHANT_IDS[0], pattern)
    TransactionCreate.model_validate(payload)  # raises on failure


def test_build_payload_respects_fraud_rate_over_large_sample() -> None:
    """At fraud_rate=0.30, ~30% of 1000 samples should be non-clean.

    Loose binomial bound: σ ≈ 14.5 → 95% CI is ~[271, 329]. The asserted
    range [200, 400] gives ~7σ slack so the test never flakes.
    """
    random.seed(42)
    non_clean = 0
    for _ in range(1000):
        _payload, label = _build_payload(
            _CUSTOMER_IDS, _MERCHANT_IDS, fraud_rate=0.30,
        )
        if label != "clean":
            non_clean += 1
    assert 200 <= non_clean <= 400, non_clean


def test_build_payload_uses_pool_ids_only() -> None:
    """The simulator must never reference an ID outside the pool it
    loaded at startup — every request would otherwise FK-violate."""
    random.seed(42)
    customer_pool = set(_CUSTOMER_IDS)
    merchant_pool = set(_MERCHANT_IDS)
    for _ in range(50):
        payload, _label = _build_payload(
            _CUSTOMER_IDS, _MERCHANT_IDS, fraud_rate=0.25,
        )
        assert payload["customer_id"] in customer_pool
        assert payload["merchant_id"] in merchant_pool
