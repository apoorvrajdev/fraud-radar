"""Unit tests for the Phase 3A rules engine.

Every test constructs in-memory ORM instances via `_make_*` helpers — no DB,
no SQLAlchemy session. Rules read their inputs from the frozen
TransactionContext, so the tests prove behaviour without any I/O.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.fraud.rules import (
    HIGH_RISK_COUNTRIES,
    Severity,
    evaluate_all,
    rule_amount_ceiling,
    rule_dormant_account_high_value,
    rule_geo_velocity_impossible,
    rule_high_risk_country,
    rule_off_hours_high_value,
    rule_velocity_burst,
)
from app.fraud.transaction_context import TransactionContext
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction


# Reference instant; deltas are easy to reason about relative to a fixed point.
NOW = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transaction(
    *,
    id: str = "tx-current",
    customer_id: str = "cust-1",
    merchant_id: str = "merch-1",
    amount: Decimal = Decimal("100.00"),
    country: str = "US",
    is_card_present: bool = True,
    created_at: datetime = NOW,
    currency: str = "USD",
    status: str = "APPROVED",
    payment_method: str = "CARD",
    idempotency_key: str = "key-1",
) -> Transaction:
    tx = Transaction()
    tx.id = id
    tx.customer_id = customer_id
    tx.merchant_id = merchant_id
    tx.amount = amount
    tx.country = country
    tx.is_card_present = is_card_present
    tx.created_at = created_at
    tx.currency = currency
    tx.status = status
    tx.payment_method = payment_method
    tx.idempotency_key = idempotency_key
    return tx


def _make_customer(
    *,
    id: str = "cust-1",
    country: str = "US",
    risk_tier: str = "LOW",
    account_age_days: int = 365,
    created_at: datetime | None = None,
) -> Customer:
    c = Customer()
    c.id = id
    c.email = f"{id}@example.com"
    c.full_name = "Test Customer"
    c.country = country
    c.risk_tier = risk_tier
    c.account_age_days = account_age_days
    c.created_at = (
        created_at if created_at is not None
        else NOW - timedelta(days=account_age_days)
    )
    return c


def _make_merchant(
    *,
    id: str = "merch-1",
    category: str = "RETAIL",
    mcc: str = "5311",
    country: str = "US",
    risk_rating: str = "LOW",
    created_at: datetime | None = None,
) -> Merchant:
    m = Merchant()
    m.id = id
    m.name = "Test Merchant"
    m.category = category
    m.mcc = mcc
    m.country = country
    m.risk_rating = risk_rating
    m.created_at = created_at if created_at is not None else NOW - timedelta(days=365)
    return m


def make_context(
    *,
    transaction: Transaction | None = None,
    customer: Customer | None = None,
    merchant: Merchant | None = None,
    recent_transactions: list[Transaction] | None = None,
) -> TransactionContext:
    return TransactionContext(
        transaction=transaction or _make_transaction(),
        customer=customer or _make_customer(),
        merchant=merchant or _make_merchant(),
        recent_transactions=recent_transactions or [],
    )


# ---------------------------------------------------------------------------
# rule_velocity_burst
# ---------------------------------------------------------------------------


def test_velocity_burst_triggers_on_3_in_120s() -> None:
    recent = [
        _make_transaction(id="tx-r1", created_at=NOW - timedelta(seconds=60)),
        _make_transaction(id="tx-r2", created_at=NOW - timedelta(seconds=120)),
    ]
    ctx = make_context(recent_transactions=recent)
    result = rule_velocity_burst(ctx)
    assert result.triggered is True
    assert result.severity == Severity.HARD_BLOCK
    assert "3 transactions" in (result.reason or "")
    assert "120s" in (result.reason or "")


def test_velocity_burst_does_not_trigger_on_2_in_120s() -> None:
    recent = [
        _make_transaction(id="tx-r1", created_at=NOW - timedelta(seconds=60)),
    ]
    ctx = make_context(recent_transactions=recent)
    result = rule_velocity_burst(ctx)
    assert result.triggered is False
    assert result.reason is None


def test_velocity_burst_boundary_3_in_121s_does_not_trigger() -> None:
    """One recent tx inside the 120s window, one just outside — total = 2 in window."""
    recent = [
        _make_transaction(id="tx-r1", created_at=NOW - timedelta(seconds=60)),
        _make_transaction(id="tx-r2", created_at=NOW - timedelta(seconds=121)),
    ]
    ctx = make_context(recent_transactions=recent)
    result = rule_velocity_burst(ctx)
    assert result.triggered is False


def test_velocity_burst_ignores_transactions_after_current() -> None:
    """Future-stamped transactions in recent_transactions must not count."""
    recent = [
        _make_transaction(id="tx-r1", created_at=NOW - timedelta(seconds=30)),
        _make_transaction(id="tx-r2", created_at=NOW + timedelta(seconds=30)),
        _make_transaction(id="tx-r3", created_at=NOW + timedelta(seconds=60)),
    ]
    ctx = make_context(recent_transactions=recent)
    result = rule_velocity_burst(ctx)
    # Only tx-r1 sits strictly before NOW → 1 in window + 1 current = 2.
    assert result.triggered is False


# ---------------------------------------------------------------------------
# rule_geo_velocity_impossible
# ---------------------------------------------------------------------------


def test_geo_velocity_triggers_on_different_country_30min_ago() -> None:
    recent = [
        _make_transaction(
            id="tx-r1", country="GB",
            created_at=NOW - timedelta(minutes=30),
        ),
    ]
    ctx = make_context(recent_transactions=recent)
    result = rule_geo_velocity_impossible(ctx)
    assert result.triggered is True
    assert result.severity == Severity.HARD_BLOCK
    assert "GB" in (result.reason or "")
    assert "30min ago" in (result.reason or "")


def test_geo_velocity_no_trigger_on_same_country_30min_ago() -> None:
    recent = [
        _make_transaction(
            id="tx-r1", country="US",
            created_at=NOW - timedelta(minutes=30),
        ),
    ]
    ctx = make_context(recent_transactions=recent)
    result = rule_geo_velocity_impossible(ctx)
    assert result.triggered is False


def test_geo_velocity_no_trigger_on_different_country_61min_ago() -> None:
    recent = [
        _make_transaction(
            id="tx-r1", country="GB",
            created_at=NOW - timedelta(minutes=61),
        ),
    ]
    ctx = make_context(recent_transactions=recent)
    result = rule_geo_velocity_impossible(ctx)
    assert result.triggered is False


# ---------------------------------------------------------------------------
# rule_amount_ceiling
# ---------------------------------------------------------------------------


def test_amount_ceiling_triggers_at_5000_01() -> None:
    tx = _make_transaction(amount=Decimal("5000.01"))
    ctx = make_context(transaction=tx)
    result = rule_amount_ceiling(ctx)
    assert result.triggered is True
    assert result.severity == Severity.REVIEW
    assert "5000.01" in (result.reason or "")


def test_amount_ceiling_does_not_trigger_at_5000_exact() -> None:
    """Threshold is strict >; 5000.00 must not trigger."""
    tx = _make_transaction(amount=Decimal("5000.00"))
    ctx = make_context(transaction=tx)
    result = rule_amount_ceiling(ctx)
    assert result.triggered is False


def test_amount_ceiling_uses_decimal_precision() -> None:
    """Sub-cent Decimal values are honoured (no float coercion)."""
    tx = _make_transaction(amount=Decimal("5000.0001"))
    ctx = make_context(transaction=tx)
    result = rule_amount_ceiling(ctx)
    assert result.triggered is True


# ---------------------------------------------------------------------------
# rule_high_risk_country
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("country", sorted(HIGH_RISK_COUNTRIES))
def test_high_risk_country_triggers_for_each_listed_country(country: str) -> None:
    tx = _make_transaction(country=country, amount=Decimal("600.00"))
    ctx = make_context(transaction=tx)
    result = rule_high_risk_country(ctx)
    assert result.triggered is True
    assert result.severity == Severity.REVIEW
    assert country in (result.reason or "")


def test_high_risk_country_does_not_trigger_below_amount_threshold() -> None:
    tx = _make_transaction(country="RU", amount=Decimal("400.00"))
    ctx = make_context(transaction=tx)
    result = rule_high_risk_country(ctx)
    assert result.triggered is False


def test_high_risk_country_does_not_trigger_for_us_even_at_high_amount() -> None:
    tx = _make_transaction(country="US", amount=Decimal("50000.00"))
    ctx = make_context(transaction=tx)
    result = rule_high_risk_country(ctx)
    assert result.triggered is False


# ---------------------------------------------------------------------------
# rule_dormant_account_high_value
# ---------------------------------------------------------------------------


def test_dormant_account_triggers_on_old_account_no_recent_tx_high_amount() -> None:
    customer = _make_customer(
        created_at=NOW - timedelta(days=200),
        account_age_days=200,
    )
    # A prior tx 200 days ago — outside the 180-day gap window.
    recent = [
        _make_transaction(
            id="tx-r1", amount=Decimal("50.00"),
            created_at=NOW - timedelta(days=200),
        ),
    ]
    tx = _make_transaction(amount=Decimal("1500.00"))
    ctx = make_context(
        transaction=tx, customer=customer, recent_transactions=recent,
    )
    result = rule_dormant_account_high_value(ctx)
    assert result.triggered is True
    assert result.severity == Severity.REVIEW
    assert "dormant" in (result.reason or "")
    assert "1500" in (result.reason or "")


def test_dormant_account_no_trigger_when_recent_tx_within_180d() -> None:
    customer = _make_customer(
        created_at=NOW - timedelta(days=200),
        account_age_days=200,
    )
    recent = [
        _make_transaction(
            id="tx-r1", amount=Decimal("50.00"),
            created_at=NOW - timedelta(days=30),
        ),
    ]
    tx = _make_transaction(amount=Decimal("1500.00"))
    ctx = make_context(
        transaction=tx, customer=customer, recent_transactions=recent,
    )
    result = rule_dormant_account_high_value(ctx)
    assert result.triggered is False


def test_dormant_account_no_trigger_when_account_too_young() -> None:
    customer = _make_customer(
        created_at=NOW - timedelta(days=50),
        account_age_days=50,
    )
    tx = _make_transaction(amount=Decimal("1500.00"))
    ctx = make_context(transaction=tx, customer=customer, recent_transactions=[])
    result = rule_dormant_account_high_value(ctx)
    assert result.triggered is False


def test_dormant_account_no_trigger_below_amount_threshold() -> None:
    customer = _make_customer(
        created_at=NOW - timedelta(days=200),
        account_age_days=200,
    )
    tx = _make_transaction(amount=Decimal("900.00"))
    ctx = make_context(transaction=tx, customer=customer, recent_transactions=[])
    result = rule_dormant_account_high_value(ctx)
    assert result.triggered is False


def test_dormant_account_uses_transaction_time_as_now() -> None:
    """The rule must reference `ctx.transaction.created_at`, not `datetime.now()`.

    Build a context where the transaction is in year 2000 and the customer
    was created 30 days before the transaction. From the rule's perspective
    (using ctx as 'now'), the account is 30 days old → no trigger.

    If the rule reached for wall-clock `datetime.now()`, the account would
    appear ~25 years old and the rule would fire. Asserting no-trigger here
    proves the determinism property.
    """
    transaction_time = datetime(2000, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    customer = _make_customer(
        created_at=transaction_time - timedelta(days=30),
        account_age_days=30,
    )
    tx = _make_transaction(
        amount=Decimal("1500.00"),
        created_at=transaction_time,
    )
    ctx = make_context(transaction=tx, customer=customer, recent_transactions=[])
    result = rule_dormant_account_high_value(ctx)
    assert result.triggered is False


# ---------------------------------------------------------------------------
# rule_off_hours_high_value
# ---------------------------------------------------------------------------


def test_off_hours_triggers_at_3am_600_card_not_present() -> None:
    at = NOW.replace(hour=3, minute=15)
    tx = _make_transaction(
        amount=Decimal("600.00"),
        is_card_present=False,
        created_at=at,
    )
    ctx = make_context(transaction=tx)
    result = rule_off_hours_high_value(ctx)
    assert result.triggered is True
    assert result.severity == Severity.REVIEW
    assert "3:00" in (result.reason or "")


def test_off_hours_no_trigger_when_card_present() -> None:
    at = NOW.replace(hour=3, minute=15)
    tx = _make_transaction(
        amount=Decimal("600.00"),
        is_card_present=True,
        created_at=at,
    )
    ctx = make_context(transaction=tx)
    result = rule_off_hours_high_value(ctx)
    assert result.triggered is False


def test_off_hours_no_trigger_at_2pm() -> None:
    at = NOW.replace(hour=14, minute=0)
    tx = _make_transaction(
        amount=Decimal("600.00"),
        is_card_present=False,
        created_at=at,
    )
    ctx = make_context(transaction=tx)
    result = rule_off_hours_high_value(ctx)
    assert result.triggered is False


def test_off_hours_no_trigger_below_amount() -> None:
    at = NOW.replace(hour=3, minute=15)
    tx = _make_transaction(
        amount=Decimal("400.00"),
        is_card_present=False,
        created_at=at,
    )
    ctx = make_context(transaction=tx)
    result = rule_off_hours_high_value(ctx)
    assert result.triggered is False


@pytest.mark.parametrize("hour", [2, 3, 4, 5])
def test_off_hours_triggers_for_each_in_window(hour: int) -> None:
    at = NOW.replace(hour=hour, minute=15)
    tx = _make_transaction(
        amount=Decimal("600.00"),
        is_card_present=False,
        created_at=at,
    )
    ctx = make_context(transaction=tx)
    result = rule_off_hours_high_value(ctx)
    assert result.triggered is True


@pytest.mark.parametrize("hour", [1, 6])
def test_off_hours_does_not_trigger_at_boundary_hours(hour: int) -> None:
    at = NOW.replace(hour=hour, minute=15)
    tx = _make_transaction(
        amount=Decimal("600.00"),
        is_card_present=False,
        created_at=at,
    )
    ctx = make_context(transaction=tx)
    result = rule_off_hours_high_value(ctx)
    assert result.triggered is False


# ---------------------------------------------------------------------------
# evaluate_all
# ---------------------------------------------------------------------------


def test_evaluate_all_returns_six_results() -> None:
    ctx = make_context()
    results = evaluate_all(ctx)
    assert len(results) == 6


def test_evaluate_all_order_is_documented() -> None:
    ctx = make_context()
    results = evaluate_all(ctx)
    assert [r.rule_name for r in results] == [
        "velocity_burst",
        "geo_velocity_impossible",
        "amount_ceiling",
        "high_risk_country",
        "dormant_account_high_value",
        "off_hours_high_value",
    ]


def test_evaluate_all_clean_transaction_triggers_nothing() -> None:
    """Default context: 365-day-old US customer, US merchant, $100 noon card-present."""
    ctx = make_context()
    results = evaluate_all(ctx)
    assert all(r.triggered is False for r in results)
    assert all(r.reason is None for r in results)
