"""Fraud rules engine — pure functions over a TransactionContext.

Rules are pure: same input always produces the same RuleResult. No database
queries, no `datetime.now()` calls, no I/O of any kind. The caller is
responsible for loading the TransactionContext; rules just read from it.
That keeps every rule deterministic and replayable, which matters for
both the audit log and the unit tests.

This module does NOT decide APPROVE / REVIEW / DECLINE — that's the
scoring service's job in Phase 3C. Each rule returns a RuleResult
describing whether it triggered, its severity (HARD_BLOCK or REVIEW),
and a human-readable reason for the audit log and API response.

Hard rules short-circuit the pipeline → DECLINE. Review rules are
advisory: the scoring service combines them with the model output
following the decision matrix in `docs/adr/PHASE_3_DESIGN.md`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from enum import Enum

from app.fraud.transaction_context import TransactionContext


HIGH_RISK_COUNTRIES: frozenset[str] = frozenset({"RU", "CN", "NG", "RO", "VE", "ID"})


class Severity(str, Enum):
    """How a rule trigger should be interpreted by the scoring service."""

    HARD_BLOCK = "HARD_BLOCK"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class RuleResult:
    """The outcome of evaluating one rule against one transaction.

    `reason` is None when the rule did not trigger, and a short human-readable
    string when it did — suitable for the audit log and the API response.
    """

    rule_name: str
    triggered: bool
    severity: Severity
    reason: str | None


# ---------------------------------------------------------------------------
# Internal thresholds — kept at module level so the rule bodies stay terse.
# ---------------------------------------------------------------------------

_VELOCITY_WINDOW = timedelta(seconds=120)
_VELOCITY_THRESHOLD = 3

_GEO_WINDOW = timedelta(minutes=60)

_AMOUNT_CEILING = Decimal("5000.00")

_HIGH_RISK_AMOUNT = Decimal("500.00")

_DORMANT_AGE = timedelta(days=180)
_DORMANT_GAP = timedelta(days=180)
_DORMANT_AMOUNT = Decimal("1000.00")

_OFF_HOURS: frozenset[int] = frozenset({2, 3, 4, 5})
_OFF_HOURS_AMOUNT = Decimal("500.00")


# ---------------------------------------------------------------------------
# Hard rules — trigger ⇒ DECLINE, ML scorer never runs
# ---------------------------------------------------------------------------


def rule_velocity_burst(ctx: TransactionContext) -> RuleResult:
    """≥3 transactions from the same customer in a 120-second window.

    Counts the current transaction plus prior transactions in
    `recent_transactions` whose `created_at` falls strictly before the
    current one and within the last 120 seconds.
    """
    name = "velocity_burst"
    cutoff = ctx.transaction.created_at - _VELOCITY_WINDOW

    in_window = [
        tx for tx in ctx.recent_transactions
        if tx.created_at < ctx.transaction.created_at
        and tx.created_at >= cutoff
    ]
    total = len(in_window) + 1  # +1 for the current transaction

    if total >= _VELOCITY_THRESHOLD:
        return RuleResult(
            rule_name=name,
            triggered=True,
            severity=Severity.HARD_BLOCK,
            reason=f"{total} transactions in 120s (threshold: 3 in 120s)",
        )
    return RuleResult(name, False, Severity.HARD_BLOCK, None)


def rule_geo_velocity_impossible(ctx: TransactionContext) -> RuleResult:
    """Same customer in two distinct countries within 60 minutes."""
    name = "geo_velocity_impossible"
    cutoff = ctx.transaction.created_at - _GEO_WINDOW
    current_country = ctx.transaction.country

    mismatched = [
        tx for tx in ctx.recent_transactions
        if tx.created_at < ctx.transaction.created_at
        and tx.created_at >= cutoff
        and tx.country != current_country
    ]

    if mismatched:
        # Pick the most recent mismatched transaction for the reason text.
        # `recent_transactions` is sorted descending by convention, but the
        # list comprehension above doesn't guarantee it preserved that
        # order, so be explicit.
        most_recent = max(mismatched, key=lambda tx: tx.created_at)
        minutes_ago = int(
            (ctx.transaction.created_at - most_recent.created_at).total_seconds() // 60
        )
        return RuleResult(
            rule_name=name,
            triggered=True,
            severity=Severity.HARD_BLOCK,
            reason=(
                f"country mismatch: {current_country} now, "
                f"{most_recent.country} {minutes_ago}min ago"
            ),
        )
    return RuleResult(name, False, Severity.HARD_BLOCK, None)


# ---------------------------------------------------------------------------
# Review rules — advisory; combined with model output per the design doc
# ---------------------------------------------------------------------------


def rule_amount_ceiling(ctx: TransactionContext) -> RuleResult:
    """Transaction amount exceeds the $5,000 ceiling (strict >)."""
    name = "amount_ceiling"
    amount = ctx.transaction.amount
    if amount > _AMOUNT_CEILING:
        return RuleResult(
            rule_name=name,
            triggered=True,
            severity=Severity.REVIEW,
            reason=f"amount ${amount} exceeds $5,000 ceiling",
        )
    return RuleResult(name, False, Severity.REVIEW, None)


def rule_high_risk_country(ctx: TransactionContext) -> RuleResult:
    """Transaction in a high-risk country with amount > $500."""
    name = "high_risk_country"
    country = ctx.transaction.country
    amount = ctx.transaction.amount
    if country in HIGH_RISK_COUNTRIES and amount > _HIGH_RISK_AMOUNT:
        return RuleResult(
            rule_name=name,
            triggered=True,
            severity=Severity.REVIEW,
            reason=f"high-risk country {country} with amount ${amount} > $500",
        )
    return RuleResult(name, False, Severity.REVIEW, None)


def rule_dormant_account_high_value(ctx: TransactionContext) -> RuleResult:
    """Dormant customer suddenly makes a high-value transaction.

    All three conditions must hold:
      * account age (`ctx.transaction.created_at - ctx.customer.created_at`)
        is greater than 180 days
      * no prior transaction in the past 180 days
      * current amount > $1,000

    "Now" is `ctx.transaction.created_at`, never wall-clock time — keeps the
    rule deterministic and replayable for tests and the audit log.
    """
    name = "dormant_account_high_value"
    now = ctx.transaction.created_at
    age = now - ctx.customer.created_at
    amount = ctx.transaction.amount

    if age <= _DORMANT_AGE:
        return RuleResult(name, False, Severity.REVIEW, None)
    if amount <= _DORMANT_AMOUNT:
        return RuleResult(name, False, Severity.REVIEW, None)

    recent_within_gap = [
        tx for tx in ctx.recent_transactions
        if tx.created_at < now and (now - tx.created_at) <= _DORMANT_GAP
    ]
    if recent_within_gap:
        return RuleResult(name, False, Severity.REVIEW, None)

    age_days = age.days
    if ctx.recent_transactions:
        last_tx = max(ctx.recent_transactions, key=lambda tx: tx.created_at)
        gap_days = (now - last_tx.created_at).days
        gap_clause = f"last tx {gap_days}d ago"
    else:
        gap_clause = "no prior transactions"

    return RuleResult(
        rule_name=name,
        triggered=True,
        severity=Severity.REVIEW,
        reason=(
            f"dormant account ({age_days}d old, {gap_clause}) "
            f"with high-value transaction ${amount}"
        ),
    )


def rule_off_hours_high_value(ctx: TransactionContext) -> RuleResult:
    """Card-not-present transaction at 2-5 AM with amount > $500.

    Hour is taken from `ctx.transaction.created_at.hour` in whatever timezone
    the timestamp carries — UTC for the synthetic dataset. Production would
    convert to the cardholder's local timezone before applying this rule.
    """
    name = "off_hours_high_value"
    hour = ctx.transaction.created_at.hour
    amount = ctx.transaction.amount
    is_card_present = ctx.transaction.is_card_present

    if (
        hour in _OFF_HOURS
        and amount > _OFF_HOURS_AMOUNT
        and not is_card_present
    ):
        return RuleResult(
            rule_name=name,
            triggered=True,
            severity=Severity.REVIEW,
            reason=(
                f"off-hours card-not-present transaction at {hour}:00 "
                f"with amount ${amount}"
            ),
        )
    return RuleResult(name, False, Severity.REVIEW, None)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def evaluate_all(ctx: TransactionContext) -> list[RuleResult]:
    """Run every rule and return all results, triggered or not.

    The caller (Phase 3C scoring service) filters and composes the
    decision per the matrix in `docs/adr/PHASE_3_DESIGN.md`. Order matters
    for testability and audit-log readability: hard rules first, then
    review rules.
    """
    return [
        rule_velocity_burst(ctx),
        rule_geo_velocity_impossible(ctx),
        rule_amount_ceiling(ctx),
        rule_high_risk_country(ctx),
        rule_dormant_account_high_value(ctx),
        rule_off_hours_high_value(ctx),
    ]
