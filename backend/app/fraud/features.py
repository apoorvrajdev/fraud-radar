"""Feature extraction from raw transactions.

The FeatureExtractor reads from a Session to compute customer-aware
velocity and history features. At inference time, it's called for one
transaction; at training time, the batch generator iterates the dataset
and calls it per row.

When the caller has already loaded the customer's recent transactions
(Phase 3C scoring service does this so the rules engine and the feature
extractor share one fetch), the optional `recent_transactions` keyword
on `extract()` and the four velocity/recency helpers lets them aggregate
in-Python instead of issuing per-call SELECT statements. Both paths
produce byte-identical `FeatureRow.values` for the same input data; see
`tests/unit/test_features_parity.py` for the safety net.

All `created_at` comparisons normalize to offset-aware UTC via
`_ensure_utc()` — SQLAlchemy returns naive datetimes from `DateTime`
columns on SQLite; the project convention is UTC throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.fraud.feature_spec import FEATURE_NAMES, N_FEATURES
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction


def _ensure_utc(ts: datetime) -> datetime:
    """Normalize a datetime to offset-aware UTC.

    SQLAlchemy's `DateTime` column returns offset-naive datetimes regardless
    of how they were inserted (SQLite has no native timezone type, so the
    tzinfo is dropped on round-trip). Recent-history comparisons mix
    freshly-constructed in-memory transactions (typically offset-aware UTC
    from `datetime.now(timezone.utc)`) with DB-loaded transactions (naive).
    Python refuses to compare or subtract across flavors, so we normalize
    everything to offset-aware UTC at every comparison point.

    Naive datetimes are assumed to represent UTC — matches the project
    convention (Phase 2A migration timestamps, the idempotency module,
    `datetime.now(timezone.utc)` everywhere else in the codebase).

    No-op when the input is already offset-aware.
    """
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


# Risk-tier encoding: lower is safer
_RISK_TIER_MAP: dict[str, float] = {
    "LOW": 0.0,
    "MEDIUM": 1.0,
    "HIGH": 2.0,
}

# Merchant categories considered high-risk for fraud
_HIGH_RISK_CATEGORIES: set[str] = {
    "JEWELRY",
    "MONEY_TRANSFER",
    "GAMBLING",
    "CRYPTO",
    "ELECTRONICS",  # high resale value
}


@dataclass
class FeatureRow:
    """A single feature vector with names attached for debuggability."""

    values: list[float]

    def as_dict(self) -> dict[str, float]:
        """Return as {feature_name: value} for logging or SHAP."""
        return dict(zip(FEATURE_NAMES, self.values, strict=True))


class FeatureExtractor:
    """Extract feature vectors from Transactions.

    Holds no per-request state — safe to instantiate once and reuse.
    All methods take a Session so the caller controls transaction lifecycle.
    """

    def extract(
        self,
        db: Session,
        tx: Transaction,
        *,
        customer: Customer | None = None,
        merchant: Merchant | None = None,
        recent_transactions: list[Transaction] | None = None,
    ) -> FeatureRow:
        """Compute the full feature vector for a single transaction.

        Pass customer/merchant if already loaded (avoids extra DB hits).

        Pass `recent_transactions` to skip the four velocity/recency SQL
        queries — useful when the caller has already loaded the customer's
        history for another purpose (e.g. the Phase 3C scoring service
        loads it once for both the rules engine and the feature extractor).
        The list must already be filtered to a single customer (this one)
        and must NOT include `tx` itself. When `recent_transactions` is
        None the helpers query the database as before.
        """
        if customer is None:
            customer = db.get(Customer, tx.customer_id)
        if merchant is None:
            merchant = db.get(Merchant, tx.merchant_id)

        if customer is None or merchant is None:
            raise ValueError(
                f"Cannot extract features: missing customer or merchant for tx {tx.id}"
            )

        values: list[float] = [
            # Transaction-level
            self._log_amount(tx.amount),
            float(tx.created_at.hour),
            float(self._is_weekend(tx.created_at)),
            float(self._is_off_hours(tx.created_at)),
            float(tx.is_card_present or False),
            # Geographic
            float(tx.country != customer.country),
            float(tx.country != merchant.country),
            # Velocity
            float(self._tx_count_within(db, tx, timedelta(hours=1), recent_transactions)),
            float(self._tx_count_within(db, tx, timedelta(hours=24), recent_transactions)),
            self._log_amount_sum_within(db, tx, timedelta(hours=24), recent_transactions),
            # Customer history
            float(customer.account_age_days or 0),
            _RISK_TIER_MAP.get(customer.risk_tier or "LOW", 0.0),
            *self._customer_amount_stats(db, tx, recent_transactions),
            float(self._days_since_last_tx(db, tx, recent_transactions)),
            # Merchant context
            _RISK_TIER_MAP.get(merchant.risk_rating or "LOW", 0.0),
            float(merchant.category in _HIGH_RISK_CATEGORIES),
        ]

        assert len(values) == N_FEATURES, (
            f"Feature vector length mismatch: got {len(values)}, expected {N_FEATURES}. "
            "FEATURE_NAMES and extract() must stay in sync."
        )
        return FeatureRow(values=values)

    # ------------------------------------------------------------------
    # Transaction-level features
    # ------------------------------------------------------------------

    @staticmethod
    def _log_amount(amount: Decimal | float) -> float:
        """log1p of amount compresses the long tail (small + huge amounts coexist)."""
        return float(math.log1p(float(amount)))

    @staticmethod
    def _is_weekend(ts: datetime) -> bool:
        """Saturday=5, Sunday=6."""
        return ts.weekday() >= 5

    @staticmethod
    def _is_off_hours(ts: datetime) -> bool:
        """2am to 5am inclusive — covers the off_hours fraud pattern."""
        return 2 <= ts.hour <= 5

    # ------------------------------------------------------------------
    # Velocity features
    # ------------------------------------------------------------------

    @staticmethod
    def _tx_count_within(
        db: Session,
        tx: Transaction,
        window: timedelta,
        recent_transactions: list[Transaction] | None = None,
    ) -> int:
        """Count of this customer's transactions before `tx` within `window`.

        When `recent_transactions` is provided, counts in-memory using the
        same `cutoff <= created_at < tx.created_at` predicate the SQL uses.
        """
        tx_ts = _ensure_utc(tx.created_at)
        cutoff = tx_ts - window
        if recent_transactions is not None:
            return sum(
                1 for t in recent_transactions
                if cutoff <= _ensure_utc(t.created_at) < tx_ts
            )
        stmt = select(Transaction).where(
            and_(
                Transaction.customer_id == tx.customer_id,
                Transaction.created_at >= cutoff,
                Transaction.created_at < tx_ts,
            )
        )
        return len(list(db.execute(stmt).scalars().all()))

    @staticmethod
    def _log_amount_sum_within(
        db: Session,
        tx: Transaction,
        window: timedelta,
        recent_transactions: list[Transaction] | None = None,
    ) -> float:
        """log1p of the total amount in the window — compresses scale.

        When `recent_transactions` is provided, sums amounts in-memory using
        the same `cutoff <= created_at < tx.created_at` predicate the SQL
        uses, then applies log1p — identical to the DB path.
        """
        tx_ts = _ensure_utc(tx.created_at)
        cutoff = tx_ts - window
        if recent_transactions is not None:
            amounts_iter = (
                float(t.amount) for t in recent_transactions
                if cutoff <= _ensure_utc(t.created_at) < tx_ts
            )
            total = sum(amounts_iter, start=0.0)
            return float(math.log1p(total))
        stmt = select(Transaction.amount).where(
            and_(
                Transaction.customer_id == tx.customer_id,
                Transaction.created_at >= cutoff,
                Transaction.created_at < tx_ts,
            )
        )
        amounts = list(db.execute(stmt).scalars().all())
        total = sum((float(a) for a in amounts), start=0.0)
        return float(math.log1p(total))

    # ------------------------------------------------------------------
    # Customer history features
    # ------------------------------------------------------------------

    @staticmethod
    def _customer_amount_stats(
        db: Session,
        tx: Transaction,
        recent_transactions: list[Transaction] | None = None,
    ) -> tuple[float, float]:
        """Return (avg_amount_30d, amount_zscore_30d).

        z-score: how many std-deviations above/below customer's 30-day mean.
        Captures the amount_anomaly fraud pattern directly.

        When `recent_transactions` is provided, filters the list with the
        same `cutoff <= created_at < tx.created_at` predicate the SQL uses
        and runs the identical mean/variance/z-score math (biased stdev,
        `(0.0, 0.0)` when fewer than two amounts are in the window).
        """
        tx_ts = _ensure_utc(tx.created_at)
        cutoff = tx_ts - timedelta(days=30)
        if recent_transactions is not None:
            amounts = [
                float(t.amount) for t in recent_transactions
                if cutoff <= _ensure_utc(t.created_at) < tx_ts
            ]
        else:
            stmt = select(Transaction.amount).where(
                and_(
                    Transaction.customer_id == tx.customer_id,
                    Transaction.created_at >= cutoff,
                    Transaction.created_at < tx_ts,
                )
            )
            amounts = [float(a) for a in db.execute(stmt).scalars().all()]
        if len(amounts) < 2:
            return (0.0, 0.0)

        mean = sum(amounts) / len(amounts)
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        stdev = math.sqrt(variance) if variance > 0 else 1.0
        zscore = (float(tx.amount) - mean) / stdev
        return (mean, zscore)

    @staticmethod
    def _days_since_last_tx(
        db: Session,
        tx: Transaction,
        recent_transactions: list[Transaction] | None = None,
    ) -> int:
        """Days since this customer's previous transaction.

        Captures the account_takeover pattern: long dormancy then high value.
        Returns 999 if no prior transaction exists (effectively "very dormant").

        When `recent_transactions` is provided, picks the most recent prior
        in-memory using the same `created_at < tx.created_at` predicate
        the SQL uses, with the same 999 fallback when no prior exists.
        """
        tx_ts = _ensure_utc(tx.created_at)
        if recent_transactions is not None:
            priors = []
            for t in recent_transactions:
                t_ts = _ensure_utc(t.created_at)
                if t_ts < tx_ts:
                    priors.append(t_ts)
            if not priors:
                return 999
            last_ts = max(priors)
            return (tx_ts - last_ts).days
        stmt = (
            select(Transaction.created_at)
            .where(
                and_(
                    Transaction.customer_id == tx.customer_id,
                    Transaction.created_at < tx_ts,
                )
            )
            .order_by(Transaction.created_at.desc())
            .limit(1)
        )
        last_ts = db.execute(stmt).scalar_one_or_none()
        if last_ts is None:
            return 999
        return (tx_ts - _ensure_utc(last_ts)).days


# Module-level convenience function -------------------------------------------

_default_extractor = FeatureExtractor()


def extract_features(
    db: Session,
    tx: Transaction,
    *,
    customer: Customer | None = None,
    merchant: Merchant | None = None,
    recent_transactions: list[Transaction] | None = None,
) -> FeatureRow:
    """Convenience wrapper using a shared FeatureExtractor instance.

    Forwards `recent_transactions` so callers that have already loaded the
    customer's history (Phase 3C scoring service) can skip the velocity
    SQL queries — see `FeatureExtractor.extract` for the contract.
    """
    return _default_extractor.extract(
        db, tx,
        customer=customer,
        merchant=merchant,
        recent_transactions=recent_transactions,
    )
