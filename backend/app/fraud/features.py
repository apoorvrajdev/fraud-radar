"""Feature extraction from raw transactions.

The FeatureExtractor reads from a Session to compute customer-aware
velocity and history features. At inference time, it's called for one
transaction; at training time, the batch generator iterates the dataset
and calls it per row.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.fraud.feature_spec import FEATURE_NAMES, N_FEATURES
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction


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
    ) -> FeatureRow:
        """Compute the full feature vector for a single transaction.

        Pass customer/merchant if already loaded (avoids extra DB hits).
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
            float(self._tx_count_within(db, tx, timedelta(hours=1))),
            float(self._tx_count_within(db, tx, timedelta(hours=24))),
            self._log_amount_sum_within(db, tx, timedelta(hours=24)),
            # Customer history
            float(customer.account_age_days or 0),
            _RISK_TIER_MAP.get(customer.risk_tier or "LOW", 0.0),
            *self._customer_amount_stats(db, tx),
            float(self._days_since_last_tx(db, tx)),
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
    def _tx_count_within(db: Session, tx: Transaction, window: timedelta) -> int:
        """Count of this customer's transactions before `tx` within `window`."""
        cutoff = tx.created_at - window
        stmt = select(Transaction).where(
            and_(
                Transaction.customer_id == tx.customer_id,
                Transaction.created_at >= cutoff,
                Transaction.created_at < tx.created_at,
            )
        )
        return len(list(db.execute(stmt).scalars().all()))

    @staticmethod
    def _log_amount_sum_within(
        db: Session, tx: Transaction, window: timedelta
    ) -> float:
        """log1p of the total amount in the window — compresses scale."""
        cutoff = tx.created_at - window
        stmt = select(Transaction.amount).where(
            and_(
                Transaction.customer_id == tx.customer_id,
                Transaction.created_at >= cutoff,
                Transaction.created_at < tx.created_at,
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
        db: Session, tx: Transaction
    ) -> tuple[float, float]:
        """Return (avg_amount_30d, amount_zscore_30d).

        z-score: how many std-deviations above/below customer's 30-day mean.
        Captures the amount_anomaly fraud pattern directly.
        """
        cutoff = tx.created_at - timedelta(days=30)
        stmt = select(Transaction.amount).where(
            and_(
                Transaction.customer_id == tx.customer_id,
                Transaction.created_at >= cutoff,
                Transaction.created_at < tx.created_at,
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
    def _days_since_last_tx(db: Session, tx: Transaction) -> int:
        """Days since this customer's previous transaction.

        Captures the account_takeover pattern: long dormancy then high value.
        Returns 999 if no prior transaction exists (effectively "very dormant").
        """
        stmt = (
            select(Transaction.created_at)
            .where(
                and_(
                    Transaction.customer_id == tx.customer_id,
                    Transaction.created_at < tx.created_at,
                )
            )
            .order_by(Transaction.created_at.desc())
            .limit(1)
        )
        last_ts = db.execute(stmt).scalar_one_or_none()
        if last_ts is None:
            return 999
        return (tx.created_at - last_ts).days


# Module-level convenience function -------------------------------------------

_default_extractor = FeatureExtractor()


def extract_features(
    db: Session,
    tx: Transaction,
    *,
    customer: Customer | None = None,
    merchant: Merchant | None = None,
) -> FeatureRow:
    """Convenience wrapper using a shared FeatureExtractor instance."""
    return _default_extractor.extract(db, tx, customer=customer, merchant=merchant)
