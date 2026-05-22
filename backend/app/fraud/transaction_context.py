"""TransactionContext — the immutable input to every rule function.

Rules in `app.fraud.rules` are pure functions that take a TransactionContext
and return a RuleResult. The caller (Phase 3C scoring service) is responsible
for loading the underlying data; rules themselves perform no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction


@dataclass(frozen=True)
class TransactionContext:
    """Everything a rule function needs to evaluate a single transaction.

    Constructed once per request by the scoring service. Frozen so it is
    hashable and immutable — rules cannot accidentally mutate state shared
    with the caller, and the same context replayed later produces the same
    rule outcomes.

    Attributes:
        transaction: the transaction being scored.
        customer: the eager-loaded Customer record for `transaction.customer_id`.
        merchant: the eager-loaded Merchant record for `transaction.merchant_id`.
        recent_transactions: prior transactions for the same customer,
            sorted by `created_at` descending. The caller controls the
            lookback window; individual rules filter further as needed.
            Must NOT include `transaction` itself.
    """

    transaction: Transaction
    customer: Customer
    merchant: Merchant
    recent_transactions: list[Transaction] = field(default_factory=list)
