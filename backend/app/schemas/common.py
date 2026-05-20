"""Common types and enums shared across Pydantic schemas."""
from typing import Literal

# Enum-like literals matching the CHECK constraints in the ORM models
RiskTier = Literal["LOW", "MEDIUM", "HIGH"]
TransactionStatus = Literal["APPROVED", "DECLINED", "PENDING_REVIEW"]
FraudDecision = Literal["APPROVE", "REVIEW", "DECLINE"]
AnalystLabel = Literal["CONFIRMED_FRAUD", "CONFIRMED_LEGIT"]
PaymentMethod = Literal["CARD", "WALLET", "ACH"]
