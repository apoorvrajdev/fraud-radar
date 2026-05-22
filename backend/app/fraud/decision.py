"""Decision — the final outcome of fraud scoring.

`PENDING` is Phase 3B scaffolding: the ingestion endpoint persists rows
before scoring lands, so something has to fill the `fraud_decision`
column in the meantime. Phase 3C replaces this with one of APPROVE,
REVIEW, or DECLINE on every scored transaction; PENDING then becomes
useful only as a transient state for rows that error mid-scoring and
need re-attempting.
"""
from __future__ import annotations

from enum import Enum


class Decision(str, Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"
    PENDING = "PENDING"
