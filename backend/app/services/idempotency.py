"""Stripe-style idempotency cache primitives.

This module owns the hash + lookup + insert operations. The endpoint
layer (`app/api/v1/transactions.py`) decides what to do with the
results — return the cached response on a hit, raise 409 on a payload
mismatch, or write a fresh entry on a miss.

Concurrency handling — two requests with the same key arriving within
milliseconds — lives in the endpoint, not here. SQLite serialises
writes through its global lock; a portfolio-scope production-Postgres
implementation would use an upsert with RETURNING.

See `docs/adr/PHASE_3_DESIGN.md` "Idempotency design" for the full
semantics including the 24-hour TTL and the canonical-JSON rationale.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.idempotency_key import IdempotencyKey
from app.schemas.transaction import TransactionCreate

TTL = timedelta(hours=24)


def hash_request(payload: TransactionCreate) -> str:
    """SHA-256 over the Pydantic-normalised JSON representation.

    Hashing `model_dump_json()` (not the raw HTTP bytes) makes the hash
    stable across whitespace differences and field reordering by
    intermediaries. Pydantic v2 serialises fields in declaration order,
    which is deterministic for a given schema version.
    """
    canonical = payload.model_dump_json()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def lookup(session: Session, key: str) -> IdempotencyKey | None:
    """Return the cached entry for `key`, or None if absent or expired.

    Expired rows are filtered out at the SQL layer — callers do not need
    to check `expires_at` themselves.
    """
    now = datetime.now(timezone.utc)
    stmt = select(IdempotencyKey).where(
        IdempotencyKey.key == key,
        IdempotencyKey.expires_at > now,
    )
    return session.execute(stmt).scalar_one_or_none()


def store(
    session: Session,
    *,
    key: str,
    request_hash: str,
    transaction_id: str,
    response_body: str,
    status_code: int,
) -> IdempotencyKey:
    """Insert a new idempotency record with a 24-hour TTL. Commits the session.

    The commit is intentional: callers that have already staged related
    rows (e.g. the new Transaction itself) get an atomic flush here.
    """
    now = datetime.now(timezone.utc)
    entry = IdempotencyKey(
        key=key,
        request_hash=request_hash,
        transaction_id=transaction_id,
        response_body=response_body,
        status_code=status_code,
        created_at=now,
        expires_at=now + TTL,
    )
    session.add(entry)
    session.commit()
    return entry
