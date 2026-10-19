"""Stripe-style idempotency cache primitives.

This module owns the hash + lookup + resolve + insert operations. The
endpoint layer (`app/api/v1/transactions.py`) maps the results to HTTP:
replay on a hit, 409 on a conflict, a fresh entry on a miss.

Concurrency — two requests with the same key arriving within
milliseconds — is handled by the endpoint through `resolve`. Both
requests miss the lookup; the first insert wins; the loser's insert
trips a unique constraint (`transactions.uq_transactions_idempotency`
or the `idempotency_keys` primary key). The endpoint rolls back and
calls `resolve` again, which now finds the winner and returns it for
replay — the "loser re-reads" design from the ADR, keyed off the
constraint rather than a rowcount.

See `docs/adr/PHASE_3_DESIGN.md` "Idempotency design" for the full
semantics including the 24-hour TTL and the canonical-JSON rationale.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.idempotency_key import IdempotencyKey
from app.schemas.transaction import TransactionCreate

TTL = timedelta(hours=24)


class IdempotencyConflictError(Exception):
    """The key is already taken and the request cannot be served as a replay.

    Raised for a different payload under a known key, and for any reuse
    of an expired key. Mapped to 409 by the router; the message explains
    which so the client can tell "fix your body" from "mint a new key".
    """


def hash_request(payload: TransactionCreate) -> str:
    """SHA-256 over the Pydantic-normalised JSON representation.

    Hashing `model_dump_json()` (not the raw HTTP bytes) makes the hash
    stable across whitespace differences and field reordering by
    intermediaries. Pydantic v2 serialises fields in declaration order,
    which is deterministic for a given schema version.
    """
    canonical = payload.model_dump_json()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _as_utc(moment: datetime) -> datetime:
    """SQLite drops tzinfo on round-trip; every stored timestamp is UTC."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def lookup(
    session: Session, key: str, *, include_expired: bool = False,
) -> IdempotencyKey | None:
    """Return the cached entry for `key`, or None if absent.

    By default expired rows are filtered out at the SQL layer, so callers
    that only want a replayable entry need not check `expires_at`.
    `include_expired=True` returns the row regardless: an expired row
    still owns its key (the `idempotency_keys` primary key and the
    `(customer_id, idempotency_key)` constraint on `transactions` never
    expire), so a caller deciding whether a key can be *reused* must see
    it.
    """
    stmt = select(IdempotencyKey).where(IdempotencyKey.key == key)
    if not include_expired:
        stmt = stmt.where(IdempotencyKey.expires_at > datetime.now(UTC))
    return session.execute(stmt).scalar_one_or_none()


def resolve(
    session: Session,
    *,
    key: str,
    request_hash: str,
    now: datetime | None = None,
) -> IdempotencyKey | None:
    """Decide what `key` + `request_hash` means against the cache.

    Returns the entry to replay when the key is live and the payload
    matches, or None when the key has never been seen (the caller
    proceeds with a fresh insert). Raises `IdempotencyConflictError`
    when the key is taken but cannot be replayed:

    * the stored request hash differs — same key, different body;
    * the entry has expired — the cached response is no longer served,
      and the key cannot be recycled either because the row keeps its
      unique-constraint slot. The client must mint a new key.

    The expiry check comes first: a dead key is dead whatever the body.

    This is the single decision point for both the normal pre-insert
    lookup and the post-`IntegrityError` recovery after a lost race, so
    the two paths cannot drift.
    """
    now = datetime.now(UTC) if now is None else _as_utc(now)

    entry = lookup(session, key, include_expired=True)
    if entry is None:
        return None
    if _as_utc(entry.expires_at) <= now:
        # Name the row the key produced: a client that never received
        # the original response can still find what was scored instead
        # of guessing whether to resubmit.
        raise IdempotencyConflictError(
            "idempotency key expired: the cached response is no longer "
            "available and the key cannot be reused; the original request "
            f"was recorded as transaction {entry.transaction_id}; send a "
            "new key for a new transaction"
        )
    if entry.request_hash != request_hash:
        raise IdempotencyConflictError(
            "idempotency key reused with different payload"
        )
    return entry


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

    A plain insert, by design: if another request committed the same
    key first, the primary key raises `IntegrityError` here (or the
    transactions constraint raised earlier at flush) and the endpoint
    recovers through `resolve`.
    """
    now = datetime.now(UTC)
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
