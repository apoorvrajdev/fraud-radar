"""Alerts-queue read service (Phase 3H).

Owns the cursor codec and the orchestration between the repository
aggregate and the per-row hydration. The router stays a one-liner
on top of this.

See `docs/adr/PHASE_3H_DESIGN.md` for the queue predicate, the
sort key, and the bucket boundaries.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Final

from sqlalchemy.orm import Session

from app.models.transaction import Transaction
from app.repositories.transaction import (
    AlertsListFilters,
    transaction_repository,
)
from app.schemas.alerts import (
    AlertItem,
    AlertsQuery,
    AlertsResponse,
    AlertsSummary,
)

_CURSOR_PADDING: Final = "="


def _as_utc(dt: datetime) -> datetime:
    """Coerce a possibly-naive datetime to tz-aware UTC.

    SQLite drops timezone info on round-trip even when the column
    is declared ``TIMESTAMP(timezone=True)``. The codebase writes
    UTC timestamps exclusively (scoring pipeline, simulator, test
    fixtures), so treating naive reads as UTC is correct and lets
    the service do arithmetic against ``datetime.now(timezone.utc)``
    without surprises.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def encode_alert_cursor(score: Decimal, ts: datetime, id_: str) -> str:
    """Pack a ``(fraud_score, created_at, id)`` keyset position into a
    urlsafe-base64 token.

    Mirrors the envelope shape of ``app.services.transactions.encode_cursor``
    but encodes a three-tuple because the alerts sort is
    ``fraud_score DESC, created_at ASC, id ASC`` — see the ADR.
    """
    payload = json.dumps(
        {"s": str(score), "ts": ts.isoformat(), "id": id_}
    )
    raw = base64.urlsafe_b64encode(payload.encode()).decode()
    return raw.rstrip(_CURSOR_PADDING)


def decode_alert_cursor(cursor: str) -> tuple[Decimal, datetime, str]:
    """Reverse of ``encode_alert_cursor``.

    Raises ``ValueError`` on malformed input so the router can map it
    to a clean 422.
    """
    padded = cursor + _CURSOR_PADDING * (-len(cursor) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode()).decode()
        payload = json.loads(raw)
        score = Decimal(str(payload["s"]))
        ts = datetime.fromisoformat(payload["ts"])
        id_ = str(payload["id"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid cursor") from exc
    return score, ts, id_


def _parse_rules(raw: str | None) -> list[str]:
    """Decode the persisted ``rules_triggered`` JSON column.

    Stored as a JSON-encoded list of strings; legacy nulls or empty
    strings collapse to an empty list. Bad JSON is treated as empty
    rather than fatal — the queue page is a read surface, never the
    place to fail an analyst's shift.
    """
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _to_item(row: Transaction, *, now: datetime) -> AlertItem:
    """Hydrate a Transaction ORM row into an AlertItem.

    ``age_seconds`` is computed at response time against the injected
    ``now`` so the value is stable for the duration of a single
    request and trivially mockable in tests.
    """
    # Defensive clamps: persisted rows are guaranteed by the queue
    # predicate to have a non-null fraud_score and a tz-aware created_at,
    # but the type system can't see that.
    score = row.fraud_score
    assert score is not None
    created_at = _as_utc(row.created_at)
    age = max(0, int((now - created_at).total_seconds()))
    return AlertItem(
        id=row.id,
        created_at=created_at,
        age_seconds=age,
        amount=row.amount,
        currency=row.currency,
        country=row.country,
        customer_id=row.customer_id,
        merchant_id=row.merchant_id,
        fraud_score=score,
        fraud_decision="REVIEW",
        rules_triggered=_parse_rules(row.rules_triggered),
    )


def list_alerts(
    db: Session,
    query: AlertsQuery,
    *,
    now: datetime | None = None,
) -> AlertsResponse:
    """Build a page of the analyst alerts queue.

    The summary block is computed against the *unfiltered* queue
    predicate so the header strip still reflects total queue health
    when the analyst narrows the visible page.
    """
    now = now or datetime.now(timezone.utc)

    pending_count, oldest, buckets = (
        transaction_repository.pending_review_summary(db)
    )
    oldest_seconds: int | None
    if oldest is None:
        oldest_seconds = None
    else:
        oldest_seconds = max(0, int((now - _as_utc(oldest)).total_seconds()))
    summary = AlertsSummary(
        pending_count=pending_count,
        oldest_pending_seconds=oldest_seconds,
        score_buckets=buckets,
    )

    cursor_tuple = (
        decode_alert_cursor(query.cursor) if query.cursor else None
    )

    # Translate the age-window parameters into absolute timestamps at
    # this service boundary so the repository stays clock-agnostic.
    max_created_at = (
        now - timedelta(seconds=query.min_age_seconds)
        if query.min_age_seconds is not None
        else None
    )
    min_created_at = (
        now - timedelta(seconds=query.max_age_seconds)
        if query.max_age_seconds is not None
        else None
    )

    filters = AlertsListFilters(
        min_score=query.min_score,
        country=query.country,
        min_created_at=min_created_at,
        max_created_at=max_created_at,
    )

    rows, next_cursor_tuple = transaction_repository.list_alerts_paginated(
        db,
        filters=filters,
        limit=query.limit,
        cursor=cursor_tuple,
        now=now,
    )

    next_cursor = (
        encode_alert_cursor(*next_cursor_tuple)
        if next_cursor_tuple is not None
        else None
    )

    return AlertsResponse(
        summary=summary,
        items=[_to_item(row, now=now) for row in rows],
        next_cursor=next_cursor,
        has_more=next_cursor is not None,
    )
