"""Service layer for the transactions list endpoint (Phase 3F).

Three responsibilities live here that the router and repository
deliberately do not own:

1. Cursor encoding — opaque base64(JSON) round-trip so the wire format
   stays decoupled from the SQL sort key.
2. Repository orchestration — translate the validated query model into
   a ``TransactionListFilters`` dataclass plus a decoded cursor tuple.
3. Response shaping — wrap the rows in the ``TransactionList`` envelope
   and re-emit the next cursor.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Final

from sqlalchemy.orm import Session

from app.repositories.transaction import (
    TransactionListFilters,
    transaction_repository,
)
from app.schemas.transaction import (
    TransactionList,
    TransactionListQuery,
    TransactionResponse,
)

_CURSOR_PADDING: Final = "="


def encode_cursor(ts: datetime, id_: str) -> str:
    """Pack a (created_at, id) keyset position into a urlsafe-base64 token."""
    payload = json.dumps({"ts": ts.isoformat(), "id": id_})
    raw = base64.urlsafe_b64encode(payload.encode()).decode()
    return raw.rstrip(_CURSOR_PADDING)


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    """Reverse of ``encode_cursor``. Raises ``ValueError`` on malformed input."""
    # Restore stripped base64 padding (length must be a multiple of 4).
    padded = cursor + _CURSOR_PADDING * (-len(cursor) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode()).decode()
        payload = json.loads(raw)
        ts = datetime.fromisoformat(payload["ts"])
        id_ = str(payload["id"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid cursor") from exc
    return ts, id_


def list_transactions(
    db: Session, query: TransactionListQuery
) -> TransactionList:
    """Return a paginated, filtered page of transactions."""
    cursor_tuple = decode_cursor(query.cursor) if query.cursor else None

    filters = TransactionListFilters(
        decision=query.decision,
        country=query.country,
        min_amount=query.min_amount,
        max_amount=query.max_amount,
        start_time=query.start_time,
        end_time=query.end_time,
        customer_id=query.customer_id,
        merchant_id=query.merchant_id,
    )

    rows, next_cursor_tuple = transaction_repository.list_paginated(
        db,
        filters=filters,
        limit=query.limit,
        cursor=cursor_tuple,
    )

    items = [TransactionResponse.model_validate(row) for row in rows]
    next_cursor = (
        encode_cursor(*next_cursor_tuple) if next_cursor_tuple else None
    )
    return TransactionList(
        items=items,
        next_cursor=next_cursor,
        has_more=next_cursor is not None,
    )
