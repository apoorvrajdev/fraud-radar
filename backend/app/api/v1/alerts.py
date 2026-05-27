"""Alerts queue endpoint (Phase 3H).

One read-only route. The write surface is reused from Phase 3G's
``POST /transactions/{id}/decision`` — submitting a verdict there
drops the row off the queue on the next refetch.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas.alerts import AlertsQuery, AlertsResponse
from app.services.alerts import list_alerts as list_alerts_service

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get(
    "",
    response_model=AlertsResponse,
    summary="List pending-review transactions with queue health summary",
)
def list_alerts(
    query: Annotated[AlertsQuery, Query()],
    db: Session = Depends(get_db),
) -> AlertsResponse:
    """Return a page of the analyst alerts queue plus a queue-wide
    summary block.

    Sort is fixed (``fraud_score DESC, created_at ASC, id ASC``) and
    not configurable — see ``docs/adr/PHASE_3H_DESIGN.md``. Malformed
    cursors map to 422.
    """
    try:
        return list_alerts_service(db, query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
