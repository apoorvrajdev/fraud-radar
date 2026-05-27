"""Repository for the append-only audit log."""
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog
from app.repositories.base import BaseRepository


class AuditRepository(BaseRepository[AuditLog]):
    """Append-only audit log writer.

    Reads are intentionally narrow — surface the trailing entries for
    one resource (used by the Phase 3G detail page). Broader analyst
    tooling uses raw queries. The repository's job is to make writes
    consistent.
    """

    model = AuditLog

    def record(
        self,
        db: Session,
        *,
        actor: str,
        action: str,
        resource_type: str,
        resource_id: str,
        payload: dict[str, Any] | None = None,
    ) -> AuditLog:
        """Write a single audit log entry.

        The payload is serialized to JSON for storage in the Text column.
        """
        entry = AuditLog(
            actor=actor,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            payload=json.dumps(payload, default=str) if payload else None,
        )
        db.add(entry)
        db.flush()
        return entry

    def recent_for_resource(
        self,
        db: Session,
        *,
        resource_type: str,
        resource_id: str,
        limit: int = 20,
    ) -> list[AuditLog]:
        """Return the trailing N audit-log entries for one resource.

        Newest first (created_at DESC, id DESC) so the detail page can
        render the most-recent action at the top without a second sort.
        """
        stmt = (
            select(AuditLog)
            .where(AuditLog.resource_type == resource_type)
            .where(AuditLog.resource_id == resource_id)
            .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
            .limit(limit)
        )
        return list(db.execute(stmt).scalars().all())


audit_repository = AuditRepository()
