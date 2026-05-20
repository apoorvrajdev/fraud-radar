"""Repository for the append-only audit log."""
import json
from typing import Any

from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog
from app.repositories.base import BaseRepository


class AuditRepository(BaseRepository[AuditLog]):
    """Append-only audit log writer.

    Reads are intentionally not provided here — analyst tooling uses raw
    queries. The repository's job is to make writes consistent.
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


audit_repository = AuditRepository()
