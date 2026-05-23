"""Audit log ORM model — append-only record of all state changes."""
from sqlalchemy import Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class AuditLog(Base, TimestampMixin):
    """Append-only audit trail.

    Every fraud decision, analyst override, and state change writes a row
    here. This table is intentionally not editable from the application
    layer — production systems would enforce this with database-level
    permissions.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index(
            "ix_audit_resource", "resource_type", "resource_id"
        ),
        Index("ix_audit_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog action={self.action} "
            f"resource={self.resource_type}:{self.resource_id}>"
        )
