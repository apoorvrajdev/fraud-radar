"""SQLAlchemy declarative base for all ORM models."""
from datetime import datetime
from sqlalchemy import TIMESTAMP, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models in the application."""
    pass


class TimestampMixin:
    """Mixin providing created_at timestamp for tables that need it."""
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.current_timestamp(),
        nullable=False,
    )
