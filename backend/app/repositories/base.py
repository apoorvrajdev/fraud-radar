"""Generic base repository providing common CRUD operations."""
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Generic repository with common CRUD operations.

    Concrete repositories inherit from this and add domain-specific queries.
    """

    model: type[ModelT]

    def get(self, db: Session, id: str) -> ModelT | None:
        """Fetch a single record by primary key, or None if not found."""
        return db.get(self.model, id)

    def list_all(
        self, db: Session, *, limit: int = 100, offset: int = 0
    ) -> list[ModelT]:
        """List all records with simple pagination."""
        stmt = select(self.model).limit(limit).offset(offset)
        return list(db.execute(stmt).scalars().all())

    def add(self, db: Session, instance: ModelT) -> ModelT:
        """Add an instance to the session and flush so it gets an ID."""
        db.add(instance)
        db.flush()
        return instance

    def delete(self, db: Session, instance: ModelT) -> None:
        """Remove an instance from the database."""
        db.delete(instance)
        db.flush()
