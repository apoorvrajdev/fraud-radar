"""Repository for Customer queries."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.customer import Customer
from app.repositories.base import BaseRepository


class CustomerRepository(BaseRepository[Customer]):
    """Data access methods for Customer entities."""

    model = Customer

    def get_by_email(self, db: Session, email: str) -> Customer | None:
        """Look up a customer by their unique email."""
        stmt = select(Customer).where(Customer.email == email)
        return db.execute(stmt).scalar_one_or_none()

    def count_by_risk_tier(self, db: Session, tier: str) -> int:
        """Count customers in a given risk tier."""
        stmt = select(Customer).where(Customer.risk_tier == tier)
        return len(list(db.execute(stmt).scalars().all()))


customer_repository = CustomerRepository()
