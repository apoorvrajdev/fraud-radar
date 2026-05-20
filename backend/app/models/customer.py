"""Customer ORM model — cardholders who initiate transactions."""
from sqlalchemy import CheckConstraint, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class Customer(Base, TimestampMixin):
    """A cardholder. Customers initiate transactions against merchants."""

    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    country: Mapped[str] = mapped_column(String(2), nullable=False)
    risk_tier: Mapped[str] = mapped_column(
        String(16), nullable=False, default="LOW"
    )
    account_age_days: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        CheckConstraint(
            "risk_tier IN ('LOW', 'MEDIUM', 'HIGH')",
            name="ck_customers_risk_tier",
        ),
        CheckConstraint(
            "length(country) = 2", name="ck_customers_country_iso2"
        ),
    )

    def __repr__(self) -> str:
        return f"<Customer id={self.id} tier={self.risk_tier}>"
