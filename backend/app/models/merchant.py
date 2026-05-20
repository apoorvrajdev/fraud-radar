"""Merchant ORM model — entities that accept payments."""
from sqlalchemy import CheckConstraint, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class Merchant(Base, TimestampMixin):
    """A merchant that accepts transactions. Includes MCC code and risk rating."""

    __tablename__ = "merchants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    mcc: Mapped[str] = mapped_column(String(4), nullable=False)
    country: Mapped[str] = mapped_column(String(2), nullable=False)
    risk_rating: Mapped[str] = mapped_column(
        String(16), nullable=False, default="LOW"
    )

    __table_args__ = (
        CheckConstraint(
            "risk_rating IN ('LOW', 'MEDIUM', 'HIGH')",
            name="ck_merchants_risk_rating",
        ),
        CheckConstraint(
            "length(country) = 2", name="ck_merchants_country_iso2"
        ),
        CheckConstraint(
            "length(mcc) = 4", name="ck_merchants_mcc_length"
        ),
    )

    def __repr__(self) -> str:
        return f"<Merchant id={self.id} category={self.category}>"
