"""FxRate ORM model — the local cache of historical FX reference rates.

One row is one published rate for one currency pair on one date, and it
means: **1 unit of `quote` = `rate` units of `base`**. `base` is the
reporting currency (`FX_BASE_CURRENCY`); `quote` is the currency a
transaction was charged in. Conversion is therefore always a
multiplication — see `docs/FX_CONTRACT.md`.

The table is a cache, not a source of truth: it can be emptied at any
time and will refill from the provider. What it must never do is hand
back a rate that could not have applied to the transaction asking for
it, which is why every read is constrained to `rate_date <= on` in
`app/repositories/fx_rate.py`.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import TIMESTAMP, CheckConstraint, Date, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class FxRate(Base):
    """One cached FX reference rate for a (base, quote, date) triple."""

    __tablename__ = "fx_rates"

    # Composite primary key — a pair has exactly one rate per date.
    base: Mapped[str] = mapped_column(String(3), primary_key=True)
    quote: Mapped[str] = mapped_column(String(3), primary_key=True)
    rate_date: Mapped[date] = mapped_column(Date, primary_key=True)

    # Money-adjacent — NEVER a Float. Numeric(18, 8) hands back a Decimal.
    rate: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)

    # When this row was retrieved, not when the rate was published. The
    # publication date is `rate_date`.
    fetched_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )

    __table_args__ = (
        CheckConstraint("rate > 0", name="ck_fx_rates_rate_positive"),
        # A same-currency conversion is the identity path: rate 1, no
        # lookup, no cache read. A row for it would never be consulted,
        # so it is a corrupt write rather than a redundant one.
        CheckConstraint("base <> quote", name="ck_fx_rates_distinct_pair"),
    )

    def __repr__(self) -> str:
        return (
            f"<FxRate 1 {self.quote} = {self.rate} {self.base} "
            f"on {self.rate_date}>"
        )
