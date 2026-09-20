"""Pydantic schemas for dashboard aggregate endpoints.

Every money field below is denominated in the **reporting currency**
(``FX_BASE_CURRENCY``, USD), not in the currencies the underlying
transactions were charged in. A row that could not be converted
contributes its original amount rather than being dropped, so a total is
"as close to the reporting currency as the FX data allowed" rather than
either a silent currency mix or an under-count. Per-transaction
currencies stay on the transaction endpoints, where they belong.

The shapes are unchanged by Phase 5F — only what the numbers mean is
stated more precisely. See ``docs/FX_CONTRACT.md``.
"""
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class StatsOverview(BaseModel):
    """Top-line KPIs displayed at the top of the dashboard."""

    model_config = ConfigDict(frozen=True)

    total_transactions_24h: int = Field(ge=0)
    approved_count_24h: int = Field(ge=0)
    declined_count_24h: int = Field(ge=0)
    pending_review_count: int = Field(ge=0)
    approved_rate: float = Field(ge=0.0, le=1.0)
    fraud_caught_amount: Decimal = Field(
        description=(
            "Total amount of declined/reviewed transactions, in the "
            "reporting currency"
        ),
    )
    avg_fraud_score: float | None = Field(default=None, ge=0.0, le=1.0)


class TimeseriesPoint(BaseModel):
    """A single point in a timeseries chart."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    transaction_count: int = Field(ge=0)
    fraud_rate: float = Field(ge=0.0, le=1.0)


class StatsTimeseries(BaseModel):
    """Timeseries of fraud rate and volume over a time window."""

    model_config = ConfigDict(frozen=True)

    window: str = Field(description="e.g. '24h', '7d'")
    points: list[TimeseriesPoint]


class CategoryBreakdown(BaseModel):
    """Aggregate counts broken down by merchant category."""

    model_config = ConfigDict(frozen=True)

    category: str
    transaction_count: int = Field(ge=0)
    declined_count: int = Field(ge=0)
    total_amount: Decimal = Field(
        description="Total transacted amount, in the reporting currency",
    )


class StatsBreakdown(BaseModel):
    """Breakdown of activity by some dimension (e.g. merchant category)."""

    model_config = ConfigDict(frozen=True)

    dimension: str = Field(description="e.g. 'category', 'country'")
    items: list[CategoryBreakdown]
