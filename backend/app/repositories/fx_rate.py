"""Repository for the FX rate cache.

Two operations, and one invariant that matters more than either of them.

The invariant: **no read ever returns a rate dated after the date it was
asked about.** Both lookups below are constrained to
``rate_date <= on``, so a transaction from last month cannot be priced
with this morning's rate no matter which resolution step is running. That
is a structural guarantee rather than a rule the caller has to remember —
see `docs/FX_CONTRACT.md`, "No silent fallback to the current rate".

This repository deliberately does **not** inherit ``BaseRepository``.
``FxRate`` has a three-column primary key, and ``BaseRepository.get()``
takes a single string id — inheriting it would ship a method that cannot
be called correctly for this model.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.fx_rate import FxRate


class FxRateRepository:
    """Reads and writes the `fx_rates` cache table."""

    model = FxRate

    def latest_on_or_before(
        self,
        db: Session,
        *,
        base: str,
        quote: str,
        on: date,
        max_age_days: int | None = None,
    ) -> FxRate | None:
        """Return the newest cached rate for the pair that could apply to `on`.

        "Could apply to `on`" means ``rate_date <= on`` — a rate published
        after the transaction is never a candidate for it.

        `max_age_days` bounds how far back the search goes. Pass it for a
        fresh-cache read (the caller wants a rate from around that date, and
        would rather ask the provider than reach further back); omit it for
        the stale fallback (the caller has already tried the provider and
        will take the most recent rate that exists, however old).

        Returns None when nothing in the cache qualifies.
        """
        stmt = (
            select(FxRate)
            .where(
                FxRate.base == base,
                FxRate.quote == quote,
                FxRate.rate_date <= on,
            )
            .order_by(FxRate.rate_date.desc())
            .limit(1)
        )
        if max_age_days is not None:
            stmt = stmt.where(FxRate.rate_date >= on - timedelta(days=max_age_days))
        return db.execute(stmt).scalar_one_or_none()

    def upsert(
        self,
        db: Session,
        *,
        base: str,
        quote: str,
        rate_date: date,
        rate: Decimal,
        fetched_at: datetime,
    ) -> FxRate:
        """Insert or refresh one cached rate. Does NOT commit.

        The caller owns the commit — on the ingestion path that is
        `idempotency.store()`, so a newly fetched rate lands in the same
        transaction as the row that fetched it. A read-then-write rather
        than a dialect-specific upsert: the statement has to work on both
        SQLite and Postgres, and this path runs at most once per currency
        per day.
        """
        existing = db.get(FxRate, (base, quote, rate_date))
        if existing is not None:
            existing.rate = rate
            existing.fetched_at = fetched_at
            db.flush()
            return existing

        row = FxRate(
            base=base,
            quote=quote,
            rate_date=rate_date,
            rate=rate,
            fetched_at=fetched_at,
        )
        db.add(row)
        db.flush()
        return row


fx_rate_repository = FxRateRepository()
