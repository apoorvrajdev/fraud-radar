"""Phase 5F — the FX rate cache repository.

Exercised against a real in-memory SQLite engine rather than a mock, so
the ``rate_date <= on`` predicate, the ordering, the age window and the
``Decimal`` round-trip through ``NUMERIC(18, 8)`` are all really tested.

The rule this file exists to hold: **no read ever returns a rate dated
after the date it was asked about.** That is what makes "converted at
today's rate" unreachable rather than merely discouraged — see
docs/FX_CONTRACT.md.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.base import Base
from app.models.fx_rate import FxRate
from app.repositories.fx_rate import fx_rate_repository

FETCHED_AT = datetime(2024, 1, 3, 9, 0, tzinfo=UTC)


@pytest.fixture
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    SessionTesting = sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False,
    )
    db = SessionTesting()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _seed(
    db: Session, *, quote: str, rate_date: date, rate: str, base: str = "USD"
) -> FxRate:
    return fx_rate_repository.upsert(
        db,
        base=base,
        quote=quote,
        rate_date=rate_date,
        rate=Decimal(rate),
        fetched_at=FETCHED_AT,
    )


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def test_upsert_inserts_a_new_rate(db_session: Session) -> None:
    _seed(db_session, quote="EUR", rate_date=date(2024, 1, 2), rate="1.1022")

    rows = list(db_session.execute(select(FxRate)).scalars().all())
    assert len(rows) == 1
    assert (rows[0].base, rows[0].quote) == ("USD", "EUR")
    assert rows[0].rate_date == date(2024, 1, 2)


def test_upsert_refreshes_an_existing_rate_in_place(db_session: Session) -> None:
    """Same (base, quote, date) is one row — a re-fetch corrects it."""
    _seed(db_session, quote="EUR", rate_date=date(2024, 1, 2), rate="1.1022")
    later = datetime(2024, 1, 4, 9, 0, tzinfo=UTC)
    fx_rate_repository.upsert(
        db_session,
        base="USD",
        quote="EUR",
        rate_date=date(2024, 1, 2),
        rate=Decimal("1.2000"),
        fetched_at=later,
    )

    rows = list(db_session.execute(select(FxRate)).scalars().all())
    assert len(rows) == 1
    assert rows[0].rate == Decimal("1.2000")
    assert rows[0].fetched_at.replace(tzinfo=UTC) == later


def test_upsert_does_not_commit(db_session: Session) -> None:
    """The caller owns the commit, so a rolled-back ingestion takes the rate with it."""
    _seed(db_session, quote="EUR", rate_date=date(2024, 1, 2), rate="1.1022")
    db_session.rollback()

    assert db_session.execute(select(FxRate)).first() is None


def test_rate_round_trips_at_full_eight_decimal_precision(
    db_session: Session,
) -> None:
    """NUMERIC(18, 8) — a thin rate must survive storage exactly."""
    _seed(db_session, quote="JPY", rate_date=date(2024, 1, 2), rate="0.00678912")
    db_session.expire_all()

    row = db_session.get(FxRate, ("USD", "JPY", date(2024, 1, 2)))
    assert row is not None
    assert row.rate == Decimal("0.00678912")


def test_non_positive_rate_is_rejected_by_the_database(
    db_session: Session,
) -> None:
    db_session.add(
        FxRate(
            base="USD",
            quote="EUR",
            rate_date=date(2024, 1, 2),
            rate=Decimal("0"),
            fetched_at=FETCHED_AT,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_same_currency_row_is_rejected_by_the_database(
    db_session: Session,
) -> None:
    """Identity never consults the cache, so such a row is a corrupt write."""
    db_session.add(
        FxRate(
            base="USD",
            quote="USD",
            rate_date=date(2024, 1, 2),
            rate=Decimal("1"),
            fetched_at=FETCHED_AT,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def test_exact_date_hit(db_session: Session) -> None:
    _seed(db_session, quote="EUR", rate_date=date(2024, 1, 2), rate="1.1022")

    found = fx_rate_repository.latest_on_or_before(
        db_session, base="USD", quote="EUR", on=date(2024, 1, 2)
    )

    assert found is not None
    assert found.rate == Decimal("1.1022")


def test_read_returns_the_newest_rate_at_or_before_the_date(
    db_session: Session,
) -> None:
    _seed(db_session, quote="EUR", rate_date=date(2024, 1, 2), rate="1.1000")
    _seed(db_session, quote="EUR", rate_date=date(2024, 1, 4), rate="1.1400")
    _seed(db_session, quote="EUR", rate_date=date(2024, 1, 3), rate="1.1200")

    found = fx_rate_repository.latest_on_or_before(
        db_session, base="USD", quote="EUR", on=date(2024, 1, 3)
    )

    assert found is not None
    assert found.rate == Decimal("1.1200")


def test_read_never_returns_a_rate_dated_after_the_request(
    db_session: Session,
) -> None:
    """The structural guarantee: a newer rate cannot price an older transaction."""
    _seed(db_session, quote="EUR", rate_date=date(2024, 6, 1), rate="1.1400")

    found = fx_rate_repository.latest_on_or_before(
        db_session, base="USD", quote="EUR", on=date(2024, 1, 2)
    )

    assert found is None


def test_read_is_scoped_to_the_pair(db_session: Session) -> None:
    _seed(db_session, quote="GBP", rate_date=date(2024, 1, 2), rate="1.2700")

    found = fx_rate_repository.latest_on_or_before(
        db_session, base="USD", quote="EUR", on=date(2024, 1, 2)
    )

    assert found is None


def test_read_is_scoped_to_the_base(db_session: Session) -> None:
    """A rate into EUR must not answer a question about converting into USD."""
    _seed(db_session, base="EUR", quote="GBP", rate_date=date(2024, 1, 2), rate="1.15")

    found = fx_rate_repository.latest_on_or_before(
        db_session, base="USD", quote="GBP", on=date(2024, 1, 2)
    )

    assert found is None


def test_empty_cache_returns_none(db_session: Session) -> None:
    assert (
        fx_rate_repository.latest_on_or_before(
            db_session, base="USD", quote="EUR", on=date(2024, 1, 2)
        )
        is None
    )


# ---------------------------------------------------------------------------
# The age window
# ---------------------------------------------------------------------------


def test_max_age_admits_a_rate_exactly_on_the_boundary(
    db_session: Session,
) -> None:
    _seed(db_session, quote="EUR", rate_date=date(2024, 1, 2), rate="1.1022")

    found = fx_rate_repository.latest_on_or_before(
        db_session,
        base="USD",
        quote="EUR",
        on=date(2024, 1, 9),
        max_age_days=7,
    )

    assert found is not None


def test_max_age_excludes_a_rate_one_day_past_the_boundary(
    db_session: Session,
) -> None:
    """Past the window the caller wants the provider, not an older rate."""
    _seed(db_session, quote="EUR", rate_date=date(2024, 1, 2), rate="1.1022")

    found = fx_rate_repository.latest_on_or_before(
        db_session,
        base="USD",
        quote="EUR",
        on=date(2024, 1, 10),
        max_age_days=7,
    )

    assert found is None


def test_omitting_max_age_reaches_back_indefinitely(
    db_session: Session,
) -> None:
    """The stale fallback: any age, but still never newer than the request."""
    _seed(db_session, quote="EUR", rate_date=date(2020, 3, 9), rate="1.1400")

    found = fx_rate_repository.latest_on_or_before(
        db_session, base="USD", quote="EUR", on=date(2024, 1, 2)
    )

    assert found is not None
    assert found.rate_date == date(2020, 3, 9)
