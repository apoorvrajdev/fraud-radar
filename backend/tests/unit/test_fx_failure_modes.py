"""Phase 5F — what FX enrichment does when the outside world misbehaves.

The conversion semantics live in `test_fx_service.py`. This file is
about everything that goes wrong: timeouts, outages, rate limits,
garbage responses, currencies the provider does not carry, and the rate
that exists but is too old to be the right answer.

Two properties are asserted repeatedly on purpose, because they are the
ones a future refactor is most likely to break quietly:

1. **No lookup ever resolves to a rate newer than the transaction.** Not
   from the cache, not from the stale fallback, not from anywhere. The
   test that seeds a *newer* rate and expects `unavailable` is the
   central one in this file.
2. **The provider is called as few times as the cache allows**, asserted
   by counting calls rather than by inspecting values.

Nothing here touches a network: every provider is a fake, and the
failures are raised rather than produced by an unreachable host.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.enrichment.fx import (
    SOURCE_CACHE,
    SOURCE_LIVE,
    SOURCE_STALE,
    SOURCE_UNAVAILABLE,
    SOURCE_UNSUPPORTED,
    FxService,
)
from app.enrichment.provider import (
    FxProviderError,
    FxResponseError,
    FxTimeoutError,
    FxUnsupportedCurrencyError,
    ProviderRate,
)
from app.models.base import Base
from app.models.fx_rate import FxRate
from app.models.transaction import Transaction
from app.repositories.fx_rate import fx_rate_repository

TX_DATE = date(2024, 1, 2)  # a Tuesday


class CountingProvider:
    """Base fake: records every call so cache behaviour can be asserted."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, date]] = []

    def fetch_rate(
        self, *, base: str, quote: str, on: date
    ) -> ProviderRate | None:
        self.calls.append((base, quote, on))
        return self._answer(base=base, quote=quote, on=on)

    def _answer(
        self, *, base: str, quote: str, on: date
    ) -> ProviderRate | None:
        raise NotImplementedError


class WorkingProvider(CountingProvider):
    def __init__(self, rate: str = "1.1022") -> None:
        super().__init__()
        self.rate = Decimal(rate)

    def _answer(
        self, *, base: str, quote: str, on: date
    ) -> ProviderRate | None:
        return ProviderRate(base=base, quote=quote, rate_date=on, rate=self.rate)


class RaisingProvider(CountingProvider):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self.exc = exc

    def _answer(
        self, *, base: str, quote: str, on: date
    ) -> ProviderRate | None:
        raise self.exc


class EmptyProvider(CountingProvider):
    """Reachable, but has no rate for the date — what a future date returns."""

    def _answer(
        self, *, base: str, quote: str, on: date
    ) -> ProviderRate | None:
        return None


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


def _service(provider: CountingProvider, **kwargs: object) -> FxService:
    defaults: dict[str, object] = {
        "provider": provider,
        "base_currency": "USD",
        "cache_max_age_days": 7,
        "enabled": True,
    }
    defaults.update(kwargs)
    return FxService(**defaults)  # type: ignore[arg-type]


def _transaction(*, amount: str = "100.00", currency: str = "EUR") -> Transaction:
    """A transient row — `enrich` only sets attributes, it never persists."""
    return Transaction(
        id="33333333-3333-3333-3333-333333333333",
        idempotency_key="fx-failure-key",
        customer_id="11111111-1111-1111-1111-111111111111",
        merchant_id="22222222-2222-2222-2222-222222222222",
        amount=Decimal(amount),
        currency=currency,
        status="PENDING_REVIEW",
        payment_method="CARD",
        country="US",
        is_card_present=True,
        created_at=datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
    )


def _seed_rate(
    db: Session, *, quote: str, rate_date: date, rate: str
) -> None:
    fx_rate_repository.upsert(
        db,
        base="USD",
        quote=quote,
        rate_date=rate_date,
        rate=Decimal(rate),
        fetched_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Transport failures, with nothing cached
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(FxTimeoutError("timed out"), id="timeout"),
        pytest.param(FxProviderError("connection refused"), id="network"),
        pytest.param(FxResponseError("garbage body"), id="malformed"),
    ],
)
def test_provider_failure_with_an_empty_cache_is_unavailable(
    db_session: Session, exc: Exception
) -> None:
    """No rate is invented — not from today, not from anywhere."""
    service = _service(RaisingProvider(exc))

    result = service.get_rate(db_session, quote="EUR", on=TX_DATE)

    assert result.source == SOURCE_UNAVAILABLE
    assert result.rate is None
    assert result.rate_date is None


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(FxTimeoutError("timed out"), id="timeout"),
        pytest.param(FxProviderError("connection refused"), id="network"),
        pytest.param(FxResponseError("garbage body"), id="malformed"),
    ],
)
def test_provider_failure_does_not_propagate(
    db_session: Session, exc: Exception
) -> None:
    """Nothing from the provider family reaches the caller."""
    service = _service(RaisingProvider(exc))

    # Would raise if the exception escaped — the assertion is that it does not.
    result = service.convert_amount(
        db_session, amount=Decimal("100.00"), currency="EUR", on=TX_DATE
    )

    assert result.amount_base is None


def test_an_empty_provider_answer_is_a_missing_rate(
    db_session: Session,
) -> None:
    provider = EmptyProvider()
    service = _service(provider)

    result = service.get_rate(db_session, quote="EUR", on=TX_DATE)

    assert result.source == SOURCE_UNAVAILABLE
    assert provider.calls == [("USD", "EUR", TX_DATE)]


def test_get_rate_does_not_swallow_an_unforeseen_error(
    db_session: Session,
) -> None:
    """`get_rate` catches the provider family and nothing else.

    Documented here so the containment below is understood as deliberate
    placement rather than a blanket `except` scattered around.
    """
    service = _service(RaisingProvider(RuntimeError("something unforeseen")))

    with pytest.raises(RuntimeError):
        service.get_rate(db_session, quote="EUR", on=TX_DATE)


def test_enrich_contains_even_an_unforeseen_error(db_session: Session) -> None:
    """The outermost boundary: nothing at all escapes into the scoring path."""
    service = _service(RaisingProvider(RuntimeError("something unforeseen")))
    tx = _transaction(currency="EUR")

    service.enrich(db_session, tx)  # must not raise

    assert tx.amount == Decimal("100.00")
    assert tx.currency == "EUR"
    assert tx.amount_base is None
    assert tx.fx_rate is None
    assert tx.fx_source == SOURCE_UNAVAILABLE


# ---------------------------------------------------------------------------
# Unsupported currencies
# ---------------------------------------------------------------------------


def test_a_provider_refusal_marks_the_currency_unsupported(
    db_session: Session,
) -> None:
    service = _service(RaisingProvider(FxUnsupportedCurrencyError("no such pair")))

    result = service.get_rate(db_session, quote="ZZZ", on=TX_DATE)

    assert result.source == SOURCE_UNSUPPORTED
    assert result.rate is None


def test_an_unsupported_currency_does_not_fall_back_to_a_stale_rate(
    db_session: Session,
) -> None:
    """A permanent refusal is a different question from a transient one.

    If the provider has stopped carrying a pair, an old rate for it is
    not a degraded answer — it is an answer to a question nobody asked.
    """
    _seed_rate(db_session, quote="ZZZ", rate_date=date(2023, 6, 1), rate="2.0")
    service = _service(RaisingProvider(FxUnsupportedCurrencyError("dropped")))

    result = service.get_rate(db_session, quote="ZZZ", on=TX_DATE)

    assert result.source == SOURCE_UNSUPPORTED
    assert result.rate is None


def test_a_transient_failure_does_fall_back_to_a_stale_rate(
    db_session: Session,
) -> None:
    """The contrast with the test above — this is the whole distinction."""
    _seed_rate(db_session, quote="EUR", rate_date=date(2023, 6, 1), rate="2.0")
    service = _service(RaisingProvider(FxTimeoutError("timed out")))

    result = service.get_rate(db_session, quote="EUR", on=TX_DATE)

    assert result.source == SOURCE_STALE
    assert result.rate == Decimal("2.00000000")


# ---------------------------------------------------------------------------
# The stale fallback, and its one hard limit
# ---------------------------------------------------------------------------


def test_stale_fallback_reports_the_date_it_actually_used(
    db_session: Session,
) -> None:
    """A converted amount from a stale rate must say which day priced it."""
    _seed_rate(db_session, quote="EUR", rate_date=date(2023, 6, 1), rate="1.05")
    service = _service(RaisingProvider(FxTimeoutError("timed out")))

    result = service.convert_amount(
        db_session, amount=Decimal("100.00"), currency="EUR", on=TX_DATE
    )

    assert result.source == SOURCE_STALE
    assert result.rate_date == date(2023, 6, 1)
    assert result.amount_base == Decimal("105.0000")


def test_stale_fallback_takes_the_most_recent_usable_rate(
    db_session: Session,
) -> None:
    _seed_rate(db_session, quote="EUR", rate_date=date(2022, 1, 3), rate="1.10")
    _seed_rate(db_session, quote="EUR", rate_date=date(2023, 6, 1), rate="1.05")
    service = _service(RaisingProvider(FxProviderError("down")))

    result = service.get_rate(db_session, quote="EUR", on=TX_DATE)

    assert result.rate_date == date(2023, 6, 1)


def test_no_silent_fallback_to_a_newer_rate(db_session: Session) -> None:
    """The central guarantee of this contract.

    The cache holds a rate — just not one that could have applied to this
    transaction. Pricing it at a later rate would be indistinguishable
    from "use today's rate", which is exactly the behaviour a historical
    conversion exists to avoid. The answer must be "no rate", not "this
    one".
    """
    _seed_rate(db_session, quote="EUR", rate_date=date(2024, 6, 1), rate="1.30")
    service = _service(RaisingProvider(FxTimeoutError("timed out")))

    result = service.get_rate(db_session, quote="EUR", on=TX_DATE)

    assert result.source == SOURCE_UNAVAILABLE
    assert result.rate is None


def test_no_silent_fallback_to_a_newer_rate_when_the_provider_is_up(
    db_session: Session,
) -> None:
    """Same guarantee on the path where the provider simply has no data."""
    _seed_rate(db_session, quote="EUR", rate_date=date(2024, 6, 1), rate="1.30")
    service = _service(EmptyProvider())

    result = service.get_rate(db_session, quote="EUR", on=TX_DATE)

    assert result.source == SOURCE_UNAVAILABLE


def test_a_newer_rate_does_not_leak_through_the_converted_amount(
    db_session: Session,
) -> None:
    """Belt and braces: the same case, asserted on the money rather than the source."""
    _seed_rate(db_session, quote="EUR", rate_date=date(2024, 6, 1), rate="1.30")
    service = _service(RaisingProvider(FxProviderError("down")))

    result = service.convert_amount(
        db_session, amount=Decimal("100.00"), currency="EUR", on=TX_DATE
    )

    assert result.amount_base is None
    assert result.amount_base != Decimal("130.0000")


# ---------------------------------------------------------------------------
# Cache behaviour — asserted by counting calls
# ---------------------------------------------------------------------------


def test_a_repeated_lookup_does_not_call_the_provider_twice(
    db_session: Session,
) -> None:
    provider = WorkingProvider()
    service = _service(provider)

    first = service.get_rate(db_session, quote="EUR", on=TX_DATE)
    second = service.get_rate(db_session, quote="EUR", on=TX_DATE)
    third = service.get_rate(db_session, quote="EUR", on=TX_DATE)

    assert len(provider.calls) == 1
    assert first.source == SOURCE_LIVE
    assert second.source == third.source == SOURCE_CACHE
    assert first.rate == second.rate == third.rate


def test_each_currency_is_fetched_separately(db_session: Session) -> None:
    provider = WorkingProvider()
    service = _service(provider)

    service.get_rate(db_session, quote="EUR", on=TX_DATE)
    service.get_rate(db_session, quote="GBP", on=TX_DATE)
    service.get_rate(db_session, quote="EUR", on=TX_DATE)

    assert [call[1] for call in provider.calls] == ["EUR", "GBP"]


def test_a_nearby_date_reuses_the_cached_rate(db_session: Session) -> None:
    """Within the freshness window the cache answers without a call."""
    _seed_rate(db_session, quote="EUR", rate_date=date(2024, 1, 2), rate="1.1022")
    provider = WorkingProvider()
    service = _service(provider)

    result = service.get_rate(db_session, quote="EUR", on=date(2024, 1, 8))

    assert result.source == SOURCE_CACHE
    assert provider.calls == []


def test_a_rate_past_the_freshness_window_triggers_a_fetch(
    db_session: Session,
) -> None:
    """Old enough that the provider should be asked, even though a rate exists."""
    _seed_rate(db_session, quote="EUR", rate_date=date(2023, 12, 1), rate="1.05")
    provider = WorkingProvider(rate="1.1022")
    service = _service(provider)

    result = service.get_rate(db_session, quote="EUR", on=TX_DATE)

    assert len(provider.calls) == 1
    assert result.source == SOURCE_LIVE
    assert result.rate == Decimal("1.10220000")


def test_a_failed_lookup_is_not_cached(db_session: Session) -> None:
    """A transient failure must not poison the next transaction's chance."""
    provider = RaisingProvider(FxTimeoutError("timed out"))
    service = _service(provider)

    service.get_rate(db_session, quote="EUR", on=TX_DATE)
    service.get_rate(db_session, quote="EUR", on=TX_DATE)

    assert len(provider.calls) == 2
    assert db_session.get(FxRate, ("USD", "EUR", TX_DATE)) is None


def test_a_fetched_rate_is_stored_quantised(db_session: Session) -> None:
    """What lands in the cache is what NUMERIC(18, 8) can hold."""
    service = _service(WorkingProvider(rate="1.123456789"))

    service.get_rate(db_session, quote="EUR", on=TX_DATE)

    row = db_session.get(FxRate, ("USD", "EUR", TX_DATE))
    assert row is not None
    assert row.rate == Decimal("1.12345679")


# ---------------------------------------------------------------------------
# The kill switch
# ---------------------------------------------------------------------------


def test_disabling_fx_stops_the_network_call(db_session: Session) -> None:
    provider = WorkingProvider()
    service = _service(provider, enabled=False)

    result = service.get_rate(db_session, quote="EUR", on=TX_DATE)

    assert provider.calls == []
    assert result.source == SOURCE_UNAVAILABLE


def test_disabling_fx_still_reads_the_cache(db_session: Session) -> None:
    """It gates the network, not the enrichment."""
    _seed_rate(db_session, quote="EUR", rate_date=TX_DATE, rate="1.1022")
    provider = WorkingProvider()
    service = _service(provider, enabled=False)

    result = service.get_rate(db_session, quote="EUR", on=TX_DATE)

    assert result.source == SOURCE_CACHE
    assert result.rate == Decimal("1.10220000")
    assert provider.calls == []


def test_disabling_fx_leaves_the_identity_path_working(
    db_session: Session,
) -> None:
    """A USD-only deployment does not need FX switched on at all."""
    provider = WorkingProvider()
    service = _service(provider, enabled=False)

    result = service.convert_amount(
        db_session, amount=Decimal("42.50"), currency="USD", on=TX_DATE
    )

    assert result.amount_base == Decimal("42.50")
    assert provider.calls == []
