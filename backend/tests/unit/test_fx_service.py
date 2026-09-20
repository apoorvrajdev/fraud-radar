"""Phase 5F — FX conversion semantics.

What a converted amount means: which date it is priced on, which source
answered, how the arithmetic rounds, and what the row looks like when the
original amount is in the reporting currency already.

The provider is a fake throughout, so nothing here touches a network.
The failure modes — timeouts, outages, malformed responses, staleness —
are exercised separately in `test_fx_failure_modes.py`.

See docs/FX_CONTRACT.md.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.enrichment.fx import (
    SOURCE_CACHE,
    SOURCE_IDENTITY,
    SOURCE_LIVE,
    SOURCE_UNAVAILABLE,
    SOURCE_UNSUPPORTED,
    FxService,
    convert,
    quantize_rate,
    rate_date_for,
)
from app.enrichment.provider import ProviderRate
from app.models.base import Base
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction

CUSTOMER_ID = "11111111-1111-1111-1111-111111111111"
MERCHANT_ID = "22222222-2222-2222-2222-222222222222"

# A Tuesday, so nothing here trips the weekend rule by accident.
TX_MOMENT = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
TX_DATE = date(2024, 1, 2)


class FakeProvider:
    """A provider that answers from a dict and counts its calls."""

    def __init__(self, rates: dict[tuple[str, date], str] | None = None) -> None:
        self.rates = rates or {}
        self.calls: list[tuple[str, str, date]] = []

    def fetch_rate(
        self, *, base: str, quote: str, on: date
    ) -> ProviderRate | None:
        self.calls.append((base, quote, on))
        raw = self.rates.get((quote, on))
        if raw is None:
            return None
        return ProviderRate(
            base=base, quote=quote, rate_date=on, rate=Decimal(raw)
        )


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
    db.add(
        Customer(
            id=CUSTOMER_ID,
            email="fx@example.test",
            full_name="FX Tester",
            country="US",
            risk_tier="LOW",
            account_age_days=365,
        )
    )
    db.add(
        Merchant(
            id=MERCHANT_ID,
            name="FX Merchant",
            category="RETAIL",
            mcc="5311",
            country="US",
            risk_rating="LOW",
        )
    )
    db.commit()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _service(provider: FakeProvider, **kwargs: object) -> FxService:
    defaults: dict[str, object] = {
        "provider": provider,
        "base_currency": "USD",
        "cache_max_age_days": 7,
        "enabled": True,
    }
    defaults.update(kwargs)
    return FxService(**defaults)  # type: ignore[arg-type]


def _transaction(
    *,
    amount: str = "100.00",
    currency: str = "USD",
    created_at: datetime | None = TX_MOMENT,
) -> Transaction:
    return Transaction(
        id="33333333-3333-3333-3333-333333333333",
        idempotency_key="fx-test-key",
        customer_id=CUSTOMER_ID,
        merchant_id=MERCHANT_ID,
        amount=Decimal(amount),
        currency=currency,
        status="PENDING_REVIEW",
        payment_method="CARD",
        country="US",
        is_card_present=True,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# Same-currency conversion
# ---------------------------------------------------------------------------


def test_same_currency_needs_no_lookup(db_session: Session) -> None:
    provider = FakeProvider()
    service = _service(provider)

    result = service.convert_amount(
        db_session, amount=Decimal("100.00"), currency="USD", on=TX_DATE
    )

    assert result.source == SOURCE_IDENTITY
    assert result.rate == Decimal(1)
    assert provider.calls == []


def test_same_currency_amount_is_copied_not_multiplied(
    db_session: Session,
) -> None:
    """Identity is deterministic by construction, not by rounding luck."""
    provider = FakeProvider()
    service = _service(provider)
    amount = Decimal("1234.5678")

    result = service.convert_amount(
        db_session, amount=amount, currency="USD", on=TX_DATE
    )

    assert result.amount_base == amount
    # The exact same Decimal, scale included — not a re-quantised copy.
    assert result.amount_base is not None
    assert result.amount_base.as_tuple() == amount.as_tuple()


def test_same_currency_is_case_insensitive(db_session: Session) -> None:
    provider = FakeProvider()
    service = _service(provider)

    result = service.convert_amount(
        db_session, amount=Decimal("10.00"), currency="usd", on=TX_DATE
    )

    assert result.source == SOURCE_IDENTITY
    assert provider.calls == []


def test_identity_follows_the_configured_reporting_currency(
    db_session: Session,
) -> None:
    provider = FakeProvider()
    service = _service(provider, base_currency="EUR")

    eur = service.convert_amount(
        db_session, amount=Decimal("10.00"), currency="EUR", on=TX_DATE
    )
    usd = service.convert_amount(
        db_session, amount=Decimal("10.00"), currency="USD", on=TX_DATE
    )

    assert eur.source == SOURCE_IDENTITY
    assert usd.source != SOURCE_IDENTITY


# ---------------------------------------------------------------------------
# Normal historical conversion
# ---------------------------------------------------------------------------


def test_historical_conversion_multiplies_by_the_rate(
    db_session: Session,
) -> None:
    provider = FakeProvider({("EUR", TX_DATE): "1.1022"})
    service = _service(provider)

    result = service.convert_amount(
        db_session, amount=Decimal("100.00"), currency="EUR", on=TX_DATE
    )

    assert result.source == SOURCE_LIVE
    assert result.rate == Decimal("1.10220000")
    assert result.amount_base == Decimal("110.2200")
    assert result.rate_date == TX_DATE


def test_conversion_asks_for_the_transaction_date(db_session: Session) -> None:
    """The historical date, not today — the whole point of the exercise."""
    provider = FakeProvider({("EUR", date(2022, 3, 15)): "1.1000"})
    service = _service(provider)

    service.convert_amount(
        db_session,
        amount=Decimal("100.00"),
        currency="EUR",
        on=date(2022, 3, 15),
    )

    assert provider.calls == [("USD", "EUR", date(2022, 3, 15))]


def test_a_fetched_rate_is_written_into_the_cache(db_session: Session) -> None:
    provider = FakeProvider({("EUR", TX_DATE): "1.1022"})
    service = _service(provider)

    service.convert_amount(
        db_session, amount=Decimal("100.00"), currency="EUR", on=TX_DATE
    )
    second = service.get_rate(db_session, quote="EUR", on=TX_DATE)

    assert second.source == SOURCE_CACHE
    assert second.rate == Decimal("1.10220000")


# ---------------------------------------------------------------------------
# Decimal precision and rounding
# ---------------------------------------------------------------------------


def test_conversion_never_touches_a_binary_float() -> None:
    """0.1 * 3 is 0.30000000000000004 in float and 0.3000 here."""
    assert convert(Decimal("0.1"), Decimal("3")) == Decimal("0.3000")


def test_conversion_is_exact_before_it_rounds() -> None:
    """A 19-digit amount times an 8-decimal rate exceeds the default context.

    Under Decimal's default 28-digit precision the product below would be
    rounded before the quantise ever ran, losing cents on a large
    transfer. The widened context makes the single rounding the quantise.
    """
    amount = Decimal("999999999999999.9999")  # NUMERIC(19, 4) at full width
    rate = Decimal("1.00000001")

    result = convert(amount, rate)

    expected = (amount * rate).quantize(Decimal("0.0001"))
    assert result == expected
    assert result == Decimal("1000000009999999.9999")


def test_conversion_rounds_half_up_at_four_places() -> None:
    # 10.00005 → the .00005 rounds away from zero, not to even.
    assert convert(Decimal("1.00"), Decimal("10.000050")) == Decimal("10.0001")
    assert convert(Decimal("1.00"), Decimal("10.000150")) == Decimal("10.0002")


def test_conversion_result_always_carries_four_decimal_places() -> None:
    result = convert(Decimal("2"), Decimal("3"))

    assert result == Decimal("6")
    assert result.as_tuple().exponent == -4


def test_rate_is_quantised_to_eight_places() -> None:
    assert quantize_rate(Decimal("1.123456789")) == Decimal("1.12345679")
    assert quantize_rate(Decimal("1.1")).as_tuple().exponent == -8


def test_a_thin_rate_survives_quantisation(db_session: Session) -> None:
    """JPY-scale rates are the case where 8 places actually matter."""
    provider = FakeProvider({("JPY", TX_DATE): "0.00678912"})
    service = _service(provider)

    result = service.convert_amount(
        db_session, amount=Decimal("15000.00"), currency="JPY", on=TX_DATE
    )

    assert result.rate == Decimal("0.00678912")
    assert result.amount_base == Decimal("101.8368")


def test_conversion_is_deterministic_for_identical_inputs(
    db_session: Session,
) -> None:
    provider = FakeProvider({("EUR", TX_DATE): "1.10225"})
    service = _service(provider)

    first = service.convert_amount(
        db_session, amount=Decimal("19.99"), currency="EUR", on=TX_DATE
    )
    second = service.convert_amount(
        db_session, amount=Decimal("19.99"), currency="EUR", on=TX_DATE
    )

    assert first.amount_base == second.amount_base == Decimal("22.0340")
    assert first.rate == second.rate
    assert first.rate_date == second.rate_date
    # Same value *and* same scale, so serialisation cannot differ either.
    assert first.amount_base is not None and second.amount_base is not None
    assert first.amount_base.as_tuple() == second.amount_base.as_tuple()
    # The money is identical; only the provenance records that the second
    # answer came from the cache the first one filled.
    assert (first.source, second.source) == (SOURCE_LIVE, SOURCE_CACHE)


# ---------------------------------------------------------------------------
# Transaction-date semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        pytest.param(
            datetime(2024, 1, 2, 0, 0, tzinfo=UTC), date(2024, 1, 2), id="tuesday"
        ),
        pytest.param(
            datetime(2024, 1, 5, 23, 59, tzinfo=UTC),
            date(2024, 1, 5),
            id="friday-late",
        ),
        pytest.param(
            datetime(2024, 1, 6, 12, 0, tzinfo=UTC),
            date(2024, 1, 5),
            id="saturday-to-friday",
        ),
        pytest.param(
            datetime(2024, 1, 7, 12, 0, tzinfo=UTC),
            date(2024, 1, 5),
            id="sunday-to-friday",
        ),
        pytest.param(
            datetime(2024, 1, 8, 0, 1, tzinfo=UTC),
            date(2024, 1, 8),
            id="monday",
        ),
    ],
)
def test_weekends_resolve_back_to_the_preceding_friday(
    moment: datetime, expected: date
) -> None:
    assert rate_date_for(moment) == expected


def test_a_naive_timestamp_is_read_as_utc() -> None:
    """SQLite hands some timestamps back without a tzinfo."""
    assert rate_date_for(datetime(2024, 1, 6, 12, 0)) == date(2024, 1, 5)


def test_a_timestamp_in_another_zone_is_converted_before_the_date_is_taken() -> None:
    """01:00 on the 8th in UTC+2 is 23:00 on the 7th in UTC — still a Sunday."""
    moment = datetime(2024, 1, 8, 1, 0, tzinfo=timezone(timedelta(hours=2)))

    assert rate_date_for(moment) == date(2024, 1, 5)


def test_a_whole_weekend_shares_one_provider_call(db_session: Session) -> None:
    """The reason weekends are normalised locally rather than at the provider."""
    friday = date(2024, 1, 5)
    provider = FakeProvider({("EUR", friday): "1.0932"})
    service = _service(provider)

    for moment in (
        datetime(2024, 1, 6, 9, 0, tzinfo=UTC),   # Saturday
        datetime(2024, 1, 7, 18, 0, tzinfo=UTC),  # Sunday
    ):
        service.convert_amount(
            db_session,
            amount=Decimal("50.00"),
            currency="EUR",
            on=rate_date_for(moment),
        )

    assert provider.calls == [("USD", "EUR", friday)]


def test_a_future_date_is_refused_without_a_call(db_session: Session) -> None:
    provider = FakeProvider()
    service = _service(provider)
    tomorrow = datetime.now(UTC).date() + timedelta(days=1)

    result = service.get_rate(db_session, quote="EUR", on=tomorrow)

    assert result.source == SOURCE_UNAVAILABLE
    assert result.rate is None
    assert provider.calls == []


# ---------------------------------------------------------------------------
# Unsupported currencies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["US", "USDX", "us1", "", "   ", "12E"])
def test_a_non_iso_code_is_unsupported_without_a_call(
    db_session: Session, code: str
) -> None:
    provider = FakeProvider()
    service = _service(provider)

    result = service.get_rate(db_session, quote=code, on=TX_DATE)

    assert result.source == SOURCE_UNSUPPORTED
    assert result.rate is None
    assert provider.calls == []


def test_an_unsupported_currency_fabricates_no_amount(
    db_session: Session,
) -> None:
    provider = FakeProvider()
    service = _service(provider)

    result = service.convert_amount(
        db_session, amount=Decimal("100.00"), currency="XX", on=TX_DATE
    )

    assert result.source == SOURCE_UNSUPPORTED
    assert result.amount_base is None
    assert result.rate is None
    assert result.rate_date is None


# ---------------------------------------------------------------------------
# Enriching a transaction row
# ---------------------------------------------------------------------------


def test_enrich_attaches_the_derived_columns(db_session: Session) -> None:
    provider = FakeProvider({("EUR", TX_DATE): "1.1022"})
    service = _service(provider)
    tx = _transaction(amount="250.00", currency="EUR")

    service.enrich(db_session, tx)

    assert tx.amount_base == Decimal("275.5500")
    assert tx.fx_rate == Decimal("1.10220000")
    assert tx.fx_rate_date == TX_DATE
    assert tx.fx_source == SOURCE_LIVE


def test_enrich_leaves_the_original_amount_untouched(
    db_session: Session,
) -> None:
    """The invariant the whole phase rests on."""
    provider = FakeProvider({("EUR", TX_DATE): "1.1022"})
    service = _service(provider)
    tx = _transaction(amount="250.00", currency="EUR")

    service.enrich(db_session, tx)

    assert tx.amount == Decimal("250.00")
    assert tx.amount.as_tuple() == Decimal("250.00").as_tuple()


def test_enrich_leaves_the_original_currency_untouched(
    db_session: Session,
) -> None:
    provider = FakeProvider({("EUR", TX_DATE): "1.1022"})
    service = _service(provider)
    tx = _transaction(currency="EUR")

    service.enrich(db_session, tx)

    assert tx.currency == "EUR"


def test_enrich_of_a_usd_row_is_the_identity_path(db_session: Session) -> None:
    provider = FakeProvider()
    service = _service(provider)
    tx = _transaction(amount="99.99", currency="USD")

    service.enrich(db_session, tx)

    assert tx.amount_base == Decimal("99.99")
    assert tx.fx_rate == Decimal(1)
    assert tx.fx_source == SOURCE_IDENTITY
    assert provider.calls == []


def test_enrich_without_a_timestamp_prices_nothing(db_session: Session) -> None:
    """No date means no historical rate — and "now" is not an answer."""
    provider = FakeProvider({("EUR", TX_DATE): "1.1022"})
    service = _service(provider)
    tx = _transaction(currency="EUR", created_at=None)

    service.enrich(db_session, tx)

    assert tx.amount_base is None
    assert tx.fx_rate is None
    assert tx.fx_rate_date is None
    assert tx.fx_source == SOURCE_UNAVAILABLE
    assert provider.calls == []


def test_an_enriched_row_persists_and_reloads(db_session: Session) -> None:
    """The CHECK constraints accept what the service writes."""
    provider = FakeProvider({("EUR", TX_DATE): "1.1022"})
    service = _service(provider)
    tx = _transaction(amount="250.00", currency="EUR")
    db_session.add(tx)

    service.enrich(db_session, tx)
    db_session.commit()
    db_session.expire_all()

    reloaded = db_session.get(Transaction, tx.id)
    assert reloaded is not None
    assert reloaded.amount == Decimal("250.00")
    assert reloaded.currency == "EUR"
    assert reloaded.amount_base == Decimal("275.5500")
    assert reloaded.fx_source == SOURCE_LIVE


def test_an_unconverted_row_persists_with_null_fx_columns(
    db_session: Session,
) -> None:
    """The pairing CHECK permits all-null; a half-converted row is the thing it forbids."""
    provider = FakeProvider()
    service = _service(provider)
    tx = _transaction(currency="EUR")
    db_session.add(tx)

    service.enrich(db_session, tx)
    db_session.commit()
    db_session.expire_all()

    reloaded = db_session.get(Transaction, tx.id)
    assert reloaded is not None
    assert reloaded.amount_base is None
    assert reloaded.fx_rate is None
    assert reloaded.fx_source == SOURCE_UNAVAILABLE
