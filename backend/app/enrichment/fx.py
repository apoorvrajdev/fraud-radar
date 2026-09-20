"""FX enrichment — converting a transaction into the reporting currency.

This module owns the business half of the FX story: which date a
transaction is priced on, which source answers, how the arithmetic is
done, and what a row looks like when none of it worked. The network half
lives in `app/enrichment/provider.py` and is injected, so everything here
is testable without a socket.

Two invariants hold everything else up:

1. **The original amount and currency are never written.** Enrichment
   only ever assigns the four derived columns, or leaves them null.
2. **Enrichment cannot stop a transaction being scored.** `enrich`
   catches everything; a provider outage produces a row with four null
   columns and an `fx_source` that says so.

See `docs/FX_CONTRACT.md` for the full contract.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, localcontext

from sqlalchemy.orm import Session

from app.config import get_settings
from app.enrichment.provider import (
    FrankfurterProvider,
    FxProviderError,
    FxRateProvider,
    FxUnsupportedCurrencyError,
)
from app.models.transaction import Transaction
from app.repositories.fx_rate import fx_rate_repository

log = logging.getLogger(__name__)

# ISO 4217 alpha-3. Anything else cannot be a currency, so it is refused
# locally rather than spent as a network round-trip.
_ISO_4217 = re.compile(r"^[A-Z]{3}$")

# Scales of the columns these values are stored in.
_RATE_SCALE = Decimal("0.00000001")   # NUMERIC(18, 8)
_AMOUNT_SCALE = Decimal("0.0001")     # NUMERIC(19, 4)

# Wide enough that the product of a 19-digit amount and an 8-decimal
# rate is computed exactly, instead of being rounded to the default
# 28-digit context part-way through and then quantised again.
_CONVERSION_PRECISION = 38

# `fx_source` vocabulary, matching ck_transactions_fx_source.
SOURCE_IDENTITY = "identity"
SOURCE_LIVE = "live"
SOURCE_CACHE = "cache"
SOURCE_STALE = "stale"
SOURCE_UNSUPPORTED = "unsupported"
SOURCE_UNAVAILABLE = "unavailable"

_SATURDAY = 5


@dataclass(frozen=True)
class FxLookup:
    """The outcome of resolving one rate.

    `rate` is None exactly when no rate was found — `source` is then
    `unsupported` or `unavailable`, and nothing is converted. `rate_date`
    is the date the rate applies to, which is never after the date that
    was asked about.
    """

    rate: Decimal | None
    rate_date: date | None
    source: str

    @property
    def resolved(self) -> bool:
        return self.rate is not None


@dataclass(frozen=True)
class FxConversion:
    """A resolved lookup plus the money it produced."""

    amount_base: Decimal | None
    rate: Decimal | None
    rate_date: date | None
    source: str


def rate_date_for(moment: datetime) -> date:
    """The calendar date a transaction is priced on, in UTC.

    Weekends normalise back to the preceding Friday. The ECB publishes on
    business days, and — this is the part that matters — the provider does
    not say so: asked for a Saturday it returns the carried-forward rate
    stamped with the Saturday. Normalising here means `fx_rate_date` is a
    day the rate could actually have been published on, and a weekend's
    transactions share one cache entry instead of inventing two.

    Public holidays are deliberately not normalised: there is no local
    TARGET calendar, and the provider already carries the rate forward.
    """
    if moment.tzinfo is None:
        # Timestamps are stored UTC by convention across this codebase;
        # SQLite hands some of them back naive.
        moment = moment.replace(tzinfo=UTC)
    day = moment.astimezone(UTC).date()
    weekday = day.weekday()  # Monday 0 … Saturday 5, Sunday 6
    if weekday >= _SATURDAY:
        return day - timedelta(days=weekday - 4)
    return day


def quantize_rate(rate: Decimal) -> Decimal:
    """Round a rate to the 8 decimal places of NUMERIC(18, 8)."""
    return rate.quantize(_RATE_SCALE, rounding=ROUND_HALF_UP)


def convert(amount: Decimal, rate: Decimal) -> Decimal:
    """Convert `amount` at `rate`, to the 4 decimal places of NUMERIC(19, 4).

    The multiplication runs under a widened Decimal context so the exact
    product is formed first and rounded once, rather than being rounded
    by the default context and then rounded again by the quantise.
    """
    with localcontext() as ctx:
        ctx.prec = _CONVERSION_PRECISION
        product = amount * rate
    return product.quantize(_AMOUNT_SCALE, rounding=ROUND_HALF_UP)


class FxService:
    """Resolves rates and converts amounts into the reporting currency.

    The provider is injected, so tests substitute a fake and never open a
    socket. `enabled` gates the network call only — with it off the cache
    still answers, and a miss reads `unavailable`.
    """

    def __init__(
        self,
        *,
        provider: FxRateProvider,
        base_currency: str = "USD",
        cache_max_age_days: int = 7,
        enabled: bool = True,
    ) -> None:
        self.base_currency = base_currency.upper()
        self._provider = provider
        self._cache_max_age_days = cache_max_age_days
        self._enabled = enabled

    # -- rate resolution ----------------------------------------------------

    def get_rate(self, db: Session, *, quote: str, on: date) -> FxLookup:
        """Resolve the rate for `quote` → the reporting currency on `on`.

        Order: identity, fresh cache, provider, stale cache, nothing. The
        first that answers wins. Never raises.
        """
        quote = (quote or "").strip().upper()

        if quote == self.base_currency:
            return FxLookup(rate=Decimal(1), rate_date=on, source=SOURCE_IDENTITY)

        if not _ISO_4217.match(quote):
            return FxLookup(rate=None, rate_date=None, source=SOURCE_UNSUPPORTED)

        # A rate for a date that has not happened yet does not exist, and
        # asking for one would spend a round-trip to be told so.
        if on > datetime.now(UTC).date():
            log.warning("FX lookup for %s refused: %s is in the future", quote, on)
            return FxLookup(rate=None, rate_date=None, source=SOURCE_UNAVAILABLE)

        cached = fx_rate_repository.latest_on_or_before(
            db,
            base=self.base_currency,
            quote=quote,
            on=on,
            max_age_days=self._cache_max_age_days,
        )
        if cached is not None:
            return FxLookup(
                rate=cached.rate, rate_date=cached.rate_date, source=SOURCE_CACHE
            )

        if self._enabled:
            try:
                fetched = self._provider.fetch_rate(
                    base=self.base_currency, quote=quote, on=on
                )
            except FxUnsupportedCurrencyError:
                # Permanent: this pair will not start working, so a stale
                # rate for it would be an answer to a different question.
                log.warning("FX provider does not carry %s->%s", quote, self.base_currency)
                return FxLookup(
                    rate=None, rate_date=None, source=SOURCE_UNSUPPORTED
                )
            except FxProviderError as exc:
                log.warning("FX provider failed for %s on %s: %s", quote, on, exc)
            else:
                if fetched is not None:
                    rate = quantize_rate(fetched.rate)
                    fx_rate_repository.upsert(
                        db,
                        base=self.base_currency,
                        quote=quote,
                        rate_date=fetched.rate_date,
                        rate=rate,
                        fetched_at=datetime.now(UTC),
                    )
                    return FxLookup(
                        rate=rate,
                        rate_date=fetched.rate_date,
                        source=SOURCE_LIVE,
                    )
                log.warning("FX provider has no rate for %s on %s", quote, on)

        # Stale fallback: the most recent rate that could have applied,
        # however old. Still bounded by `rate_date <= on`, so this is
        # always an older rate — never the current one.
        stale = fx_rate_repository.latest_on_or_before(
            db, base=self.base_currency, quote=quote, on=on
        )
        if stale is not None:
            log.warning(
                "FX falling back to a stale %s rate from %s for %s",
                quote,
                stale.rate_date,
                on,
            )
            return FxLookup(
                rate=stale.rate, rate_date=stale.rate_date, source=SOURCE_STALE
            )

        return FxLookup(rate=None, rate_date=None, source=SOURCE_UNAVAILABLE)

    # -- conversion ---------------------------------------------------------

    def convert_amount(
        self, db: Session, *, amount: Decimal, currency: str, on: date
    ) -> FxConversion:
        """Convert one amount, reporting which source priced it.

        A same-currency amount is copied verbatim rather than multiplied
        by one, so the identity path is deterministic by construction and
        cannot be perturbed by a rounding rule.
        """
        lookup = self.get_rate(db, quote=currency, on=on)
        if lookup.source == SOURCE_IDENTITY:
            return FxConversion(
                amount_base=amount,
                rate=Decimal(1),
                rate_date=lookup.rate_date,
                source=SOURCE_IDENTITY,
            )
        if lookup.rate is None:
            return FxConversion(
                amount_base=None,
                rate=None,
                rate_date=None,
                source=lookup.source,
            )
        return FxConversion(
            amount_base=convert(amount, lookup.rate),
            rate=lookup.rate,
            rate_date=lookup.rate_date,
            source=lookup.source,
        )

    # -- the ingestion entry point -----------------------------------------

    def enrich(self, db: Session, tx: Transaction) -> FxConversion:
        """Attach the derived FX columns to `tx`. Never raises.

        `tx.amount` and `tx.currency` are read and never written. On any
        failure — including one this module did not anticipate — the four
        derived columns are left describing the absence of a rate, and
        the caller carries on and scores the transaction.
        """
        try:
            if tx.created_at is None:
                # Cannot price a transaction without knowing when it
                # happened, and guessing "now" would be exactly the
                # silent current-rate fallback this contract forbids.
                log.warning("FX skipped for tx %s: no created_at", tx.id)
                result = _unavailable()
            else:
                result = self.convert_amount(
                    db,
                    amount=tx.amount,
                    currency=tx.currency,
                    on=rate_date_for(tx.created_at),
                )
        # Deliberately bare: an enrichment that can raise is an enrichment
        # that can stop a transaction being scored, which is the one thing
        # this path must never do.
        except Exception:
            log.exception("FX enrichment failed for tx %s", tx.id)
            result = _unavailable()

        tx.amount_base = result.amount_base
        tx.fx_rate = result.rate
        tx.fx_rate_date = result.rate_date
        tx.fx_source = result.source
        return result

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release the provider's connection pool, if it holds one."""
        close = getattr(self._provider, "close", None)
        if callable(close):
            close()


def _unavailable() -> FxConversion:
    return FxConversion(
        amount_base=None, rate=None, rate_date=None, source=SOURCE_UNAVAILABLE
    )


# ---------------------------------------------------------------------------
# Module-level singleton — built on first use, mirroring the explainer.
# ---------------------------------------------------------------------------

_singleton: FxService | None = None


def build_fx_service() -> FxService:
    """Construct an FxService from settings. Opens no connection."""
    settings = get_settings()
    provider = FrankfurterProvider(
        base_url=settings.fx_api_base_url,
        timeout_seconds=settings.fx_timeout_seconds,
    )
    return FxService(
        provider=provider,
        base_currency=settings.fx_base_currency,
        cache_max_age_days=settings.fx_cache_max_age_days,
        enabled=settings.fx_enabled,
    )


def get_fx_service() -> FxService:
    """FastAPI dependency: the process-wide FX service.

    Built lazily rather than in the lifespan hook, because unlike the
    model artifacts there is nothing to load and nothing that can fail at
    startup. Tests override this dependency to inject a fake provider.
    """
    global _singleton
    if _singleton is None:
        _singleton = build_fx_service()
    return _singleton


def shutdown_fx_service() -> None:
    """Close the pooled HTTP client, if one was ever opened."""
    global _singleton
    if _singleton is not None:
        _singleton.close()
        _singleton = None


def reset_fx_service_for_tests() -> None:
    """Test helper — clears the cached singleton so each test starts fresh."""
    global _singleton
    _singleton = None
