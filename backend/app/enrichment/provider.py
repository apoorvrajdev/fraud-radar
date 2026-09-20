"""The external FX rate provider — the only code here that opens a socket.

This module knows how to ask one question over HTTP ("what was the rate
for this pair on this date?") and how to refuse an answer it does not
trust. It knows nothing about transactions, Decimal money arithmetic,
caching or fallbacks — those live in `app/enrichment/fx.py`. The split is
the point: the business rules are testable without a network stack, and
the network handling is testable without a database.

Every failure leaves this module as a typed exception from the
`FxProviderError` family. Nothing propagates an `httpx` type outward, so
the service layer never has to know which client library is underneath.

Provider: Frankfurter (https://frankfurter.dev), an open API over ECB
reference rates. No key, no quota, self-hostable — `FX_API_BASE_URL`
points at whichever deployment you want.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

import httpx

_RATES_PATH = "/v2/rates"

# Frankfurter answers a bad currency code with 422. 400 and 404 are
# included for a self-hosted deployment that words the same refusal
# differently. Every other 4xx — 429 above all — is transient and is
# treated as a provider failure, which means the caller may fall back to
# a cached rate. A permanent refusal must not do that: see
# `FxUnsupportedCurrencyError`.
_PERMANENT_REFUSAL_STATUSES = frozenset({400, 404, 422})

# `fx_rates.rate` is NUMERIC(18, 8): eight decimal places leave ten
# integer digits.
_MAX_STORABLE_RATE = Decimal(10) ** 10


class FxProviderError(Exception):
    """Base class for every FX provider failure.

    Catching this catches "the provider did not give us a usable rate"
    without caring why. The subclasses exist for the two cases where the
    caller's response differs.
    """


class FxTimeoutError(FxProviderError):
    """The provider did not answer within the configured timeout."""


class FxResponseError(FxProviderError):
    """The provider answered with something we will not price money from.

    Raised for a non-2xx status, a body that is not the expected shape,
    a rate that is not a positive number, a date that will not parse, a
    pair that is not the one we asked for, and — importantly — a rate
    stamped with a date *later* than the one requested.
    """


class FxUnsupportedCurrencyError(FxProviderError):
    """The provider rejected the currency pair itself.

    Kept apart from the rest of the family because it is permanent. A
    timeout may succeed on the next transaction; a currency the provider
    does not carry will not. The caller records `unsupported` and does
    **not** reach for a stale cached rate.
    """


@dataclass(frozen=True)
class ProviderRate:
    """One rate as the provider reported it.

    `rate` means: 1 unit of `quote` = `rate` units of `base`, matching the
    direction fixed in `docs/FX_CONTRACT.md`. `rate_date` is the date the
    provider stamped the rate with, which this module guarantees is not
    after the date that was asked for.
    """

    base: str
    quote: str
    rate_date: date
    rate: Decimal


class FxRateProvider(Protocol):
    """What the FX service needs from a rate source.

    Implemented by `FrankfurterProvider` in production and by fakes in
    the tests. `fetch_rate` returns None when the provider is reachable
    and simply has no rate for that date; it raises an `FxProviderError`
    when the lookup failed.
    """

    def fetch_rate(
        self, *, base: str, quote: str, on: date
    ) -> ProviderRate | None:  # pragma: no cover - protocol declaration
        ...


class FrankfurterProvider:
    """Historical rate lookups against a Frankfurter deployment.

    One pooled `httpx.Client` is reused across calls so a burst of
    non-USD traffic does not pay for a TLS handshake per transaction. It
    is created lazily, so importing this module — which the API does at
    startup — opens nothing.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    # -- client lifecycle ---------------------------------------------------

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout_seconds)
        return self._client

    def close(self) -> None:
        """Close the pooled client if this provider created it.

        Called from the app's lifespan teardown. A client handed in by a
        caller (a test transport, say) belongs to that caller and is left
        alone.
        """
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    # -- the lookup ---------------------------------------------------------

    def fetch_rate(
        self, *, base: str, quote: str, on: date
    ) -> ProviderRate | None:
        """Fetch the rate for `quote` → `base` on `on`, or None if there is none.

        Note the parameter inversion against the wire format. Frankfurter
        returns *quote units per one base unit*; this contract wants
        *reporting-currency units per one transaction-currency unit*. So
        the transaction's currency is sent as Frankfurter's `base` and the
        reporting currency as its `quotes`, and the response is validated
        to name that pair back.
        """
        params = {
            "base": quote,
            "quotes": base,
            "date": on.isoformat(),
        }
        try:
            response = self._get_client().get(
                f"{self._base_url}{_RATES_PATH}",
                params=params,
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise FxTimeoutError(
                f"FX lookup for {quote}->{base} on {on} timed out "
                f"after {self._timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise FxProviderError(
                f"FX lookup for {quote}->{base} on {on} failed: {exc}"
            ) from exc

        if response.status_code in _PERMANENT_REFUSAL_STATUSES:
            raise FxUnsupportedCurrencyError(
                f"Provider does not carry {quote}->{base} "
                f"(HTTP {response.status_code})"
            )
        if response.status_code >= 400:
            raise FxResponseError(
                f"FX lookup for {quote}->{base} on {on} returned "
                f"HTTP {response.status_code}"
            )

        return self._parse(response.text, base=base, quote=quote, on=on)

    # -- response validation ------------------------------------------------

    @staticmethod
    def _parse(
        body: str, *, base: str, quote: str, on: date
    ) -> ProviderRate | None:
        """Validate one response body into a ProviderRate, or None if empty.

        `parse_float=Decimal` is load-bearing: it takes the rate from wire
        bytes straight to Decimal without an intervening binary float, so
        no rounding error is baked in before the money arithmetic starts.
        """
        try:
            payload = json.loads(body, parse_float=Decimal)
        except json.JSONDecodeError as exc:
            raise FxResponseError(
                f"FX response for {quote}->{base} on {on} is not JSON"
            ) from exc

        # An empty array is the provider saying "reachable, no rate for
        # that date" — a future date, or one before the series starts.
        # That is a missing rate, not a malformed response.
        if isinstance(payload, list) and not payload:
            return None

        if not isinstance(payload, list) or len(payload) != 1:
            raise FxResponseError(
                f"FX response for {quote}->{base} on {on} is not a "
                f"single-rate array"
            )

        entry = payload[0]
        if not isinstance(entry, dict):
            raise FxResponseError(
                f"FX response for {quote}->{base} on {on} is not an object"
            )

        # Frankfurter's base/quote are this contract's quote/base — see
        # the docstring on `fetch_rate`.
        got_quote = _require_str(entry, "base", base=base, quote=quote, on=on)
        got_base = _require_str(entry, "quote", base=base, quote=quote, on=on)
        if got_base.upper() != base.upper() or got_quote.upper() != quote.upper():
            raise FxResponseError(
                f"FX response names {got_quote}->{got_base}, asked for "
                f"{quote}->{base}"
            )

        raw_date = _require_str(entry, "date", base=base, quote=quote, on=on)
        try:
            rate_date = date.fromisoformat(raw_date)
        except ValueError as exc:
            raise FxResponseError(
                f"FX response for {quote}->{base} carries an unparseable "
                f"date {raw_date!r}"
            ) from exc

        # A rate published after the transaction cannot price it. The
        # provider has never been observed to do this; if it ever did, an
        # inverted comparison somewhere would otherwise convert money at a
        # future rate in silence.
        if rate_date > on:
            raise FxResponseError(
                f"FX response for {quote}->{base} is dated {rate_date}, "
                f"after the requested {on}"
            )

        rate = _require_decimal(entry, "rate", base=base, quote=quote, on=on)
        if rate <= 0:
            raise FxResponseError(
                f"FX response for {quote}->{base} on {on} carries a "
                f"non-positive rate {rate}"
            )
        # `fx_rates.rate` is NUMERIC(18, 8) — ten integer digits. A rate
        # above that cannot be stored, and finding out at INSERT time
        # would surface a database error from inside an enrichment that
        # is supposed to be incapable of disrupting anything.
        if rate >= _MAX_STORABLE_RATE:
            raise FxResponseError(
                f"FX response for {quote}->{base} on {on} carries a rate "
                f"{rate} too large for NUMERIC(18, 8)"
            )

        return ProviderRate(
            base=base.upper(),
            quote=quote.upper(),
            rate_date=rate_date,
            rate=rate,
        )


def _require_str(
    entry: dict[str, Any], key: str, *, base: str, quote: str, on: date
) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise FxResponseError(
            f"FX response for {quote}->{base} on {on} is missing a "
            f"usable {key!r}"
        )
    return value


def _require_decimal(
    entry: dict[str, Any], key: str, *, base: str, quote: str, on: date
) -> Decimal:
    value = entry.get(key)
    # `parse_float=Decimal` leaves whole numbers as int, so both are
    # legitimate here. A bool is an int in Python and is not a rate.
    if isinstance(value, bool) or not isinstance(value, Decimal | int):
        raise FxResponseError(
            f"FX response for {quote}->{base} on {on} is missing a "
            f"numeric {key!r}"
        )
    try:
        rate = Decimal(value)
    except (InvalidOperation, ValueError) as exc:  # pragma: no cover - defensive
        raise FxResponseError(
            f"FX response for {quote}->{base} on {on} carries an "
            f"unusable {key!r}"
        ) from exc
    if not rate.is_finite():
        raise FxResponseError(
            f"FX response for {quote}->{base} on {on} carries a "
            f"non-finite {key!r}"
        )
    return rate
