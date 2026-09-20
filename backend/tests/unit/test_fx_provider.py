"""Phase 5F — FrankfurterProvider: request shape and response validation.

Every test drives the provider through an `httpx.MockTransport`, so the
suite exercises the real client code — URL building, parameter encoding,
status handling, body parsing — without a socket. Nothing here needs the
internet, and nothing here is pinned to the live provider's data.

The response bodies are copies of what the live v2 API actually returned
when the provider was chosen (see docs/FX_CONTRACT.md).
"""
from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from typing import Any

import httpx
import pytest

from app.enrichment.provider import (
    FrankfurterProvider,
    FxProviderError,
    FxResponseError,
    FxTimeoutError,
    FxUnsupportedCurrencyError,
)

BASE_URL = "https://fx.example.test"
ON = date(2024, 1, 2)


def _provider(
    handler: Any, *, base_url: str = BASE_URL, timeout: float = 2.0
) -> FrankfurterProvider:
    """Build a provider wired to a MockTransport running `handler`."""
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout)
    return FrankfurterProvider(
        base_url=base_url, timeout_seconds=timeout, client=client
    )


def _json_handler(payload: Any, status: int = 200) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=json.dumps(payload))

    return handler


def _body_handler(body: str, status: int = 200) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    return handler


def _rate_payload(
    *,
    api_base: str = "EUR",
    api_quote: str = "USD",
    rate: str = "1.1022",
    on: str = "2024-01-02",
) -> list[dict[str, Any]]:
    """One live-shaped v2 record. `api_base`/`api_quote` are the wire names."""
    # json.dumps of a Decimal fails, and the wire carries a bare JSON
    # number, so the rate is embedded as a float here on purpose — this is
    # exactly the value the provider has to recover without float damage.
    return [
        {
            "date": on,
            "base": api_base,
            "quote": api_quote,
            "rate": float(rate),
        }
    ]


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_request_targets_the_v2_rates_endpoint_with_inverted_pair() -> None:
    """The transaction currency goes in `base`, the reporting one in `quotes`.

    Frankfurter reports quote-per-base; this contract stores
    reporting-per-transaction. Getting this inversion wrong would invert
    every converted amount, so it is pinned here.
    """
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, text=json.dumps(_rate_payload()))

    provider = _provider(handler)
    provider.fetch_rate(base="USD", quote="EUR", on=ON)

    assert seen["path"] == "/v2/rates"
    assert seen["url"].startswith(BASE_URL)
    assert seen["params"] == {
        "base": "EUR",      # the transaction's currency
        "quotes": "USD",    # the reporting currency
        "date": "2024-01-02",
    }


def test_base_url_trailing_slash_does_not_double_up() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, text=json.dumps(_rate_payload()))

    provider = _provider(handler, base_url=BASE_URL + "/")
    provider.fetch_rate(base="USD", quote="EUR", on=ON)

    assert seen["path"] == "/v2/rates"


def test_requests_for_different_dates_are_distinct() -> None:
    """Deterministic parameters: the date asked for is the date sent."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked = request.url.params["date"]
        seen.append(asked)
        # Echo the date back, as the live API does.
        return httpx.Response(200, text=json.dumps(_rate_payload(on=asked)))

    provider = _provider(handler)
    provider.fetch_rate(base="USD", quote="EUR", on=date(2024, 1, 2))
    provider.fetch_rate(base="USD", quote="EUR", on=date(2023, 6, 30))

    assert seen == ["2024-01-02", "2023-06-30"]


# ---------------------------------------------------------------------------
# Successful parse
# ---------------------------------------------------------------------------


def test_successful_lookup_returns_the_rate_as_exact_decimal() -> None:
    """The wire number reaches Decimal without passing through a float.

    `Decimal(1.1022)` is 1.10219999999999996...; the equality below only
    holds if the body was parsed with `parse_float=Decimal`.
    """
    provider = _provider(_json_handler(_rate_payload(rate="1.1022")))

    result = provider.fetch_rate(base="USD", quote="EUR", on=ON)

    assert result is not None
    assert result.rate == Decimal("1.1022")
    # Four decimal places — not the ~50 that constructing a Decimal from
    # the binary float 1.1022 would drag in.
    assert result.rate.as_tuple().exponent == -4
    assert result.base == "USD"
    assert result.quote == "EUR"
    assert result.rate_date == date(2024, 1, 2)


def test_lowercase_pair_in_the_response_is_accepted() -> None:
    """Case is not signal — a self-hosted deployment may lowercase codes."""
    provider = _provider(
        _json_handler(_rate_payload(api_base="eur", api_quote="usd"))
    )

    result = provider.fetch_rate(base="USD", quote="EUR", on=ON)

    assert result is not None
    assert (result.base, result.quote) == ("USD", "EUR")


def test_whole_number_rate_is_accepted() -> None:
    """A rate of exactly 1 arrives as a JSON int, not a float."""
    provider = _provider(
        _body_handler(
            '[{"date":"2024-01-02","base":"EUR","quote":"USD","rate":1}]'
        )
    )

    result = provider.fetch_rate(base="USD", quote="EUR", on=ON)

    assert result is not None
    assert result.rate == Decimal("1")


def test_rate_dated_earlier_than_requested_is_accepted() -> None:
    """A provider that reports the true business day is fine — it is older."""
    provider = _provider(
        _json_handler(_rate_payload(on="2023-12-29"))
    )

    result = provider.fetch_rate(base="USD", quote="EUR", on=ON)

    assert result is not None
    assert result.rate_date == date(2023, 12, 29)


# ---------------------------------------------------------------------------
# Missing rate — reachable provider, no data
# ---------------------------------------------------------------------------


def test_empty_array_is_a_missing_rate_not_an_error() -> None:
    """What the live API returns for a future date. Missing, not malformed."""
    provider = _provider(_json_handler([]))

    assert provider.fetch_rate(base="USD", quote="EUR", on=ON) is None


# ---------------------------------------------------------------------------
# Transport failures
# ---------------------------------------------------------------------------


def test_timeout_raises_fx_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider = _provider(handler)

    with pytest.raises(FxTimeoutError):
        provider.fetch_rate(base="USD", quote="EUR", on=ON)


def test_connect_error_raises_fx_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    provider = _provider(handler)

    with pytest.raises(FxProviderError) as exc:
        provider.fetch_rate(base="USD", quote="EUR", on=ON)
    # Not a timeout, and not a permanent refusal — a plain transient.
    assert not isinstance(exc.value, FxTimeoutError | FxUnsupportedCurrencyError)


def test_httpx_errors_do_not_escape_the_provider() -> None:
    """No `httpx` type reaches the caller — the client library stays here."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    provider = _provider(handler)

    with pytest.raises(FxProviderError) as exc:
        provider.fetch_rate(base="USD", quote="EUR", on=ON)
    assert not isinstance(exc.value, httpx.HTTPError)


# ---------------------------------------------------------------------------
# Status handling — permanent refusal vs transient failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 404, 422])
def test_permanent_refusal_statuses_mean_unsupported_currency(
    status: int,
) -> None:
    """The live API answers an unknown code with 422.

    This is classified apart from the transient failures because the
    service must not answer it with a stale cached rate — the pair is not
    going to start working.
    """
    provider = _provider(_json_handler({"message": "not found"}, status=status))

    with pytest.raises(FxUnsupportedCurrencyError):
        provider.fetch_rate(base="USD", quote="ZZZ", on=ON)


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_transient_statuses_are_provider_errors_not_unsupported(
    status: int,
) -> None:
    """429 especially: rate-limited is "try later", not "never works"."""
    provider = _provider(_json_handler({"message": "slow down"}, status=status))

    with pytest.raises(FxProviderError) as exc:
        provider.fetch_rate(base="USD", quote="EUR", on=ON)
    assert not isinstance(exc.value, FxUnsupportedCurrencyError)


# ---------------------------------------------------------------------------
# Malformed responses
# ---------------------------------------------------------------------------


def test_non_json_body_is_a_response_error() -> None:
    provider = _provider(_body_handler("<html>502 Bad Gateway</html>"))

    with pytest.raises(FxResponseError):
        provider.fetch_rate(base="USD", quote="EUR", on=ON)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"rates": {"USD": 1.1}}, id="object-not-array"),
        pytest.param(["EUR"], id="array-of-strings"),
        pytest.param(
            [
                {"date": "2024-01-02", "base": "EUR", "quote": "USD", "rate": 1.1},
                {"date": "2024-01-02", "base": "EUR", "quote": "GBP", "rate": 0.9},
            ],
            id="two-rates-for-one-question",
        ),
    ],
)
def test_wrong_envelope_shape_is_a_response_error(payload: Any) -> None:
    provider = _provider(_json_handler(payload))

    with pytest.raises(FxResponseError):
        provider.fetch_rate(base="USD", quote="EUR", on=ON)


@pytest.mark.parametrize("missing", ["date", "base", "quote", "rate"])
def test_missing_field_is_a_response_error(missing: str) -> None:
    entry = _rate_payload()[0]
    del entry[missing]
    provider = _provider(_json_handler([entry]))

    with pytest.raises(FxResponseError):
        provider.fetch_rate(base="USD", quote="EUR", on=ON)


def test_response_naming_a_different_pair_is_refused() -> None:
    """The guard that would catch the provider flipping its rate direction."""
    provider = _provider(
        _json_handler(_rate_payload(api_base="USD", api_quote="EUR"))
    )

    with pytest.raises(FxResponseError, match="asked for"):
        provider.fetch_rate(base="USD", quote="EUR", on=ON)


def test_unparseable_date_is_a_response_error() -> None:
    provider = _provider(_json_handler(_rate_payload(on="02/01/2024")))

    with pytest.raises(FxResponseError):
        provider.fetch_rate(base="USD", quote="EUR", on=ON)


def test_rate_dated_after_the_request_is_refused() -> None:
    """A future rate can never price a past transaction — not even one day."""
    provider = _provider(_json_handler(_rate_payload(on="2024-01-03")))

    with pytest.raises(FxResponseError, match="after the requested"):
        provider.fetch_rate(base="USD", quote="EUR", on=ON)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param('"1.1022"', id="string"),
        pytest.param("null", id="null"),
        pytest.param("true", id="bool"),
    ],
)
def test_non_numeric_rate_is_a_response_error(body: str) -> None:
    provider = _provider(
        _body_handler(
            '[{"date":"2024-01-02","base":"EUR","quote":"USD",'
            f'"rate":{body}}}]'
        )
    )

    with pytest.raises(FxResponseError):
        provider.fetch_rate(base="USD", quote="EUR", on=ON)


@pytest.mark.parametrize("rate", ["0", "-1.1022"])
def test_non_positive_rate_is_refused(rate: str) -> None:
    provider = _provider(
        _body_handler(
            '[{"date":"2024-01-02","base":"EUR","quote":"USD",'
            f'"rate":{rate}}}]'
        )
    )

    with pytest.raises(FxResponseError, match="non-positive"):
        provider.fetch_rate(base="USD", quote="EUR", on=ON)


def test_rate_too_large_for_the_column_is_refused() -> None:
    """Refused here rather than as a database error inside enrichment."""
    provider = _provider(
        _body_handler(
            '[{"date":"2024-01-02","base":"EUR","quote":"USD",'
            '"rate":1e12}]'
        )
    )

    with pytest.raises(FxResponseError, match="too large"):
        provider.fetch_rate(base="USD", quote="EUR", on=ON)


def test_non_finite_rate_is_refused() -> None:
    """Python's json accepts bare Infinity; money arithmetic does not."""
    provider = _provider(
        _body_handler(
            '[{"date":"2024-01-02","base":"EUR","quote":"USD",'
            '"rate":Infinity}]'
        )
    )

    with pytest.raises(FxResponseError):
        provider.fetch_rate(base="USD", quote="EUR", on=ON)


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------


def test_close_leaves_an_injected_client_alone() -> None:
    """A client the caller supplied belongs to the caller."""
    client = httpx.Client(transport=httpx.MockTransport(_json_handler([])))
    provider = FrankfurterProvider(
        base_url=BASE_URL, timeout_seconds=1.0, client=client
    )

    provider.close()

    assert not client.is_closed


def test_construction_opens_nothing() -> None:
    """Importing and building the provider must not create a client.

    The API constructs this at startup; a socket opened there would be a
    startup dependency on an external service.
    """
    provider = FrankfurterProvider(base_url=BASE_URL, timeout_seconds=1.0)

    assert provider._client is None
    provider.close()  # a no-op, and must not raise
