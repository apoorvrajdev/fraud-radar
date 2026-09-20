# FX Enrichment Contract

**Phase 5F.** What a converted amount on a Fraud Radar transaction means, and what
it means when there isn't one.

This is a contract, not a methodology record. The benchmark ADRs exist because a
measurement method has to be frozen before the data is seen; nothing here has that
property. The rules below are short enough to hold in your head, and the tests in
`backend/tests/unit/test_fx_*.py` and `backend/tests/integration/test_fx_ingestion.py`
hold them to it.

## The one-line version

Every transaction keeps the amount and the currency it was submitted with. FX
enrichment only ever *adds* a derived reporting figure beside them, and when it
cannot, it says so instead of guessing.

## Reporting currency

USD, from `FX_BASE_CURRENCY`. Every `amount_base` in the database is denominated in
it.

It is a setting because the code should not hard-code a business choice, but
changing it does not re-price anything: the `amount_base` values already stored were
computed against the old base and would silently start being read as the new one.
Moving the reporting currency is a backfill migration, not a config flip.

## Currency representation

ISO 4217 alpha-3, uppercase — the same values `transactions.currency` already
carries, unchanged by this phase.

## Rate direction

One rate is stored per (base, quote, date), and it means:

> **1 unit of `quote` = `rate` units of `base`.**

So conversion is always a multiplication:

```
amount_base = amount × fx_rate
```

There is no division anywhere in the conversion path, and no pair of "which way
round is this rate" helpers. `fx_rates.base` is the reporting currency; `fx_rates.quote`
is the currency the cardholder was charged in.

## Transaction-date semantics

A transaction is converted at the rate for **the calendar date it was created**, in
UTC: `tx.created_at` → UTC → `.date()`. A naive timestamp is read as UTC, matching
the rest of the codebase. A transaction with no timestamp is not converted.

Saturdays and Sundays are normalised back to the preceding Friday **before** the
lookup. Two reasons, and the second is the load-bearing one:

1. The ECB publishes on TARGET business days, so a weekend has no rate of its own.
2. The provider does not tell you that. Asked for a Saturday, Frankfurter returns
   the carried-forward rate stamped **with the Saturday's date** — it looks like a
   Saturday observation and is not one. Normalising locally means the date stored in
   `fx_rate_date` is a day the rate could actually have been published on, and a
   weekend's transactions share one cache entry instead of two invented ones.

Public holidays are not normalised — there is no local holiday calendar, and adding
one would be a third-party dependency to answer a question the provider already
answers by carrying the rate forward. So `fx_rate_date` is the date we resolved
against, which is always a weekday and is usually, but not always, the date the rate
was actually published.

A date in the future is refused locally, with no network call.

## Precision and rounding

Money never touches a binary float on this path.

- The provider's JSON is parsed with `parse_float=Decimal`, so a rate goes from
  wire bytes to `Decimal` without an intervening `float`.
- Rates are quantised to **8 decimal places**, `ROUND_HALF_UP`, matching
  `NUMERIC(18, 8)`.
- The multiplication runs inside a `decimal.localcontext(prec=38)` so the product
  of a 19-digit amount and an 8-decimal rate is computed exactly rather than being
  rounded by the default 28-digit context part-way through.
- The product is then quantised to **4 decimal places**, `ROUND_HALF_UP`, matching
  `NUMERIC(19, 4)` — the same scale the original `amount` column uses.

Display rounding to 2 dp stays where it already was, in the stats service at the API
boundary. The database keeps 4.

## Same-currency conversion

A transaction already in the reporting currency is converted with no lookup, no
cache read, no network call and no arithmetic:

```
fx_rate      = 1
amount_base  = amount        (copied verbatim — not multiplied, not re-quantised)
fx_rate_date = the resolved date
fx_source    = "identity"
```

This is the overwhelmingly common path in this system, and it is deterministic by
construction rather than by a rate that happens to equal 1.

## Resolution order

For anything else, in order, stopping at the first that answers:

| # | Step | `fx_source` |
|---|---|---|
| 1 | Cache: the latest rate for the pair with `rate_date ≤ on`, no older than 7 days | `cache` |
| 2 | Provider: one HTTP GET, 2 s timeout, result written into the cache | `live` |
| 3 | Stale fallback: the latest cached rate for the pair with `rate_date ≤ on`, any age | `stale` |
| 4 | Nothing | `unavailable` |

`FX_ENABLED=false` removes step 2 only. The cache still resolves; a miss reads
`unavailable`. It gates the network, not the enrichment.

## No silent fallback to the current rate

Every cache read in the table above — the fresh one and the stale one — is
constrained to `rate_date ≤ on`. A rate dated after the transaction can never be
selected for it, so "last month's transaction priced at today's rate" is not a
discouraged path, it is an unreachable one.

The stale fallback is therefore never *newer* than the transaction: it is an older
rate, and the row records that it is old by carrying `fx_source = "stale"` and the
`fx_rate_date` it actually used. Anyone reading a converted amount can see which of
the four sources produced it.

## Failure behaviour

| Condition | Result |
|---|---|
| Currency is not three uppercase letters | `unsupported`, no network call |
| Provider rejects the pair (400 / 404 / 422) | `unsupported`, no stale fallback — the pair will not start working |
| Provider times out | stale fallback, else `unavailable` |
| Network error, 429, or 5xx | stale fallback, else `unavailable` |
| Provider returns an empty result for the date | stale fallback, else `unavailable` |
| Malformed response (see below) | stale fallback, else `unavailable` |
| Requested date is in the future | `unavailable`, no network call |
| Transaction has no timestamp | `unavailable`, no network call |

A response is malformed if it is not JSON, is not a single-element array, is missing
a field, names a different base or quote than was asked for, carries a
non-numeric or non-positive rate, has an unparseable date, or is stamped with a date
**later** than the one requested.

`unsupported` is kept apart from `unavailable` on purpose: one says this currency
will never convert, the other says this conversion did not happen today. They want
different responses from an operator.

**Nothing here raises into the scoring path.** FX enrichment is wrapped so that a
provider outage cannot stop a transaction being scored, decided, persisted or
audited. A transaction that could not be converted is a completely normal row with
four null columns.

## What is stored

Four additive, nullable columns on `transactions`, and one cache table:

```
transactions.amount_base    NUMERIC(19,4)   the derived figure, in the reporting currency
transactions.fx_rate        NUMERIC(18,8)   the rate it was derived with
transactions.fx_rate_date   DATE            the date that rate is for
transactions.fx_source      VARCHAR(16)     identity | live | cache | stale | unsupported | unavailable

fx_rates(base, quote, rate_date, rate, fetched_at)    PRIMARY KEY (base, quote, rate_date)
```

`amount` and `currency` are never written by this path. Three CHECK constraints keep
the derived fields coherent: `fx_source` is one of the six values, `fx_rate > 0`
where present, and `amount_base` and `fx_rate` are either both set or both null — a
half-converted row cannot exist.

`fx_source` is also copied into the scoring decision's audit payload whenever it is
not `identity`, so a decision taken on a row priced with a stale rate, or with no
rate at all, is visible in the audit trail rather than only in the transaction row.

## Provenance

Rates come from [Frankfurter](https://frankfurter.dev), an open-source API over
European Central Bank reference rates. No key, no quota, self-hostable — point
`FX_API_BASE_URL` at your own deployment and nothing else changes. The endpoint used
is `GET /v2/rates?base=…&quotes=…&date={YYYY-MM-DD}`.

Note that Frankfurter's `base` and this contract's `base` are opposite ends of the
same rate. Frankfurter returns *quote units per one base unit*, and what we want is
*reporting-currency units per one transaction-currency unit* — so the transaction's
currency goes in Frankfurter's `base` parameter and the reporting currency goes in
its `quotes` parameter. Converting a EUR transaction asks for
`?base=EUR&quotes=USD`, which answers `{"base":"EUR","quote":"USD","rate":1.1022}`,
and 1.1022 is exactly the multiplier this contract stores. The provider validates
that the response names the pair it asked for, so an API that ever flipped this
would fail loudly rather than invert every converted amount.

No FX data is committed to this repository. Rates are cached at runtime in the
`fx_rates` table of the local (gitignored) SQLite database. Licence and citation
details are in [`DATA_LICENSES.md`](DATA_LICENSES.md).

## How it is displayed

Phase 5G renders this contract. One `Amount` component
(`frontend/src/components/ui/Amount.tsx`) owns every per-transaction
money display, so the rule that the original value is authoritative holds
by construction rather than by each call site agreeing:

- A transaction already in the reporting currency renders as one line.
- A converted one shows the original first, with the derived figure
  below it marked `≈` and visually subordinate, carrying the rate and
  its date on hover. It can never be mistaken for the charged amount.
- A `stale`-priced row says so and shows which day's rate was used.
- An unconverted row shows the original alone, with the reason on hover.
  This state is rendered quietly: a row with no rate is normal, not an
  error.

`frontend/src/lib/format.ts` keeps the two kinds of money apart —
`formatMoneyIn(value, currency)` for a transaction, `formatMoney` for an
aggregate the backend already summed in the reporting currency — and
says which is which at the top of the file. A currency code `Intl` does
not recognise degrades to `125.00 XYZ` rather than rendering a dollar
sign, because the backend accepts any three-letter code.

`GET /api/v1/model` reports the reporting currency alongside the model
identity, so the dashboard labels its aggregates from the backend's
setting instead of a second hard-coded copy.

## Deliberately not done

- **The model does not see `amount_base`.** The 17-feature production registry is
  unchanged, the served model is unchanged, and no benchmark run knows FX exists.
  Whether a base-currency amount helps the classifier is an evaluation question, and
  answering it by assumption would put an untested feature into a scored decision.
- **Amount *filters* on the transactions list still filter on `amount`** — what the
  cardholder was actually charged. Only the reporting aggregates (volume and the
  country breakdown) use `COALESCE(amount_base, amount)`, because summing mixed
  currencies is the thing that is actually wrong.
- **No retries, no circuit breaker, no background refresh.** The cache plus the
  stale fallback is the entire resilience story. A retry against a 2-second timeout
  on a synchronous ingestion path buys a second failure, not a success.
- **The offline benchmark tracks have no FX hook.** Every offline dataset is
  single-currency, so a hook there would be dead code inside the benchmark tooling.
