# Data Sources, Licences and Provenance

Fraud Radar trains and evaluates on datasets it does not own. This file records what each source is, under what terms it is used, and what of it ever enters this repository.

**Status:** Sparkov retrieved and verified on 2026-09-17; its SHA-256 digests are not yet pinned in the manifest. Nothing else external has been downloaded. Licence rows below are what the source advertises publicly; each is confirmed at retrieval time (Phase 5B) and the confirmed wording, with its retrieval date, is written into the dataset's `DatasetProvenance.license` field and into every run that uses it.

---

## What is committed, and what is not

| Committed | Never committed |
|---|---|
| SHA-256 of every raw input file | The raw data files themselves |
| Derived statistics: counts, rates, percentiles, quality reports | Any row-level extract of a third-party dataset |
| Metrics, model cards, run metadata | Feature matrices built from third-party data (cached locally, gitignored) |

`backend/ml/data/raw/` and `backend/ml/data/cache/` are gitignored in full. Because raw bytes are absent, the file hash recorded in `run.json` is the durable link between a published number and the data that produced it — anyone can re-download, hash, and compare.

---

## Sources

### Fraud Radar synthetic generator (in-house)

| | |
|---|---|
| **Origin** | Synthetic |
| **Location** | `backend/ml/synthesis/`, reproducible from `seed=42` |
| **Licence** | MIT, with this repository |
| **Label** | `is_fraud`, assigned by the injection pattern that generated the row |
| **Used for** | The v1 baseline model currently served by the API |
| **Caveat** | Features were designed alongside the generator, so its numbers measure learnability of the generator, not detectability of fraud. This is the closed loop Phase 5 exists to break. |

### Sparkov — external synthetic benchmark (adapter shipped, 5B)

**This is simulated data.** It must never be described as real card transactions. Its value is that someone else's generator produced it, so it tests whether features and rules designed against this project's own simulator transfer at all.

| | |
|---|---|
| **Origin** | Synthetic. Kaggle's own subtitle: "Simulated Credit Card Transactions generated using Sparkov" |
| **Source** | Kaggle `kartik2112/fraud-detection`, published 2020-08-05 by Kartik Shenoy |
| **Licence** | **CC0: Public Domain**, confirmed 2026-09-16 via the Kaggle dataset API (`licenseName`) |
| **Generator** | [Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation) by Brandon Harris — **MIT**, confirmed 2026-09-16 via the GitHub API |
| **Citation** | Kartik Shenoy, "Credit Card Transactions Fraud Detection Dataset", Kaggle, 2020. Generated with Sparkov Data Generation (Brandon Harris). |
| **Files** | `fraudTrain.csv` (351,238,196 bytes) and `fraudTest.csv` (150,354,339 bytes), sizes from the Kaggle files API; the retrieved files have exactly these sizes |
| **Advertised coverage** | Kaggle's dataset description: 1 Jan 2019 – 31 Dec 2020, the cards of 1,000 customers, a pool of 800 merchants. This is the publisher's description of the generated data, not counts observed in the files. |
| **Observed coverage** | Wall clock 2019-01-01 00:00:18 – 2020-12-31 23:59:34. Across both files: **999 cards** and **693 merchant names**; the adapter builds **697 merchants**, one canonical entity per (name, canonical category). See *Observed on retrieval* below. |
| **Schema** | 23 columns: an unnamed index plus the 22 named columns listed in `ml/datasets/sparkov.py`, matched exactly in both retrieved files |
| **Label** | `is_fraud`; 1 = a transaction the generator produced as fraudulent |
| **Used for** | Training, evaluation, temporal-drift experiment, rules audit (5C onward) |
| **Derived and committed** | Quality report, metrics, model card, feature-importance summaries |

**Observed on retrieval** (2026-09-17). Computed from the retrieved bytes with the adapter's own reader, timestamp parser, exclusion gate and merchant identity. The adapter recomputes every count from the bytes it reads at each load, and the loader refuses to run if the header does not match the expected schema.

| | `fraudTrain.csv` | `fraudTest.csv` | Both files |
|---|---|---|---|
| Rows | 1,296,675 | 555,719 | 1,852,394 |
| Frauds (rate) | 7,506 (0.579%) | 2,145 (0.386%) | 9,651 (0.521%) |
| First wall-clock timestamp | 2019-01-01 00:00:18 | 2020-06-21 12:14:25 | 2019-01-01 00:00:18 |
| Last wall-clock timestamp | 2020-06-21 12:13:37 | 2020-12-31 23:59:34 | 2020-12-31 23:59:34 |
| Cards | 983 | 924 | 999 |
| Merchant names | 693 | 693 | 693 |

- **Entity counts.** 999 cards and 693 merchant names are the observed counts across both files. 697 is the adapter-level count of canonical merchants: a merchant is identified by (name, canonical category), and 4 of the 693 names appear under two canonical categories. Kaggle's 1,000 cards and 800 merchants are the publisher's advertised description, not observed counts; neither the data nor the adapter is adjusted to match them.
- **Where the files meet.** The test file begins 48 seconds after the train file ends. Neither file has a row inside the other's range, and no transaction number appears in both. 908 cards appear in both files, 75 only in train and 16 only in test.
- **Exclusions.** None: 0 rows under every exclusion reason.
- The row counts and fraud rates agree with the figures previously known only from secondary sources.

**Preprocessing applied by the adapter**, each recorded in `DatasetProvenance.preprocessing`:

- Card numbers, merchant names and transaction numbers are hashed to UUIDs (`uuid5`); no card-like value is stored or logged.
- The generator prefixes every merchant name with `fraud_`, on legitimate and fraudulent rows alike. It is stripped: it carries no signal, and leaving a token spelling "fraud" in a merchant name invites a future text model to latch onto it.
- The 14 source categories are folded onto this project's 12-category taxonomy. Category carries merchant *type*; the channel is carried separately by `is_card_present`, which is why the `_pos` and `_net` variants of a category share one canonical bucket. `health_fitness` has no exact equivalent and is mapped to `ENTERTAINMENT` — a discretionary recreation service at the same medium risk level, in the same 7xxx MCC family as health clubs. `ONLINE_SERVICE` was rejected: its MCC 5968 (direct marketing/subscription) asserts a card-not-present channel, while `health_fitness` carries no `_pos`/`_net` suffix and is treated as card-present. This remains a judgment call and is recorded as one in the field inventory.
- Cardholder name, address, coordinates, job and date of birth are dropped: no canonical field, and no reason to carry personal-looking detail.
- Risk tier and account age have no source equivalent and are filled with inert constants rather than plausible-looking values. Deriving an account age from the birth date would invent signal.
- **`trans_date_trans_time` is the authoritative transaction timestamp; `unix_time` is a diagnostic only** ([Phase 5D decision 17](adr/PHASE_5D_BENCHMARK_METHODOLOGY.md#17-on-sparkov-trans_date_trans_time-is-the-only-clock)). `unix_time` is not used for feature extraction, chronological ordering, splitting, customer history windows, temporal drift analysis or any other temporal modelling, and it is not reconstructed or corrected with an inferred offset. The quality report keeps its full offset distribution as a diagnostic.
  - *Earlier interpretation, from the generator's source code before the files were retrieved — not supported by the published files.* The generator samples an hour of day from a shopping daypart, builds a naive `datetime`, and derives the epoch from it with `.timestamp()` ([`profile_weights.py`](https://github.com/namebrandon/Sparkov_Data_Generation/blob/master/profile_weights.py)), which resolves a naive datetime against the local timezone of the machine that runs it. On that reading the epoch would differ from the wall clock by that machine's timezone offset and nothing else.
  - *Observed in the published files, 2026-09-17.* Every row's `unix_time` is exactly 2,556 or 2,557 whole days earlier than its wall clock, with no hours component, which puts the epochs in 2012–2013. The offset is 2,557 days through 2019-02-27 and from 2020-03-01 on, 2,556 days from 2019-03-01 through 2020-02-28, and both on 2019-02-28, a wall-clock date holding rows from `unix_time` dates 2012-02-28 and 2012-02-29. Wall-clock 2020-02-29 holds no rows. `fraudTrain.csv` is stored in `unix_time` order, so its wall clock steps backwards once. The full evidence is under [The `unix_time` offset](adr/PHASE_5D_BENCHMARK_METHODOLOGY.md#the-unix_time-offset).
- Timestamps are parsed once, with the explicit `%Y-%m-%d %H:%M:%S` format, and normalised to UTC. Ambiguous strings such as `01/02/2019` are excluded rather than guessed at.

### ULB Credit Card Fraud — real-world benchmark (planned, 5E)

| | |
|---|---|
| **Origin** | Real, anonymised |
| **Source** | Kaggle `mlg-ulb/creditcardfraud` — 284,807 European card transactions over two days in September 2013, 492 frauds (0.172%) |
| **Licence** | Open Database–family licence as displayed on the dataset card — **to be confirmed at retrieval** |
| **Attribution** | Machine Learning Group, Université Libre de Bruxelles, with Worldline |
| **Citation** | Dal Pozzolo, Caelen, Johnson and Bontempi, "Calibrating Probability with Undersampling for Unbalanced Classification", IEEE SSCI, 2015 |
| **Label** | `Class` |
| **Used for** | A separate benchmark track only. Features are PCA components of undisclosed inputs, so entity identity, merchants and geography are unrecoverable and no velocity feature can honestly be built on them. Never promoted to the serving API. |

### Frankfurter — FX reference rates (planned, 5F)

| | |
|---|---|
| **Origin** | Public reference data |
| **Source** | `https://frankfurter.dev` — open API over European Central Bank daily reference rates |
| **Licence** | Open API over public ECB data; no key required, self-hostable |
| **Used for** | Converting a transaction amount to a base currency for display and analytics at ingestion time. Cached locally. Not a model input. |
| **Failure policy** | Never blocks scoring: on timeout or outage the transaction is scored and stored with the FX source marked stale or unavailable. |

---

## Sources deliberately excluded

**IEEE-CIS.** The richest real card-not-present dataset available, but competition-rules licensing restricts redistribution and use, and its semantics are masked. Documented as possible future research; not used in Phase 5.

**FRED and market-data APIs** (Alpha Vantage, Finnhub, FMP). None carries card transactions or fraud labels, so none can train, evaluate or enrich a fraud model. A macro tile on the dashboard would imply a relationship to fraud decisions that does not exist.

---

## Adding a source

1. Record it here before writing the adapter: name, URL, licence as displayed, retrieval date, citation, what is derived from it, what is committed.
2. Add expected filenames and SHA-256 digests to the download manifest (5B).
3. Populate `DatasetProvenance` in the adapter so every run carries licence and hashes with it.
4. If the licence forbids redistribution of derived artifacts, say so here and keep those artifacts out of the repository.
