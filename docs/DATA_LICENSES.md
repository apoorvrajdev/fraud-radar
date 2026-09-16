# Data Sources, Licences and Provenance

Fraud Radar trains and evaluates on datasets it does not own. This file records what each source is, under what terms it is used, and what of it ever enters this repository.

**Status:** Phase 5A. Nothing external has been downloaded yet. Licence rows below are what the source advertises publicly; each is confirmed at retrieval time (Phase 5B) and the confirmed wording, with its retrieval date, is written into the dataset's `DatasetProvenance.license` field and into every run that uses it.

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
| **Files** | `fraudTrain.csv` (351,238,196 bytes) and `fraudTest.csv` (150,354,339 bytes), sizes from the Kaggle files API |
| **Coverage** | 1 Jan 2019 – 31 Dec 2020, 1,000 cards, 800 merchants (Kaggle description) |
| **Schema** | 23 columns: an unnamed index plus the 22 named columns listed in `ml/datasets/sparkov.py` |
| **Label** | `is_fraud`; 1 = a transaction the generator produced as fraudulent |
| **Used for** | Training, evaluation, temporal-drift experiment, rules audit (5C onward) |
| **Derived and committed** | Quality report, metrics, model card, feature-importance summaries |

**Reported but not independently verified** (the corpus has not been retrieved here, so these come from secondary sources and are recomputed at load time rather than trusted): row counts of 1,296,675 and 555,719, fraud prevalence of roughly 0.58% and 0.39%, and the exact boundary date between the two files. The adapter computes all counts from the bytes it reads, and the loader refuses to run if the header does not match the expected schema.

**Preprocessing applied by the adapter**, each recorded in `DatasetProvenance.preprocessing`:

- Card numbers, merchant names and transaction numbers are hashed to UUIDs (`uuid5`); no card-like value is stored or logged.
- The generator prefixes every merchant name with `fraud_`, on legitimate and fraudulent rows alike. It is stripped: it carries no signal, and leaving a token spelling "fraud" in a merchant name invites a future text model to latch onto it.
- The 14 source categories are folded onto this project's 12-category taxonomy. Category carries merchant *type*; the channel is carried separately by `is_card_present`, which is why the `_pos` and `_net` variants of a category share one canonical bucket. `health_fitness` has no exact equivalent and is mapped to `ENTERTAINMENT` — a discretionary recreation service at the same medium risk level, in the same 7xxx MCC family as health clubs. `ONLINE_SERVICE` was rejected: its MCC 5968 (direct marketing/subscription) asserts a card-not-present channel, while `health_fitness` carries no `_pos`/`_net` suffix and is treated as card-present. This remains a judgment call and is recorded as one in the field inventory.
- Cardholder name, address, coordinates, job and date of birth are dropped: no canonical field, and no reason to carry personal-looking detail.
- Risk tier and account age have no source equivalent and are filled with inert constants rather than plausible-looking values. Deriving an account age from the birth date would invent signal.

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
