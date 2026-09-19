# Data Sources, Licences and Provenance

Fraud Radar trains and evaluates on datasets it does not own. This file records what each source is, under what terms it is used, and what of it ever enters this repository.

**Status:** Sparkov retrieved and verified on 2026-09-17; the SHA-256 digests of both files are pinned in `backend/ml/data/manifest.json`, so a load of any other bytes is refused. ULB retrieved and verified on 2026-09-18; the SHA-256 digest of its one file is pinned in the same manifest. Nothing else external has been downloaded. Licence rows below are what the source advertises publicly; each is confirmed at retrieval time (Phase 5B) and the confirmed wording, with its retrieval date, is written into the dataset's `DatasetProvenance.license` field and into every run that uses it.

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

### ULB Credit Card Fraud — real-world benchmark (manifest entry, 5E)

**This is real card-transaction data, anonymised by its publisher.** It is the one dataset in Phase 5 that no generator produced. Apart from the amount, a relative time and the label, every column is a PCA component of inputs the publisher withholds, so it is used on its own track, under the method fixed in [`PHASE_5E_ULB_BENCHMARK_METHODOLOGY.md`](adr/PHASE_5E_ULB_BENCHMARK_METHODOLOGY.md), and never promoted to the serving API.

The attribution, citation and descriptions below come from the source's own metadata, read through the Kaggle API on 2026-09-18. The licence comes from the same metadata and the dataset page's own wording, and its terms from the full texts of both licences it names, read on 2026-09-19. The file was retrieved and verified on 2026-09-18; what its bytes showed is recorded under [Observed at acquisition](adr/PHASE_5E_ULB_BENCHMARK_METHODOLOGY.md#observed-at-acquisition-2026-09-18), and it matches the advertised size, row count and fraud count.

| | |
|---|---|
| **Origin** | Real, anonymised. Kaggle's subtitle: "Anonymized credit card transactions labeled as fraudulent or genuine" |
| **Source** | Kaggle `mlg-ulb/creditcardfraud`, "Credit Card Fraud Detection", published by the Machine Learning Group of the Université Libre de Bruxelles (ULB) |
| **Licence** | **ODbL-1.0 for the database, DbCL-1.0 for its contents.** The Kaggle dataset API returns one licence, `DbCL-1.0` (`licenseName`, confirmed 2026-09-18), and the dataset page names that same selection "Database: Open Database, Contents: Database Contents" (read 2026-09-19). The [Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/) covers the individual contents and requires compliance with the [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/), which covers the database. See *Licence terms* below. |
| **Attribution** | Per the dataset description, collected and analysed during a research collaboration of Worldline and the Machine Learning Group of ULB |
| **Citation** | The works the publisher asks to be cited, listed below |
| **Files** | `creditcard.csv`, 150,828,752 bytes, the size the Kaggle files API advertises and the size retrieved; the API dates the file 2019-09-20. SHA-256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`, pinned in the manifest. |
| **Advertised coverage** | Transactions made by credit cards in September 2013 by European cardholders, over two days: 492 frauds out of 284,807 transactions (0.172%). This is the publisher's description, not counts observed in the file. |
| **Schema, as described** | `Time`: the seconds elapsed between each transaction and the first transaction in the dataset. `V1`–`V28`: principal components obtained with PCA; the original features are withheld for confidentiality. `Amount`: the transaction amount. `Class`: 1 for fraud, 0 otherwise. |
| **Label** | `Class`; 1 = fraud, 0 otherwise, as the description defines it |
| **Not stated by the publisher** | The currency of `Amount`; the calendar dates of the two days; which rows the PCA was fitted on; how the labels were established |
| **Used for** | The ULB track only, with its own featureset `ulb_pca_v1`. Never promoted to the serving API ([Phase 5E decisions 4 and 12](adr/PHASE_5E_ULB_BENCHMARK_METHODOLOGY.md#4-ulb-has-its-own-featureset-ulb_pca_v1-outside-the-production-registry)). |
| **Derived and committed** | The file digest. Once each exists: quality-report aggregates, run records and the generated ULB card, each carrying the notice below. Never rows or matrices. |

**Licence terms**, settled on 2026-09-19 from the full texts of both licences at opendatacommons.org and from the source's own metadata, before any ULB quality report, run record or card is committed ([Phase 5E decision 16](adr/PHASE_5E_ULB_BENCHMARK_METHODOLOGY.md#16-the-raw-file-is-external-and-local-only)). A bare section number refers to the licence last named before it.

- **What each licence covers.** The DbCL grants a copyright licence over the contents (DbCL §2.1), incorporates the ODbL's definitions (§1.0), states "You must comply with the ODbL" (§2.2), and leaves rights in the database to "the ODbL covering the Database" (§2.3). The dataset page names that database licence. Every use of ULB here is therefore held to the ODbL.
- **Committing is public use.** This repository is public, so anything pushed to it is used "Publicly" as the ODbL defines it (ODbL §1.0).
- **Committed ULB records are Produced Works.** Counts, rates, quality-report aggregates, metrics, thresholds, calibration summaries, run metadata, the ULB card and the ULB figures in this repository's documentation result from querying the contents (ODbL §1.0, "Produced Work"). Share-alike does not reach them (ODbL §4.5b), so they are published under this repository's MIT licence, but each carries the notice below (ODbL §4.3). The MIT licence does not relicense the ULB database or any database derived from it. The code is outside the ODbL (§2.3a).
- **The in-memory matrix is a Derivative Database.** The ULB track arranges the file's contents into the `ulb_pca_v1` matrix and adds a row-position id and an elapsed-time stamp to each row (ODbL §1.0, "Derivative Database"). Using it internally is not public (§4.5c), but publishing a Produced Work made from it counts as publicly using it (§4.4c). It is therefore held under the ODbL only, and the method of making it is offered below (§4.6).
- **The database and any Derivative Database are never conveyed.** The raw file, the matrix and anything else holding rows stay local and gitignored. Publishing any of them would bring in the ODbL's notice (§4.2) and share-alike (§4.4) terms; this repository does not publish them, whatever those terms would allow.
- **Trained models.** The ODbL does not address them. A model's split points are quantiles of feature values, not rows, so a model is treated as a Produced Work. `model.json` stays gitignored; a model ever published would carry the notice and the method offer.
- **Insubstantial parts.** The fold-boundary `Time` values in a run record are an insubstantial part of the contents (ODbL §6.2).
- **Not covered by either licence.** Privacy and data-protection rights in the contents (ODbL §2.4); committing no rows keeps this repository clear of them. Kaggle's own Terms of Use were not examined, because the page renders only in a browser.
- **The citation list** below is the publisher's request, not a licence term, and it does not replace the notice.

**Notice** (ODbL §4.3). Every published ULB-derived work carries it: this section, the acquisition findings in the Phase 5E record, each ULB run's provenance through the manifest's licence field, and, once they exist, the ULB card and the README's ULB table.

> Contains information from the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) database of the Machine Learning Group, ULB, which is available under the [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/); its contents are under the [Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/).

**Method offer** (ODbL §4.6b). Every ULB record computed through the track's loader or matrix is a Produced Work from a Derivative Database. The method of making that Derivative Database from the pinned file is offered, free of charge and in machine-readable form, as this repository's code: the ULB track in `backend/ml/tracks/ulb/`, at the commit each run records as `code_version` in its `run.json`. It adds no contents from outside the file. Those commits are pushed with or before the records they produced, and history is never rewritten to remove them.

**Citation requested by the publisher**, as listed on the dataset page on 2026-09-18:

1. Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
2. Andrea Dal Pozzolo, Olivier Caelen, Yann-Ael Le Borgne, Serge Waterschoot and Gianluca Bontempi. Learned lessons in credit card fraud detection from a practitioner perspective. Expert Systems with Applications, 41(10), 4915–4928, 2014.
3. Andrea Dal Pozzolo, Giacomo Boracchi, Olivier Caelen, Cesare Alippi and Gianluca Bontempi. Credit card fraud detection: a realistic modeling and a novel learning strategy. IEEE Transactions on Neural Networks and Learning Systems, 29(8), 3784–3797, 2018.
4. Andrea Dal Pozzolo. Adaptive Machine Learning for Credit Card Fraud Detection. PhD thesis, ULB Machine Learning Group, supervised by G. Bontempi.
5. Fabrizio Carcillo, Andrea Dal Pozzolo, Yann-Aël Le Borgne, Olivier Caelen, Yannis Mazzer and Gianluca Bontempi. Scarff: a scalable framework for streaming credit card fraud detection with Spark. Information Fusion, 41, 182–194, 2018.
6. Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen and Gianluca Bontempi. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization. International Journal of Data Science and Analytics, 5(4), 285–300, 2018.
7. Bertrand Lebichot, Yann-Aël Le Borgne, Liyun He, Frederic Oblé and Gianluca Bontempi. Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection. INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, 78–88, 2019.
8. Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen, Frederic Oblé and Gianluca Bontempi. Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection. Information Sciences, 2019.
9. Yann-Aël Le Borgne and Gianluca Bontempi. Reproducible Machine Learning for Credit Card Fraud Detection — Practical Handbook.
10. Bertrand Lebichot, Gianmarco Paldino, Wissam Siblini, Liyun He, Frederic Oblé and Gianluca Bontempi. Incremental learning strategies for credit cards fraud detection. International Journal of Data Science and Analytics.

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
