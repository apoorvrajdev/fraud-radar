"""Phase 5D — the rules audit: production rules on a run's rows, with serving's context.

Two fixtures:

- A small hand-built dataset in which every one of the six rules fires, with
  customers that carry account-open timestamps. The same rows are written to
  SQLite, so every audit context and every rule result can be compared with
  what the live scoring service loads and evaluates for the same transaction.
- A Sparkov-format corpus trained into a fixture run through the real adapter,
  batch builder and feature cache. Its special rows are placed so that which
  rule fires on which row is known in advance: three-transaction bursts inside
  120 seconds, amounts above the $5,000 ceiling, card-not-present rows above
  $500 at 03:00, and near misses of each. Sparkov customers have no
  account-open timestamp, so the dormant-account rule is not evaluable there.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.fraud.rules import evaluate_all
from app.fraud.transaction_context import TransactionContext
from app.models.base import Base
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction
from app.services import scoring
from ml import loading, train
from ml.datasets.base import CanonicalDataset, DataOrigin, DatasetProvenance
from ml.datasets.sparkov import SparkovAdapter, transaction_id_for
from ml.experiments import rules_audit
from ml.experiments.rules_audit import (
    RULES,
    Evaluability,
    RulesAuditError,
    audit_rows,
    rule_evaluability,
    serving_contexts,
)
from ml.features.batch import HISTORY_WINDOW
from ml.paths import RAW_DATA_DIR
from ml.run_verification import RunVerificationError
from ml.splits import chronological_split
from ml.tuning import TuningResult
from tests.unit.test_dataset_sparkov import _row, _write_csv, fixture_adapter
from tests.unit.test_features_batch_parity import _customer, _merchant, _tx
from tests.unit.test_run_verification import train_fixture_run

ANCHOR = datetime(2025, 3, 1, 12, 0, tzinfo=UTC)
EVALUABLE_WITHOUT_DORMANT = [name for name, _ in RULES if name != "dormant_account_high_value"]


# ---------------------------------------------------------------------------
# A hand-built dataset where every rule fires
# ---------------------------------------------------------------------------


def _entities(*, with_account_open: bool = True) -> tuple[dict[str, Customer], dict[str, Merchant]]:
    opened = {"c-a": 900, "c-b": 400, "c-c": 30}
    customers = {
        "c-a": _customer("c-a", country="US", risk_tier="LOW", age_days=900),
        "c-b": _customer("c-b", country="US", risk_tier="LOW", age_days=400),
        "c-c": _customer("c-c", country="GB", risk_tier="MEDIUM", age_days=30),
    }
    if with_account_open:
        for customer_id, days in opened.items():
            customers[customer_id].created_at = ANCHOR - timedelta(days=days)
    merchants = {
        "m-1": _merchant("m-1", country="US", category="GROCERY", risk="LOW"),
    }
    return customers, merchants


def _transactions() -> list[Transaction]:
    def tx(tx_id: str, customer: str, amount: str, at: datetime, **options: Any) -> Transaction:
        return _tx(tx_id, customer_id=customer, merchant_id="m-1", amount=amount, at=at, **options)

    rows = [
        tx("a-old-1", "c-a", "40.00", ANCHOR - timedelta(days=40)),
        tx("a-old-2", "c-a", "55.00", ANCHOR - timedelta(days=20)),
        tx("a-old-3", "c-a", "61.00", ANCHOR - timedelta(days=10)),
        # A burst: the third row inside 120 seconds fires velocity_burst.
        tx("a-burst-1", "c-a", "20.00", ANCHOR - timedelta(hours=2)),
        tx("a-burst-2", "c-a", "21.00", ANCHOR - timedelta(hours=2) + timedelta(seconds=50)),
        tx("a-burst-3", "c-a", "22.00", ANCHOR - timedelta(hours=2) + timedelta(seconds=100)),
        # Two rows at the same instant: neither is in the other's history.
        tx("a-tie-1", "c-a", "30.00", ANCHOR - timedelta(hours=1)),
        tx("a-tie-2", "c-a", "31.00", ANCHOR - timedelta(hours=1)),
        # Another country within 60 minutes fires geo_velocity_impossible.
        tx("a-abroad", "c-a", "80.00", ANCHOR - timedelta(minutes=30), country="FR"),
        tx("a-ceiling", "c-a", "6000.00", ANCHOR),
        tx("a-risky", "c-a", "800.00", ANCHOR + timedelta(minutes=5), country="RU"),
        tx("a-night", "c-a", "700.00", ANCHOR + timedelta(hours=15), card_present=False),
        # An old row evicted from the 180-day window, then a dormant high-value return.
        tx("b-old", "c-b", "25.00", ANCHOR - timedelta(days=300)),
        tx("b-return", "c-b", "2500.00", ANCHOR - timedelta(days=1)),
        # A young account's high-value row: not dormant.
        tx("c-young", "c-c", "1500.00", ANCHOR - timedelta(days=3)),
    ]
    rows.sort(key=lambda row: (row.created_at, row.id))
    return rows


def _dataset(*, with_account_open: bool = True) -> CanonicalDataset:
    customers, merchants = _entities(with_account_open=with_account_open)
    rows = _transactions()
    frauds = {"a-burst-3", "a-ceiling", "b-return"}
    return CanonicalDataset(
        customers=customers,
        merchants=merchants,
        transactions=rows,
        labels={row.id: int(row.id in frauds) for row in rows},
        provenance=DatasetProvenance(
            name="audit-fixture",
            version="v1",
            origin=DataOrigin.SYNTHETIC,
            source_url="",
            citation="fixture",
            license="CC0",
            label_field="is_fraud",
            label_definition="1 = fraud",
        ),
    ).validate()


def _labels(dataset: CanonicalDataset) -> np.ndarray:
    return np.asarray([dataset.labels[tx.id] for tx in dataset.transactions], dtype=np.int64)


@pytest.fixture
def db_session() -> Iterator[Session]:
    """The same rows in SQLite, for the live scoring service to load."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()
    customers, merchants = _entities()
    session.add_all([*customers.values(), *merchants.values(), *_transactions()])
    session.commit()
    try:
        yield session
    finally:
        session.close()


def _utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Parity with the scoring service
# ---------------------------------------------------------------------------


def test_the_audit_window_is_the_serving_window() -> None:
    assert HISTORY_WINDOW == scoring._RECENT_HISTORY_WINDOW == timedelta(days=180)


def test_the_audit_rules_are_evaluate_all_in_order(db_session: Session) -> None:
    """Every production rule, in the order the scoring service runs them."""
    for tx in db_session.query(Transaction).all():
        context = scoring._load_context(db_session, tx)

        assert [rule(context) for _, rule in RULES] == evaluate_all(context)
        assert [name for name, _ in RULES] == [result.rule_name for result in evaluate_all(context)]


def test_each_context_holds_what_the_scoring_service_loads(db_session: Session) -> None:
    dataset = _dataset()

    contexts = dict(serving_contexts(dataset))

    assert len(contexts) == dataset.n_rows
    for index, tx in enumerate(dataset.transactions):
        served = scoring._load_context(db_session, db_session.get(Transaction, tx.id))
        audited = contexts[index]
        assert audited.transaction is tx
        assert {prior.id for prior in audited.recent_transactions} == {
            prior.id for prior in served.recent_transactions
        }, tx.id
        # Newest first. Rows sharing an instant may come in either order in SQL.
        assert [prior.created_at for prior in audited.recent_transactions] == [
            _utc(prior.created_at) for prior in served.recent_transactions
        ], tx.id


def test_each_rule_result_is_the_one_the_scoring_service_evaluates(db_session: Session) -> None:
    """Rule, outcome and severity match exactly.

    Reason texts are compared only for presence: SQLite stores amounts at four
    decimal places, so the served reason reads $2500.0000 where the canonical
    row reads $2500.00. The audit records no reason text.
    """
    dataset = _dataset()

    def outcomes(results: list[Any]) -> list[tuple[str, bool, str, bool]]:
        return [
            (result.rule_name, result.triggered, result.severity.value, result.reason is not None)
            for result in results
        ]

    for index, context in serving_contexts(dataset):
        tx = dataset.transactions[index]
        served = evaluate_all(scoring._load_context(db_session, db_session.get(Transaction, tx.id)))
        audited = [rule(context) for _, rule in RULES]
        assert outcomes(audited) == outcomes(served), tx.id


def test_the_fixture_fires_every_rule(db_session: Session) -> None:
    fired = {
        result.rule_name
        for tx in db_session.query(Transaction).all()
        for result in evaluate_all(scoring._load_context(db_session, tx))
        if result.triggered
    }

    assert fired == {name for name, _ in RULES}


def test_a_row_is_never_in_its_own_history_nor_a_same_instant_rows() -> None:
    contexts = {
        context.transaction.id: context for _, context in serving_contexts(_dataset())
    }

    for tie, other in (("a-tie-1", "a-tie-2"), ("a-tie-2", "a-tie-1")):
        history = {prior.id for prior in contexts[tie].recent_transactions}
        assert tie not in history
        assert other not in history
    assert {"a-tie-1", "a-tie-2"} <= {
        prior.id for prior in contexts["a-abroad"].recent_transactions
    }


def test_a_row_older_than_the_window_is_not_in_the_history() -> None:
    contexts = {
        context.transaction.id: context for _, context in serving_contexts(_dataset())
    }

    assert contexts["b-return"].recent_transactions == []


def test_only_the_wanted_rows_are_yielded_with_history_from_every_row() -> None:
    dataset = _dataset()
    burst_3 = next(i for i, tx in enumerate(dataset.transactions) if tx.id == "a-burst-3")

    yielded = list(serving_contexts(dataset, rows={burst_3}))

    assert [index for index, _ in yielded] == [burst_3]
    history = {prior.id for prior in yielded[0][1].recent_transactions}
    assert {"a-burst-1", "a-burst-2"} <= history


# ---------------------------------------------------------------------------
# Evaluability, decided from the data
# ---------------------------------------------------------------------------


def test_every_rule_is_evaluable_when_every_customer_has_an_account_open_timestamp() -> None:
    customers, _ = _entities()

    assert rule_evaluability(list(customers.values())) == tuple(
        Evaluability(name, True) for name, _ in RULES
    )


def test_the_dormant_rule_is_not_evaluable_when_no_customer_has_an_account_open_timestamp() -> None:
    customers, _ = _entities(with_account_open=False)

    decisions = {decision.rule: decision for decision in rule_evaluability(list(customers.values()))}

    assert decisions["dormant_account_high_value"] == Evaluability(
        "dormant_account_high_value", False, "not evaluable: no account-open timestamp"
    )
    assert all(decisions[name].evaluable for name in EVALUABLE_WITHOUT_DORMANT)


def test_account_open_timestamps_on_some_customers_only_are_refused() -> None:
    customers, _ = _entities(with_account_open=False)
    customers["c-a"].created_at = ANCHOR - timedelta(days=900)

    with pytest.raises(RulesAuditError, match="1 of 3 customers carry 'created_at'"):
        rule_evaluability(list(customers.values()))


def test_a_rule_that_is_not_evaluable_is_never_called(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_run(context: TransactionContext) -> Any:
        raise AssertionError("the dormant-account rule ran")

    patched = tuple(
        (name, must_not_run if name == "dormant_account_high_value" else rule)
        for name, rule in RULES
    )
    monkeypatch.setattr(rules_audit, "RULES", patched)
    dataset = _dataset(with_account_open=False)

    counts = audit_rows(dataset, _labels(dataset))

    assert [entry.rule for entry in counts.rules] == EVALUABLE_WITHOUT_DORMANT


def test_an_exception_from_an_evaluable_rule_is_never_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RuleBrokeError(Exception):
        pass

    def broken(context: TransactionContext) -> Any:
        raise RuleBrokeError("a genuine failure")

    patched = tuple(
        (name, broken if name == "amount_ceiling" else rule) for name, rule in RULES
    )
    monkeypatch.setattr(rules_audit, "RULES", patched)
    dataset = _dataset()

    with pytest.raises(RuleBrokeError):
        audit_rows(dataset, _labels(dataset))


def test_the_dormant_rule_would_raise_on_customers_without_an_account_open_timestamp() -> None:
    """Why evaluability is decided before any rule runs, not by catching this."""
    dataset = _dataset(with_account_open=False)
    _, context = next(serving_contexts(dataset))
    dormant = dict(RULES)["dormant_account_high_value"]

    with pytest.raises((AttributeError, TypeError)):
        dormant(context)


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_the_counts_follow_the_rule_results_and_the_labels(db_session: Session) -> None:
    dataset = _dataset()
    labels = _labels(dataset)
    expected_fired: dict[str, list[int]] = {name: [] for name, _ in RULES}
    expected_outcomes: dict[str, list[int]] = {"DECLINE": [], "REVIEW": [], "APPROVE": []}
    for index, tx in enumerate(dataset.transactions):
        served = evaluate_all(scoring._load_context(db_session, db_session.get(Transaction, tx.id)))
        for result in served:
            if result.triggered:
                expected_fired[result.rule_name].append(int(labels[index]))
        severities = {result.severity.value for result in served if result.triggered}
        outcome = "DECLINE" if "HARD_BLOCK" in severities else "REVIEW" if severities else "APPROVE"
        expected_outcomes[outcome].append(int(labels[index]))

    counts = audit_rows(dataset, labels)

    assert (counts.rows, counts.frauds) == (15, 3)
    for entry in counts.rules:
        assert (entry.rows_fired, entry.fired_on_fraud) == (
            len(expected_fired[entry.rule]),
            sum(expected_fired[entry.rule]),
        ), entry.rule
    assert counts.outcomes == {
        outcome: (len(values), sum(values)) for outcome, values in expected_outcomes.items()
    }


def test_an_empty_population_is_refused() -> None:
    dataset = _dataset()

    with pytest.raises(RulesAuditError, match="empty"):
        audit_rows(dataset, _labels(dataset), rows=set())


def test_labels_that_do_not_cover_the_dataset_are_refused() -> None:
    dataset = _dataset()

    with pytest.raises(RulesAuditError, match="14 labels were given for 15"):
        audit_rows(dataset, _labels(dataset)[:-1])


# ---------------------------------------------------------------------------
# A trained Sparkov fixture run
# ---------------------------------------------------------------------------

CORPUS_START = datetime(2019, 1, 1, 8, 0)
CATEGORIES = ("grocery_pos", "shopping_net", "travel", "gas_transport", "food_dining")
RUN = "sparkov-audit-fixture"


@dataclass(frozen=True)
class CorpusRow:
    trans_num: str
    at: datetime
    card: int
    category: str
    amount: str
    fraud: bool
    fires: frozenset[str]

    @property
    def transaction_id(self) -> str:
        return transaction_id_for(self.trans_num)


def _base_time(index: int) -> datetime:
    return CORPUS_START + timedelta(hours=5 * index)


def _corpus_rows() -> list[CorpusRow]:
    """Every row with the rules it is designed to fire, sorted as the adapter sorts them."""
    rows = [
        CorpusRow(
            trans_num=f"b{i}",
            at=_base_time(i),
            card=i % 3,
            category=CATEGORIES[i % 5],
            amount=f"{20 + (i * 37) % 280}.00",
            fraud=i % 7 == 0,
            fires=frozenset(),
        )
        for i in range(100)
    ]
    for i in (12, 48, 83, 95):  # bursts: the third row inside 120 seconds fires
        for step, fires in ((40, frozenset()), (80, frozenset({"velocity_burst"}))):
            rows.append(
                CorpusRow(
                    trans_num=f"burst{i}-{step}",
                    at=_base_time(i) + timedelta(seconds=step),
                    card=i % 3,
                    category=CATEGORIES[i % 5],
                    amount="35.00",
                    fraud=step == 80 and i in (12, 95),
                    fires=fires,
                )
            )
    for i in (20, 60, 88, 97):  # above the ceiling, card present
        rows.append(
            CorpusRow(
                trans_num=f"ceiling{i}",
                at=_base_time(i) + timedelta(minutes=30),
                card=(i + 1) % 3,
                category="shopping_pos",
                amount="6000.00",
                fraud=i in (20, 97),
                fires=frozenset({"amount_ceiling"}),
            )
        )
    for i in (30, 70, 92, 98):  # card not present, above $500, at 03:15
        rows.append(
            CorpusRow(
                trans_num=f"night{i}",
                at=datetime.combine(_base_time(i).date(), datetime.min.time()) + timedelta(hours=3, minutes=15),
                card=(i + 2) % 3,
                category="shopping_net",
                amount="750.00",
                fraud=i in (30, 92),
                fires=frozenset({"off_hours_high_value"}),
            )
        )
    near_misses = (
        ("miss-present", 40, 15, "grocery_pos", "750.00"),  # card present at night
        ("miss-small", 41, 45, "misc_net", "450.00"),  # not above $500
    )
    for trans_num, i, minute, category, amount in near_misses:
        rows.append(
            CorpusRow(
                trans_num=trans_num,
                at=datetime.combine(_base_time(i).date(), datetime.min.time()) + timedelta(hours=3, minutes=minute),
                card=(i + 2) % 3,
                category=category,
                amount=amount,
                fraud=False,
                fires=frozenset(),
            )
        )
    rows.sort(key=lambda row: (row.at, row.transaction_id))
    return rows


def _write_corpus(root: Path) -> Path:
    corpus = root / "raw" / "sparkov"
    corpus.mkdir(parents=True)
    _write_csv(
        corpus / "fraudTrain.csv",
        [
            _row(
                index,
                trans_num=row.trans_num,
                cc_num=f"433300000000000{row.card}",
                category=row.category,
                amt=row.amount,
                timestamp=row.at.strftime("%Y-%m-%d %H:%M:%S"),
                is_fraud="1" if row.fraud else "0",
            )
            for index, row in enumerate(_corpus_rows())
        ],
    )
    return corpus


def _stub_tune(*_: object, **__: object) -> TuningResult:
    return TuningResult(
        best_params={"n_estimators": 30, "max_depth": 3, "learning_rate": 0.3},
        best_score=0.5,
        cv_results_summary={"mean_test_score": [0.5], "std_test_score": [0.0]},
    )


@dataclass(frozen=True)
class Trained:
    root: Path
    corpus: Path
    adapter: SparkovAdapter

    @property
    def runs_root(self) -> Path:
        return self.root / "runs"

    @property
    def cache_root(self) -> Path:
        return self.root / "cache"


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    root = tmp_path_factory.mktemp("audit")
    corpus = _write_corpus(root)
    adapter = fixture_adapter(root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loading, "get_adapter", lambda name: adapter)
        patch.setattr(train, "tune_hyperparameters", _stub_tune)
        patch.setattr(train, "RUNS_ROOT", root / "runs")
        train.main(
            [
                "--dataset", "sparkov",
                "--root", str(corpus),
                "--cache-root", str(root / "cache"),
                "--run-name", RUN,
                "--n-iter", "1",
                "--cv-splits", "2",
                "--target-fpr", "0.05",
            ]
        )
    return Trained(root=root, corpus=corpus, adapter=adapter)


@dataclass
class Workspace:
    trained: Trained
    runs_root: Path
    adapter_roots: list[Path]

    @property
    def directory(self) -> Path:
        return self.runs_root / RUN

    def audit(self, rows: str = "all", run: str = RUN) -> rules_audit.RulesAudit:
        return rules_audit.audit_run(
            run,
            rows,
            runs_root=self.runs_root,
            root=self.trained.corpus,
            cache_root=self.trained.cache_root,
        )

    def edit(self, filename: str, change: Callable[[dict[str, Any]], None]) -> None:
        path = self.directory / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        change(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def ws(trained: Trained, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    shutil.copytree(trained.runs_root, tmp_path / "runs")
    roots: list[Path] = []
    real_load = SparkovAdapter.load_detailed

    def load_spy(self: SparkovAdapter, source: Path, **options: Any) -> Any:
        roots.append(Path(source))
        return real_load(self, source, **options)

    monkeypatch.setattr(loading, "get_adapter", lambda name: trained.adapter)
    monkeypatch.setattr(SparkovAdapter, "load_detailed", load_spy)
    return Workspace(trained=trained, runs_root=tmp_path / "runs", adapter_roots=roots)


def _expected(rows: list[CorpusRow]) -> dict[str, Any]:
    fired = {name: [row for row in rows if name in row.fires] for name in EVALUABLE_WITHOUT_DORMANT}

    def outcome(row: CorpusRow) -> str:
        if "velocity_burst" in row.fires:
            return "DECLINE"
        return "REVIEW" if row.fires else "APPROVE"

    by_outcome = {
        name: [row for row in rows if outcome(row) == name] for name in ("DECLINE", "REVIEW", "APPROVE")
    }
    return {
        "row_count": len(rows),
        "fraud_count": sum(row.fraud for row in rows),
        "fired": {name: (len(hits), sum(hit.fraud for hit in hits)) for name, hits in fired.items()},
        "by_outcome": {
            name: {
                "rows": len(members),
                "frauds": sum(member.fraud for member in members),
                "legitimate": sum(not member.fraud for member in members),
            }
            for name, members in by_outcome.items()
        },
    }


def _population_rows(rows: str) -> list[CorpusRow]:
    corpus = _corpus_rows()
    if rows == "all":
        return corpus
    splits = chronological_split(np.asarray([row.at for row in corpus], dtype=object))
    return [corpus[index] for index in getattr(splits, rows)]


@pytest.mark.parametrize("rows", ["all", "train", "val", "test"])
def test_the_audit_counts_the_designed_firings_in_each_population(
    ws: Workspace, rows: str
) -> None:
    expected = _expected(_population_rows(rows))

    payload = ws.audit(rows).payload

    assert payload["population"]["row_count"] == expected["row_count"]
    assert payload["population"]["fraud_count"] == expected["fraud_count"]
    counted = {rule["rule"]: rule for rule in payload["rules"] if rule["evaluable"]}
    assert list(counted) == EVALUABLE_WITHOUT_DORMANT
    for name, (hits, fraud_hits) in expected["fired"].items():
        assert (counted[name]["rows_fired"], counted[name]["fired_on_fraud"]) == (hits, fraud_hits)
        assert counted[name]["rows_evaluated"] == expected["row_count"]
        assert counted[name]["precision"] == (fraud_hits / hits if hits else None)
        assert counted[name]["fraud_recall"] == (
            fraud_hits / expected["fraud_count"] if expected["fraud_count"] else None
        )
    assert payload["rules_only_outcome"]["by_outcome"] == expected["by_outcome"]


def test_the_designed_rules_fire_somewhere_in_every_fold() -> None:
    """Otherwise the per-fold counts above would prove little."""
    for rows in ("train", "val", "test"):
        members = _population_rows(rows)
        assert any(row.fires for row in members), rows
        assert 0 < sum(row.fraud for row in members) < len(members), rows


def test_the_dormant_rule_is_reported_as_not_evaluable_and_left_out_of_the_outcome(
    ws: Workspace,
) -> None:
    payload = ws.audit().payload

    (dormant,) = [rule for rule in payload["rules"] if rule["rule"] == "dormant_account_high_value"]
    outcome = payload["rules_only_outcome"]

    assert dormant == {
        "rule": "dormant_account_high_value",
        "evaluable": False,
        "reason": "not evaluable: no account-open timestamp",
    }
    assert outcome["rules_used"] == EVALUABLE_WITHOUT_DORMANT
    assert outcome["rules_not_evaluable"] == ["dormant_account_high_value"]
    assert outcome["note"].startswith("Computed from the 5 evaluable rules of 6.")
    assert "no account-open timestamp" in outcome["note"]
    assert [rule["rule"] for rule in payload["rules"]] == [name for name, _ in RULES]


def test_every_audit_states_its_row_population(ws: Workspace) -> None:
    record = json.loads((ws.directory / "run.json").read_text(encoding="utf-8"))
    corpus = _corpus_rows()

    everything = ws.audit("all").payload["population"]
    test_fold = ws.audit("test").payload["population"]

    assert everything["run"] == RUN
    assert everything["rows"] == "all"
    assert everything["row_count"] == len(corpus) == record["dataset"]["row_count"]
    assert everything["first_timestamp"] == corpus[0].at.replace(tzinfo=UTC).isoformat()
    assert everything["last_timestamp"] == corpus[-1].at.replace(tzinfo=UTC).isoformat()
    assert everything["recorded_fold_period"] is None
    assert everything["test_fold_transaction_ids_sha256"] is None
    assert everything["dataset"]["name"] == "sparkov"
    assert "prevalence" in everything["note"]

    recorded_test = next(split for split in record["splits"] if split["name"] == "test")
    test_rows = _population_rows("test")
    assert test_fold["rows"] == "test"
    assert test_fold["recorded_fold_period"] == recorded_test
    assert test_fold["first_timestamp"] == recorded_test["start"]
    assert test_fold["last_timestamp"] == recorded_test["end"]
    assert test_fold["test_fold_transaction_count"] == len(test_rows)
    assert test_fold["test_fold_transaction_ids_sha256"] == (
        record["test_fold_identity"]["transaction_ids_sha256"]
    )
    assert test_fold["fraud_rate"] == test_fold["fraud_count"] / test_fold["row_count"]


def test_each_population_writes_its_own_file_and_nothing_else_changes(ws: Workspace) -> None:
    def snapshot() -> dict[str, str]:
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in ws.directory.iterdir()
        }

    before = snapshot()

    written = [ws.audit(rows).written.name for rows in ("all", "test")]

    after = snapshot()
    assert written == ["rules_audit.json", "rules_audit_test.json"]
    assert {name: digest for name, digest in after.items() if name in before} == before
    assert set(after) - set(before) == set(written)
    for name in written:
        assert "NaN" not in (ws.directory / name).read_text(encoding="utf-8")


def test_the_audit_verifies_the_run_before_counting(ws: Workspace) -> None:
    payload = ws.audit().payload

    assert payload["verification"] == {
        "metrics_reproduced": True,
        "model_digest_verified": True,
        "test_fold_identity_verified": True,
    }


def test_a_run_that_does_not_verify_is_refused_and_nothing_is_written(ws: Workspace) -> None:
    def nudge(metrics: dict[str, Any]) -> None:
        metrics["test_roc_auc"] = float(np.nextafter(metrics["test_roc_auc"], 0.0))

    ws.edit("metrics.json", nudge)

    with pytest.raises(RunVerificationError, match="do not reproduce"):
        ws.audit()

    assert not (ws.directory / "rules_audit.json").exists()


def test_a_synthetic_run_is_refused_before_its_data_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synthetic = train_fixture_run(tmp_path)
    loads: list[object] = []
    monkeypatch.setattr(loading, "load_dataset_with_csv_labels", lambda *a, **k: loads.append(a))

    with pytest.raises(RulesAuditError, match="no canonical dataset"):
        rules_audit.audit_run(
            "synthetic-fixture", runs_root=synthetic.runs_root, cache_root=tmp_path / "cache"
        )

    assert loads == []


def test_unknown_rows_are_refused(ws: Workspace) -> None:
    with pytest.raises(RulesAuditError, match="not one of all, train, val, test"):
        ws.audit("holdout")


def test_the_real_sparkov_corpus_is_never_read(ws: Workspace) -> None:
    ws.audit()

    assert ws.adapter_roots
    for root in ws.adapter_roots:
        assert ws.trained.root in root.parents
        assert RAW_DATA_DIR not in root.parents


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_the_cli_writes_the_audit_into_the_run(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rules_audit, "RUNS_ROOT", ws.runs_root)

    rules_audit.main(
        [
            "--run-name", RUN,
            "--rows", "val",
            "--root", str(ws.trained.corpus),
            "--cache-root", str(ws.trained.cache_root),
        ]
    )

    payload = json.loads((ws.directory / "rules_audit_val.json").read_text(encoding="utf-8"))
    assert payload["population"]["rows"] == "val"


def test_the_cli_exits_with_an_error_when_the_run_does_not_verify(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(rules_audit, "RUNS_ROOT", ws.runs_root)
    ws.edit("run.json", lambda record: record["library_versions"].update(numpy="0.0.1"))

    with pytest.raises(SystemExit) as refused:
        rules_audit.main(["--run-name", RUN, "--root", str(ws.trained.corpus)])

    assert refused.value.code == 1
    assert "was not run" in caplog.text


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--run-name", "../escape"], id="unsafe-name"),
        pytest.param(["--run-name", "r", "--rows", "holdout"], id="unknown-rows"),
        pytest.param([], id="no-run-name"),
    ],
)
def test_the_cli_refuses_arguments_before_any_work(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(rules_audit, "audit_run", lambda *a, **k: calls.append(a))

    with pytest.raises(SystemExit) as refused:
        rules_audit.main(argv)

    assert refused.value.code == 2
    assert calls == []
