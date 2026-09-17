"""Rules audit: how the production rules fire on a run's rows, with the context serving builds.

The Phase 5D methodology's rules audit (decisions 4 and 10 and the frozen
protocol). Every audited row is evaluated by the production rule functions,
unmodified, on the context the scoring service builds for it: that
customer's transactions in the 180 days before it, newest first, excluding
the row itself — `app.services.scoring._load_context`, reproduced in memory
from the canonical dataset. The history comes from every row of the run's
dataset, as it would in production; only the audited rows are counted.

For each rule the audit reports the rows it fired on, how many of those were
frauds, its precision and its fraud recall. The rules-only outcome of a row
uses the evaluable rules alone: a triggered hard-block rule gives DECLINE,
otherwise a triggered review rule gives REVIEW, otherwise APPROVE. It is
compared with the labels.

Whether a rule can be evaluated at all is decided from the data, before any
rule runs. `dormant_account_high_value` reads the customer's account-open
timestamp; where no customer carries one, the rule is reported as not
evaluable and left out of the rules-only outcome, which says so. Where only
some customers carry one, the audit refuses rather than guess. No exception
from any rule is ever caught.

Rule precision depends on prevalence, and prevalence differs between a corpus
and any fold of it, so every audit states its row population: which run,
which rows, how many, and which period. The run is verified first, exactly as
a run's analysis verifies it, so the rows audited are the run's own.

Usage:
    cd backend
    uv run python -m ml.experiments.rules_audit --run-name sparkov_v1_full --rows all \\
        --full-corpus
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.fraud.explainer import load_explainer
from app.fraud.rules import (
    RuleResult,
    Severity,
    rule_amount_ceiling,
    rule_dormant_account_high_value,
    rule_geo_velocity_impossible,
    rule_high_risk_country,
    rule_off_hours_high_value,
    rule_velocity_burst,
)
from app.fraud.transaction_context import TransactionContext
from app.models.customer import Customer
from app.models.transaction import Transaction
from ml.artifacts import collect_library_versions, utc_now_iso
from ml.datasets.base import CanonicalDataset
from ml.features.batch import HISTORY_WINDOW
from ml.loading import DataRequest, load_run_data
from ml.paths import FEATURE_CACHE_DIR, RUNS_ROOT, InvalidRunNameError, validate_run_name
from ml.run_verification import (
    RunVerificationError,
    VerifiedRun,
    check_recorded_run,
    read_recorded_run,
    verify_run,
)
from ml.runs import TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT, current_git_commit

log = logging.getLogger("ml.experiments.rules_audit")

# Bumped when the shape of a rules audit file changes.
RULES_AUDIT_VERSION = "1"

ALL_ROWS = "all"
AUDITABLE_ROWS = (ALL_ROWS, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

APPROVE = "APPROVE"
REVIEW = "REVIEW"
DECLINE = "DECLINE"

# The production rules, in the order `app.fraud.rules.evaluate_all` runs them.
# Listed rather than called through `evaluate_all` because a rule that cannot
# be evaluated on a dataset must not be run at all; a test pins this list to
# `evaluate_all`, so the two cannot drift apart.
RULES: tuple[tuple[str, Callable[[TransactionContext], RuleResult]], ...] = (
    ("velocity_burst", rule_velocity_burst),
    ("geo_velocity_impossible", rule_geo_velocity_impossible),
    ("amount_ceiling", rule_amount_ceiling),
    ("high_risk_country", rule_high_risk_country),
    ("dormant_account_high_value", rule_dormant_account_high_value),
    ("off_hours_high_value", rule_off_hours_high_value),
)

# The customer fields a rule reads that a dataset may not carry, and what the
# rule is reported as lacking when no customer carries them.
CUSTOMER_FIELDS_REQUIRED: Mapping[str, tuple[str, str]] = {
    "dormant_account_high_value": ("created_at", "no account-open timestamp"),
}

CONTEXT_DEFINITION = (
    "That customer's transactions in the 180 days before the row, newest first, excluding the "
    "row itself: the context app.services.scoring builds for a transaction it scores."
)

POPULATION_NOTE = (
    "Rule precision depends on prevalence, which differs between a corpus and any fold of it. "
    "Every count and rate in this file describes exactly these rows."
)

OUTCOME_RULE = (
    "A triggered hard-block rule gives DECLINE; otherwise a triggered review rule gives REVIEW; "
    "otherwise APPROVE."
)


class RulesAuditError(RunVerificationError):
    """The audit cannot be run on these rows as the methodology defines it."""


@dataclass(frozen=True)
class Evaluability:
    """Whether a rule can be evaluated on a dataset, decided from the data."""

    rule: str
    evaluable: bool
    reason: str | None = None


def rule_evaluability(customers: Collection[Customer]) -> tuple[Evaluability, ...]:
    """Decide, from the customers alone, which rules can be evaluated.

    A rule that reads a customer field is evaluable only if every customer
    carries it, and not evaluable if none does. Anything in between is
    refused: the rule could be evaluated for some rows only, which the
    methodology does not define.
    """
    decisions = []
    for name, _ in RULES:
        requirement = CUSTOMER_FIELDS_REQUIRED.get(name)
        if requirement is None:
            decisions.append(Evaluability(name, True))
            continue
        field, lacking = requirement
        carrying = sum(1 for customer in customers if getattr(customer, field) is not None)
        if carrying == 0:
            decisions.append(Evaluability(name, False, f"not evaluable: {lacking}"))
        elif carrying == len(customers):
            decisions.append(Evaluability(name, True))
        else:
            raise RulesAuditError(
                f"{carrying} of {len(customers)} customers carry {field!r}, so rule {name!r} "
                "could be evaluated for some rows only. The methodology defines a rule as "
                "evaluable or not evaluable on a dataset, not per row."
            )
    return tuple(decisions)


def serving_contexts(
    dataset: CanonicalDataset, rows: Collection[int] | None = None
) -> Iterator[tuple[int, TransactionContext]]:
    """Each wanted row with the context the scoring service would build for it.

    Walks every transaction in the dataset's chronological order, keeping one
    180-day window per customer, so each row's history is drawn from the whole
    dataset. Only the rows in `rows` are yielded; all of them when None.
    """
    windows: dict[str, deque[Transaction]] = {}
    for index, tx in enumerate(dataset.transactions):
        window = windows.setdefault(tx.customer_id, deque())
        cutoff = tx.created_at - HISTORY_WINDOW
        while window and window[0].created_at < cutoff:
            window.popleft()
        if rows is None or index in rows:
            yield index, TransactionContext(
                transaction=tx,
                customer=dataset.customers[tx.customer_id],
                merchant=dataset.merchants[tx.merchant_id],
                recent_transactions=[
                    prior for prior in reversed(window) if prior.created_at < tx.created_at
                ],
            )
        window.append(tx)


@dataclass(frozen=True)
class RuleCounts:
    rule: str
    severity: str
    rows_fired: int
    fired_on_fraud: int


@dataclass(frozen=True)
class AuditCounts:
    """The audit of one row population, before anything about the run is attached."""

    rows: int
    frauds: int
    evaluability: tuple[Evaluability, ...]
    rules: tuple[RuleCounts, ...]
    outcomes: Mapping[str, tuple[int, int]]  # outcome -> (rows, frauds)


def audit_rows(
    dataset: CanonicalDataset, labels: np.ndarray, rows: Collection[int] | None = None
) -> AuditCounts:
    """Evaluate the evaluable rules on `rows` of `dataset`, with `labels` in dataset order."""
    if len(labels) != dataset.n_rows:
        raise RulesAuditError(
            f"{len(labels)} labels were given for {dataset.n_rows} transactions."
        )
    evaluability = rule_evaluability(list(dataset.customers.values()))
    evaluable = {decision.rule for decision in evaluability if decision.evaluable}
    rules = [(name, rule) for name, rule in RULES if name in evaluable]

    fired = {name: 0 for name, _ in rules}
    fired_on_fraud = {name: 0 for name, _ in rules}
    severities: dict[str, str] = {}
    outcomes = {outcome: [0, 0] for outcome in (DECLINE, REVIEW, APPROVE)}
    audited = frauds = 0

    for index, context in serving_contexts(dataset, rows):
        label = int(labels[index])
        audited += 1
        frauds += label
        hard_block = review = False
        for name, rule in rules:
            result = rule(context)
            severities[name] = result.severity.value
            if not result.triggered:
                continue
            fired[name] += 1
            fired_on_fraud[name] += label
            hard_block = hard_block or result.severity == Severity.HARD_BLOCK
            review = review or result.severity == Severity.REVIEW
        outcome = DECLINE if hard_block else REVIEW if review else APPROVE
        outcomes[outcome][0] += 1
        outcomes[outcome][1] += label

    if audited == 0:
        raise RulesAuditError("The row population is empty, so there is nothing to audit.")
    return AuditCounts(
        rows=audited,
        frauds=frauds,
        evaluability=evaluability,
        rules=tuple(
            RuleCounts(name, severities[name], fired[name], fired_on_fraud[name])
            for name, _ in rules
        ),
        outcomes={outcome: (count[0], count[1]) for outcome, count in outcomes.items()},
    )


@dataclass(frozen=True)
class RulesAudit:
    """One audited row population of a verified run, and where its result was written."""

    verified: VerifiedRun
    rows: str
    counts: AuditCounts
    payload: dict[str, Any]
    written: Path


def audit_filename(rows: str) -> str:
    """`rules_audit.json` for all of a run's rows; one file per fold otherwise."""
    return "rules_audit.json" if rows == ALL_ROWS else f"rules_audit_{rows}.json"


def audit_run(
    run_name: str,
    rows: str = ALL_ROWS,
    *,
    runs_root: Path = RUNS_ROOT,
    root: Path | None = None,
    cache_root: Path = FEATURE_CACHE_DIR,
    full_corpus: bool = False,
) -> RulesAudit:
    """Verify a run, audit the rules on the chosen rows of its dataset, and write the result."""
    if rows not in AUDITABLE_ROWS:
        raise RulesAuditError(f"Rows {rows!r} are not one of {', '.join(AUDITABLE_ROWS)}.")
    recorded = read_recorded_run(run_name, runs_root=runs_root)
    check_recorded_run(recorded)
    try:
        request = DataRequest.from_run_record(
            recorded.record, root=root, cache_root=cache_root, full_corpus=full_corpus
        )
        data = load_run_data(request, with_provenance=True, with_canonical=True)
    except ValueError as exc:
        raise RulesAuditError(f"Run {run_name!r} cannot be audited: {exc}") from exc
    if data.canonical is None:  # pragma: no cover - requested above
        raise RulesAuditError(f"Run {run_name!r} was loaded without its canonical dataset.")

    verified = verify_run(recorded, data, load_explainer(recorded.directory))
    dataset = data.canonical
    if [tx.id for tx in dataset.transactions] != verified.ds.transaction_ids:
        raise RulesAuditError(
            f"Run {run_name!r}: the canonical transactions are not the verified matrix's rows, "
            "in its order."
        )

    indices = _population(verified, rows)
    counts = audit_rows(
        dataset, verified.ds.y, None if indices is None else frozenset(int(i) for i in indices)
    )
    payload = audit_payload(
        verified,
        rows,
        counts,
        indices=indices,
        generated_at=utc_now_iso(),
        code_version=current_git_commit(),
        library_versions=collect_library_versions(),
    )
    written = _write_json(recorded.directory / audit_filename(rows), payload)
    return RulesAudit(verified=verified, rows=rows, counts=counts, payload=payload, written=written)


def audit_payload(
    verified: VerifiedRun,
    rows: str,
    counts: AuditCounts,
    *,
    indices: np.ndarray | None,
    generated_at: str,
    code_version: str | None,
    library_versions: dict[str, str],
) -> dict[str, Any]:
    """The contents of a rules audit file."""
    record = verified.recorded.record
    timestamps = verified.ds.timestamps if indices is None else verified.ds.timestamps[indices]
    identity = record.test_fold_identity if rows == TEST_SPLIT else None
    fold = record.split(rows) if rows != ALL_ROWS else None
    evaluable = [decision.rule for decision in counts.evaluability if decision.evaluable]
    not_evaluable = [decision for decision in counts.evaluability if not decision.evaluable]
    counted = {entry.rule: entry for entry in counts.rules}

    rules = []
    for decision in counts.evaluability:
        if not decision.evaluable:
            rules.append({"rule": decision.rule, "evaluable": False, "reason": decision.reason})
            continue
        entry = counted[decision.rule]
        rules.append(
            {
                "rule": entry.rule,
                "evaluable": True,
                "severity": entry.severity,
                "rows_evaluated": counts.rows,
                "rows_fired": entry.rows_fired,
                "fired_on_fraud": entry.fired_on_fraud,
                "precision": (
                    entry.fired_on_fraud / entry.rows_fired if entry.rows_fired else None
                ),
                "fraud_recall": entry.fired_on_fraud / counts.frauds if counts.frauds else None,
            }
        )

    return {
        "rules_audit_version": RULES_AUDIT_VERSION,
        "run": verified.recorded.name,
        "population": {
            "run": verified.recorded.name,
            "rows": rows,
            "row_count": counts.rows,
            "fraud_count": counts.frauds,
            "fraud_rate": counts.frauds / counts.rows,
            "first_timestamp": min(timestamps).isoformat(),
            "last_timestamp": max(timestamps).isoformat(),
            "recorded_fold_period": None if fold is None else fold.to_dict(),
            "test_fold_transaction_count": None if identity is None else identity.transaction_count,
            "test_fold_transaction_ids_sha256": (
                None if identity is None else identity.transaction_ids_sha256
            ),
            "dataset": {
                "name": record.dataset.name,
                "version": record.dataset.version,
                "files": dict(record.dataset.files),
                "subsample": (
                    None if record.dataset.subsample is None else record.dataset.subsample.to_dict()
                ),
                "row_count": record.dataset.row_count,
            },
            "note": POPULATION_NOTE,
        },
        "context": {
            "history_window_days": HISTORY_WINDOW.days,
            "definition": CONTEXT_DEFINITION,
            "history_drawn_from": "every row of the run's dataset",
        },
        "rules": rules,
        "rules_only_outcome": {
            "rules_used": evaluable,
            "rules_not_evaluable": [decision.rule for decision in not_evaluable],
            "note": (
                f"Computed from the {len(evaluable)} evaluable rules of {len(RULES)}."
                + (
                    " Not evaluable: "
                    + "; ".join(f"{d.rule} ({d.reason})" for d in not_evaluable)
                    + "."
                    if not_evaluable
                    else ""
                )
            ),
            "decision_rule": OUTCOME_RULE,
            "by_outcome": {
                outcome: {"rows": total, "frauds": frauds, "legitimate": total - frauds}
                for outcome, (total, frauds) in counts.outcomes.items()
            },
        },
        "verification": {
            "metrics_reproduced": True,
            "model_digest_verified": verified.model_digest_verified,
            "test_fold_identity_verified": verified.test_fold_identity_verified,
        },
        "provenance": {
            "generated_at_utc": generated_at,
            "code_version": code_version,
            "library_versions": dict(library_versions),
            "run_code_version": record.code_version,
        },
    }


def _population(verified: VerifiedRun, rows: str) -> np.ndarray | None:
    if rows == ALL_ROWS:
        return None
    folds = {
        TRAIN_SPLIT: verified.splits.train,
        VAL_SPLIT: verified.splits.val,
        TEST_SPLIT: verified.splits.test,
    }
    return np.asarray(folds[rows])


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the production rules on a verified run's rows"
    )
    parser.add_argument("--run-name", required=True, help="Run whose dataset is audited")
    parser.add_argument(
        "--rows",
        default=ALL_ROWS,
        choices=AUDITABLE_ROWS,
        help="Which of the run's rows to audit (default: all)",
    )
    parser.add_argument("--root", type=Path, default=None, help="Directory holding the source files")
    parser.add_argument("--cache-root", type=Path, default=FEATURE_CACHE_DIR)
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Allow an unsubsampled load where the adapter guards against one",
    )
    args = parser.parse_args(argv)
    try:
        validate_run_name(args.run_name)
    except InvalidRunNameError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    args = parse_args(argv)
    try:
        audit = audit_run(
            args.run_name,
            args.rows,
            runs_root=RUNS_ROOT,
            root=args.root,
            cache_root=args.cache_root,
            full_corpus=args.full_corpus,
        )
    except RunVerificationError as exc:
        log.error("Rules audit of run %s was not run: %s", args.run_name, exc)
        sys.exit(1)

    population = audit.payload["population"]
    log.info(
        "Audited %d %s rows of run %s (%d frauds)",
        population["row_count"],
        population["rows"],
        population["run"],
        population["fraud_count"],
    )
    for rule in audit.payload["rules"]:
        if rule["evaluable"]:
            log.info(
                "  %-28s fired=%d  on fraud=%d  precision=%s  fraud recall=%s",
                rule["rule"],
                rule["rows_fired"],
                rule["fired_on_fraud"],
                rule["precision"],
                rule["fraud_recall"],
            )
        else:
            log.info("  %-28s %s", rule["rule"], rule["reason"])
    log.info("Wrote %s", audit.written)


if __name__ == "__main__":
    main()
