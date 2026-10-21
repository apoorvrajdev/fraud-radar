"""SQLite transaction control in `app/db/session.py`.

The sqlite3 driver's default mode emits BEGIN only ahead of INSERT,
UPDATE and DELETE. A SAVEPOINT issued before one of those starts a
transaction of its own, so releasing it commits — before the caller
commits, and beyond the reach of the caller's rollback. `build_engine`
issues BEGIN ahead of such a savepoint so it always nests inside the
caller's transaction, and changes nothing else about when the driver
begins: reads outside a savepoint still hold no lock.

Committed state is read through a raw sqlite3 connection on a file-backed
database. A second connection sees committed rows only, so these tests
detect a premature commit instead of asking the session about itself.
"""
from __future__ import annotations

import sqlite3
import threading
import types
from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db import build_engine
from app.db.session import SessionLocal, _sqlite_savepoint
from app.db.session import engine as application_engine
from app.enrichment.fx import SOURCE_LIVE, FxService
from app.enrichment.provider import ProviderRate
from app.models.base import Base
from app.models.fx_rate import FxRate

RATE_DATE = date(2024, 1, 2)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "transactions.db"


@pytest.fixture
def engine(db_path: Path) -> Iterator[Engine]:
    engine = build_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    # The application's session factory, bound to the test engine, so the
    # session is configured exactly as a request's is.
    db = SessionLocal(bind=engine)
    try:
        yield db
    finally:
        db.close()


def _rate(quote: str, rate: str = "1.1") -> FxRate:
    return FxRate(
        base="USD",
        quote=quote,
        rate_date=RATE_DATE,
        rate=Decimal(rate),
        fetched_at=datetime.now(UTC),
    )


def _committed(db_path: Path) -> list[str]:
    """The quotes on disk, seen by a connection the session knows nothing about."""
    raw = sqlite3.connect(db_path)
    try:
        return [row[0] for row in raw.execute("SELECT quote FROM fx_rates ORDER BY quote")]
    finally:
        raw.close()


def _read_something(db: Session) -> None:
    """A SELECT: the driver alone would still not have begun a transaction."""
    db.execute(select(func.count()).select_from(FxRate)).scalar_one()


def _traced(engine: Engine) -> list[str]:
    """Every statement SQLite executes on the engine's next connections."""
    statements: list[str] = []

    @event.listens_for(engine, "connect")
    def trace(dbapi_connection: sqlite3.Connection, _record: object) -> None:
        dbapi_connection.set_trace_callback(statements.append)

    engine.dispose()  # so the next connection is traced from its first statement
    return statements


def _transaction_statements(statements: list[str]) -> list[str]:
    return [s.strip() for s in statements if not s.startswith(("SELECT", "PRAGMA", "INSERT"))]


def _driver(connection: Connection) -> sqlite3.Connection:
    return cast(sqlite3.Connection, connection.connection.driver_connection)


class _Provider:
    def fetch_rate(self, *, base: str, quote: str, on: date) -> ProviderRate | None:
        return ProviderRate(base=base, quote=quote, rate_date=on, rate=Decimal("1.1022"))


# ---------------------------------------------------------------------------
# A savepoint nests inside the caller's transaction
# ---------------------------------------------------------------------------


def test_releasing_a_savepoint_does_not_commit_before_the_caller(
    session: Session, db_path: Path
) -> None:
    """The savepoint is the first write of the transaction. `begin_nested`
    flushes pending state first, so a row added before it would hand the
    driver its BEGIN and hide the fault; the outer row is added after the
    release instead. Nothing may reach disk until the caller commits."""
    session.begin()
    _read_something(session)

    with session.begin_nested():
        session.add(_rate("EUR"))
        session.flush()
    session.add(_rate("GBP"))
    session.flush()

    assert _committed(db_path) == []
    session.commit()
    assert _committed(db_path) == ["EUR", "GBP"]


def test_rolling_back_the_caller_discards_a_released_savepoint(
    session: Session, db_path: Path
) -> None:
    _read_something(session)
    with session.begin_nested():
        session.add(_rate("EUR"))
        session.flush()
    session.add(_rate("GBP"))
    session.flush()

    session.rollback()

    assert _committed(db_path) == []
    assert session.execute(select(func.count()).select_from(FxRate)).scalar_one() == 0


def test_rolling_back_a_savepoint_keeps_the_outer_work_pending(
    session: Session, db_path: Path
) -> None:
    """A genuine constraint violation inside the savepoint — a zero rate
    against ck_fx_rates_rate_positive — is undone alone. The row staged
    before it stays pending, the session stays usable, and only the
    caller's commit makes that row visible."""
    session.add(_rate("GBP"))
    session.flush()

    with pytest.raises(IntegrityError, match="CHECK constraint failed"), session.begin_nested():
        session.add(_rate("EUR", rate="0"))
        session.flush()

    assert session.get(FxRate, ("USD", "EUR", RATE_DATE)) is None
    assert session.get(FxRate, ("USD", "GBP", RATE_DATE)) is not None
    assert _committed(db_path) == []
    session.commit()
    assert _committed(db_path) == ["GBP"]


# ---------------------------------------------------------------------------
# The mechanism: BEGIN ahead of a savepoint the driver has not begun for
# ---------------------------------------------------------------------------


def test_begin_is_issued_just_ahead_of_an_outermost_savepoint(engine: Engine) -> None:
    statements = _traced(engine)

    with SessionLocal(bind=engine) as db:
        _read_something(db)
        with db.begin_nested():
            db.add(_rate("EUR"))
            db.flush()
        db.rollback()

    assert _transaction_statements(statements) == [
        "BEGIN IMMEDIATE", "SAVEPOINT sa_savepoint_1", "RELEASE SAVEPOINT sa_savepoint_1", "ROLLBACK",
    ]


def test_a_savepoint_after_a_write_reuses_the_drivers_transaction(engine: Engine) -> None:
    """The driver began ahead of the first INSERT; a second BEGIN would be
    an error. The savepoint nests in the transaction already open."""
    statements = _traced(engine)

    with SessionLocal(bind=engine) as db:
        db.add(_rate("GBP"))
        db.flush()
        with db.begin_nested():
            db.add(_rate("EUR"))
            db.flush()
        db.rollback()

    assert _transaction_statements(statements) == [
        "BEGIN", "SAVEPOINT sa_savepoint_1", "RELEASE SAVEPOINT sa_savepoint_1", "ROLLBACK",
    ]


def test_nested_savepoints_share_one_begin(engine: Engine, db_path: Path) -> None:
    statements = _traced(engine)

    with SessionLocal(bind=engine) as db:
        _read_something(db)
        with db.begin_nested():
            db.add(_rate("EUR"))
            db.flush()
            with db.begin_nested():
                db.add(_rate("GBP"))
                db.flush()
        db.rollback()

    assert [s for s in statements if s.startswith("BEGIN")] == ["BEGIN IMMEDIATE"]
    assert _committed(db_path) == []


def test_a_session_that_only_read_does_not_block_a_writer(
    session: Session, db_path: Path
) -> None:
    """The driver's own behaviour outside a savepoint is kept: a SELECT
    begins nothing, so an open session that has only read holds no lock
    and another connection can commit at once. (Taking over BEGIN for
    every transaction would make this commit wait on the reader and, for
    a second writer, fail outright.)"""
    _read_something(session)

    writer = sqlite3.connect(db_path, timeout=0.5)
    try:
        writer.execute(
            "INSERT INTO fx_rates (base, quote, rate_date, rate, fetched_at) "
            "VALUES ('USD', 'EUR', '2024-01-02', 1.1, '2024-01-03 09:00:00')"
        )
        writer.commit()
    finally:
        writer.close()

    assert _committed(db_path) == ["EUR"]
    assert session.execute(select(func.count()).select_from(FxRate)).scalar_one() == 1


def test_a_savepoint_first_write_blocked_by_another_writer_holds_no_lock(db_path: Path) -> None:
    """The guard takes the write lock up front, through the busy handler.
    When another writer keeps it past the timeout, the savepoint fails
    before any statement ran inside it: this session holds no lock, so
    the other writer commits at once, and the session is still the
    caller's to use and commit. (A deferred BEGIN would let the SELECT
    inside the savepoint open a read transaction whose lock then blocks
    the other writer's commit.)"""
    engine = build_engine(f"sqlite:///{db_path}", connect_args={"timeout": 0.2})
    Base.metadata.create_all(engine)
    other = sqlite3.connect(db_path, timeout=0.2)
    try:
        other.execute("BEGIN IMMEDIATE")
        other.execute(
            "INSERT INTO fx_rates (base, quote, rate_date, rate, fetched_at) "
            "VALUES ('USD', 'GBP', '2024-01-02', 1.1, '2024-01-03 09:00:00')"
        )
        with SessionLocal(bind=engine) as db:
            service = FxService(provider=_Provider(), base_currency="USD")
            with pytest.raises(OperationalError, match="database is locked"):
                service.get_rate(db, quote="EUR", on=RATE_DATE)

            other.commit()  # would raise "database is locked" if this session held a lock

            db.add(_rate("CHF"))
            db.flush()
            db.commit()
        assert _committed(db_path) == ["CHF", "GBP"]
    finally:
        other.close()
        engine.dispose()


# ---------------------------------------------------------------------------
# The fx_rates write — the one savepoint in the application
# ---------------------------------------------------------------------------


def test_a_fetched_fx_rate_is_committed_only_with_the_caller(
    session: Session, db_path: Path
) -> None:
    """`get_rate` may be the first write a session makes. The ingestion
    path happens to flush the transaction row first; the service must not
    depend on it."""
    service = FxService(provider=_Provider(), base_currency="USD")

    lookup = service.get_rate(session, quote="EUR", on=RATE_DATE)

    assert lookup.source == SOURCE_LIVE
    assert _committed(db_path) == []
    session.commit()
    assert _committed(db_path) == ["EUR"]


def test_a_fetched_fx_rate_is_discarded_with_the_caller(
    session: Session, db_path: Path
) -> None:
    service = FxService(provider=_Provider(), base_currency="USD")
    lookup = service.get_rate(session, quote="EUR", on=RATE_DATE)
    assert lookup.source == SOURCE_LIVE

    session.rollback()

    assert _committed(db_path) == []


# ---------------------------------------------------------------------------
# Where the guard applies
# ---------------------------------------------------------------------------


def test_the_driver_keeps_its_own_transaction_control(engine: Engine) -> None:
    """Only the savepoint guard is added; the driver stays in its default
    mode on every connection the pool hands out."""
    with engine.connect() as first:
        assert _driver(first).isolation_level is not None
    with engine.connect() as again:
        assert _driver(again).isolation_level is not None
    assert event.contains(engine, "savepoint", _sqlite_savepoint)


def test_the_application_engine_is_built_by_the_factory() -> None:
    """The guard is on the engine the application uses exactly when that
    engine is SQLite — whatever DATABASE_URL this checkout runs with."""
    is_sqlite = application_engine.dialect.name == "sqlite"
    assert event.contains(application_engine, "savepoint", _sqlite_savepoint) is is_sqlite


def test_an_in_memory_static_pool_engine_gets_the_same_control() -> None:
    """The test suites' shape of engine: one in-memory connection shared by
    every session, where a premature commit would survive a rollback."""
    engine = build_engine("sqlite:///:memory:", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    try:
        with SessionLocal(bind=engine) as db:
            _read_something(db)
            with db.begin_nested():
                db.add(_rate("EUR"))
                db.flush()
            db.rollback()
        with SessionLocal(bind=engine) as other:
            assert other.execute(select(func.count()).select_from(FxRate)).scalar_one() == 0
    finally:
        engine.dispose()


def test_connect_args_are_merged_with_the_sqlite_defaults(db_path: Path) -> None:
    """A caller's connect_args add to the factory's, so the connection can
    still be used from FastAPI's worker threads."""
    engine = build_engine(f"sqlite:///{db_path}", connect_args={"timeout": 1.5})
    errors: list[BaseException] = []
    try:
        with engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA busy_timeout").scalar_one() == 1500
            raw = _driver(conn)

            def use_from_another_thread() -> None:
                try:
                    raw.execute("SELECT 1").fetchone()
                except BaseException as exc:  # reported to the test thread
                    errors.append(exc)

            worker = threading.Thread(target=use_from_another_thread)
            worker.start()
            worker.join()
    finally:
        engine.dispose()
    assert errors == []


def test_other_backends_are_left_to_their_driver() -> None:
    """A PEP 249 driver begins on its own; the SQLite guard must not reach it.
    The dialect is built on a stub module so no Postgres driver is needed."""
    stub = types.ModuleType("stub_pg8000")
    stub.__version__ = "1.30.0"  # type: ignore[attr-defined]
    stub.paramstyle = "format"  # type: ignore[attr-defined]
    for name in (
        "Error", "DatabaseError", "OperationalError", "InterfaceError",
        "IntegrityError", "ProgrammingError", "DataError", "NotSupportedError",
        "InternalError",
    ):
        setattr(stub, name, type(name, (Exception,), {}))

    engine = build_engine("postgresql+pg8000://user:secret@localhost/fraud", module=stub)

    assert engine.dialect.name == "postgresql"
    assert not event.contains(engine, "savepoint", _sqlite_savepoint)
