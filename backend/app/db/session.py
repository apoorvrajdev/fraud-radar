"""SQLAlchemy engine and session factory."""
from __future__ import annotations

import sqlite3
from collections.abc import Generator
from typing import Any, cast

from sqlalchemy import create_engine, event, make_url
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

settings = get_settings()


def _sqlite_savepoint(connection: Connection, _name: str | None) -> None:
    """Make sure a real transaction is open before a SAVEPOINT is issued.

    The sqlite3 driver's default ("legacy") mode emits BEGIN only ahead
    of INSERT, UPDATE and DELETE — never ahead of SELECT or SAVEPOINT. A
    savepoint issued before any write therefore starts a transaction of
    its own, and releasing it commits: the work inside it is on disk
    before the caller commits, and the caller's rollback can no longer
    undo it. Issuing BEGIN here, only when the driver has not begun yet,
    nests the savepoint in a transaction that lasts until the caller's
    commit or rollback. Nothing else changes: the driver still decides
    when a plain write begins, so reads outside a savepoint hold no lock
    and writers queue through SQLite's busy handler exactly as before.

    BEGIN IMMEDIATE, because a savepoint exists to protect a write.
    Taking the write lock up front keeps the busy handler in play: a
    second writer waits its turn. A deferred BEGIN would let the first
    SELECT inside the savepoint open a read transaction, and SQLite
    refuses to retry a lock upgrade from inside one — the write fails at
    once with "database is locked", and the read lock then held blocks
    the other writer's commit until this session rolls back.

    SQLAlchemy's documented alternative — disabling the driver's control
    and emitting BEGIN at the start of every transaction — was measured
    and rejected: it makes every session hold a shared lock from its
    first SELECT, so a writer that overlaps another fails at once with
    "database is locked" (SQLite skips the busy handler for a lock
    upgrade inside a read transaction) and a long read stalls commits.
    """
    driver_connection = cast(sqlite3.Connection, connection.connection.driver_connection)
    if (
        driver_connection.isolation_level is None
        or getattr(driver_connection, "autocommit", None) is True  # Python 3.12+ form
    ):
        return  # driver-level autocommit was asked for; leave it alone
    if driver_connection.in_transaction:
        return  # the driver already began, ahead of an earlier write
    connection.exec_driver_sql("BEGIN IMMEDIATE")


def build_engine(url: str, **kwargs: Any) -> Engine:
    """Create an engine for `url` with the application's settings.

    Every SQLite engine — the application's, and a test's whenever the
    code under test may reach a savepoint — is built here so the
    savepoint guard above applies. Other backends begin a transaction on
    the first statement as PEP 249 requires and get no hook.

    `kwargs` reach `create_engine` and override the defaults here;
    `connect_args` are merged with the SQLite defaults, not replaced.
    """
    is_sqlite = make_url(url).get_backend_name() == "sqlite"
    # SQLite-specific: connect_args needed for FastAPI multi-threaded usage
    connect_args: dict[str, Any] = {"check_same_thread": False} if is_sqlite else {}
    connect_args.update(kwargs.pop("connect_args", {}))
    kwargs.setdefault("echo", False)
    engine = create_engine(url, connect_args=connect_args, **kwargs)
    if is_sqlite:
        event.listen(engine, "savepoint", _sqlite_savepoint)
    return engine


engine = build_engine(settings.database_url)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session.

    The session is closed automatically when the request finishes.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
