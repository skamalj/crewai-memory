"""Postgres (pgvector) StorageBackend: the shared contract suite against a real PostgreSQL."""
import os
import uuid

import pytest
from sqlalchemy import create_engine, text

from crewai_memory_core.contract import *  # noqa: F401,F403
from crewai_memory_core.contract import DIMS
from crewai_memory_postgres import PostgresMemoryBackend

URL = os.environ.get("SQL_TEST_URL", "postgresql+psycopg2://postgres:postgres@localhost:5433/postgres")


def _available() -> bool:
    try:
        with create_engine(URL).begin() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _available(), reason=f"PostgreSQL+pgvector not reachable: {URL}")


@pytest.fixture(scope="module")
def _store():
    tname = f"crewai_mem_test_{uuid.uuid4().hex[:8]}"
    b = PostgresMemoryBackend(URL, table_name=tname, dimensions=DIMS)
    yield b
    with b._engine.begin() as c:
        c.execute(text(f'DROP TABLE IF EXISTS "{tname}"'))
    b.close()


@pytest.fixture()
def backend(_store):
    _store.reset()
    yield _store
    _store.reset()


def test_native_vector_path_used(backend, monkeypatch):
    from crewai_memory_core.contract import _seed, EMB
    _seed(backend)
    called = []
    orig = backend._vector_search
    monkeypatch.setattr(backend, "_vector_search", lambda *a, **k: (called.append(1), orig(*a, **k))[1])
    assert backend.search(EMB(["sushi"])[0], scope_prefix="/crew", limit=5)
    assert called
    with backend._engine.connect() as c:
        idx = c.execute(text(f"select indexdef from pg_indexes where tablename='{backend._t.name}'")).all()
    assert any("hnsw" in r[0] for r in idx)
