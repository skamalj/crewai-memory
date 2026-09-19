"""PostgreSQL (pgvector) ``StorageBackend`` for CrewAI unified ``Memory``.

One table with a ``vector(dims)`` column and an HNSW cosine index. ``search``
is a single SQL query ordered by ``embedding <=> query`` with the scope subtree
filtered in ``WHERE``. Built on ``crewai-memory-core``.

Requires the ``vector`` extension on the server; ``CREATE EXTENSION IF NOT
EXISTS vector`` is run at startup.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Column, Float, Index, MetaData, String, Table, create_engine, delete, select, text
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.engine import Engine

from crewai_memory_core import MemoryBackend, ROW_FIELDS, norm_scope

__all__ = ["PostgresMemoryBackend"]


def _escape_like(v: str) -> str:
    return v.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class PostgresMemoryBackend(MemoryBackend):
    """CrewAI ``StorageBackend`` on PostgreSQL + pgvector.

    Example:
        ```python
        from crewai.memory import Memory
        from crewai_memory_postgres import PostgresMemoryBackend

        backend = PostgresMemoryBackend("postgresql://user:pass@host:5432/db", dimensions=3072)
        memory = Memory(storage=backend)
        ```
    """

    supports_native_vector_search = True

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        table_name: str = "crewai_memory",
        dimensions: int = 3072,
        engine: Optional[Engine] = None,
    ) -> None:
        if engine is None and url is None:
            raise ValueError("Provide either 'url' or 'engine'")
        self.dimensions = dimensions
        self._engine = engine or create_engine(url)  # type: ignore[arg-type]
        with self._engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        md = MetaData()
        self._t = Table(
            table_name, md,
            Column("id", String, primary_key=True),
            Column("scope", String, nullable=False, index=True),
            Column("content", String, nullable=False),
            Column("categories", JSONB, nullable=False),
            Column("metadata", JSONB, nullable=False),
            Column("importance", Float, nullable=False),
            Column("created_at", String, nullable=False),
            Column("last_accessed", String, nullable=False),
            Column("source", String, nullable=True),
            Column("private", Boolean, nullable=False),
            Column("embedding", Vector(dimensions), nullable=True),
        )
        Index(f"{table_name}_embedding_hnsw", self._t.c.embedding, postgresql_using="hnsw",
              postgresql_with={"m": 16, "ef_construction": 64}, postgresql_ops={"embedding": "vector_cosine_ops"})
        md.create_all(self._engine)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _row(r) -> dict:
        row = {f: getattr(r, f) for f in ROW_FIELDS}
        row["embedding"] = list(r.embedding) if r.embedding is not None else None
        return row

    def _scope_clause(self, scope_prefix: Optional[str]):
        p = norm_scope(scope_prefix)
        if p == "/":
            return None
        t = self._t
        return (t.c.scope == p) | t.c.scope.like(f"{_escape_like(p)}/%", escape="\\")

    # ------------------------------------------------------------------ primitives
    def _put(self, rows: List[dict]) -> None:
        t = self._t
        with self._engine.begin() as conn:
            for row in rows:
                values = {f: row.get(f) for f in ROW_FIELDS}
                values["scope"] = norm_scope(row["scope"])
                values["categories"] = list(row.get("categories") or [])
                values["metadata"] = dict(row.get("metadata") or {})
                values["private"] = bool(row.get("private", False))
                stmt = pg_insert(t).values(**values).on_conflict_do_update(
                    index_elements=[t.c.id], set_={k: v for k, v in values.items() if k != "id"}
                )
                conn.execute(stmt)

    def _get(self, record_id: str) -> Optional[dict]:
        with self._engine.connect() as conn:
            r = conn.execute(select(self._t).where(self._t.c.id == record_id)).first()
        return self._row(r) if r else None

    def _delete_ids(self, ids: List[str]) -> int:
        with self._engine.begin() as conn:
            return conn.execute(delete(self._t).where(self._t.c.id.in_(ids))).rowcount

    def _scan(self, scope_prefix: Optional[str]) -> List[dict]:
        stmt = select(self._t)
        clause = self._scope_clause(scope_prefix)
        if clause is not None:
            stmt = stmt.where(clause)
        with self._engine.connect() as conn:
            return [self._row(r) for r in conn.execute(stmt).all()]

    def _vector_search(
        self, vector: List[float], scope_prefix: Optional[str], limit: int
    ) -> List[Tuple[dict, float]]:
        t = self._t
        dist = t.c.embedding.cosine_distance(vector).label("distance")
        stmt = select(t, dist).where(t.c.embedding.isnot(None))
        clause = self._scope_clause(scope_prefix)
        if clause is not None:
            stmt = stmt.where(clause)
        stmt = stmt.order_by(dist).limit(limit)
        with self._engine.connect() as conn:
            return [(self._row(r), 1.0 - float(r.distance)) for r in conn.execute(stmt).all()]

    def close(self) -> None:
        self._engine.dispose()
