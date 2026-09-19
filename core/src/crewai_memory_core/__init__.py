"""Shared core for CrewAI unified-memory ``StorageBackend`` implementations.

CrewAI 1.10+ has one ``Memory`` engine (LLM analysis, consolidation, composite
scoring) and a pluggable ``crewai.memory.storage.backend.StorageBackend``
protocol underneath it. ``MemoryBackend`` implements the *whole* protocol —
save / search / delete / update / get_record / list_records / get_scope_info /
list_scopes / list_categories / count / reset / touch_records and the async
variants — on top of five primitives a datastore supplies:

    _put(rows)                       -> None          upsert rows by id
    _get(record_id)                  -> row | None
    _delete_ids(ids)                 -> int
    _scan(scope_prefix)              -> list[row]     every row in the scope subtree
    _vector_search(vector, scope_prefix, limit) -> list[(row, score)]   (optional)

A ``row`` is a plain dict with the ``MemoryRecord`` fields (``id``, ``content``,
``scope``, ``categories``, ``metadata``, ``importance``, ``created_at``,
``last_accessed``, ``source``, ``private``) plus ``embedding``. ``score`` is a
cosine similarity, higher is better. The default ``_vector_search`` ranks
``_scan`` rows in Python, so a backend works before it has a native path.

Scope semantics: ``scope_prefix="/a"`` matches ``/a`` and everything under
``/a/...`` but not ``/ab``; ``"/"`` or ``None`` matches everything.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from crewai.memory.types import MemoryRecord, ScopeInfo

__all__ = ["MemoryBackend", "record_to_row", "row_to_record", "in_scope", "norm_scope", "cosine_similarity"]

ROW_FIELDS = (
    "id", "content", "scope", "categories", "metadata", "importance",
    "created_at", "last_accessed", "source", "private", "embedding",
)


def norm_scope(scope: Optional[str]) -> str:
    """Normalise a scope path: leading slash, no trailing slash, ``/`` for root."""
    s = (scope or "/").strip()
    if not s.startswith("/"):
        s = "/" + s
    s = s.rstrip("/")
    return s or "/"


def in_scope(scope: str, prefix: Optional[str]) -> bool:
    """True if ``scope`` is ``prefix`` itself or lies under it."""
    p = norm_scope(prefix)
    if p == "/":
        return True
    s = norm_scope(scope)
    return s == p or s.startswith(p + "/")


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _parse_dt(val: Any) -> datetime:
    if isinstance(val, datetime):
        return val
    if val is None:
        return datetime.utcnow()
    return datetime.fromisoformat(str(val).replace("Z", "+00:00"))


def record_to_row(record: MemoryRecord) -> dict:
    return {
        "id": record.id,
        "content": record.content,
        "scope": norm_scope(record.scope),
        "categories": list(record.categories or []),
        "metadata": dict(record.metadata or {}),
        "importance": float(record.importance),
        "created_at": record.created_at.isoformat(),
        "last_accessed": record.last_accessed.isoformat(),
        "source": record.source,
        "private": bool(record.private),
        "embedding": list(record.embedding) if record.embedding else None,
    }


def row_to_record(row: dict) -> MemoryRecord:
    return MemoryRecord(
        id=str(row["id"]),
        content=str(row.get("content", "")),
        scope=norm_scope(row.get("scope")),
        categories=list(row.get("categories") or []),
        metadata=dict(row.get("metadata") or {}),
        importance=float(row.get("importance", 0.5)),
        created_at=_parse_dt(row.get("created_at")),
        last_accessed=_parse_dt(row.get("last_accessed")),
        embedding=list(row["embedding"]) if row.get("embedding") else None,
        source=row.get("source") or None,
        private=bool(row.get("private", False)),
    )


def _matches(row: dict, categories: Optional[List[str]], metadata_filter: Optional[Dict[str, Any]],
             older_than: Optional[datetime] = None) -> bool:
    if categories and not any(c in (row.get("categories") or []) for c in categories):
        return False
    if metadata_filter:
        md = row.get("metadata") or {}
        if not all(md.get(k) == v for k, v in metadata_filter.items()):
            return False
    if older_than is not None and _parse_dt(row.get("created_at")) >= older_than:
        return False
    return True


class MemoryBackend:
    """A CrewAI ``StorageBackend`` implemented over five datastore primitives."""

    supports_native_vector_search: bool = False

    # ------------------------------------------------------------------ primitives
    def _put(self, rows: List[dict]) -> None:
        raise NotImplementedError

    def _get(self, record_id: str) -> Optional[dict]:
        raise NotImplementedError

    def _delete_ids(self, ids: List[str]) -> int:
        raise NotImplementedError

    def _scan(self, scope_prefix: Optional[str]) -> List[dict]:
        """Every row whose scope is ``scope_prefix`` or under it (all rows for "/" / None)."""
        raise NotImplementedError

    def _vector_search(
        self, vector: List[float], scope_prefix: Optional[str], limit: int
    ) -> List[Tuple[dict, float]]:
        """Up to ``limit`` ``(row, cosine_similarity)`` pairs in the scope subtree, best first."""
        scored = [
            (row, cosine_similarity(vector, row["embedding"]))
            for row in self._scan(scope_prefix)
            if row.get("embedding")
        ]
        scored.sort(key=lambda rs: -rs[1])
        return scored[:limit]

    # ------------------------------------------------------------------ StorageBackend
    def save(self, records: List[MemoryRecord]) -> None:
        if records:
            self._put([record_to_row(r) for r in records])

    def update(self, record: MemoryRecord) -> None:
        self._put([record_to_row(record)])

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        row = self._get(record_id)
        return row_to_record(row) if row else None

    def search(
        self,
        query_embedding: List[float],
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        if not query_embedding:
            return []
        oversample = limit * 3 if (categories or metadata_filter) else limit
        hits = self._vector_search(list(query_embedding), scope_prefix, oversample)
        out: List[Tuple[MemoryRecord, float]] = []
        for row, score in hits:
            if not in_scope(row["scope"], scope_prefix):
                continue
            if not _matches(row, categories, metadata_filter):
                continue
            if score >= min_score:
                out.append((row_to_record(row), float(score)))
            if len(out) >= limit:
                break
        return out

    def delete(
        self,
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        record_ids: Optional[List[str]] = None,
        older_than: Optional[datetime] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        if record_ids and not (categories or metadata_filter or older_than):
            return self._delete_ids(list(record_ids))
        rows = self._scan(scope_prefix)
        if record_ids:
            wanted = set(record_ids)
            rows = [r for r in rows if r["id"] in wanted]
        ids = [r["id"] for r in rows if _matches(r, categories, metadata_filter, older_than)]
        return self._delete_ids(ids) if ids else 0

    def list_records(self, scope_prefix: Optional[str] = None, limit: int = 200, offset: int = 0) -> List[MemoryRecord]:
        rows = sorted(self._scan(scope_prefix), key=lambda r: _parse_dt(r.get("created_at")), reverse=True)
        return [row_to_record(r) for r in rows[offset : offset + limit]]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        scope = norm_scope(scope)
        rows = self._scan(scope)
        cats: set = set()
        oldest = newest = None
        children: set = set()
        child_prefix = "/" if scope == "/" else scope + "/"
        for r in rows:
            sc = norm_scope(r["scope"])
            if sc.startswith(child_prefix):
                first = sc[len(child_prefix):].split("/", 1)[0]
                if first:
                    children.add(child_prefix + first)
            cats.update(r.get("categories") or [])
            dt = _parse_dt(r.get("created_at"))
            oldest = dt if oldest is None or dt < oldest else oldest
            newest = dt if newest is None or dt > newest else newest
        return ScopeInfo(
            path=scope, record_count=len(rows), categories=sorted(cats),
            oldest_record=oldest, newest_record=newest, child_scopes=sorted(children),
        )

    def list_scopes(self, parent: str = "/") -> List[str]:
        parent = norm_scope(parent)
        child_prefix = "/" if parent == "/" else parent + "/"
        children: set = set()
        for r in self._scan(parent):
            sc = norm_scope(r["scope"])
            if sc.startswith(child_prefix) and sc != parent:
                first = sc[len(child_prefix):].split("/", 1)[0]
                if first:
                    children.add(child_prefix + first)
        return sorted(children)

    def list_categories(self, scope_prefix: Optional[str] = None) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self._scan(scope_prefix):
            for c in r.get("categories") or []:
                counts[c] = counts.get(c, 0) + 1
        return counts

    def count(self, scope_prefix: Optional[str] = None) -> int:
        return len(self._scan(scope_prefix))

    def reset(self, scope_prefix: Optional[str] = None) -> None:
        ids = [r["id"] for r in self._scan(scope_prefix)]
        if ids:
            self._delete_ids(ids)

    def touch_records(self, record_ids: List[str]) -> None:
        """Bump ``last_accessed`` (called by ``Memory.recall``; best-effort)."""
        now = datetime.utcnow().isoformat()
        rows = []
        for rid in record_ids:
            row = self._get(rid)
            if row:
                row["last_accessed"] = now
                rows.append(row)
        if rows:
            self._put(rows)

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    # ------------------------------------------------------------------ async surface
    async def asave(self, records: List[MemoryRecord]) -> None:
        await asyncio.to_thread(self.save, records)

    async def asearch(self, query_embedding: List[float], scope_prefix: Optional[str] = None,
                      categories: Optional[List[str]] = None, metadata_filter: Optional[Dict[str, Any]] = None,
                      limit: int = 10, min_score: float = 0.0) -> List[Tuple[MemoryRecord, float]]:
        return await asyncio.to_thread(
            self.search, query_embedding, scope_prefix, categories, metadata_filter, limit, min_score
        )

    async def adelete(self, scope_prefix: Optional[str] = None, categories: Optional[List[str]] = None,
                      record_ids: Optional[List[str]] = None, older_than: Optional[datetime] = None,
                      metadata_filter: Optional[Dict[str, Any]] = None) -> int:
        return await asyncio.to_thread(
            self.delete, scope_prefix, categories, record_ids, older_than, metadata_filter
        )
