"""Google Firestore ``StorageBackend`` for CrewAI unified ``Memory``.

One collection, one document per ``MemoryRecord`` (doc id = record id) with the
embedding stored as a Firestore ``Vector``. ``search`` uses native
``find_nearest`` (cosine) after a scope pre-filter; a scope *subtree* needs two
queries (the exact scope, and the ``scope/…`` range), merged by similarity.
Built on ``crewai-memory-core``.

Firestore needs a composite **vector index** ``scope ASC + embedding (flat)``;
with ``create_index=True`` (default) it is created through the Admin API if
missing (asynchronous build — vector queries fail with FAILED_PRECONDITION
until READY).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from crewai_memory_core import MemoryBackend, ROW_FIELDS, norm_scope

__all__ = ["FirestoreMemoryBackend"]

logger = logging.getLogger(__name__)
_HIGH = chr(0xF8FF)


class FirestoreMemoryBackend(MemoryBackend):
    """CrewAI ``StorageBackend`` on Firestore with native vector search.

    Example:
        ```python
        from crewai.memory import Memory
        from crewai_memory_firestore import FirestoreMemoryBackend

        backend = FirestoreMemoryBackend(project_id="my-project", collection="crewai_memory", dimensions=3072)
        memory = Memory(storage=backend)
        ```
    """

    supports_native_vector_search = True

    def __init__(
        self,
        *,
        project_id: str,
        collection: str = "crewai_memory",
        dimensions: int = 3072,
        client: Optional["firestore.Client"] = None,
        create_index: bool = True,
        database: str = "(default)",
    ) -> None:
        self.dimensions = dimensions
        self._project = project_id
        self._database = database
        self._collection = collection
        self._client = client or firestore.Client(project=project_id, database=database)
        self._col = self._client.collection(collection)
        if create_index:
            self._ensure_vector_index()

    # ------------------------------------------------------------------ index bootstrap
    def _ensure_vector_index(self) -> None:
        from google.cloud import firestore_admin_v1 as fa

        admin = fa.FirestoreAdminClient()
        parent = f"projects/{self._project}/databases/{self._database}/collectionGroups/{self._collection}"
        for ix in admin.list_indexes(parent=parent):
            paths = [f.field_path for f in ix.fields]
            if "embedding" in paths and "scope" in paths:
                return
        index = fa.Index(
            query_scope=fa.Index.QueryScope.COLLECTION,
            fields=[
                fa.Index.IndexField(field_path="scope", order=fa.Index.IndexField.Order.ASCENDING),
                fa.Index.IndexField(
                    field_path="embedding",
                    vector_config=fa.Index.IndexField.VectorConfig(
                        dimension=self.dimensions, flat=fa.Index.IndexField.VectorConfig.FlatIndex()
                    ),
                ),
            ],
        )
        try:
            admin.create_index(parent=parent, index=index)
            logger.info("FirestoreMemoryBackend: creating vector index on %s (async)", self._collection)
        except Exception as exc:  # already exists / racing creation
            logger.info("FirestoreMemoryBackend: vector index create skipped: %s", exc)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _row(d: dict) -> dict:
        row = {f: d.get(f) for f in ROW_FIELDS}
        row["scope"] = norm_scope(d.get("scope"))
        emb = d.get("embedding")
        row["embedding"] = list(emb.to_map_value()["value"]) if isinstance(emb, Vector) else (list(emb) if emb else None)
        return row

    def _scope_queries(self, scope_prefix: Optional[str]):
        """Firestore has no OR-with-range on one field, so a subtree is two queries."""
        p = norm_scope(scope_prefix)
        if p == "/":
            # Every scope starts with "/", so a range on `scope` selects all rows while
            # still using the scope+embedding composite index (a filter-less
            # find_nearest would need a separate embedding-only index).
            return [self._col.where(filter=FieldFilter("scope", ">=", "/"))]
        return [
            self._col.where(filter=FieldFilter("scope", "==", p)),
            self._col.where(filter=FieldFilter("scope", ">=", p + "/")).where(filter=FieldFilter("scope", "<", p + "/" + _HIGH)),
        ]

    # ------------------------------------------------------------------ primitives
    def _put(self, rows: List[dict]) -> None:
        batch = self._client.batch()
        for row in rows:
            doc = {f: row.get(f) for f in ROW_FIELDS if f != "embedding"}
            doc["scope"] = norm_scope(row["scope"])
            doc["categories"] = list(row.get("categories") or [])
            doc["metadata"] = dict(row.get("metadata") or {})
            doc["private"] = bool(row.get("private", False))
            if row.get("embedding"):
                doc["embedding"] = Vector(row["embedding"])
            batch.set(self._col.document(row["id"]), doc)
        batch.commit()

    def _get(self, record_id: str) -> Optional[dict]:
        snap = self._col.document(record_id).get()
        return self._row(snap.to_dict()) if snap.exists else None

    def _delete_ids(self, ids: List[str]) -> int:
        n = 0
        batch = self._client.batch()
        for rid in ids:
            ref = self._col.document(rid)
            if ref.get().exists:
                batch.delete(ref)
                n += 1
        batch.commit()
        return n

    def _scan(self, scope_prefix: Optional[str]) -> List[dict]:
        rows: List[dict] = []
        for q in self._scope_queries(scope_prefix):
            rows.extend(self._row(d.to_dict()) for d in q.stream())
        return rows

    def _vector_search(
        self, vector: List[float], scope_prefix: Optional[str], limit: int
    ) -> List[Tuple[dict, float]]:
        out: List[Tuple[dict, float]] = []
        for q in self._scope_queries(scope_prefix):
            vq = q.find_nearest(
                vector_field="embedding", query_vector=Vector(vector),
                distance_measure=DistanceMeasure.COSINE, limit=limit, distance_result_field="_distance",
            )
            for d in vq.stream():
                data = d.to_dict()
                dist = data.pop("_distance", None)
                out.append((self._row(data), 1.0 - float(dist) if dist is not None else 0.0))
        out.sort(key=lambda rs: -rs[1])
        return out[:limit]
