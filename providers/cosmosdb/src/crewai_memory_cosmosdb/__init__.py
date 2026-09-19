"""Azure Cosmos DB (NoSQL) ``StorageBackend`` for CrewAI unified ``Memory``.

A container partitioned by ``/scope`` holding one document per ``MemoryRecord``
(``id`` = record id) with a **vector embedding policy** on ``/embedding``
(cosine) and a ``diskANN`` vector index. ``search`` is a single SQL query:
``ORDER BY VectorDistance(c.embedding, @q)`` with the scope subtree filtered in
``WHERE``. Built on ``crewai-memory-core``.

The account needs the ``EnableNoSQLVectorSearch`` capability; the vector policy
can only be set at container creation.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from crewai_memory_core import MemoryBackend, ROW_FIELDS, norm_scope

__all__ = ["CosmosDBMemoryBackend"]

_PROJ = ", ".join(f'c["{f}"]' for f in ROW_FIELDS if f != "embedding")


class CosmosDBMemoryBackend(MemoryBackend):
    """CrewAI ``StorageBackend`` on Azure Cosmos DB with native vector search.

    Example:
        ```python
        from crewai.memory import Memory
        from crewai_memory_cosmosdb import CosmosDBMemoryBackend

        backend = CosmosDBMemoryBackend(endpoint=..., key=..., database_name="crewai",
                                        container_name="memory", dimensions=3072)
        memory = Memory(storage=backend)
        ```
    """

    supports_native_vector_search = True

    def __init__(
        self,
        *,
        endpoint: str,
        key: str,
        database_name: str,
        container_name: str,
        dimensions: int = 3072,
        vector_index_type: str = "diskANN",
    ) -> None:
        self.dimensions = dimensions
        client = CosmosClient(endpoint, credential=key)
        db = client.create_database_if_not_exists(database_name)
        self._container = db.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/scope"),
            vector_embedding_policy={
                "vectorEmbeddings": [
                    {"path": "/embedding", "dataType": "float32", "distanceFunction": "cosine", "dimensions": dimensions}
                ]
            },
            indexing_policy={
                "automatic": True,
                "indexingMode": "consistent",
                "includedPaths": [{"path": "/*"}],
                "excludedPaths": [{"path": "/embedding/*"}, {"path": '/"_etag"/?'}],
                "vectorIndexes": [{"path": "/embedding", "type": vector_index_type}],
            },
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _row(doc: dict) -> dict:
        row = {f: doc.get(f) for f in ROW_FIELDS}
        row["scope"] = norm_scope(doc.get("scope"))
        return row

    @staticmethod
    def _scope_where(scope_prefix: Optional[str]) -> Tuple[str, list]:
        p = norm_scope(scope_prefix)
        if p == "/":
            return "true", []
        return '(c["scope"] = @p OR STARTSWITH(c["scope"], @pc))', [
            {"name": "@p", "value": p}, {"name": "@pc", "value": p + "/"},
        ]

    def _query(self, sql: str, params: list) -> List[dict]:
        return list(self._container.query_items(query=sql, parameters=params, enable_cross_partition_query=True))

    # ------------------------------------------------------------------ primitives
    def _put(self, rows: List[dict]) -> None:
        for row in rows:
            existing = self._get(row["id"])
            if existing and existing["scope"] != norm_scope(row["scope"]):
                self._delete_doc(row["id"], existing["scope"])   # scope moved: partition changes
            doc = {f: row.get(f) for f in ROW_FIELDS}
            doc["scope"] = norm_scope(row["scope"])
            if not doc.get("embedding"):
                doc.pop("embedding", None)
            self._container.upsert_item(doc)

    def _get(self, record_id: str) -> Optional[dict]:
        docs = self._query('SELECT TOP 1 * FROM c WHERE c["id"] = @id', [{"name": "@id", "value": record_id}])
        return self._row(docs[0]) if docs else None

    def _delete_doc(self, record_id: str, scope: str) -> bool:
        try:
            self._container.delete_item(item=record_id, partition_key=scope)
            return True
        except CosmosResourceNotFoundError:
            return False

    def _delete_ids(self, ids: List[str]) -> int:
        n = 0
        for rid in ids:
            row = self._get(rid)
            if row and self._delete_doc(rid, row["scope"]):
                n += 1
        return n

    def _scan(self, scope_prefix: Optional[str]) -> List[dict]:
        where, params = self._scope_where(scope_prefix)
        return [self._row(d) for d in self._query(f"SELECT * FROM c WHERE {where}", params)]

    def _vector_search(
        self, vector: List[float], scope_prefix: Optional[str], limit: int
    ) -> List[Tuple[dict, float]]:
        where, params = self._scope_where(scope_prefix)
        sql = (
            f'SELECT TOP @k {_PROJ}, VectorDistance(c["embedding"], @q) AS score '
            f'FROM c WHERE {where} AND IS_DEFINED(c["embedding"]) '
            f'ORDER BY VectorDistance(c["embedding"], @q)'
        )
        params = params + [{"name": "@q", "value": vector}, {"name": "@k", "value": int(limit)}]
        out: List[Tuple[dict, float]] = []
        for d in self._query(sql, params):
            row = self._row(d)
            out.append((row, float(d["score"])))   # cosine similarity, higher is better
        return out
