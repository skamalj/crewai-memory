"""Amazon DynamoDB ``StorageBackend`` for CrewAI unified ``Memory``.

One table, ``PK`` = scope path, ``SK`` = record id, with a DynamoDB **vector
index** on ``embedding`` (cosine) whose search schema declares ``PK`` as an
inline filter. ``search`` runs DynamoDB's native ``SearchVectors`` — one call
per concrete scope under the requested prefix (DynamoDB allows only equality on
a string search-schema attribute) — and the core merges by similarity and
applies category / metadata filters. Built on ``crewai-memory-core``.

Requires ``boto3>=1.43.78`` and a region where DynamoDB vector search is
available. The vector index can only be declared at table creation.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, List, Optional, Tuple

import boto3
from boto3.dynamodb.conditions import Attr, Key
from boto3.dynamodb.types import TypeDeserializer

from crewai_memory_core import MemoryBackend, ROW_FIELDS, in_scope, norm_scope

__all__ = ["DynamoDBMemoryBackend"]


def _plain(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def _ddb(obj: Any) -> Any:
    if isinstance(obj, float):
        return Decimal(repr(obj))
    if isinstance(obj, dict):
        return {k: _ddb(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ddb(v) for v in obj]
    return obj


class DynamoDBMemoryBackend(MemoryBackend):
    """CrewAI ``StorageBackend`` on DynamoDB with native vector search.

    Example:
        ```python
        from crewai.memory import Memory
        from crewai_memory_dynamodb import DynamoDBMemoryBackend

        backend = DynamoDBMemoryBackend(table_name="crewai-memory", dimensions=3072)
        memory = Memory(storage=backend)          # embedder default: OpenAI text-embedding-3-large (3072)
        ```

    ``dimensions`` must match the embedder configured on ``Memory``.
    """

    supports_native_vector_search = True

    def __init__(
        self,
        table_name: str,
        *,
        dimensions: int = 3072,
        index_name: str = "vector_index",
        region_name: Optional[str] = None,
        boto_session: Optional["boto3.Session"] = None,
        endpoint_url: Optional[str] = None,
    ) -> None:
        session = boto_session or boto3.Session(region_name=region_name)
        self._client = session.client("dynamodb", endpoint_url=endpoint_url)
        self._resource = session.resource("dynamodb", endpoint_url=endpoint_url)
        self.table_name = table_name
        self.dimensions = dimensions
        self._index_name = index_name
        self._deser = TypeDeserializer()
        self._ensure_table()
        self.table = self._resource.Table(table_name)

    # ------------------------------------------------------------------ setup
    def _ensure_table(self) -> None:
        try:
            self._client.describe_table(TableName=self.table_name)
            return
        except self._client.exceptions.ResourceNotFoundException:
            pass
        self._client.create_table(
            TableName=self.table_name,
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "by_id",
                    "KeySchema": [{"AttributeName": "SK", "KeyType": "HASH"}],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            VectorIndexes=[
                {
                    "IndexName": self._index_name,
                    "VectorAttribute": {"AttributeName": "embedding"},
                    "Dimensions": self.dimensions,
                    "DistanceFunction": "COSINE",
                    "Projection": {"ProjectionType": "ALL"},
                    "SearchSchema": [{"AttributeName": "PK", "SearchSchemaElementType": "INLINE_FILTER"}],
                }
            ],
        )
        self._client.get_waiter("table_exists").wait(TableName=self.table_name)

    # ------------------------------------------------------------------ rows
    def _to_row(self, item: dict) -> dict:
        item = _plain(item)
        row = {f: item.get(f) for f in ROW_FIELDS}
        row["id"] = item["SK"]
        row["scope"] = item["PK"]
        row["embedding"] = [float(v) for v in item["embedding"]] if item.get("embedding") else None
        return row

    def _to_item(self, row: dict) -> dict:
        item = {"PK": norm_scope(row["scope"]), "SK": row["id"]}
        for f in ROW_FIELDS:
            if f in ("id", "scope", "embedding"):
                continue
            if row.get(f) is not None:
                item[f] = _ddb(row[f])
        if row.get("embedding"):
            item["embedding"] = [Decimal(repr(float(v))) for v in row["embedding"]]
        return item

    def _paginate(self, fn, **kwargs) -> List[dict]:
        items: List[dict] = []
        while True:
            resp = fn(**kwargs)
            items.extend(resp.get("Items", []))
            lek = resp.get("LastEvaluatedKey")
            if not lek:
                return items
            kwargs["ExclusiveStartKey"] = lek

    # ------------------------------------------------------------------ primitives
    def _put(self, rows: List[dict]) -> None:
        with self.table.batch_writer() as bw:
            for row in rows:
                existing = self._get(row["id"])
                if existing and norm_scope(existing["scope"]) != norm_scope(row["scope"]):
                    bw.delete_item(Key={"PK": existing["scope"], "SK": row["id"]})   # scope moved
                bw.put_item(Item=self._to_item(row))

    def _get(self, record_id: str) -> Optional[dict]:
        resp = self.table.query(IndexName="by_id", KeyConditionExpression=Key("SK").eq(record_id), Limit=1)
        items = resp.get("Items", [])
        return self._to_row(items[0]) if items else None

    def _delete_ids(self, ids: List[str]) -> int:
        n = 0
        with self.table.batch_writer() as bw:
            for rid in ids:
                row = self._get(rid)
                if row:
                    bw.delete_item(Key={"PK": row["scope"], "SK": rid})
                    n += 1
        return n

    def _scan(self, scope_prefix: Optional[str]) -> List[dict]:
        p = norm_scope(scope_prefix)
        if p == "/":
            items = self._paginate(self.table.scan)
        else:
            items = self._paginate(self.table.query, KeyConditionExpression=Key("PK").eq(p))
            items += self._paginate(self.table.scan, FilterExpression=Attr("PK").begins_with(p + "/"))
        return [self._to_row(i) for i in items]

    def _scopes_under(self, scope_prefix: Optional[str]) -> List[str]:
        p = norm_scope(scope_prefix)
        kwargs: dict = {"ProjectionExpression": "PK"}
        if p != "/":
            kwargs["FilterExpression"] = Attr("PK").eq(p) | Attr("PK").begins_with(p + "/")
        return sorted({i["PK"] for i in self._paginate(self.table.scan, **kwargs)})

    def _vector_search(
        self, vector: List[float], scope_prefix: Optional[str], limit: int
    ) -> List[Tuple[dict, float]]:
        scopes = self._scopes_under(scope_prefix)
        if not scopes:
            return []
        qv = [{"N": repr(float(v))} for v in vector]
        out: List[Tuple[dict, float]] = []
        for sc in scopes:
            resp = self._client.search_vectors(
                TableName=self.table_name,
                IndexName=self._index_name,
                SearchVector=qv,
                TopK=max(1, min(100, int(limit))),   # SearchVectors accepts TopK in [1, 100]
                SearchConditionExpression="#pk = :s",
                ExpressionAttributeNames={"#pk": "PK"},
                ExpressionAttributeValues={":s": {"S": sc}},
            )
            for r in resp.get("SearchResults", []):
                item = {k: self._deser.deserialize(v) for k, v in r["Item"].items()}
                # DynamoDB returns the cosine *distance* (0 = identical).
                out.append((self._to_row(item), 1.0 - float(r.get("Score", 1.0))))
        out.sort(key=lambda rs: -rs[1])
        return out[:limit]
