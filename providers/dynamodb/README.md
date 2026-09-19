# crewai-memory-dynamodb

An **Amazon DynamoDB** `StorageBackend` for [CrewAI](https://docs.crewai.com/en/concepts/memory)'s unified `Memory` — hierarchical scopes, categories, metadata filters, importance/recency, and **native vector search** via DynamoDB `SearchVectors`. No external vector database.

```bash
pip install crewai-memory-dynamodb
```

```python
from crewai import Crew
from crewai.memory import Memory
from crewai_memory_dynamodb import DynamoDBMemoryBackend

backend = DynamoDBMemoryBackend(table_name="crewai-memory", dimensions=3072)   # match your embedder
memory = Memory(storage=backend)                                               # CrewAI does LLM analysis, scoping, scoring
crew = Crew(agents=[...], tasks=[...], memory=memory)

memory.remember("The customer prefers email over phone", scope="/customers/acme", categories=["preference"])
memory.recall("how should we contact acme?", scope="/customers")
```

Or register it once for every `Crew(memory=True)`:

```python
from crewai.memory.storage.factory import set_memory_storage_factory
set_memory_storage_factory(lambda spec: DynamoDBMemoryBackend("crewai-memory", dimensions=3072))
```

## How it works

- One table: `PK` = scope path, `SK` = record id, plus a `by_id` GSI and a **vector index** on `embedding` (cosine, `dimensions`) whose search schema declares `PK` as an inline filter. Auto-created (`PAY_PER_REQUEST`); the vector index can only be declared at creation, so use a new table to change `dimensions`.
- `search` runs native `SearchVectors` once per concrete scope under the requested prefix (DynamoDB allows only equality on a string search-schema attribute), merges by similarity (`1 - distance`), then applies `categories` / `metadata_filter` / `min_score` on the candidates (oversampled ×3 when filtering).
- Scope tree, category counts, `list_records`, `delete(older_than=...)`, `reset`, `touch_records` are query/scan based — sized for agent-memory volumes.
- `dimensions` must equal the `Memory` embedder's output size (CrewAI default `text-embedding-3-large` = 3072; Bedrock Titan v2 = 1024).

Requires `boto3>=1.43.78` and a region where DynamoDB vector search is available. Permissions: `DescribeTable`, `CreateTable`, `GetItem`, `PutItem`, `DeleteItem`, `BatchWriteItem`, `Query`, `Scan`, `SearchVectors`.

Docs: <https://skamalj.github.io/agentstate-reducer/> · part of [crewai-memory](https://github.com/skamalj/crewai-memory)

## License

MIT
