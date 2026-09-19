# crewai-memory

**`StorageBackend`** implementations for [CrewAI](https://docs.crewai.com/en/concepts/memory)'s unified `Memory` (CrewAI ≥ 1.10) on managed cloud databases — hierarchical scopes, categories, metadata filters, importance/recency, and **native vector search** on every backend. CrewAI's engine keeps doing the LLM analysis, scope inference, consolidation and composite scoring; these packages replace the local LanceDB store with a shared, durable one.

📖 **Full documentation:** **<https://skamalj.github.io/agentstate-reducer/>**

## Packages

| Package | Backend | Vector search via |
|---|---|---|
| `crewai-memory-core` | shared base — build your own with 5 primitives | portable cosine fallback |
| `crewai-memory-dynamodb` | Amazon DynamoDB | native `SearchVectors` |
| `crewai-memory-postgres` | PostgreSQL | pgvector (`<=>`, HNSW) |
| `crewai-memory-cosmosdb` | Azure Cosmos DB (NoSQL) | `VectorDistance` (diskANN) |
| `crewai-memory-firestore` | Google Firestore | `find_nearest` (vector index) |

```bash
pip install crewai-memory-dynamodb     # or -postgres / -cosmosdb / -firestore
```

## Usage

```python
from crewai import Crew
from crewai.memory import Memory
from crewai_memory_dynamodb import DynamoDBMemoryBackend   # or PostgresMemoryBackend / CosmosDBMemoryBackend / FirestoreMemoryBackend

backend = DynamoDBMemoryBackend(table_name="crewai-memory", dimensions=3072)   # match the Memory embedder
memory = Memory(storage=backend)
crew = Crew(agents=[...], tasks=[...], memory=memory)

memory.remember("The customer prefers email over phone", scope="/customers/acme", categories=["preference"])
memory.recall("how should we contact acme?", scope="/customers")
```

Register once for every `Crew(memory=True)` instead:

```python
from crewai.memory.storage.factory import set_memory_storage_factory
set_memory_storage_factory(lambda spec: DynamoDBMemoryBackend("crewai-memory", dimensions=3072))
```

`dimensions` must equal the embedder's output size (CrewAI default `text-embedding-3-large` = 3072; Bedrock Titan v2 = 1024). Vector indexes on DynamoDB and Cosmos are fixed at table/container creation; Firestore's composite index is created via the Admin API (asynchronous build); Postgres needs the `pgvector` extension. See each package README.

## Architecture

A [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) monorepo. `crewai-memory-core` implements the **whole** `StorageBackend` protocol — `save` / `search` / `delete` / `update` / `get_record` / `list_records` / `get_scope_info` / `list_scopes` / `list_categories` / `count` / `reset` / `touch_records` and the async variants — over five primitives (`_put`, `_get`, `_delete_ids`, `_scan`, optional `_vector_search`). Providers implement only those against their datastore.

```
core/                  crewai-memory-core        (MemoryBackend base, FakeEmbedder, contract tests)
providers/dynamodb/    crewai-memory-dynamodb
providers/postgres/    crewai-memory-postgres
providers/cosmosdb/    crewai-memory-cosmosdb
providers/firestore/   crewai-memory-firestore
```

`crewai_memory_core.contract` is one importable test suite that every provider runs against its real backend — including an end-to-end pass through CrewAI's `Memory` engine with a deterministic embedder and zero LLM calls.

## Development

```bash
uv sync                                   # Python 3.12 (pinned: CrewAI's ChromaDB dependency breaks on 3.14)
uv run pytest core/tests                  # offline
uv run pytest providers/postgres/tests    # needs Postgres + pgvector (SQL_TEST_URL)
uv run pytest providers/dynamodb/tests    # needs AWS creds; CREWAI_DDB_REGION (default us-east-1)
uv run pytest providers/cosmosdb/tests    # needs COSMOS_ENDPOINT/COSMOS_KEY (+ EnableNoSQLVectorSearch)
uv run pytest providers/firestore/tests   # needs GCP_PROJECT + ADC
```

## License

MIT
