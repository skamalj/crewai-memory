# crewai-memory-cosmosdb

An **Azure Cosmos DB (NoSQL)** `StorageBackend` for [CrewAI](https://docs.crewai.com/en/concepts/memory)'s unified `Memory` — hierarchical scopes, categories, metadata filters, importance/recency, and **native vector search** via `VectorDistance` (diskANN index).

```bash
pip install crewai-memory-cosmosdb
```

```python
from crewai import Crew
from crewai.memory import Memory
from crewai_memory_cosmosdb import CosmosDBMemoryBackend

backend = CosmosDBMemoryBackend(
    endpoint="https://<acct>.documents.azure.com:443/", key="<key>",
    database_name="crewai", container_name="memory", dimensions=3072,   # match your embedder
)
memory = Memory(storage=backend)
crew = Crew(agents=[...], tasks=[...], memory=memory)
```

Or once for every `Crew(memory=True)`: `set_memory_storage_factory(lambda spec: CosmosDBMemoryBackend(...))`.

## How it works

- A container partitioned by `/scope`, one document per record (`id` = record id), created with a vector embedding policy on `/embedding` (cosine, `dimensions`) and a `diskANN` vector index (`vector_index_type="quantizedFlat"|"flat"` to change). The policy is fixed at container creation — use a new container to change `dimensions`.
- `search` is one SQL query: `ORDER BY VectorDistance(c.embedding, @q)` with the scope subtree in `WHERE`; the score is the cosine similarity. `categories` / `metadata_filter` / `min_score` are applied on the candidates.
- Scope tree, category counts, listing, `delete(older_than=...)`, `reset` are cross-partition SQL queries.
- The account needs **Vector Search for NoSQL**: `az cosmosdb update -n <acct> -g <rg> --capabilities EnableServerless EnableNoSQLVectorSearch` (list all existing capabilities; propagation can take a few minutes).

Docs: <https://skamalj.github.io/agentstate-reducer/> · part of [crewai-memory](https://github.com/skamalj/crewai-memory)

## License

MIT
