# crewai-memory-postgres

A **PostgreSQL (pgvector)** `StorageBackend` for [CrewAI](https://docs.crewai.com/en/concepts/memory)'s unified `Memory` — hierarchical scopes, categories, metadata filters, importance/recency, and **native vector search** (`<=>`, HNSW index).

```bash
pip install crewai-memory-postgres
```

```python
from crewai import Crew
from crewai.memory import Memory
from crewai_memory_postgres import PostgresMemoryBackend

backend = PostgresMemoryBackend("postgresql://user:pass@host:5432/db", dimensions=3072)   # match your embedder
memory = Memory(storage=backend)
crew = Crew(agents=[...], tasks=[...], memory=memory)
```

Or once for every `Crew(memory=True)`: `set_memory_storage_factory(lambda spec: PostgresMemoryBackend(...))`.

## How it works

- One table (`crewai_memory` by default) with `id`, `scope`, `content`, `categories` / `metadata` (JSONB), `importance`, timestamps, `source`, `private`, and `embedding vector(dims)`; a B-tree on `scope` and an HNSW cosine index on `embedding`. Auto-created; `CREATE EXTENSION IF NOT EXISTS vector` is run at startup.
- `search` is one SQL query ordered by `embedding <=> query` with the scope subtree in `WHERE` (`scope = p OR scope LIKE 'p/%'`); score is `1 - cosine_distance`. `categories` / `metadata_filter` / `min_score` are applied on the candidates.
- Requires the [pgvector](https://github.com/pgvector/pgvector) extension on the server (`apt install postgresql-16-pgvector`, or built in on RDS / Cloud SQL / Azure Database).

Docs: <https://skamalj.github.io/agentstate-reducer/> · part of [crewai-memory](https://github.com/skamalj/crewai-memory)

## License

MIT
