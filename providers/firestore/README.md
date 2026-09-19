# crewai-memory-firestore

A **Google Firestore** `StorageBackend` for [CrewAI](https://docs.crewai.com/en/concepts/memory)'s unified `Memory` — hierarchical scopes, categories, metadata filters, importance/recency, and **native vector search** via `find_nearest`.

```bash
pip install crewai-memory-firestore
```

```python
from crewai import Crew
from crewai.memory import Memory
from crewai_memory_firestore import FirestoreMemoryBackend

backend = FirestoreMemoryBackend(project_id="my-project", collection="crewai_memory", dimensions=3072)  # match your embedder
memory = Memory(storage=backend)
crew = Crew(agents=[...], tasks=[...], memory=memory)
```

Or once for every `Crew(memory=True)`: `set_memory_storage_factory(lambda spec: FirestoreMemoryBackend(...))`. Authentication uses Application Default Credentials.

## How it works

- One collection, one document per record (doc id = record id) with the embedding stored as a Firestore `Vector`.
- `search` uses native `find_nearest` (cosine) after a scope pre-filter. A scope *subtree* is two queries (the exact scope, and the `scope/…` range) merged by similarity; score is `1 - distance`. `categories` / `metadata_filter` / `min_score` are applied on the candidates.
- Firestore requires a **composite vector index** `scope ASC + embedding (flat, dims)`. With `create_index=True` (default) the backend creates it through the Admin API if missing; the build is asynchronous and vector queries fail with `FAILED_PRECONDITION` until it is READY (a few minutes). Needs `datastore.indexes.create`/`list`, or create it yourself and pass `create_index=False`:

```bash
gcloud firestore indexes composite create --collection-group=crewai_memory --query-scope=COLLECTION \
  --field-config=order=ASCENDING,field-path=scope \
  --field-config=vector-config='{"dimension":"3072","flat":"{}"}',field-path=embedding
```

Docs: <https://skamalj.github.io/agentstate-reducer/> · part of [crewai-memory](https://github.com/skamalj/crewai-memory)

## License

MIT
