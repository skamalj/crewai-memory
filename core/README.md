# crewai-memory-core

Shared core for building [CrewAI](https://docs.crewai.com/en/concepts/memory) unified-memory **`StorageBackend`** implementations. CrewAI 1.10+ has one `Memory` engine (LLM analysis, consolidation, scope inference, composite scoring) over a pluggable storage protocol; this core implements the *whole* protocol — `save` / `search` / `delete` / `update` / `get_record` / `list_records` / `get_scope_info` / `list_scopes` / `list_categories` / `count` / `reset` / `touch_records` and the async variants — so a backend only supplies five primitives:

```python
from crewai_memory_core import MemoryBackend

class MyBackend(MemoryBackend):
    def _put(self, rows): ...                    # upsert rows by id
    def _get(self, record_id): ...               # -> row | None
    def _delete_ids(self, ids): ...              # -> int deleted
    def _scan(self, scope_prefix): ...           # every row in the scope subtree
    # optional — native ANN; default ranks _scan rows by cosine in Python
    def _vector_search(self, vector, scope_prefix, limit): ...   # -> [(row, score), ...]
```

A `row` is a dict of the `MemoryRecord` fields plus `embedding`. Scope paths are hierarchical (`/company/team/user`); a prefix matches the scope itself and everything under it, never a sibling that merely shares characters (`/a` does not match `/ab`).

Use a backend with CrewAI:

```python
from crewai import Crew
from crewai.memory import Memory

memory = Memory(storage=MyBackend(...))
crew = Crew(agents=[...], tasks=[...], memory=memory)     # or set_memory_storage_factory(...) once at startup
```

`crewai_memory_core.testing` ships `FakeEmbedder` (deterministic, no network — pass as `Memory(embedder=...)`) and `InMemoryBackend`; `crewai_memory_core.contract` is an importable test suite every provider runs, including an end-to-end pass through CrewAI's real `Memory` engine with zero LLM calls.

Concrete backends: [`crewai-memory-dynamodb`](https://pypi.org/project/crewai-memory-dynamodb/), [`crewai-memory-postgres`](https://pypi.org/project/crewai-memory-postgres/), [`crewai-memory-cosmosdb`](https://pypi.org/project/crewai-memory-cosmosdb/), [`crewai-memory-firestore`](https://pypi.org/project/crewai-memory-firestore/).

Docs: <https://skamalj.github.io/agentstate-reducer/>

## License

MIT
