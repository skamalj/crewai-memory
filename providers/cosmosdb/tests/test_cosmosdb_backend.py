"""Cosmos DB StorageBackend: the shared contract suite against a real Cosmos account."""
import os
import uuid

import pytest

from crewai_memory_core.contract import *  # noqa: F401,F403
from crewai_memory_core.contract import DIMS
from crewai_memory_cosmosdb import CosmosDBMemoryBackend

EP = os.environ.get("COSMOS_ENDPOINT")
KEY = os.environ.get("COSMOS_KEY")
DB = os.environ.get("CREWAI_COSMOS_DB", "crewai_memory_test")

pytestmark = pytest.mark.skipif(not (EP and KEY), reason="COSMOS_ENDPOINT/COSMOS_KEY not set")


@pytest.fixture(scope="module")
def _store():
    cname = f"mem_{uuid.uuid4().hex[:8]}"
    b = CosmosDBMemoryBackend(endpoint=EP, key=KEY, database_name=DB, container_name=cname, dimensions=DIMS)
    yield b
    from azure.cosmos import CosmosClient
    CosmosClient(EP, credential=KEY).get_database_client(DB).delete_container(cname)


@pytest.fixture()
def backend(_store):
    _store.reset()
    yield _store
    _store.reset()


def test_native_vector_path_used(backend, monkeypatch):
    from crewai_memory_core.contract import _seed, EMB
    _seed(backend)
    seen = []
    orig = backend._container.query_items

    def spy(query, **kw):
        seen.append(query)
        return orig(query=query, **kw)

    monkeypatch.setattr(backend._container, "query_items", spy)
    assert backend.search(EMB(["sushi"])[0], scope_prefix="/crew", limit=5)
    assert any("VectorDistance" in q for q in seen)
