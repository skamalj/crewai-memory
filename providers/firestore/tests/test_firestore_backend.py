"""Firestore StorageBackend: the shared contract suite against a real Firestore database.

The backend creates the composite vector index (scope ASC + embedding flat) via
the Admin API; index builds are asynchronous so the fixture waits for READY and
the collection name is fixed per DIMS so the index is reused across runs.
"""
import os
import time

import pytest

from crewai_memory_core.contract import *  # noqa: F401,F403
from crewai_memory_core.contract import DIMS
from crewai_memory_firestore import FirestoreMemoryBackend

PROJECT = os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or "gcdeveloper-new"
COLLECTION = f"crewai_mem_test_d{DIMS}"


def _available() -> bool:
    try:
        from google.cloud import firestore
        firestore.Client(project=PROJECT).collection("x").limit(1).get()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="Firestore not reachable")


def _wait_index_ready(timeout: int = 600) -> None:
    from google.cloud import firestore_admin_v1 as fa
    admin = fa.FirestoreAdminClient()
    parent = f"projects/{PROJECT}/databases/(default)/collectionGroups/{COLLECTION}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        for ix in admin.list_indexes(parent=parent):
            paths = [f.field_path for f in ix.fields]
            if "embedding" in paths and "scope" in paths and ix.state == fa.Index.State.READY:
                return
        time.sleep(10)
    raise RuntimeError("vector index not READY in time")


@pytest.fixture(scope="module")
def _store():
    b = FirestoreMemoryBackend(project_id=PROJECT, collection=COLLECTION, dimensions=DIMS)
    _wait_index_ready()
    yield b
    b.reset()


@pytest.fixture()
def backend(_store):
    _store.reset()
    yield _store
    _store.reset()


def test_native_vector_path_used(backend, monkeypatch):
    from crewai_memory_core.contract import _seed, EMB
    _seed(backend)
    called = []
    orig = backend._vector_search
    monkeypatch.setattr(backend, "_vector_search", lambda *a, **k: (called.append(1), orig(*a, **k))[1])
    assert backend.search(EMB(["sushi"])[0], scope_prefix="/crew", limit=5)
    assert called
