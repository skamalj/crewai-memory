"""DynamoDB StorageBackend: the shared contract suite against a real table (us-east-1)."""
import os
import time
import uuid

import boto3
import pytest

from crewai_memory_core.contract import *  # noqa: F401,F403
from crewai_memory_core.contract import DIMS
from crewai_memory_dynamodb import DynamoDBMemoryBackend

REGION = os.environ.get("CREWAI_DDB_REGION", "us-east-1")


def _aws_available() -> bool:
    try:
        boto3.client("sts", region_name=REGION).get_caller_identity()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _aws_available(), reason="AWS credentials not available")


@pytest.fixture(scope="module")
def _table():
    name = f"crewai_mem_test_{uuid.uuid4().hex[:8]}"
    b = DynamoDBMemoryBackend(name, dimensions=DIMS, region_name=REGION)
    ddb = boto3.client("dynamodb", region_name=REGION)
    for _ in range(60):
        d = ddb.describe_table(TableName=name)["Table"]
        vi = d.get("VectorIndexes") or []
        gsi = d.get("GlobalSecondaryIndexes") or []
        if vi and all(v.get("IndexStatus", "ACTIVE") == "ACTIVE" for v in vi) and all(g.get("IndexStatus") == "ACTIVE" for g in gsi):
            break
        time.sleep(5)
    yield b
    try:
        ddb.delete_table(TableName=name)
    except Exception:
        pass


@pytest.fixture()
def backend(_table):
    _table.reset()
    yield _table
    _table.reset()


def test_native_vector_path_used(backend, monkeypatch):
    from crewai_memory_core.contract import _seed, EMB
    _seed(backend)
    calls = []
    orig = backend._client.search_vectors

    def spy(**kw):
        calls.append(kw["SearchConditionExpression"])
        return orig(**kw)

    monkeypatch.setattr(backend._client, "search_vectors", spy)
    hits = backend.search(EMB(["sushi"])[0], scope_prefix="/crew/support/user", limit=5)
    assert calls and all(c == "#pk = :s" for c in calls) and len(calls) == 2   # one call per child scope
    assert hits
