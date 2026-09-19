"""Core contract on the dict-backed InMemoryBackend (offline)."""
import pytest

from crewai_memory_core.contract import *  # noqa: F401,F403
from crewai_memory_core import in_scope, norm_scope
from crewai_memory_core.testing import InMemoryBackend


@pytest.fixture()
def backend():
    return InMemoryBackend()


def test_norm_and_in_scope():
    assert norm_scope(None) == "/" and norm_scope("a/b/") == "/a/b" and norm_scope("/") == "/"
    assert in_scope("/a/b", "/a") and in_scope("/a", "/a") and not in_scope("/ab", "/a")
    assert in_scope("/anything", "/") and in_scope("/x", None)
