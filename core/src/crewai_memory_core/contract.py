"""Reusable ``StorageBackend`` contract tests.

A provider's test module supplies a ``backend`` fixture (fresh, empty store)
and does ``from crewai_memory_core.contract import *``. Every test below then
runs against that backend, including an end-to-end pass through CrewAI's real
``Memory`` engine with the deterministic ``FakeEmbedder`` (no LLM: all record
fields are provided so the encoding flow takes its zero-LLM fast path, and
recall uses ``depth="shallow"``).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest
from crewai.memory.types import MemoryRecord

from .testing import FakeEmbedder

DIMS = 64
EMB = FakeEmbedder(DIMS)

__all__ = [n for n in dir() if n.startswith("test_")]  # filled after definitions


def _rec(content, scope="/", categories=None, importance=0.5, metadata=None, source=None, private=False, created=None):
    return MemoryRecord(
        content=content, scope=scope, categories=categories or [], importance=importance,
        metadata=metadata or {}, source=source, private=private,
        embedding=EMB([content])[0], **({"created_at": created} if created else {}),
    )


def _seed(b):
    recs = [
        _rec("the user loves sushi and japanese food", "/crew/support/user/kamal", ["food", "pref"], 0.9, {"lang": "en"}),
        _rec("the user writes python every day", "/crew/support/user/kamal", ["tech"], 0.6, {"lang": "en"}),
        _rec("the user lives by the sea in a big city", "/crew/support/user/kamal", ["place"], 0.4, {"lang": "fr"}),
        _rec("priya loves sushi", "/crew/support/user/priya", ["food"], 0.5),
        _rec("company policy: remote work allowed", "/company", ["policy"], 0.8),
    ]
    b.save(recs)
    return recs


def _wait(fn, cond, tries=15, delay=1.0):
    for _ in range(tries):
        out = fn()
        if cond(out):
            return out
        time.sleep(delay)
    return fn()


# ── save / get / update / delete ──────────────────────────────────────────────


def test_save_and_get_roundtrip(backend):
    r = _rec("hello world", "/a/b", ["x"], 0.7, {"k": 1, "s": "v"}, source="u1", private=True)
    backend.save([r])
    got = _wait(lambda: backend.get_record(r.id), lambda g: g is not None)
    assert got.id == r.id and got.content == "hello world" and got.scope == "/a/b"
    assert got.categories == ["x"] and got.metadata == {"k": 1, "s": "v"}
    assert got.importance == pytest.approx(0.7) and got.source == "u1" and got.private is True
    assert abs((got.created_at - r.created_at).total_seconds()) < 1


def test_get_missing_is_none(backend):
    assert backend.get_record("nope") is None


def test_update_replaces_record(backend):
    r = _rec("v1", "/a")
    backend.save([r])
    r2 = r.model_copy(update={"content": "v2", "importance": 0.99, "embedding": EMB(["v2"])[0]})
    backend.update(r2)
    got = _wait(lambda: backend.get_record(r.id), lambda g: g and g.content == "v2")
    assert got.content == "v2" and got.importance == pytest.approx(0.99)
    assert backend.count() == 1


def test_delete_by_ids(backend):
    recs = _seed(backend)
    _wait(lambda: backend.count(), lambda n: n == 5)
    assert backend.delete(record_ids=[recs[0].id, recs[1].id]) == 2
    assert _wait(lambda: backend.count(), lambda n: n == 3) == 3


def test_delete_by_scope_categories_metadata_and_age(backend):
    _seed(backend)
    old = _rec("ancient fact", "/crew/support/user/kamal", ["old"], created=datetime.utcnow() - timedelta(days=400))
    backend.save([old])
    _wait(lambda: backend.count(), lambda n: n == 6)
    assert backend.delete(scope_prefix="/crew/support/user/kamal", categories=["tech"]) == 1
    assert backend.delete(scope_prefix="/crew", metadata_filter={"lang": "fr"}) == 1
    assert backend.delete(older_than=datetime.utcnow() - timedelta(days=30)) == 1
    assert _wait(lambda: backend.count(), lambda n: n == 3) == 3
    assert backend.delete(scope_prefix="/crew/support/user/kamal") == 1   # remaining kamal record
    assert backend.count("/company") == 1


def test_reset_scope_and_all(backend):
    _seed(backend)
    _wait(lambda: backend.count(), lambda n: n == 5)
    backend.reset("/crew/support/user")
    assert _wait(lambda: backend.count(), lambda n: n == 1) == 1
    backend.reset()
    assert _wait(lambda: backend.count(), lambda n: n == 0) == 0


# ── scopes / categories / listing ─────────────────────────────────────────────


def test_scope_prefix_is_path_aware(backend):
    backend.save([_rec("a", "/a"), _rec("ab", "/ab"), _rec("a-child", "/a/child")])
    _wait(lambda: backend.count(), lambda n: n == 3)
    assert backend.count("/a") == 2          # /a and /a/child, not /ab
    assert backend.count("/ab") == 1
    assert backend.count("/") == 3 and backend.count(None) == 3


def test_list_scopes_immediate_children(backend):
    _seed(backend)
    _wait(lambda: backend.count(), lambda n: n == 5)
    assert backend.list_scopes("/") == ["/company", "/crew"]
    assert backend.list_scopes("/crew/support") == ["/crew/support/user"]
    assert backend.list_scopes("/crew/support/user") == ["/crew/support/user/kamal", "/crew/support/user/priya"]
    assert backend.list_scopes("/nowhere") == []


def test_get_scope_info(backend):
    _seed(backend)
    _wait(lambda: backend.count(), lambda n: n == 5)
    info = backend.get_scope_info("/crew/support/user")
    assert info.path == "/crew/support/user" and info.record_count == 4
    assert set(info.categories) == {"food", "pref", "tech", "place"}
    assert info.child_scopes == ["/crew/support/user/kamal", "/crew/support/user/priya"]
    assert info.oldest_record is not None and info.newest_record is not None
    assert backend.get_scope_info("/empty").record_count == 0


def test_list_categories_counts(backend):
    _seed(backend)
    _wait(lambda: backend.count(), lambda n: n == 5)
    assert backend.list_categories() == {"food": 2, "pref": 1, "tech": 1, "place": 1, "policy": 1}
    assert backend.list_categories("/crew/support/user/kamal") == {"food": 1, "pref": 1, "tech": 1, "place": 1}


def test_list_records_newest_first_with_paging(backend):
    recs = [_rec(f"r{i}", "/x", created=datetime.utcnow() - timedelta(minutes=10 - i)) for i in range(5)]
    backend.save(recs)
    _wait(lambda: backend.count(), lambda n: n == 5)
    page1 = backend.list_records("/x", limit=2)
    page2 = backend.list_records("/x", limit=2, offset=2)
    assert [r.content for r in page1] == ["r4", "r3"] and [r.content for r in page2] == ["r2", "r1"]


# ── search ────────────────────────────────────────────────────────────────────


def test_search_ranks_by_similarity_with_scores(backend):
    _seed(backend)
    q = EMB(["sushi and japanese food"])[0]
    hits = _wait(lambda: backend.search(q, scope_prefix="/crew/support/user/kamal", limit=3), lambda h: len(h) == 3)
    assert hits[0][0].content.startswith("the user loves sushi")
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True) and all(-1.0 <= s <= 1.0001 for s in scores)


def test_search_scope_prefix_covers_subtree_only(backend):
    _seed(backend)
    q = EMB(["sushi"])[0]
    hits = _wait(lambda: backend.search(q, scope_prefix="/crew/support/user", limit=10), lambda h: len(h) == 4)
    assert {r.scope for r, _ in hits} == {"/crew/support/user/kamal", "/crew/support/user/priya"}
    assert all(r.scope.startswith("/crew") for r, _ in backend.search(q, scope_prefix="/crew", limit=10))
    assert backend.search(q, scope_prefix="/nowhere", limit=10) == []
    assert len(backend.search(q, limit=10)) == 5


def test_search_categories_metadata_and_min_score(backend):
    _seed(backend)
    q = EMB(["the user"])[0]
    food = _wait(lambda: backend.search(q, scope_prefix="/crew", categories=["food"], limit=10), lambda h: len(h) == 2)
    assert all("food" in r.categories for r, _ in food)
    en = backend.search(q, scope_prefix="/crew", metadata_filter={"lang": "en"}, limit=10)
    assert len(en) == 2 and all(r.metadata.get("lang") == "en" for r, _ in en)
    assert backend.search(q, scope_prefix="/crew", min_score=0.999, limit=10) == []


def test_search_limit(backend):
    _seed(backend)
    q = EMB(["user"])[0]
    assert len(_wait(lambda: backend.search(q, limit=2), lambda h: len(h) == 2)) == 2


def test_touch_records_bumps_last_accessed(backend):
    r = _rec("touch me", "/t")
    backend.save([r])
    _wait(lambda: backend.get_record(r.id), lambda g: g is not None)
    before = backend.get_record(r.id).last_accessed
    time.sleep(0.01)
    backend.touch_records([r.id])
    after = _wait(lambda: backend.get_record(r.id).last_accessed, lambda a: a > before)
    assert after > before


async def test_async_surface(backend):
    r = _rec("async fact", "/async", ["a"])
    await backend.asave([r])
    _wait(lambda: backend.count("/async"), lambda n: n == 1)
    hits = await backend.asearch(EMB(["async fact"])[0], scope_prefix="/async", limit=5)
    assert hits and hits[0][0].id == r.id
    assert await backend.adelete(record_ids=[r.id]) == 1


# ── end to end through CrewAI's Memory engine (no LLM) ────────────────────────


def test_crewai_memory_engine_roundtrip(backend):
    from crewai.memory import Memory

    mem = Memory(storage=backend, embedder=EMB, consolidation_threshold=1.0)   # 1.0 disables LLM consolidation
    try:
        r1 = mem.remember("the user loves sushi", scope="/user/kamal", categories=["food"], importance=0.9)
        r2 = mem.remember("the user writes python", scope="/user/kamal", categories=["tech"], importance=0.6)
        mem.remember("weather is sunny", scope="/misc", categories=["weather"], importance=0.1)
        assert r1 is not None and r2 is not None
        _wait(lambda: backend.count(), lambda n: n == 3)

        matches = _wait(lambda: mem.recall("what food does the user like?", scope="/user/kamal", depth="shallow", limit=2),
                        lambda m: len(m) == 2)
        assert matches[0].record.content == "the user loves sushi"
        assert 0.0 <= matches[0].score <= 1.0 and "semantic" in matches[0].match_reasons

        assert mem.list_scopes("/") == ["/misc", "/user"]
        assert mem.info("/user/kamal").record_count == 2
        assert mem.forget(scope="/misc") == 1
        assert _wait(lambda: backend.count(), lambda n: n == 2) == 2
    finally:
        mem.close()


__all__ = [n for n in list(globals()) if n.startswith("test_")]
