"""Test helpers shared by the provider suites (and handy for users' tests).

``FakeEmbedder`` is a deterministic, dependency-free embedder with CrewAI's
embedder call signature (``list[str] -> list[list[float]]``): a hashed
bag-of-words projected into ``dims`` dimensions and L2-normalised. Texts that
share words are close; no network call is made. Pass it as
``Memory(embedder=FakeEmbedder(64))``.

``InMemoryBackend`` is the smallest ``MemoryBackend`` — a dict — used to test
the core machinery without any datastore.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List, Optional

from . import MemoryBackend, in_scope

__all__ = ["FakeEmbedder", "InMemoryBackend"]

_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    def __init__(self, dims: int = 64) -> None:
        self.dims = dims

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dims
        for word in _WORD.findall(text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else [1.0 / math.sqrt(self.dims)] * self.dims

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]


class InMemoryBackend(MemoryBackend):
    def __init__(self) -> None:
        self._rows: Dict[str, dict] = {}

    def _put(self, rows: List[dict]) -> None:
        for r in rows:
            self._rows[r["id"]] = dict(r)

    def _get(self, record_id: str) -> Optional[dict]:
        r = self._rows.get(record_id)
        return dict(r) if r else None

    def _delete_ids(self, ids: List[str]) -> int:
        n = 0
        for i in ids:
            if self._rows.pop(i, None) is not None:
                n += 1
        return n

    def _scan(self, scope_prefix: Optional[str]) -> List[dict]:
        return [dict(r) for r in self._rows.values() if in_scope(r["scope"], scope_prefix)]
