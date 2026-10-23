"""BM25 lexical baseline (rank_bm25.BM25Okapi). No model or cloud calls."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from ..chunking import Chunk
from .base import Hit, Retriever

_TOKEN = re.compile(r"[a-z0-9]+")
# Small stopword list: keeps the baseline dependency-free.
_STOP = frozenset(
    "a an and are as at be but by for from has have he her his in is it its of on or "
    "she that the their they this to was were which who will with what when where how "
    "did does do been than also about after before both according reported".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks], k1=self.k1, b=self.b)

    def retrieve(self, query: str, k: int) -> list[Hit]:
        assert self._bm25 is not None, "call index() first"
        scores = self._bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [Hit(self._chunks[i], float(scores[i])) for i in top]
