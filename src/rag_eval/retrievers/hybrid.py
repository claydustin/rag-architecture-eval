"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Combines a lexical (BM25) and a dense retriever. RRF fuses by *rank position*,
not raw score — crucial because BM25 scores and cosine similarities live on
different scales and can't be added directly. Each candidate's fused score is

    score(d) = sum over retrievers of  1 / (k + rank_r(d))

where rank_r(d) is d's 1-based rank in retriever r (lower = better) and k is a
small constant (canonical default 60) that damps the influence of the very top
ranks so one retriever can't dominate.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet..."
"""
from __future__ import annotations

from .base import Hit, Retriever
from .bm25 import BM25Retriever
from .dense import DenseRetriever


class HybridRetriever(Retriever):
    name = "hybrid_rrf"

    def __init__(
        self,
        rrf_k: int = 60,
        bm25_params: dict | None = None,
        dense_params: dict | None = None,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
    ):
        self.rrf_k = rrf_k
        # Allow injecting prebuilt retrievers (handy in notebooks / tests);
        # otherwise construct from params.
        self.bm25 = bm25 or BM25Retriever(**(bm25_params or {}))
        self.dense = dense or DenseRetriever(**(dense_params or {}))

    def index(self, chunks) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def _rrf(self, hits_a, hits_b, k: int) -> list[Hit]:
        """Fuse two ranked Hit lists by RRF. Returns Hits ordered by fused score,
        carrying the fused score and the chunk (first-seen chunk object)."""
        scores: dict[str, float] = {}
        chunk_of: dict[str, object] = {}
        for hits in (hits_a, hits_b):
            for rank, h in enumerate(hits, start=1):
                cid = h.chunk.chunk_id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
                chunk_of.setdefault(cid, h.chunk)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [Hit(chunk_of[cid], score) for cid, score in ordered]

    def retrieve(self, query: str, k: int) -> list[Hit]:
        # Fetch deeper from each arm than k so the fusion has candidates to work
        # with (a doc ranked low in one arm can still win on combined score).
        fetch = max(k * 3, k)
        a = self.bm25.retrieve(query, fetch)
        b = self.dense.retrieve(query, fetch)
        return self._rrf(a, b, self.rrf_k)[:k]
