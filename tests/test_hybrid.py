"""Hybrid RRF fusion tests — pure logic, no cloud."""
from __future__ import annotations

from rag_eval.chunking import Chunk
from rag_eval.retrievers import HybridRetriever
from rag_eval.retrievers.base import Hit


def _hit(cid, doc, score=0.0):
    return Hit(Chunk(cid, doc, f"text-{cid}"), score)


def test_rrf_rewards_agreement():
    """A doc ranked decently by BOTH arms should beat a doc ranked #1 by only
    one arm, once ranks are deep enough — RRF rewards cross-retriever agreement."""
    h = HybridRetriever(rrf_k=60, bm25=object(), dense=object())  # arms unused here
    # arm A order: a, b, c ; arm B order: b, a, d
    a = [_hit("a", "da"), _hit("b", "db"), _hit("c", "dc")]
    b = [_hit("b", "db"), _hit("a", "da"), _hit("d", "dd")]
    fused = h._rrf(a, b, 60)
    ids = [hh.chunk.chunk_id for hh in fused]
    # a and b both appear in both lists near the top -> they lead; c and d only once.
    assert set(ids[:2]) == {"a", "b"}
    assert ids[-2:] == ["c", "d"] or ids[-2:] == ["d", "c"]


def test_rrf_scores_sum_reciprocals():
    h = HybridRetriever(rrf_k=10, bm25=object(), dense=object())
    a = [_hit("x", "dx")]            # rank 1 in A
    b = [_hit("y", "dy"), _hit("x", "dx")]  # x is rank 2 in B
    fused = {hh.chunk.chunk_id: hh.score for hh in h._rrf(a, b, 10)}
    # x: 1/(10+1) + 1/(10+2) ; y: 1/(10+1)
    assert abs(fused["x"] - (1/11 + 1/12)) < 1e-9
    assert abs(fused["y"] - (1/11)) < 1e-9
    # x scored by both -> ranks above y
    assert fused["x"] > fused["y"]
