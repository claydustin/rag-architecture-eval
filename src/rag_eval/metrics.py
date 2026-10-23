"""Document-level retrieval metrics with binary relevance.

`ranked` is the unique doc_ids a retriever returned, best first.
`relevant` is the set of gold evidence docs for a query (2-4 for MultiHop-RAG).
"""
from __future__ import annotations

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of gold docs found in the top k."""
    return len(set(ranked[:k]) & relevant) / len(relevant)


def all_hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1 if EVERY gold doc is in the top k. The strict multi-hop metric: an
    answer needing all hops can't be fully grounded otherwise."""
    return float(relevant <= set(ranked[:k]))


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first gold doc in the top k."""
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


METRICS = {"recall": recall_at_k, "all_hit": all_hit_at_k, "mrr": mrr_at_k, "ndcg": ndcg_at_k}
