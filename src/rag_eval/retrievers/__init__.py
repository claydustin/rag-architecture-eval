from .base import Hit, Retriever, ranked_doc_ids
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever

REGISTRY: dict[str, type[Retriever]] = {
    "bm25": BM25Retriever,
    "dense_bedrock": DenseRetriever,
    "hybrid_rrf": HybridRetriever,
}

__all__ = [
    "Hit",
    "Retriever",
    "ranked_doc_ids",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "REGISTRY",
]
