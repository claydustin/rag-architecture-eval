"""Dense retrieval: Titan embeddings + FAISS exact inner-product search.

Vectors are L2-normalized so inner product == cosine similarity. The index is
exact (IndexFlatIP) — the corpus is small (~609 docs / a few thousand chunks),
so there's no need for an approximate index and we avoid its recall loss.

Embeddings are produced by the cached TitanEmbedder, so re-running an
experiment with the same chunks hits the disk cache instead of Bedrock.
"""
from __future__ import annotations

import faiss
import numpy as np

from ..bedrock import TitanEmbedder
from ..chunking import Chunk
from .base import Hit, Retriever


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class DenseRetriever(Retriever):
    name = "dense_bedrock"

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        profile: str | None = None,
        embedder: TitanEmbedder | None = None,
    ):
        self.embedder = embedder or TitanEmbedder(
            model_id=model_id, region=region, profile=profile
        )
        self._chunks: list[Chunk] = []
        self._index: faiss.Index | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        vecs = _normalize(self.embedder.embed([c.text for c in chunks]))
        dim = vecs.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs)

    def retrieve(self, query: str, k: int) -> list[Hit]:
        assert self._index is not None, "call index() first"
        q = _normalize(self.embedder.embed_query(query).reshape(1, -1))
        scores, idx = self._index.search(q, min(k, len(self._chunks)))
        return [
            Hit(self._chunks[i], float(s))
            for i, s in zip(idx[0], scores[0])
            if i != -1
        ]
