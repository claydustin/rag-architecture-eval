"""Common retriever interface. Every architecture (BM25, dense, hybrid,
reranked, multi-query, agentic) implements `retrieve` and is evaluated by the
same harness."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(ABC):
    name: str = "base"

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[Hit]:
        """Return up to k chunks, best first."""


def ranked_doc_ids(hits: list[Hit]) -> list[str]:
    """Collapse chunk hits to unique doc_ids in rank order (first occurrence wins)."""
    return list(dict.fromkeys(h.chunk.doc_id for h in hits))
