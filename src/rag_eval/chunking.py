"""Chunking strategies. Every chunk keeps its parent doc_id so retrieval can be
scored at the document level regardless of how documents were split."""
from __future__ import annotations

from dataclasses import dataclass

from .data import Doc


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def chunk_words(docs: list[Doc], size: int = 256, overlap: int = 32,
                prepend_title: bool = True) -> list[Chunk]:
    """Fixed-size word windows. Word count is a cheap, model-agnostic proxy for
    tokens (~1.3 tokens/word in English). size<=0 means one chunk per doc."""
    if size > 0 and not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    chunks: list[Chunk] = []
    for d in docs:
        words = d.body.split()
        if size <= 0:
            windows = [words]
        else:
            step = size - overlap
            windows = [words[i:i + size] for i in range(0, max(len(words) - overlap, 1), step)]
        for j, w in enumerate(windows):
            text = " ".join(w)
            if prepend_title:
                text = f"{d.title} ({d.source}, {d.published_at[:10]})\n{text}"
            chunks.append(Chunk(f"{d.doc_id}::{j}", d.doc_id, text))
    return chunks
