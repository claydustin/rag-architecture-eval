"""Load MultiHop-RAG (yixuantt/MultiHopRAG) and normalize it.

Gold labels link queries to articles by *title* (titles are unique in the
corpus, verified), so the article title is used as the doc_id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

HF_DATASET = "yixuantt/MultiHopRAG"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Doc:
    doc_id: str          # == title
    title: str
    body: str
    source: str
    published_at: str
    category: str
    url: str


@dataclass(frozen=True)
class Query:
    qid: int
    query: str
    answer: str
    question_type: str
    relevant_doc_ids: tuple[str, ...] = field(default_factory=tuple)  # deduped, order kept


def _cache(name: str, subset: str) -> pd.DataFrame:
    """Download once, then read the local parquet copy."""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        from datasets import load_dataset

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        load_dataset(HF_DATASET, subset, split="train").to_pandas().to_parquet(path)
    return pd.read_parquet(path)


def load_corpus() -> list[Doc]:
    df = _cache("corpus", "corpus")
    return [
        Doc(r.title, r.title, r.body, r.source, str(r.published_at), r.category, r.url)
        for r in df.itertuples()
    ]


def load_queries() -> list[Query]:
    df = _cache("queries", "MultiHopRAG")
    out = []
    for i, r in enumerate(df.itertuples()):
        rel = tuple(dict.fromkeys(e["title"] for e in r.evidence_list))
        out.append(Query(i, r.query, r.answer, r.question_type, rel))
    return out
