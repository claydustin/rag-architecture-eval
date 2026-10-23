"""Run a retrieval experiment from a YAML config and log results.

    .venv/bin/python -m rag_eval.run_eval configs/bm25.yaml

Outputs (results/):
  runs.csv                       one row per run: config + overall + per-type metrics
  <run_name>_per_query.csv       per-query metrics, for error analysis
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from .chunking import chunk_words
from .data import load_corpus, load_queries
from .metrics import METRICS
from .retrievers import REGISTRY, ranked_doc_ids

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def evaluate(cfg: dict) -> tuple[dict, pd.DataFrame]:
    docs = load_corpus()
    # null_query has no gold evidence -> excluded from retrieval metrics
    # (it is scored later in answer-level eval: should the agent abstain?).
    queries = [q for q in load_queries() if q.relevant_doc_ids]
    if cfg.get("limit"):
        queries = queries[: cfg["limit"]]

    chunks = chunk_words(docs, **cfg.get("chunking", {}))
    r = REGISTRY[cfg["retriever"]](**cfg.get("retriever_params", {}))
    t0 = time.perf_counter()
    r.index(chunks)
    index_s = time.perf_counter() - t0

    ks = cfg.get("ks", [1, 3, 5, 10, 20])
    # Retrieve enough chunks that the top max(k) *documents* are filled,
    # since several chunks of one article can crowd the list.
    fetch = cfg.get("fetch_chunks", max(ks) * 5)

    rows, latencies = [], []
    for q in queries:
        t = time.perf_counter()
        ranked = ranked_doc_ids(r.retrieve(q.query, fetch))
        latencies.append(time.perf_counter() - t)
        rel = set(q.relevant_doc_ids)
        row = {"qid": q.qid, "question_type": q.question_type, "n_relevant": len(rel)}
        for k in ks:
            for m, fn in METRICS.items():
                row[f"{m}@{k}"] = fn(ranked, rel, k)
        rows.append(row)

    per_q = pd.DataFrame(rows)
    metric_cols = [c for c in per_q.columns if "@" in c]
    summary = {
        "run": cfg["name"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": json.dumps(cfg, sort_keys=True),
        "n_queries": len(per_q),
        "n_chunks": len(chunks),
        "index_s": round(index_s, 2),
        "p50_query_ms": round(pd.Series(latencies).median() * 1000, 1),
        **per_q[metric_cols].mean().round(4).to_dict(),
    }
    by_type = per_q.groupby("question_type")[metric_cols].mean().round(4)
    for qt, s in by_type.iterrows():
        for c in metric_cols:
            summary[f"{qt}/{c}"] = s[c]
    return summary, per_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    cfg = yaml.safe_load(ap.parse_args().config.read_text())

    summary, per_q = evaluate(cfg)
    RESULTS.mkdir(exist_ok=True)
    per_q.to_csv(RESULTS / f"{cfg['name']}_per_query.csv", index=False)
    runs = RESULTS / "runs.csv"
    df = pd.DataFrame([summary])
    if runs.exists():
        df = pd.concat([pd.read_csv(runs), df], ignore_index=True)
    df.to_csv(runs, index=False)

    keys = [k for k in summary if "/" not in k and "@" in k and k.split("@")[1] in ("5", "10")]
    print(f"{summary['run']}: {summary['n_queries']} queries, {summary['n_chunks']} chunks, "
          f"p50 {summary['p50_query_ms']} ms")
    for k in keys:
        print(f"  {k:<12} {summary[k]:.4f}")
    print("  by type (recall@10 / all_hit@10):")
    for qt in sorted({k.split('/')[0] for k in summary if '/' in k}):
        print(f"    {qt:<18} {summary[f'{qt}/recall@10']:.4f} / {summary[f'{qt}/all_hit@10']:.4f}")


if __name__ == "__main__":
    main()
