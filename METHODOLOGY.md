# RAG Architecture Eval — Methodology & Findings

Mobile Insights – Advanced Analytics · evaluation of retrieval architectures on
the MultiHopRAG benchmark, built on AWS (SageMaker + Bedrock + Terraform).

## TL;DR (the recommendation)

On the benchmark's *native* queries, cheap lexical **BM25 looks best** — which
would wrongly suggest "don't bother with embeddings." But those queries are
unrealistically keyword-rich (they were LLM-generated *from* the source
articles). When rewritten to resemble **real user questions**, BM25's advantage
**collapses** while dense (embedding) retrieval barely moves. **Hybrid retrieval
(BM25 + dense fused via Reciprocal Rank Fusion) is the robust choice** — best on
native queries *and* most resilient on realistic ones, winning every question
type.

## What was built

- **Dataset:** `yixuantt/MultiHopRAG` — 609 news articles, 2,556 multi-hop
  questions with gold evidence documents (title-level). Null queries (301)
  excluded from retrieval metrics.
- **Retrievers** (`rag_eval` package, shared interface):
  - **BM25** — lexical (rank_bm25), no model.
  - **Dense** — Amazon Titan Text Embeddings V2 (1024-dim) + FAISS exact
    inner-product search (cosine via normalization).
  - **Hybrid** — Reciprocal Rank Fusion (RRF, k=60) over BM25 + dense.
- **Metrics:** doc-level recall@k, all_hit@k (strict: every gold doc retrieved),
  MRR@k, nDCG@k. k ∈ {1,3,5,10,20}.
- **Infrastructure (Terraform):** S3 state backend + budget guardrail
  (bootstrap), S3 data bucket, SageMaker Studio domain + execution role.
  Embeddings/FAISS/eval run interactively in SageMaker Studio notebooks against
  data in S3; Bedrock for embeddings (Titan) and query rewriting (Amazon Nova).
- **Chunking:** 256-word windows, 32-word overlap, title prepended.

## The degraded-query experiment

**Motivation:** MultiHopRAG's questions inherit exact proper nouns, dates, and
even source-outlet names from the articles they were generated from — handing
lexical retrieval an unrealistic advantage (the "question" is nearly a
paraphrase of the answer passage). Real users don't phrase questions that way.

**Method:** every non-null query (2,255) was rewritten with Amazon Nova Lite to
be shorter, more casual, with softened/dropped proper nouns and **removed
source-outlet references** (a real user doesn't cite which outlet reported a
story). The **gold evidence documents were kept identical** — only the question
phrasing changed — so scoring is apples-to-apples with the original run.

**Honesty note on the manipulation check:** we first measured "fraction of query
tokens found in the gold docs" as a proxy for lexical overlap; it went *up* on
degraded queries. That proxy is **length-confounded** — shortening a query to
common words raises the fraction even as it gets vaguer — so we do **not** rely
on it. The direct retrieval results below are the ground truth, and they show
BM25's signal did degrade as intended.

## Results

### recall@10 — original vs degraded queries

| recall@10  | BM25  | Dense | Hybrid |
|------------|-------|-------|--------|
| original   | 0.859 | 0.801 | 0.862  |
| degraded   | 0.758 | 0.757 | **0.806** |
| Δ (drop)   | −0.101 | −0.044 | −0.056 |

- On native queries, BM25 ≈ hybrid > dense.
- On degraded queries, **BM25 drops 2.3× as much as dense** (−0.101 vs −0.044);
  the BM25→dense gap collapses from +0.058 to +0.001.
- **Hybrid wins the degraded set outright (0.806)** — +0.05 over either arm.

### recall@10 by question type — DEGRADED queries

| recall@10 (degraded) | BM25  | Dense | Hybrid |
|----------------------|-------|-------|--------|
| comparison           | 0.832 | 0.778 | **0.847** |
| inference            | 0.658 | 0.716 | **0.751** |
| temporal             | 0.789 | 0.782 | **0.821** |

- **Inference (semantic-reasoning) queries** are decisive: BM25 0.658 < dense
  0.716 < **hybrid 0.751**. Fusion beats both arms on the hardest type.
- Hybrid is best on every degraded question type.

### recall@10 by question type — ORIGINAL queries (reference)

| recall@10 (original) | BM25  | Dense | Hybrid |
|----------------------|-------|-------|--------|
| comparison           | 0.913 | 0.828 | 0.898  |
| inference            | 0.783 | 0.744 | **0.808** |
| temporal             | 0.888 | 0.839 | 0.882  |

Hybrid also wins inference on native queries, and leads at recall@20
(0.972 vs BM25 0.951 / dense 0.921) with all_hit@20 0.922.

## Interpretation

1. A naive leaderboard on this benchmark says "use BM25." That conclusion is an
   **artifact of unrealistically keyword-rich queries.**
2. Real user queries are closer to the *degraded* set. There, **semantic
   retrieval earns its keep**, and **hybrid dominates**.
3. **Recommendation: hybrid (BM25 + dense via RRF).** Best on native queries,
   most robust on realistic ones, wins every question type on the realistic set.

## Caveats / honest limitations

- Degraded queries are an **LLM-produced proxy** for real user traffic, not
  actual logs — directional evidence, not a production measurement.
- Single embedding model (Titan V2) and one benchmark (MultiHopRAG, news
  articles). A different corpus (e.g. support tickets / internal docs) could
  shift the result — the harness is corpus-agnostic and ready to re-run.
- Retrieval eval only; answer-generation quality (end-to-end) is a separate,
  later evaluation (generate answers with an LLM, judge faithfulness).

## Reproduce

Notebooks (run in SageMaker Studio, in order):
1. `notebooks/data_processing.ipynb` — chunk corpus → S3; embed → FAISS → S3.
2. `notebooks/query_degradation.ipynb` — generate the degraded query set (Nova).
3. `notebooks/bm25_eval.ipynb`, `dense_eval.ipynb`, `hybrid_eval.ipynb`
   (toggle `USE_DEGRADED`) — the six result cells above.

Infra: `terraform/bootstrap` (once), then `terraform/envs/dev`.
