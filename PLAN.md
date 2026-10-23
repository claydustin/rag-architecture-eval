# RAG Architecture Eval — Project Plan

## Goal
Compare RAG architectures (BM25 → dense → hybrid → rerank → query transforms →
agentic) on a multi-hop QA benchmark, deployed on AWS via Terraform. Build in a
personal AWS account as a proof of concept, then hand off to the company account.

Working style: build incrementally to learn. Explain each step; one change at a
time; run the eval after every step; commit each phase to git.

## Dataset: `yixuantt/MultiHopRAG` (Hugging Face)
- Corpus: 609 news articles (`title, body, source, published_at, category, url, author`).
  Titles are unique → title is the doc_id. Median body ≈ 1,300 words.
- Queries: 2,556 (`query, answer, question_type, evidence_list`).
  Types: comparison 856, inference 816, temporal 583, null 301.
- Gold labels: `evidence_list[].title` (2–4 per query). Dedupe titles per query.
  All evidence titles exist in the corpus.
- `null_query` has no evidence → exclude from retrieval metrics; score at the
  answer level (agent should abstain).

## Current state (already scaffolded, NOT yet run)
- `.venv` (Python 3.13 via uv), `requirements.txt` (pinned), `pyproject.toml`
- `src/rag_eval/data.py` — load + cache dataset to `data/*.parquet`, normalize
- `src/rag_eval/chunking.py` — fixed word windows w/ overlap; chunk keeps doc_id
- `src/rag_eval/retrievers/base.py` — `Retriever` interface (`index`, `retrieve`), `ranked_doc_ids`
- `src/rag_eval/retrievers/bm25.py` — BM25 baseline (rank_bm25)
- `src/rag_eval/metrics.py` — recall@k, all_hit@k, MRR@k, nDCG@k (doc-level, binary)
- `src/rag_eval/run_eval.py` — YAML-config runner → `results/runs.csv` + per-query CSV
- `configs/bm25.yaml`, `configs/bm25_fulldoc.yaml`
- `tests/test_metrics.py`

Next action: `.venv/bin/python -m pytest -q`, then
`PYTHONPATH=src .venv/bin/python -m rag_eval.run_eval configs/bm25.yaml`.

## Phases

### Phase 0 — Personal AWS + Terraform bootstrap
- [ ] IAM Identity Center/IAM user (not root); CLI profile `personal`
- [ ] Install Terraform (`brew install terraform` or `tfenv`)
- [ ] `infra/bootstrap/`: S3 state bucket (native locking, `use_lockfile`, TF ≥1.10) + AWS Budget alarm
- [ ] Enable Bedrock model access in us-east-1 (embedding model, Claude, optional Cohere Rerank).
      Anthropic may require a one-time console use-case form (not Terraform-managed).
- [ ] Smoke test: `aws bedrock-runtime invoke-model` embedding call succeeds
- Done when: embedding call works and state backend exists.

### Phase 1 — Data + eval harness (local, no LLM)  ✅ scaffolded
- [ ] Run tests; read and understand each module
- Done when: tests pass and runner works end to end.

### Phase 2 — BM25 baseline (local)
- [ ] Run chunked and full-doc BM25 configs
- [ ] Error analysis on per-query CSV by question type
- Done when: baseline numbers logged in `results/runs.csv`.

### Phase 3 — Dense retrieval on Bedrock
- [ ] Dense retriever via `langchain-aws` embeddings + FAISS; cache embeddings to disk
- [ ] Compare vs BM25; sweep chunk size/overlap
- [ ] Terraform: data bucket + least-privilege IAM; ingest job (ECR + ECS Fargate or
      SageMaker Processing) that chunks, embeds, writes index to S3
- Done when: dense vs BM25 comparison by question type, ingest runs in cloud.

### Phase 4 — Better retrieval (one variable per run)
- [ ] Hybrid: BM25 + dense with reciprocal rank fusion
- [ ] Reranker (Cohere Rerank on Bedrock or cross-encoder)
- [ ] Query transforms: multi-query, HyDE
- [ ] Query decomposition into sub-questions

### Phase 5 — Agentic RAG (LangGraph)
- [ ] Agent with search tool: retrieve → grade sufficiency → re-query; cap iterations
- [ ] Log retrieval metrics on union of evidence + steps, tokens, latency

### Phase 6 — Answer-level eval (sample ~200 queries)
- [ ] Generate answers with Claude per architecture
- [ ] Exact/fuzzy match vs gold answer; LLM-judge faithfulness; null-query abstention rate
- [ ] Terraform: reuse container image as cloud eval job → results to S3

### Phase 7 — POC deliverable
- [ ] Terraform: Lambda + API Gateway query API **with auth** (API key minimum; IAM/Cognito better)
- [ ] Plotly report: accuracy vs cost vs latency per architecture, by question type
- [ ] `METHODOLOGY.md` + one-page summary with recommendation
- [ ] `envs/company.tfvars` path for company account (check platform team conventions)

## Cloud architecture
```
Browser/demo ──► API Gateway (auth) ──► Lambda: RAG agent (LangGraph)
                                          ├──► Bedrock: Claude
                                          ├──► Bedrock: embeddings + rerank
                                          └──► Vector store (S3 Vectors or FAISS index in S3)

Batch eval job (ECS Fargate / SageMaker Processing) → results to S3 → Plotly report
```
Avoid OpenSearch Serverless in the personal account (idle minimum cost); cite it
as the at-scale option. Verify current pricing on AWS pricing pages.

## Terraform layout
```
infra/
  bootstrap/        # state bucket + budget (local state, applied once)
  modules/
    rag_api/        # Lambda + API Gateway + IAM
    batch_job/      # ECS/SageMaker job + IAM
  envs/
    personal/       # main.tf, backend.tf, personal.tfvars
```

## Conventions / guardrails
- Change one variable per experiment; every run logged with its config.
- Cache embeddings; never re-embed unchanged chunks.
- `.gitignore`: `.venv/`, `data/`, `results/` (optional), `*.tfstate*`, `.terraform/`, `*.tfvars` with secrets.
- No company data or company AWS accounts until POC is approved.
- `terraform destroy` when idle (except state bucket).
