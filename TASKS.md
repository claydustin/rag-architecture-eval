# RAG Architecture Eval — Task List

Infra-first execution order. Only Step 1 runs locally; everything else runs in
the cloud (SageMaker Studio for interactive dev, ECS Fargate for batch eval,
Lambda for the query API).

**Guardrails:** stop SageMaker Studio when idle · `terraform destroy` idle
resources except the state bucket · keep and containerize the existing
`rag_eval` harness, don't rewrite it.

---

## Phase 1 — Harness validation
- [x] Run the test suite and confirm it passes (`.venv/bin/python -m pytest -q` → 5 passed)

## Step 1 — Data (local, one time)
- [x] Download HF MultiHopRAG data locally, normalize, cache to `data/*.parquet`
  <!-- Verified: corpus 609 docs (unique titles), queries 2556 (comparison 856 /
       inference 816 / temporal 583 / null 301); 0 missing evidence titles;
       evidence/query 2–4; null queries have no evidence. -->


## Step 2 — Terraform / AWS
- [x] 2a. Bootstrap — S3 state bucket (native locking, `use_lockfile`, TF ≥1.10) + AWS Budget alarm
  <!-- Applied to account 875699230934 (sandbox), profile `dev`. Bucket
       rag-arch-eval-tfstate-875699230934: versioning on, AES256, public access
       blocked, prevent_destroy. Budget rag-arch-eval-monthly $20/mo → claymd.aws@gmail.com. -->

- [x] 2b. Data plane — S3 data bucket (IAM roles deferred to each consumer: SageMaker→3b, Fargate→5, Lambda→10)
  <!-- terraform/envs/dev/ with remote S3 backend (key envs/dev/terraform.tfstate,
       use_lockfile). Bucket rag-arch-eval-data-875699230934: versioning on,
       AES256, public access blocked. Prefixes (implicit): data/ indexes/ results/. -->

- [x] 2c. Bedrock model access (us-east-1): embeddings + Claude verified working
  <!-- Account verification hold CLEARED (no email — it lifts silently; re-invoke
       is the real check). Verified by live invoke:
         - Titan Embed V2 (amazon.titan-embed-text-v2:0) → 1024-dim vector OK
         - Claude Sonnet 4.5 → OK, but ONLY via cross-region inference profile ID
           `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (bare model ID throws
           ValidationException: on-demand not supported — use inference profile).
       No Anthropic use-case form was required. Cohere Rerank untested (optional,
       Step 7b can fall back to a cross-encoder). -->


## Step 3 — Cloud load + playground
- [x] 3a. Upload parquet data to the S3 data bucket
  <!-- s3://rag-arch-eval-data-875699230934/data/{corpus,queries}.parquet.
       Byte-for-byte match to local (3829613 / 630906), AES256-encrypted. -->
- [x] 3b. SageMaker Studio playground (stop-when-idle) — **applied**
  <!-- Domain d-ygahoxw7ihmu (InService, IAM auth), user profile `dev` (InService),
       exec role rag-arch-eval-sagemaker-exec. Native idle shutdown ENABLED, 60-min
       (AWS floor for min_idle_timeout is 60, not 15). URL:
       https://d-ygahoxw7ihmu.studio.us-east-1.sagemaker.aws
       REMEMBER: stop the JupyterLab app when done (idle-shutdown is a backstop). -->


## Step 4 — Build the Python RAG tools
- [ ] Extend the existing `rag_eval` package (don't rewrite); design for Fargate + Lambda

## Step 5 — SageMaker Processing eval  (architecture change: SageMaker Processing, NOT Docker/Fargate)
- [x] 5a. SageMaker Processing scaffolding (no Dockerfile/ECR)
  <!-- src/processing_entry.py (runs in-job, uses SM input/output channels,
       stages parquet into DATA_DIR, calls evaluate()); scripts/run_processing.py
       (local launcher: FrameworkProcessor on sklearn image, source_dir=src/,
       data+config input channels, results->s3://.../results/processing/<job>/);
       src/requirements.txt (job deps). Pinned sagemaker==2.257.7 (v2 classic
       Processing API; v3 removed it). 7/7 tests pass. First real run = 5b. -->
- [x] 5b. BM25 baseline run IN SAGEMAKER via interactive Studio notebooks (not a Processing job)
  <!-- PIVOT: after 3 failed Processing-job submits (Py3.9 sklearn image; then
       non-root SageMaker Distribution image broke FrameworkProcessor code-staging;
       then PyTorch DLC exit 1), switched to interactive Studio notebooks — same
       "in SageMaker" goal, seconds-per-iteration instead of slow batch loops.
       notebooks/data_processing.ipynb (S3 corpus -> chunk -> write chunks/chunks.parquet)
       + notebooks/bm25_eval.ipynb (load chunks -> BM25Retriever -> metrics by
       question type -> results/notebook/). Both import the uploaded rag_eval
       package. BM25 eval RAN SUCCESSFULLY in Studio. run_processing.py kept for a
       later "promote to batch job" once logic is proven. -->

### WORKFLOW PIVOT (session 2): interactive Studio notebooks
- Develop in notebooks (cell-by-cell, fast), not blind Processing-job submits.
- `rag_eval` package uploaded next to notebooks in Studio (file upload, lands on EFS).
- Install deps per-kernel with `%pip install faiss-cpu==1.15.1` etc.
- Processing jobs = the later "promote to headless batch" step, not the dev loop.

### TEARDOWN (end of session 2026-10-20)
- SageMaker domain d-ygahoxw7ihmu + `dev` user profile DESTROYED (targeted destroy)
  to avoid idle charges. Verified gone. No running apps.
- KEPT: state bucket, data bucket (parquet intact), SageMaker exec IAM role (all ~$0 idle).
- To resume: `terraform apply` in terraform/envs/dev re-creates the domain (~5 min).

## Step 6 — Dense retrieval on Bedrock  (IN PROGRESS — session task list drifted; this is NOT done yet)
- [ ] Titan embeddings + FAISS, index to S3; compare vs BM25; sweep chunk size/overlap
  <!-- Approach: extend data_processing.ipynb with a dense section (embed chunks
       via TitanEmbedder -> build FAISS -> write s3://.../indexes/), then a new
       dense_eval.ipynb mirroring bm25_eval.ipynb. Retrieval-only eval (gold docs
       from evidence_list); answer-quality/LLM-judge is Step 9 (deliberately later,
       once multiple retrievers exist to compare end-to-end). -->

<!-- EVAL PHILOSOPHY NOTE (from user Q): retrieval metrics (recall@k/all_hit@k)
     need gold docs, which real projects rarely have. The plan handles the
     realistic no-gold-docs case in Step 9 (answer-match + LLM-judge faithfulness,
     RAGAS-style). Retrieval eval first = clean/diagnostic; answer eval later =
     realistic/end-to-end. Keeping them separate is a feature: lets us see
     retrieval recall vs end-to-end accuracy independently. -->


## Step 7 — Better retrieval ladder (one variable per run)
- [x] 7a. Hybrid BM25 + dense (reciprocal rank fusion)
  <!-- rag_eval/retrievers/hybrid.py: HybridRetriever, RRF (1/(k+rank) summed,
       k=60) over BM25 + dense arms; registered hybrid_rrf. 2 unit tests (9 total
       pass). notebooks/hybrid_eval.ipynb reuses the S3 FAISS index (no re-embed),
       scores same metrics. No Claude needed — runnable now. Can also point at
       degraded_queries for the full {bm25,dense,hybrid}x{orig,degraded} matrix. -->
- [ ] Reranker (Cohere Rerank on Bedrock or cross-encoder)  → next no-Claude option (cross-encoder path)
- [ ] Reranker (Cohere Rerank on Bedrock or cross-encoder)
- [ ] Query transforms (multi-query, HyDE)
- [ ] Query decomposition into sub-questions

## Step 8 — Agentic RAG (LangGraph)
- [ ] Agent search tool: retrieve → grade sufficiency → re-query (iteration cap); log steps/tokens/latency

## Step 9 — Answer-level eval (~200-query sample)
- [ ] Claude answers per architecture; exact/fuzzy match vs gold; LLM-judge faithfulness; null-query abstention
- [ ] Reuse the ECR image as a cloud eval job writing results to S3

## Step 10 — POC deliverable
- [ ] Lambda + API Gateway query API **with auth** (API key min; IAM/Cognito better)
- [ ] Plotly report: accuracy vs cost vs latency per architecture, by question type
- [ ] `METHODOLOGY.md` + one-page summary with recommendation
- [ ] `envs/company.tfvars` handoff path (check platform team conventions)
- [ ] `terraform destroy` idle resources except the state bucket
