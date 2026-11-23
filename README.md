# Tender OS

Hybrid RAG search and grounded Q&A over **6,899** cleaned public Malaysian government tender notices across **15 portals**, built on PostgreSQL `pgvector` and YTL ILMU (`bge-m3`, `bge-reranker`, `ilmu-mini-v3.3`, `ilmu-v3.1`).

Tender OS helps procurement teams search English and Bahasa Melayu tender notices, evaluate deterministic MOF Kod Bidang and CIDB grade eligibility badges, and ask grounded questions with verifiable `[source:reference]` citations, closing dates, and inline cost/latency traces.

## Architecture

The Q&A pipeline follows the pattern described in *Ryt AI* (EMNLP 2025 Industry Track): deterministic intent routing, hybrid RAG with cross-encoder reranking, a pre-generation abstention guardrail before synthesis, cheap-first model escalation, and deterministic post-generation verification.

```text
Question / Query
  |
  +--> route_query() regex ----------> [exact_code] (6-digit MOF / portal ref)
  |                                          |
  +--> [semantic] ---------------------------+
                                             v
  Postgres 'simple' GIN FTS + pgvector cosine (1024d bge-m3, RRF k=60)
                                             |
                                             v
  YTL ILMU bge-reranker (top-20 pool, lowercased) + deterministic EligibilityVerdict
                                             |
          +--> missing ILMU_API_KEY / spend cap / retrieval error --> Degraded pg_fts summary (0 chat calls)
          +--> 0 hits ----------------------------------------------> Abstain: zero_hits (0 chat calls)
          +--> rerank < 0.04 (or missing) and no exact match -------> Abstain: rerank_below_floor (0 chat calls)
                                             |
                                             v
  Single grounded synthesis call: ilmu-mini-v3.3 -> ilmu-v3.1 fallback (15s deadline)
  over angle-bracket-sanitized <untrusted_tender_records>
                                             |
                                             v
  verify_grounding() -> checks [source:reference] IDs + YYYY-MM-DD closing dates + inline QueryTrace
```

Key engineering boundaries:

- **Single router boundary:** `src/tender_os/ilmu/router.py` is the only caller of `https://api.ilmu.ai/v1`, controlled by a single `ILMU_API_KEY` environment variable (verified by CI).
- **Atomic MYT daily spend cap:** `src/tender_os/ilmu/spend.py` reserves estimated micro-RM in Postgres `spend_ledger` with one atomic conditional `UPDATE` before each remote ILMU call, settling actual token spend or conservatively charging in-flight ambiguous failures.
- **Explicit degraded mode:** Without `ILMU_API_KEY`, or when `TENDER_OS_DAILY_CAP_MYR` is exhausted, `/v1/search` and `/v1/ask` fall back cleanly to PostgreSQL keyword search (`pg_fts`) and return HTTP 200 with `degraded_mode: true`.
- **Deterministic eligibility, never silent LLM filtering:** `evaluate_eligibility()` parses MOF `DAN`/`ATAU` clauses (stripping parenthetical Malay scope descriptions) and CIDB grade ranges (`G4 dan ke atas`, `G3-G6`) into `eligible`, `check_code`, or `mismatch` badges.

## Quickstart

### 1. Run with Docker Compose

Start PostgreSQL (`pgvector/pgvector:pg16`, bound to `127.0.0.1:5439`) and the FastAPI server (port `8000`):

```bash
export ILMU_API_KEY="your-ilmu-key"
uv run python scripts/build_embeddings.py   # optional: builds data/embeddings.f32 (28 MB, ~RM 0.262 assumed)
docker compose up -d --build --wait
```

Without `ILMU_API_KEY` (or without gitignored `data/embeddings.f32`), the container still boots, ingests all `6,899` public notices, warns cleanly on stderr that vector search is inactive, and serves `/v1/search` and degraded `/v1/ask` via PostgreSQL `pg_fts`.

### 2. Try the API

```bash
# Service health, row counts, and today's MYT spend
curl -s http://127.0.0.1:8000/health | python3 -m json.tool

# Hybrid + reranked search with state alias ("Penang" -> "Pulau Pinang" + federal title fallback)
curl -s -X POST http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "hospital air conditioning maintenance", "filters": {"location": "Penang"}, "limit": 3}'

# Look up a notice whose reference contains slashes or inner brackets
curl -s "http://127.0.0.1:8000/v1/tenders/jkr:JKR/CJ/24961/2026%20%5BADENDUM%5D"

# Grounded Q&A with inline QueryTrace and deterministic citation checks
curl -s -X POST http://127.0.0.1:8000/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Are there tenders for halal cooked meal catering for a junior science boarding school (MRSM)?"}'
```

### 3. Local CLI and checks

```bash
uv sync
export TENDER_OS_DATABASE_URL="postgresql://tenderos:tenderos@127.0.0.1:5439/tenderos"

uv run tender-os init-db
uv run tender-os ingest --skip-if-loaded
uv run tender-os ask "Which public tenders require MOF Kod Bidang 210110?"

uv run ruff check src tests scripts evals
uv run mypy src
uv run pytest -v
uv run python scripts/check_readme_numbers.py
```

## Empirical design choices

### Embedding and reranker bake-off (1,000 notices, 45 queries in `evals/pr0/`)

Before building application code, we benchmarked YTL ILMU models against local CPU ONNX baselines on `1,000` sampled notices and `45` dev queries (`25` English-to-Malay cross-lingual, `10` Malay keyword, `10` exact MOF/ref queries) because first-stage embedders feed a cross-encoder reranker (`evals/pr0/summary.json`):

| Stage / Candidate | Overall R@50 | Overall MRR@10 | Cross-Lingual R@50 | Cross-Lingual MRR@10 | Latency | Decision |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| YTL `bge-m3` lowercased (`vector(1024)`) | 0.9833 | 0.7821 | 1.0000 | 0.8240 | 87.4 ms p50 (1-query) | Selected embedder |
| YTL `bge-m3` raw ALL-CAPS | 0.9537 | 0.7369 | 1.0000 | 0.7230 | 118.0 ms p50 (1-query) | Rejected (+25.7% tokens) |
| YTL `ilmu-embedding-v1` lowercased (4096d) | 0.8630 | 0.6208 | 1.0000 | 0.7074 | 188.6 ms p50 (1-query) | Rejected (weak exact match) |
| Local `MiniLM-L12-v2` lowercased (384d) | 0.7500 | 0.5670 | 0.8800 | 0.6167 | 1.31 ms avg | Rejected (-0.2333 R@50 vs `bge-m3`) |
| YTL `bge-reranker` lowercased (top-20 pool) | 0.9556 | 0.9222 | 0.9200 | 0.8600 | 108.9 ms p50 / 213.8 ms p95 | Selected reranker (15x faster) |
| Local `bge-reranker-v2-m3-int8` CPU | 0.9556 | 0.9278 | 0.9200 | 0.8700 | 1621.7 ms p50 / 3274.8 ms p95 | Rejected (15x slower on CPU) |

Lowercasing ALL-CAPS Bahasa Melayu portal titles reduced `bge-m3` embedding token usage by 20.5% (`83,361` vs `104,820` tokens on 1,000 notices, and `524,734` tokens for all `6,899` notices at `~RM 0.262` assumed conservative cost) while lifting cross-lingual MRR@10 by `+0.101`. Choosing hosted YTL `bge-m3` + `bge-reranker` also removes all local `onnxruntime`/`torch` dependencies from the production container while keeping rerank quality within 0.0056 MRR@10 of local CPU.

### Postgres vector query plan

`migrations/001_init.sql` creates an HNSW index (`m=16, ef_construction=64`), and `PostgresStore.search_vector()` sets `SET LOCAL hnsw.ef_search = max(limit * 2, 64)` plus `SET LOCAL hnsw.iterative_scan = relaxed_order` so filtered HNSW queries do not truncate early as the corpus grows. On the current `6,899`-row corpus, `EXPLAIN (ANALYZE, BUFFERS)` shows PostgreSQL choosing an exact sequential scan (`Seq Scan` plus top-N heapsort) in **21.6 ms**, giving 100% exact cosine recall with no approximation loss.

## Evaluation results (6,899 notices, untouched holdout vs dev)

All retrieval and agent evaluations ran live against PostgreSQL + YTL ILMU (`evals/report.json`). Holdout splits (`evals/holdout_30.jsonl`, `evals/agent_holdout_10.jsonl`) were frozen in PR1 with zero query or tender ID overlap against both `dev_30` and `agent_dev_10` (and zero overlap against `evals/pr0/queries_45.jsonl`), and were never used to tune thresholds. Note that `dev_30` draws 28 of its queries from the earlier `evals/pr0/queries_45.jsonl` tuning set so it reflects in-sample dev behavior.

### Retrieval quality on untouched holdout (`holdout_30`, n=30)

Buckets: `8` exact MOF code / portal reference, `7` Malay keyword, `15` English-to-Malay cross-lingual (`22` distinct target sets across 30 queries; 1 query shifts overall MRR by `0.033`).

| Mode | Overall MRR@10 | Overall R@10 | Overall R@50 | Exact MRR@10 | Malay MRR@10 | Cross-Lingual MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pg_fts` | 0.5575 | 0.6333 | 0.6889 | 1.0000 | 0.9286 | 0.1483 |
| `vector` | 0.6634 | 0.7583 | 0.8361 | 0.1054 | 1.0000 | 0.8039 |
| `hybrid` | 0.7206 | 0.9250 | 1.0000 | 0.9375 | 0.9286 | 0.5078 |
| `hybrid_rerank` | 0.9500 | 1.0000 | 1.0000 | 0.8750 | 1.0000 | 0.9667 |

Tradeoff note: on `holdout_30`, `hybrid_rerank` lifts cross-lingual MRR@10 from `0.5078` to `0.9667` (`+0.4589`), while slightly dropping `exact_code_ref` MRR@10 from `0.9375` to `0.8750` when the cross-encoder reorders ties inside multi-notice exact-code sets (Recall@10 stays `1.0000`).

### Retrieval quality on development split (`dev_30`, n=30)

| Mode | Overall MRR@10 | Overall R@10 | Overall R@50 | Exact MRR@10 | Malay MRR@10 | Cross-Lingual MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pg_fts` | 0.4500 | 0.4905 | 0.6000 | 1.0000 | 0.7857 | 0.0000 |
| `vector` | 0.6239 | 0.6730 | 0.7171 | 0.3042 | 1.0000 | 0.6189 |
| `hybrid` | 0.6112 | 0.8516 | 0.9000 | 0.9375 | 0.7857 | 0.3557 |
| `hybrid_rerank` | 0.8319 | 0.8881 | 0.9000 | 0.9375 | 1.0000 | 0.6972 |

`dev_30` scores lower than `holdout_30` because `3/30` queries (`d21`, `d22`, `d23`) use broad English descriptions (`mosque cleaning`, `grass cutting at quarters`, `PKNS covered walkway`) where other valid notices across the full `6,899`-row corpus outranked the single labelled notice.

### Search latency (5 interleaved trials x 6 stratified dev queries, limit=10)

| Mode | Samples | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pg_fts` | 30 | 3.5 | 4.6 | 0.4 | 4.7 |
| `vector` | 30 | 101.4 | 191.9 | 82.6 | 194.2 |
| `hybrid` | 30 | 103.6 | 210.5 | 80.0 | 234.5 |
| `hybrid_rerank` | 30 | 225.4 | 1127.2 | 150.1 | 1521.6 |

### End-to-end grounded `/v1/ask` evaluation (exact counts)

Definitions:
- `Answerable Hits`: count of answerable questions where at least one `expected_ids` tender appeared in `verified_citations`.
- `Verified Cites`: count of `[source:reference]` citations in model answers whose ID and quoted closing date matched the retrieved Postgres record (verified against retrieved records, not a human relevance judgment).

| Split | Answerable Hits | OOS Abstentions | Pre-Gen OOS (0 chat calls) | Verified Cites | Unverified IDs | Date Mismatches | Fallbacks | p50 (ms) | Spend (MYR) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Untouched holdout (n=10) | 7/7 | 3/3 | 2/3 | 16 | 0 | 0 | 0 | 1189.3 | RM 0.015767 |
| Dev (n=10) | 5/7 | 3/3 | 3/3 | 13 | 0 | 0 | 0 | 912.6 | RM 0.014359 |
| Combined (n=20) | 12/14 | 6/6 | 5/6 | 29 | 0 | 0 | 0 | 1111.8 | RM 0.030126 |

Total spend across the full retrieval + latency + 20-case agent evaluation run was **RM 0.204205** (under the RM 2.00 budget cap).

Notes on `dev_10` misses and post-gen abstention:
- `ad06` (`covered walkway link ... PKNS`) scored `0.0029` on `bge-reranker` (below the `0.04` floor), triggering `rerank_below_floor` safely with 0 chat calls.
- `ad07` (`cluster pineapple plantation development in Sarawak`) cleared retrieval at `0.2348`, then `ilmu-mini-v3.3` conservatively returned `model_abstained` with 0 unverified claims.
- `ah08` (`nuclear-powered submarine reactor cores`) scored `0.3972` against top retrieved records, and `ilmu-mini-v3.3` abstained cleanly during generation as a second line of defense.

## Known limits and security notes

- **Known-item eval bias:** Query sets are constructed from known public notices rather than live user logs, and small-n counts (`n=7` / `n=3`) carry wide statistical uncertainty, so agent outcomes are reported as exact counts rather than percentages.
- **Untrusted portal data:** Scraped tender titles, agencies, and eligibility strings are fenced inside `<untrusted_tender_records>` with angle brackets neutralized and treated as read-only synthesis data.
- **Pricing disclosure:** `ilmu-mini-v3.3` (`RM 0.20` / `RM 1.20` per 1M tokens) and `ilmu-v3.1` (`RM 4.00` / `RM 16.00` per 1M tokens) use published YTL ILMU rates (`pricing_basis="published"`). Public per-token pricing for `bge-m3` and `bge-reranker` is not listed in YTL's docs, so the spend ledger applies a conservative **RM 0.50 / 1M tokens** constant (`pricing_basis="assumed_conservative"`).
- **Spend cap scope:** The Postgres `spend_ledger` conditional `UPDATE` prevents concurrent races and settles or releases in a `finally` block, though an abrupt OS `SIGKILL` leaves that call's small reserved hold on the MYT daily row until UTC+8 midnight.
- **Port binding & rate limiters:** `docker-compose.yml` binds PostgreSQL strictly to `127.0.0.1:5439` and publishes the API on port `8000`; `/v1/*` rate limits are in-memory per uvicorn worker process.
- **Corpus scope:** Snapshot of 6,899 cleaned public Malaysian procurement notices across 15 portals with emails stripped and no private client data.
