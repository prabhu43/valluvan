# Re-ranking (Cross-Encoder) in Valluvan

Valluvan's default retrieval mode is **`rerank`**: a two-stage pipeline where a
cheap retriever proposes candidates and a **cross-encoder** re-scores them for
precision. This document explains the technique, how it is implemented, and the
measured results. It complements
[`retrieval-evaluation.md`](retrieval-evaluation.md) (which compares *all* modes)
by focusing on re-ranking alone.

Corresponds to **Phase 7** and the course best-practice criterion
*"document re-ranking (evaluated)"*.

## The problem re-ranking solves

Dense and sparse search are **bi-encoders**: the query and every kural are turned
into vectors *independently*, then compared by similarity. This is fast (you can
pre-compute all 1,330 kural vectors once) but lossy — the model never sees the
query and a candidate *side by side*, so it can't reason about their specific
interaction.

The Thirukkural makes this especially hard: each **chapter (adhigaram) packs 10
near-identical kurals** on the same theme. Dense search reliably finds the *right
chapter* but often can't tell which of the 10 sibling verses best answers the
exact question.

## The technique: two-stage retrieval

```
        user question
             │
             ▼
   ┌───────────────────────┐   stage 1: cheap, high-recall
   │  dense search (top-20) │   (bi-encoder over Qdrant)
   └───────────┬───────────┘
               │  20 candidate kurals
               ▼
   ┌───────────────────────┐   stage 2: expensive, high-precision
   │  cross-encoder re-rank │   reads (query, candidate) TOGETHER
   └───────────┬───────────┘
               │  re-scored
               ▼
        top-5 kurals → RAG
```

- **Stage 1 (recall):** dense search pulls a *wide* pool of `RERANK_CANDIDATES=20`
  candidates. The goal here is only to not miss the right verse.
- **Stage 2 (precision):** a **cross-encoder** takes the query and each candidate
  *as a pair* and outputs a single relevance score. Because it reads both texts
  jointly, it judges relevance far more accurately — at the cost of one model run
  per candidate, which is why we only apply it to 20, not all 1,330.

## Implementation

- **Reranker module:** [`rag/rerank.py`](../rag/rerank.py)
  - Model: `Xenova/ms-marco-MiniLM-L-6-v2` (a small English cross-encoder), run
    **locally via `fastembed`'s `TextCrossEncoder`** — ONNX, no API cost, no
    external calls. Configurable via `RERANK_MODEL`.
  - Each candidate is scored on its **English** fields
    (`translation_en` + `explanation_en`). An English cross-encoder is the right
    fit because the user's questions are English too.
  - Returns the top-`n` kurals, each annotated with a `rerank_score`.
  - The model is loaded lazily (only when re-ranking is actually used).
- **Wiring:** [`rag/search.py`](../rag/search.py) exposes it as the `rerank` mode
  (`reranked_search`), registered in `SEARCHERS`. Tunables:
  `RERANK_CANDIDATES` (pool size, default 20) and `RERANK_BASE_MODE` (first-stage
  retriever, default `dense`).
- **Default:** `RETRIEVAL_MODE=rerank` in `rag/rag.py` / `.env.example`.

## Results

Evaluated on the **600 synthetic `(question → kural_no)` pairs** in
`data/ground_truth.json`, `k = 5` (reproduce with `make eval-retrieval`; raw
numbers in `data/eval_retrieval_results.json`):

| mode        | hit@5  | MRR@5  | chapHit@5 | chapMRR@5 |
|-------------|:------:|:------:|:---------:|:---------:|
| **rerank** 🏆 | **0.388** | **0.256** | 0.597     | 0.417     |
| dense       | 0.337  | 0.211  | 0.590     | **0.420** |
| hybrid      | 0.313  | 0.189  | 0.560     | 0.363     |
| sparse      | 0.162  | 0.103  | 0.300     | 0.198     |

**Interpretation**

- Re-ranking lifts exact **hit@5 by +15%** (0.337 → 0.388) and **MRR@5 by +21%**
  (0.211 → 0.256) over the dense baseline.
- **Chapter-level metrics stay flat** (chapMRR 0.420 vs 0.417). This is exactly
  the expected signature of a good reranker: stage 1 already lands the correct
  *chapter*, so the reranker can't add chapter-level recall — instead it promotes
  the **precise** kural among its ~10 near-identical siblings, which shows up as a
  pure **exact-match** gain.
- The cost is modest: 20 short cross-encoder scorings per query, all local.

> **Reproducibility note.** The dense baseline here (0.337) is higher than an
> earlier run (0.243) because `fastembed` was upgraded to 0.8.0, whose MiniLM uses
> **mean pooling** instead of CLS pooling. The collection was re-ingested so
> stored and query vectors are consistent. Re-ranking's *relative* lift holds
> regardless of the baseline.

## A note on query rewriting

The other Phase 7 best practice — LLM **query rewriting** — was evaluated in the
same harness and **hurt** retrieval on this benchmark (it over-generalizes
already-specific questions to broad themes). It ships as an off-by-default toggle.
Full analysis and numbers are in
[`retrieval-evaluation.md`](retrieval-evaluation.md#query-rewriting-phase-7).

## Trade-offs & future work

- **Latency:** re-ranking adds one small model pass over 20 candidates. It is
  local and fast, but for very high throughput the pool size (`RERANK_CANDIDATES`)
  can be tuned down.
- **Language:** the current cross-encoder scores English text. A multilingual
  reranker (`jinaai/jina-reranker-v2-base-multilingual`) could rerank on the Tamil
  verse directly — worth trying for Tamil-language queries.
- **Bigger pool:** raising `RERANK_CANDIDATES` trades speed for the chance to
  recover verses dense search ranked just outside the top-20.

## TL;DR

Re-ranking is Valluvan's default because it is the single best-performing
retrieval mode: a local, no-API cross-encoder that promotes the *precise* kural
among near-identical siblings, improving exact hit-rate and MRR without hurting
chapter-level recall.
