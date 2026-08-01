# Retrieval Evaluation

This document records how Valluvan's retrieval layer was evaluated, the results,
and the decisions that followed. It covers **Phase 5** (dense vs sparse vs hybrid) and the retrieval **best practices** added in **Phase 7** (cross-encoder
re-ranking and LLM query rewriting). It maps to the course criteria *"multiple
retrieval approaches evaluated, best one used"*, *"hybrid search (evaluated)"*,
*"document re-ranking (evaluated)"*, and *"user query rewriting (evaluated)"*.

## Goal

Valluvan can retrieve kurals four ways (see [hybrid-search.md](hybrid-search.md)):

- **dense** — semantic vector search (`paraphrase-multilingual-MiniLM-L12-v2`)
- **sparse** — lexical BM25 (`Qdrant/bm25`)
- **hybrid** — dense + sparse fused with Reciprocal Rank Fusion (RRF)
- **rerank** — dense recall of a wide candidate pool, then a **cross-encoder**
  re-scores those candidates (see [Re-ranking](#re-ranking-phase-7))

We need to measure which mode retrieves the *right* kural best, so the RAG answer
is grounded in the most relevant verses.

## Method

### 1. Ground truth (`eval/ground_truth.py`)

We can't hand-label thousands of questions, so we synthesize a gold set:

- Sample **200 kurals** (seeded, reproducible) from all 1,330.
- For each, ask the LLM (Groq `llama-3.3-70b-versatile`) to write **3 natural
  user questions** that this specific kural answers — varied phrasing, some
  direct, some situational, without quoting the verse or its number.
- This yields **600 `(question → kural_no)` pairs** in `data/ground_truth.json`.

The generator is **incremental and resumable** (skips already-processed kurals)
and **rate-limit tolerant** (retries with backoff on Groq 429s).

### 2. Metrics (`eval/eval_retrieval.py`)

For each question we run each mode and locate the gold kural in the top-k
(`k = 5`). We report two granularities:

- **Exact** — the *exact* gold kural appears in top-k.
  - `hit_rate@k` — fraction of questions where it appears.
  - `MRR@k` — mean reciprocal rank (1/rank, 0 if absent).
- **Chapter (adhigaram)** — *any* kural from the gold kural's chapter appears in
  top-k (`chapHit@k`, `chapMRR@k`).

**Why the chapter-level metric matters:** the Thirukkural groups 10 kurals per
chapter (adhigaram), all on the same theme. For an assistant, returning a sibling
verse from the correct chapter is still a **correct, useful answer**. Exact-match
therefore *understates* practical quality; the chapter metric complements it.

## Results

600 questions, `k = 5` (reproduce with `make eval-retrieval`; raw output in
`data/eval_retrieval_results.json`):

| mode        | hit@5  | MRR@5  | chapHit@5 | chapMRR@5 |
|-------------|:------:|:------:|:---------:|:---------:|
| **rerank** 🏆 | **0.388** | **0.256** | 0.597     | 0.417     |
| dense       | 0.337  | 0.211  | 0.590     | **0.420** |
| hybrid      | 0.313  | 0.189  | 0.560     | 0.363     |
| sparse      | 0.162  | 0.103  | 0.300     | 0.198     |

> **Note on the dense baseline.** These dense numbers (hit@5 0.337) are higher
> than an earlier run (0.243) because `fastembed` was upgraded to 0.8.0, whose
> MiniLM uses **mean pooling** instead of CLS pooling — a strictly better
> sentence representation. The collection was re-ingested so stored and query
> vectors are consistent. Sparse/hybrid shifted accordingly.

## Findings (Phase 5: dense vs sparse vs hybrid)

1. **Dense beats sparse and hybrid.** Semantic embeddings handle paraphrased,
   situational questions best — exactly the kind of query Valluvan receives.
2. **Sparse (BM25) is weak here.** Synthesized questions deliberately avoid
   quoting the verse, so lexical overlap is low; keyword matching has little to
   grab onto.
3. **Hybrid underperforms dense.** Because sparse is so weak, RRF fusion pulls
   strong dense results *down* rather than up. Hybrid helps when both retrievers
   are individually competent; here the lexical signal is too noisy.
4. **Absolute exact hit-rates are modest by design.** With ~10 near-duplicate
   sibling kurals per theme competing, exact-match is intentionally hard. The
   chapter-level result (**~59%** for dense) better reflects real usefulness. For
   reference, random retrieval would score ~0.4% (5/1330), so dense is ~85×
   better than chance at exact match.

## Re-ranking (Phase 7)

**Idea.** Dense/sparse retrieval scores the query and each document
*independently* (a bi-encoder). A **cross-encoder** instead reads the query and a
candidate *together* and outputs a direct relevance score — far more precise, but
too slow to run over all 1,330 kurals. The standard fix is **two-stage retrieval**:

1. cheap retriever (dense) pulls a wide pool — `RERANK_CANDIDATES=20`;
2. the cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, local ONNX, no API cost)
   re-scores those 20 and we keep the top 5.

Implemented in [`rag/rerank.py`](../rag/rerank.py); wired as the `rerank` mode in
`rag/search.py`. An English cross-encoder is appropriate because both the user's
question and the kural translation/explanation it is scored against are English.

**Result.** Re-ranking is the **best mode**: it lifts exact **hit@5 0.337 → 0.388
(+15%)** and **MRR@5 0.211 → 0.256 (+21%)** over dense, while chapter-level stays
essentially tied (chapMRR 0.420 vs 0.417). This is exactly the expected behaviour:
the first stage already finds the *right chapter*, and the cross-encoder's job is
to promote the *precise* kural among its ~10 near-identical siblings — which shows
up as a big exact-match gain with flat chapter-level numbers.

## Query rewriting (Phase 7)

**Idea.** Real users type messy, first-person questions. LLM query rewriting
([`rag/query_rewrite.py`](../rag/query_rewrite.py)) asks a small, fast model
(`llama-3.1-8b-instant`) to distil the question into a concise, theme-focused
retrieval query before searching. Evaluated on a 60-question sample (rewrites
cached to `data/rewrite_cache.json`; run `make eval-rewrite`):

| variant             | hit@5  | MRR@5  | chapHit@5 | chapMRR@5 |
|---------------------|:------:|:------:|:---------:|:---------:|
| rerank (raw)        | 0.567  | 0.356  | 0.667     | 0.488     |
| dense (raw)         | 0.383  | 0.216  | 0.600     | 0.419     |
| rerank (rewritten)  | 0.217  | 0.122  | 0.450     | 0.301     |
| dense (rewritten)   | 0.117  | 0.067  | 0.383     | 0.278     |

**Result: rewriting *hurts* on this benchmark** (rerank hit@5 0.567 → 0.217).
Why? The synthesized questions are already clean and *specific* to one kural.
Rewriting generalises them to a broad theme — e.g. *"Why does my ex still think
about me even after we had a fight?"* becomes *"Unresolved emotions after a
breakup, lingering attachment, and past love."* That matches many sibling verses
but discards the detail that pinpoints the **exact** gold kural, so exact-match
collapses (and even chapter-level drops).

This is a genuine, informative negative result: query rewriting is valuable when
the *raw* input is vague or keyword-poor, but counter-productive when questions
are already well-formed — which our eval set is by construction.

## Decisions

- **Default retrieval mode = `rerank`** (`RETRIEVAL_MODE=rerank`, consumed by
  `rag/rag.py`). It wins on exact hit-rate and MRR; the +1–2 model calls per query
  are cheap and local.
- **Query rewriting ships but is OFF by default** (`REWRITE_QUERY=false`). It is
  available as a toggle for genuinely vague free-text, backed by the evidence
  above rather than removed outright.
- All four modes remain available and are exercised by the evaluation, so every
  choice is evidence-based and easy to revisit.

### Qualitative check

For *"What does Thirukkural say about controlling anger?"* the `rerank` mode
returns kurals entirely from the **Restraining Anger** chapter (#301, #302, #305,
…), producing a tightly grounded answer.

## Future work

- **Multilingual reranker:** try `jinaai/jina-reranker-v2-base-multilingual` to
  rerank on the Tamil verse directly (current reranker uses the English fields).
- **Conditional rewriting:** only rewrite when the query looks vague (short /
  keyword-poor), instead of always — to capture rewriting's upside without its
  downside on specific questions.
- **Larger / human-checked ground truth:** grow beyond the 200-kural sample and
  spot-check generated questions for quality.
