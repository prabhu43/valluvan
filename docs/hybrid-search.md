# Why Valluvan Uses Hybrid Search (Two Vectors per Kural)

Valluvan stores **two vectors** for every kural in Qdrant: a **dense** vector and
a **sparse** vector. This document explains why.

## The core problem: two different notions of "relevance"

When someone searches, a document can be "relevant" in two very different ways:

1. **Lexical relevance (keyword match)** — the document contains the *actual
   words* you typed.
2. **Semantic relevance (meaning match)** — the document is *about the same
   idea*, even if it uses completely different words.

No single vector captures both well. That is why we store two.

## Vector #1 — Dense vector (semantic)

The embedding produced by the `paraphrase-multilingual-MiniLM-L12-v2` model: a
list of 384 floats that encodes **meaning**. Similar meanings map to nearby
vectors.

- ✅ Query "how to control anger" matches a verse about "restraining wrath" —
  different words, same idea.
- ✅ Works across languages (a Tamil query can match an English verse).
- ❌ Weakness: vague on exact/rare terms. Proper names, specific keywords, or
  very terse text can get "averaged away."

Observed in Valluvan: the Tamil query "கோபத்தை எப்படி கட்டுப்படுத்துவது?" (how to
control anger) drifted to the *Feigned Anger* (love-quarrel) chapter instead of
*Restraint of Anger* (வெகுளாமை) — a classic dense-only limitation on terse text.

## Vector #2 — Sparse vector (keyword / BM25)

A classic keyword score expressed in vector form. A sparse vector is mostly
zeros, with non-zero weights only at the dimensions for words actually present.
It uses the same math as traditional search engines (BM25).

- ✅ Nails exact-word matches: query "wealth" strongly hits verses literally
  containing "wealth."
- ✅ Great for names, rare or technical terms, and precise vocabulary.
- ❌ Weakness: no understanding of meaning. "anger" and "wrath" are unrelated to
  it; it won't match synonyms or across languages.

## Why store both → hybrid search

Dense and sparse have **complementary weaknesses**. Hybrid search runs both and
fuses the rankings with **Reciprocal Rank Fusion (RRF)**, so a kural that scores
well on *either* signal rises to the top.

| Query type                     | Dense wins | Sparse wins |
|--------------------------------|:----------:|:-----------:|
| "control anger" (synonyms)     | ✅         | ❌          |
| "wealth" (exact word)          | ~          | ✅          |
| Tamil query, English verse     | ✅         | ❌          |
| rare proper noun               | ❌         | ✅          |

### How RRF works (briefly)

Each retriever returns a ranked list. For a document at rank `r` in a list, RRF
adds `1 / (k + r)` to its score (with a small constant `k`, commonly 60).
Scores from both lists are summed. Documents ranked highly by either retriever —
and especially by both — end up on top. RRF needs no score calibration between
the two very different scoring scales, which is why it is the default fusion
method in Qdrant.

## Why this matters specifically for Valluvan

1. **The rubric rewards it.** "Hybrid search (combining text and vector)" is a
   best-practice point, and "multiple retrieval approaches evaluated" is worth
   points. Storing both vectors lets the evaluation phase compare
   *dense-only vs sparse-only vs hybrid* and demonstrate that hybrid wins.
2. **Thirukkural is terse.** Two-line verses give dense embeddings little to
   work with, so the lexical signal genuinely helps.
3. **Qdrant does it natively.** One collection, two named vectors, one query
   call with RRF fusion — no extra infrastructure or services.

## Cost

Negligible. Sparse vectors are tiny (a handful of non-zero weights) and are
computed locally by fastembed's BM25 implementation — no API cost, minimal
storage and latency overhead.

## Implementation in Valluvan

- Ingestion (`ingestion/ingest.py`) computes and stores both a named `dense`
  vector and a named `sparse` vector per kural.
- Retrieval (`rag/search.py`) exposes `dense`, `sparse`, and `hybrid` modes;
  hybrid uses Qdrant's Query API with `prefetch` + `FusionQuery(RRF)`.
