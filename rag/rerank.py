"""Cross-encoder re-ranking for Valluvan.

Dense/sparse/hybrid retrieval is fast but scores a query and a document
*independently* (bi-encoder). A cross-encoder instead reads the query and each
candidate *together*, so it judges relevance far more precisely — at the cost of
running the model once per candidate. The standard best-practice pattern is
therefore two-stage retrieval:

    1. cheap retriever  -> pull a wider candidate pool (e.g. top-20)
    2. cross-encoder    -> re-score those candidates and keep the best top-k

Because the user's questions and the kural translations/explanations are in
English, a small English cross-encoder (ms-marco-MiniLM) reranks well and stays
fast + local (ONNX, no API cost). Swap via the RERANK_MODEL env var.

See docs/retrieval-evaluation.md for the measured lift.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")


@lru_cache(maxsize=1)
def _encoder():
    # Imported lazily so the rest of the app doesn't pay the import/model cost
    # unless re-ranking is actually used.
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=RERANK_MODEL)


def _doc_text(k: dict) -> str:
    """The text a candidate is judged on (English, matches the query)."""
    if k.get("type") == "knowledge":
        return f"{k['title']} {k['text']}"
    return f"{k['translation_en']} {k['explanation_en']}"


def rerank(query: str, kurals: list[dict], top_n: int = 5) -> list[dict]:
    """Re-order candidate kurals by cross-encoder relevance to the query.

    Returns the top_n kurals, each annotated with its `rerank_score`.
    """
    if not kurals:
        return []
    scores = list(_encoder().rerank(query, [_doc_text(k) for k in kurals]))
    ranked = sorted(zip(scores, kurals), key=lambda p: p[0], reverse=True)
    out = []
    for score, k in ranked[:top_n]:
        k = {**k, "rerank_score": float(score)}
        out.append(k)
    return out
