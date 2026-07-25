"""Retrieval layer for Valluvan.

Exposes three retrieval modes over the Qdrant `thirukkural` collection:
  - dense  : semantic vector search (MiniLM)
  - sparse : lexical BM25 search
  - hybrid : Qdrant Query API with prefetch + RRF fusion of dense + sparse

See docs/hybrid-search.md for the rationale.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import FusionQuery, Fusion, Prefetch, SparseVector

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION = os.getenv("QDRANT_COLLECTION", "thirukkural")
DENSE_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")


@lru_cache(maxsize=1)
def _clients():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    dense = TextEmbedding(model_name=DENSE_MODEL)
    sparse = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return client, dense, sparse


def _dense_vec(query: str):
    _, dense, _ = _clients()
    return list(dense.embed([query]))[0].tolist()


def _sparse_vec(query: str) -> SparseVector:
    _, _, sparse = _clients()
    sv = list(sparse.embed([query]))[0]
    return SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())


def _payloads(points) -> list[dict]:
    return [p.payload for p in points]


def dense_search(query: str, limit: int = 5) -> list[dict]:
    client, _, _ = _clients()
    res = client.query_points(
        collection_name=COLLECTION,
        query=_dense_vec(query),
        using="dense",
        limit=limit,
        with_payload=True,
    )
    return _payloads(res.points)


def sparse_search(query: str, limit: int = 5) -> list[dict]:
    client, _, _ = _clients()
    res = client.query_points(
        collection_name=COLLECTION,
        query=_sparse_vec(query),
        using="sparse",
        limit=limit,
        with_payload=True,
    )
    return _payloads(res.points)


def hybrid_search(query: str, limit: int = 5, prefetch_limit: int = 20) -> list[dict]:
    """Dense + sparse retrieval fused with Reciprocal Rank Fusion (RRF)."""
    client, _, _ = _clients()
    res = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=_dense_vec(query), using="dense", limit=prefetch_limit),
            Prefetch(query=_sparse_vec(query), using="sparse", limit=prefetch_limit),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return _payloads(res.points)


SEARCHERS = {
    "dense": dense_search,
    "sparse": sparse_search,
    "hybrid": hybrid_search,
}

# Number of first-stage candidates fed to the cross-encoder before re-ranking.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
RERANK_BASE_MODE = os.getenv("RERANK_BASE_MODE", "dense")


def reranked_search(
    query: str,
    limit: int = 5,
    base_mode: str = None,
    candidates: int = None,
) -> list[dict]:
    """Two-stage retrieval: wide first-stage recall, then cross-encoder re-rank.

    Pulls `candidates` results from a cheap retriever (`base_mode`) and re-scores
    them with a cross-encoder, returning the best `limit`. See rag/rerank.py.
    """
    from rag.rerank import rerank  # lazy: only load the model if reranking is used

    base_mode = base_mode or RERANK_BASE_MODE
    candidates = candidates or RERANK_CANDIDATES
    pool = SEARCHERS[base_mode](query, limit=candidates)
    return rerank(query, pool, top_n=limit)


SEARCHERS["rerank"] = reranked_search


def search(query: str, mode: str = "hybrid", limit: int = 5) -> list[dict]:
    if mode not in SEARCHERS:
        raise ValueError(f"unknown mode {mode!r}; choose from {list(SEARCHERS)}")
    return SEARCHERS[mode](query, limit=limit)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What does Thirukkural say about controlling anger?"
    for mode in ("dense", "sparse", "hybrid", "rerank"):
        print(f"\n===== {mode.upper()} =====")
        for k in search(q, mode=mode, limit=3):
            if k.get("type") == "knowledge":
                print(f"  [ref] {k['title']}")
                continue
            verse = k["translation_en"].replace("\n", " ")
            print(f"  #{k['kural_no']:>4} [{k['adhigaram_en']}] {verse}")
