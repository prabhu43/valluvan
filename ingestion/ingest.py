"""Ingest the canonical Thirukkural dataset into Qdrant with hybrid vectors.

Each kural is stored with TWO named vectors so we can do hybrid search
(see docs/hybrid-search.md):
  - "dense"  : semantic embedding (paraphrase-multilingual-MiniLM-L12-v2, 384-d)
  - "sparse" : lexical BM25 vector (Qdrant/bm25)

The embedding text combines the (terse) couplet with its English translation and
prose explanation so short verses gain enough semantic body. The full record is
stored as the payload so retrieval can cite Tamil + translation + commentary and
filter by section / adhigaram.

Run (with Qdrant up):  python ingestion/ingest.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

DATA = Path(__file__).resolve().parent.parent / "data" / "thirukkural.json"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "thirukkural")
DENSE_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
VECTOR_SIZE = 384


def build_embed_text(k: dict) -> str:
    """Text used to compute both the dense and sparse vectors for a kural."""
    return (
        f"{k['adhigaram_en']} ({k['section_en']}). "
        f"{k['kural_ta']} "
        f"{k['translation_en']} "
        f"{k['explanation_en']}"
    )


def load_kurals() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))


def main() -> None:
    kurals = load_kurals()
    print(f"Loaded {len(kurals)} kurals")

    # Idempotent one-shot ingestion (used by the docker `ingest` service): skip
    # the expensive re-embed if the collection is already fully populated, unless
    # FORCE_REINGEST=true. Local re-ingests (make ingest) can force via that env.
    if os.getenv("FORCE_REINGEST", "false").lower() not in ("1", "true", "yes"):
        client = QdrantClient(url=QDRANT_URL)
        if (
            client.collection_exists(COLLECTION)
            and client.count(collection_name=COLLECTION).count == len(kurals)
        ):
            print(
                f"'{COLLECTION}' already has {len(kurals)} points at {QDRANT_URL} "
                "— skipping ingest (set FORCE_REINGEST=true to rebuild)."
            )
            return

    texts = [build_embed_text(k) for k in kurals]

    print(f"Loading dense model:  {DENSE_MODEL}")
    dense_embedder = TextEmbedding(model_name=DENSE_MODEL)
    print(f"Loading sparse model: {SPARSE_MODEL}")
    sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL)

    print("Computing dense embeddings ...")
    dense_vectors = list(dense_embedder.embed(texts))
    print("Computing sparse embeddings ...")
    sparse_vectors = list(sparse_embedder.embed(texts))

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )

    points = []
    for k, dv, sv in zip(kurals, dense_vectors, sparse_vectors):
        points.append(
            PointStruct(
                id=k["kural_no"],
                vector={
                    "dense": dv.tolist(),
                    "sparse": SparseVector(
                        indices=sv.indices.tolist(), values=sv.values.tolist()
                    ),
                },
                payload=k,
            )
        )
    client.upsert(collection_name=COLLECTION, points=points)

    count = client.count(collection_name=COLLECTION).count
    print(f"Upserted {count} points (dense+sparse) into '{COLLECTION}' at {QDRANT_URL}")


if __name__ == "__main__":
    main()
