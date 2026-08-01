"""Ingest the canonical Thirukkural dataset into Qdrant with hybrid vectors.

Two kinds of documents are ingested into the SAME collection:
  - 1,330 kurals (type="kural")     — the individual couplets
  - curated reference notes (type="knowledge", data/knowledge.json) — facts
    ABOUT Thiruvalluvar and the Thirukkural (how many kurals, book structure,
    the Kanyakumari statue, ...) so meta-questions the couplets don't answer
    on their own are still retrievable.

Each document is stored with TWO named vectors so we can do hybrid search
(see docs/hybrid-search.md):
  - "dense"  : semantic embedding (paraphrase-multilingual-MiniLM-L12-v2, 384-d)
  - "sparse" : lexical BM25 vector (Qdrant/bm25)

For a kural, the embedding text combines the (terse) couplet with its English
translation and prose explanation so short verses gain enough semantic body; for
a knowledge doc it combines the title and the fact text. The full record is
stored as the payload so retrieval can cite Tamil + translation + commentary (or
the reference title + source) and filter by type / section / adhigaram.

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
KNOWLEDGE = Path(__file__).resolve().parent.parent / "data" / "knowledge.json"
# Point-ID offset for reference/knowledge documents so they never collide with
# the 1..1330 kural IDs.
KNOWLEDGE_ID_BASE = 100_001
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION = os.getenv("QDRANT_COLLECTION", "thirukkural")
DENSE_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
VECTOR_SIZE = 384


def build_embed_text(k: dict) -> str:
    """Text used to compute both the dense and sparse vectors for a document."""
    if k.get("type") == "knowledge":
        # Reference note about Thiruvalluvar / the Thirukkural itself (not a kural).
        # Include Tamil title/text (when present) so the multilingual embedder
        # also matches Tamil-script meta-questions, not just English ones.
        parts = [k["title"], k["text"], k.get("title_ta", ""), k.get("text_ta", "")]
        return " ".join(p for p in parts if p).strip()
    return (
        f"{k['adhigaram_en']} ({k['section_en']}). "
        f"{k['kural_ta']} "
        f"{k['translation_en']} "
        f"{k['explanation_en']}"
    )


def load_kurals() -> list[dict]:
    kurals = json.loads(DATA.read_text(encoding="utf-8"))
    for k in kurals:
        k.setdefault("type", "kural")
    return kurals


def load_knowledge() -> list[dict]:
    """Load curated reference documents about Thiruvalluvar and the Thirukkural.

    These give the assistant meta-knowledge (e.g. how many kurals exist, who
    Thiruvalluvar was, the book structure) that the individual couplets do not
    contain. They are ingested into the same collection with type='knowledge'.
    """
    if not KNOWLEDGE.exists():
        return []
    docs = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))
    for i, d in enumerate(docs, start=1):
        d["type"] = "knowledge"
        d["knowledge_no"] = i
    return docs


def point_id(doc: dict) -> int:
    """Stable Qdrant point ID: kurals keep 1..1330, knowledge docs use a high range."""
    if doc.get("type") == "knowledge":
        return KNOWLEDGE_ID_BASE + int(doc["knowledge_no"])
    return doc["kural_no"]


def main() -> None:
    kurals = load_kurals()
    knowledge = load_knowledge()
    docs = kurals + knowledge
    print(f"Loaded {len(kurals)} kurals + {len(knowledge)} knowledge docs "
          f"= {len(docs)} documents")

    # Idempotent one-shot ingestion (used by the docker `ingest` service): skip
    # the expensive re-embed if the collection is already fully populated, unless
    # FORCE_REINGEST=true. Local re-ingests (make ingest) can force via that env.
    if os.getenv("FORCE_REINGEST", "false").lower() not in ("1", "true", "yes"):
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        if (
            client.collection_exists(COLLECTION)
            and client.count(collection_name=COLLECTION).count == len(docs)
        ):
            print(
                f"'{COLLECTION}' already has {len(docs)} points at {QDRANT_URL} "
                "— skipping ingest (set FORCE_REINGEST=true to rebuild)."
            )
            return

    texts = [build_embed_text(d) for d in docs]

    print(f"Loading dense model:  {DENSE_MODEL}")
    dense_embedder = TextEmbedding(model_name=DENSE_MODEL)
    print(f"Loading sparse model: {SPARSE_MODEL}")
    sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL)

    print("Computing dense embeddings ...")
    dense_vectors = list(dense_embedder.embed(texts))
    print("Computing sparse embeddings ...")
    sparse_vectors = list(sparse_embedder.embed(texts))

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
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
    for d, dv, sv in zip(docs, dense_vectors, sparse_vectors):
        points.append(
            PointStruct(
                id=point_id(d),
                vector={
                    "dense": dv.tolist(),
                    "sparse": SparseVector(
                        indices=sv.indices.tolist(), values=sv.values.tolist()
                    ),
                },
                payload=d,
            )
        )
    # Upsert in batches so a large single write does not time out against
    # remote (Qdrant Cloud) instances.
    batch_size = 256
    for start in range(0, len(points), batch_size):
        client.upsert(
            collection_name=COLLECTION,
            points=points[start : start + batch_size],
            wait=True,
        )

    count = client.count(collection_name=COLLECTION).count
    print(f"Upserted {count} points (dense+sparse) into '{COLLECTION}' at {QDRANT_URL}")


if __name__ == "__main__":
    main()
