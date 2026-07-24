# 📿 Valluvan — Thirukkural Wisdom Assistant

> Ask any life or ethics question and get an answer grounded in the **1,330
> couplets (kurals)** of Thiruvalluvar, with citations.

Valluvan is an end-to-end **Retrieval-Augmented Generation (RAG)** application
built for the [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp)
course project. It ingests the Thirukkural (a 2,000-year-old Tamil classic on
ethics, polity, and love), indexes it for **hybrid search**, and uses an LLM to
answer modern questions grounded in the most relevant verses — citing each kural
it relies on.

---

## Who was Thiruvalluvar, and what is the Thirukkural?

<img src="images/thiruvalluvar.jpg" alt="Traditional portrait of Thiruvalluvar" width="240" align="right" />

**Thiruvalluvar** (often simply **Valluvar**) was a celebrated Tamil poet and
philosopher, traditionally believed to have lived in South India between roughly
the 4th century BCE and the 5th century CE. He is one of the most revered figures
in Tamil culture — a 133-foot statue of him stands off the coast of Kanyakumari,
at the southern tip of India.

His single surviving work, the **Thirukkural** (திருக்குறள், "sacred couplets"),
is a masterpiece of world literature. It is a collection of **1,330 couplets
(kurals)** — each just two short lines — organized into **133 chapters
(adhigarams)** of 10 kurals each, grouped under three sections (**paals**):

| Section (Paal)        | Theme                          | Kurals |
|-----------------------|--------------------------------|--------|
| **Aram** (அறம்)       | Virtue / ethics                | 380    |
| **Porul** (பொருள்)    | Wealth, polity & society       | 700    |
| **Inbam** (இன்பம்)    | Love                           | 250    |

What makes the Thirukkural remarkable is its **secular, universal wisdom** — it
speaks to honesty, friendship, leadership, self-control, kindness, and love in a
way that remains strikingly relevant ~2,000 years later, across religions and
cultures. It has been translated into more than 40 languages.

Valluvan brings this timeless wisdom to your fingertips: ask a modern question,
and it answers grounded in the most relevant kurals — with citations.

> *Image: traditional representation of Thiruvalluvar
> ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Thiruvalluvar_(Likely_Representation).jpg), public domain).*

---

## Why this project?

The Thirukkural's wisdom is timeless, but the verses are terse, archaic, and
spread across 133 chapters — and most people can't read classical Tamil.
Valluvan lets anyone ask a plain-language question (English **or** Tamil) and
receive a grounded, cited answer.

---

## Architecture

```
                ┌────────────────────────────────────────────────┐
   question ───▶│  rag/search.py   dense recall → cross-encoder    │
                │        │         re-rank (default: rerank mode)   │
                │        ▼                                        │
                │   Qdrant (docker)  ── 1,330 kurals, 2 vectors   │
                │        │                                        │
                │        ▼                                        │
                │  rag/rag.py   grounded prompt → LLM (Groq)      │──▶ answer
                └────────────────────────────────────────────────┘         + cited kurals
```

| Concern           | Technology                                                |
|-------------------|-----------------------------------------------------------|
| Vector DB         | Qdrant (Docker) — named `dense` + `sparse` vectors        |
| Dense embeddings  | `paraphrase-multilingual-MiniLM-L12-v2` (Tamil + English) |
| Sparse / keyword  | `Qdrant/bm25` — hybrid via Reciprocal Rank Fusion (RRF)   |
| Re-ranker         | `ms-marco-MiniLM-L-6-v2` cross-encoder (default mode)     |
| LLM               | Groq `llama-3.3-70b-versatile` (swappable via env)        |
| Interface         | Streamlit chat UI (`app/streamlit_app.py`)                |
| Monitoring        | Postgres + Grafana (planned)                              |

See [`docs/hybrid-search.md`](docs/hybrid-search.md) for why we store two vectors
per kural, and [`docs/retrieval-evaluation.md`](docs/retrieval-evaluation.md) for
the retrieval evaluation (dense / sparse / hybrid / rerank + query rewriting) and
the resulting decisions.

---

## Tech Stack

### Large Language Model (answer generation)

- **Default:** [Groq](https://console.groq.com/) serving
  **`llama-3.3-70b-versatile`** — Meta's Llama 3.3 70B, accessed through Groq's
  fast inference API. Used to read the retrieved kurals and write a grounded,
  cited answer (`rag/rag.py`).
- **Why Groq:** generous free tier, very low latency (~1–2 s), and an
  **OpenAI-compatible API**, so the same client code works for other providers.
- **Swappable via `.env`** (`LLM_PROVIDER`, `LLM_MODEL`) with zero code changes:
  - `openai` → e.g. `gpt-4o-mini`
  - `ollama` → fully local, e.g. `llama3.1`
- **Evaluation model:** `llama-3.1-8b-instant` is also used for bulk offline jobs
  (ground-truth question generation, LLM-as-judge) to stay within daily quotas.
- **Prompting:** two prompt variants (`concise`, `structured`) are defined and
  compared with an LLM-as-judge — see
  [`docs/llm-evaluation.md`](docs/llm-evaluation.md).

### Vector database

- **[Qdrant](https://qdrant.tech/)** (v1.10.1), run as a **Docker container** via
  `docker-compose.yml`. Data persists in a named Docker volume
  (`qdrant_storage`).
- One collection (`thirukkural`) holds all **1,330 kurals**, each stored with
  **two named vectors** plus the full record as payload (Tamil verse,
  transliteration, translations, commentary, section/chapter metadata).
- Qdrant performs **native hybrid search**: dense + sparse retrieval fused with
  **Reciprocal Rank Fusion (RRF)** in a single query (its Query API), so no
  external search engine is needed.
- Ports: `6333` (REST + web dashboard at `http://localhost:6333/dashboard`),
  `6334` (gRPC).

### Embedders & reranker

The embedders and cross-encoder reranker all run **locally** via
[`fastembed`](https://github.com/qdrant/fastembed) — no API cost, no external
calls during ingestion, search, or re-ranking.

| Stage    | Model                                                        | Dim | Role                                             |
|----------|--------------------------------------------------------------|-----|--------------------------------------------------|
| `dense`  | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`| 384 | Semantic similarity; multilingual (Tamil + English) |
| `sparse` | `Qdrant/bm25`                                                | —   | Lexical / keyword matching (BM25)                |
| rerank   | `Xenova/ms-marco-MiniLM-L-6-v2` (cross-encoder)              | —   | Re-scores the top candidates for precision       |

- The **dense** model is multilingual, so a question in **English or Tamil** can
  match the relevant verse regardless of the language it was written in.
- The **sparse** BM25 model captures exact-term overlap; combining the two gives
  hybrid search (rationale in [`docs/hybrid-search.md`](docs/hybrid-search.md)).
- The **cross-encoder reranker** reads the query and each candidate *together*
  for a precise relevance score — used in two-stage retrieval (dense recall →
  rerank). It measurably beats plain dense; see
  [`docs/reranking.md`](docs/reranking.md).
- Embedding text per kural fuses the terse couplet with its English translation
  and prose explanation so short verses gain enough semantic body.

### Supporting tools

| Layer              | Technology                                             |
|--------------------|--------------------------------------------------------|
| Ingestion          | Python script (`ingestion/ingest.py`) + `fastembed`    |
| RAG / retrieval    | `qdrant-client`, OpenAI-compatible client (`openai`)   |
| Data prep          | `pandas` + `pyarrow` (reads the HF parquet)            |
| Interface          | **Streamlit** chat UI (`app/streamlit_app.py`)         |
| Monitoring         | Postgres + Grafana (planned)                           |
| Orchestration      | Docker Compose                                          |

---

## Dataset

Source: [`yuvarajvelmurugan/thirukkural`](https://huggingface.co/datasets/yuvarajvelmurugan/thirukkural)
on Hugging Face — a fully bilingual, complete (1,330 rows, zero nulls) edition
with Tamil + English for every section, chapter (adhigaram), verse, translation,
and commentary. The Thirukkural itself is in the **public domain**.

- Raw source: `data/thirukkural_hf.parquet`
- Canonical, normalized records: `data/thirukkural.json` (produced by
  `ingestion/normalize.py`)

---

## Prerequisites

- **Docker** (for Qdrant)
- **Python 3.10+** and [`uv`](https://github.com/astral-sh/uv)
- A free **Groq API key** — get one at https://console.groq.com/keys

---

## Quick start

Every step has a `make` target. Run `make help` to see them all.

```bash
# 1. Create the virtualenv and install dependencies
make venv
make install

# 2. Create your .env and add your Groq key
make env
#    then edit .env and set: GROQ_API_KEY=gsk-...

# 3. Start Qdrant (vector database)
make qdrant-up

# 4. Ingest the Thirukkural into Qdrant (dense + sparse vectors)
make ingest

# 5. Ask Valluvan a question
make rag Q="How can I control my anger?"
```

---

## How to run each step

### Environment & dependencies
```bash
make venv       # create .venv via uv
make install    # install pinned deps from requirements.txt
make env        # copy .env.example -> .env (then add GROQ_API_KEY)
```

### Qdrant (vector database)
```bash
make qdrant-up    # start the Qdrant container
make qdrant-logs  # tail its logs
make dashboard    # print the web dashboard URL (http://localhost:6333/dashboard)
make qdrant-down  # stop it (data is kept in a Docker volume)
```
Qdrant runs as a Docker container defined in `docker-compose.yml`. Your data
persists in the `qdrant_storage` volume across restarts; it is only deleted by
`make clean`.

### Data preparation (optional — output is committed)
```bash
make normalize   # regenerate data/thirukkural.json from the raw parquet
```

### Ingestion
```bash
make ingest      # embed all 1,330 kurals (dense + sparse) and upsert to Qdrant
```

### Retrieval (compare modes)
```bash
make search Q="about true friendship"
# prints top results for DENSE vs SPARSE vs HYBRID side by side
```

### Ask Valluvan (full RAG)
```bash
make rag Q="What does Thirukkural teach about leadership?"
# retrieves kurals, builds a grounded prompt, calls the LLM,
# and prints an answer that cites kural numbers + telemetry
```

### Evaluation
```bash
make eval-ground      # generate the retrieval ground-truth dataset (LLM)
make eval-retrieval   # hit-rate / MRR for dense / sparse / hybrid / rerank
make eval-rewrite     # measure LLM query rewriting (raw vs rewritten, sampled)
make eval-llm         # LLM-as-judge comparison of prompt variants
```

### Interface — Streamlit chat UI
```bash
make qdrant-up   # ensure the vector DB is running
make app         # launch the Streamlit chat UI at http://localhost:8501
```
Ask a life/ethics question; Valluvan replies grounded in the Thirukkural and
**cites the kurals it used**. Each answer includes:
- a **📖 Sources** expander (Tamil verse, transliteration, English translation
  and meaning, chapter) for every cited kural,
- **telemetry** (retrieval mode, prompt variant, model, latency, tokens), and
- **👍/👎 feedback** buttons — logged via `app/storage.py` for monitoring.

The sidebar lets you switch retrieval mode / prompt variant and `k` on the fly
(defaults are the evaluation winners: `rerank` + `concise`).

### Full stack / teardown
```bash
make up          # start all services via docker-compose
make down        # stop all services (keeps data)
make clean       # stop AND delete data volumes (re-run make ingest afterwards)
```

---

## Configuration

All configuration is via `.env` (see `.env.example`). Key variables:

| Variable            | Purpose                                    | Default                                                    |
|---------------------|--------------------------------------------|------------------------------------------------------------|
| `LLM_PROVIDER`      | `groq` \| `openai` \| `ollama`             | `groq`                                                     |
| `LLM_MODEL`         | Chat model name                            | `llama-3.3-70b-versatile`                                  |
| `GROQ_API_KEY`      | Groq API key                               | —                                                          |
| `RETRIEVAL_MODE`    | `rerank` \| `dense` \| `sparse` \| `hybrid`| `rerank`                                                  |
| `REWRITE_QUERY`     | Rewrite messy input before retrieval       | `false`                                                    |
| `PROMPT_VARIANT`    | `concise` \| `structured`                  | `concise`                                                  |
| `QDRANT_URL`        | Qdrant endpoint                            | `http://localhost:6333`                                    |
| `QDRANT_COLLECTION` | Collection name                            | `thirukkural`                                              |
| `EMBEDDING_MODEL`   | Dense embedding model                      | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `SPARSE_MODEL`      | Sparse (BM25) model                        | `Qdrant/bm25`                                              |
| `RERANK_MODEL`      | Cross-encoder reranker                     | `Xenova/ms-marco-MiniLM-L-6-v2`                            |

To switch the LLM to OpenAI: set `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-4o-mini`,
and `OPENAI_API_KEY`. For a fully local setup use `LLM_PROVIDER=ollama`.

---

## Project status

- [x] Data ingestion (bilingual Thirukkural → Qdrant, dense + sparse)
- [x] Hybrid retrieval (dense / sparse / RRF)
- [x] Grounded RAG with kural citations (Groq)
- [x] Retrieval evaluation (dense / sparse / hybrid / rerank → rerank)
- [x] LLM evaluation (prompt variants via LLM-as-judge → concise)
- [x] Best practices: hybrid search, cross-encoder re-ranking, query rewriting (all evaluated)
- [x] Streamlit chat UI (cited sources, telemetry, 👍/👎 feedback)
- [ ] Monitoring (Postgres + Grafana)
- [ ] Full containerization & cloud deployment

---

## License / attribution

The Thirukkural is in the public domain. Dataset courtesy of the Hugging Face
dataset linked above. This project is built for educational purposes as part of
the DataTalksClub LLM Zoomcamp.
