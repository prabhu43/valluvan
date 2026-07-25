# Cloud deployment (Phase 12 — bonus)

Valluvan is fully env-driven, so the same code that runs locally via
`docker compose` also runs on free managed services with **no code changes** —
only environment variables/secrets differ.

## Architecture

```
Streamlit Community Cloud  ──  runs app/streamlit_app.py (public HTTPS URL)
        │
        ├──►  Qdrant Cloud       vector DB (1,330 kurals, dense + sparse)
        ├──►  Groq API           LLM (llama-3.3-70b-versatile)
        └──►  Neon Postgres      monitoring log  ──►  Grafana Cloud (dashboard)
```

| Service            | Free tier            | Holds                              |
|--------------------|----------------------|------------------------------------|
| Streamlit Cloud    | 1 app, ~1 GB RAM     | the Streamlit UI                   |
| Qdrant Cloud       | 1 GB cluster         | `thirukkural` collection           |
| Groq               | daily token quota    | LLM inference                      |
| Neon (or Supabase) | 500 MB Postgres      | `conversations` table              |
| Grafana Cloud      | free tier            | the monitoring dashboard           |

Secrets are **never committed**. Locally they live in `.env`; on Streamlit Cloud
they go in the app's **Secrets** panel (injected as environment variables).

---

## 1. Qdrant Cloud (vector database)

1. Create a free cluster at <https://cloud.qdrant.io>.
2. Copy the **cluster URL** (`https://<id>.<region>.aws.cloud.qdrant.io:6333`)
   and create an **API key**.
3. Populate it **from your laptop** (Streamlit Cloud has no ingestion step):

   ```bash
   # in .env — point the ingester at the cloud cluster
   QDRANT_URL=https://<id>.<region>.aws.cloud.qdrant.io:6333
   QDRANT_API_KEY=<your-qdrant-cloud-api-key>

   FORCE_REINGEST=true make ingest      # embeds + upserts 1,330 kurals
   ```

   Re-run without `FORCE_REINGEST` to confirm it reports "already has 1330
   points — skipping".

## 2. Managed Postgres — Neon (monitoring)

Any managed Postgres works (the app is provider-agnostic). We use **Neon**:

1. Sign up at <https://neon.tech> (**Continue with GitHub** is easiest) and
   create a project. Neon auto-creates a database (`neondb`) and a role.
2. On the project dashboard open **Connection Details** and copy the host,
   database, user, and password. A Neon connection string looks like:

   ```
   postgresql://<user>:<password>@ep-<id>-pooler.<region>.aws.neon.tech/neondb?sslmode=require
   ```

   Map those to our env vars:

   | env var             | Neon value                                   |
   |---------------------|----------------------------------------------|
   | `POSTGRES_HOST`     | `ep-<id>-pooler.<region>.aws.neon.tech`      |
   | `POSTGRES_PORT`     | `5432`                                       |
   | `POSTGRES_DB`       | `neondb`                                     |
   | `POSTGRES_USER`     | `<user>`                                     |
   | `POSTGRES_PASSWORD` | `<password>`                                 |
   | `POSTGRES_SSLMODE`  | `require`                                    |

3. No migration needed — the `conversations` table is auto-created on first
   write (`CREATE TABLE IF NOT EXISTS`, see `app/db.py`).

   > Use the **pooled** host (`-pooler`) for a serverless app. `sslmode=require`
   > is mandatory for Neon.

   > _Supabase alternative:_ same env vars — host `db.<ref>.supabase.co`, db
   > `postgres`, user `postgres`, port `5432` (or `6543` via the session pooler),
   > `sslmode=require`.

## 3. Streamlit Community Cloud (the app)

1. Push this repo to **public** GitHub.
2. At <https://share.streamlit.io> → **New app** → pick the repo/branch and set
   the entrypoint to **`app/streamlit_app.py`**. Streamlit Cloud installs the
   lean **`app/requirements.txt`** (next to the entrypoint), not the heavier root
   file.
3. Open **Advanced settings → Secrets** and paste (TOML — top-level keys are
   also exposed as environment variables, which our `os.getenv` code reads):

   ```toml
   GROQ_API_KEY   = "gsk-…"

   QDRANT_URL     = "https://<id>.<region>.aws.cloud.qdrant.io:6333"
   QDRANT_API_KEY = "<qdrant-cloud-api-key>"

   MONITORING_DB   = "postgres"
   POSTGRES_HOST     = "ep-<id>-pooler.<region>.aws.neon.tech"
   POSTGRES_PORT     = "5432"
   POSTGRES_DB       = "neondb"
   POSTGRES_USER     = "<neon-user>"
   POSTGRES_PASSWORD = "<neon-password>"
   POSTGRES_SSLMODE  = "require"

   RETRIEVAL_MODE  = "rerank"
   PROMPT_VARIANT  = "concise"
   ```

4. Deploy. First boot downloads the embedding + reranker models into the app's
   model cache (slow the first time, cached after).

   > **Memory tip:** the free 1 GB tier is tight when `rerank` loads the dense +
   > sparse + cross-encoder models. If the app OOMs, set `RETRIEVAL_MODE = "dense"`
   > in Secrets to skip the reranker (dense was a close second in evaluation).

## 4. Grafana Cloud (dashboard)

1. Create a free stack at <https://grafana.com/products/cloud>.
2. **Connections → Add new connection → PostgreSQL**, pointing at the same
   Neon database (host/user/password above, **TLS/SSL = required**).
3. Import `monitoring/grafana/dashboards/valluvan.json` (**Dashboards → New →
   Import**) and select the Neon datasource. The panels query the same
   `conversations` table the app writes to.

---

## Verifying the deployment

- Open the Streamlit URL, ask a question → you should get a grounded answer with
  cited kurals.
- Rate it 👍/👎 → a row appears in Neon (`select count(*) from conversations`).
- The Grafana Cloud dashboard shows the conversation, latency, tokens, and
  feedback within one refresh interval.
