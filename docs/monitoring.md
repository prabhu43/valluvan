# Monitoring (Postgres + Grafana)

Valluvan logs every answered question and its user feedback, then visualises
usage and quality in a Grafana dashboard. This corresponds to **Phase 9** and the
course criterion *"monitoring — user feedback collection + a dashboard with 5+
charts"*.

## What is collected

Every call to `rag.answer()` returns rich telemetry, which the Streamlit UI
persists via the storage seam ([`app/storage.py`](../app/storage.py)) for each
interaction:

| Field | Meaning |
|-------|---------|
| `question`, `answer` | the user's question and Valluvan's grounded reply |
| `kural_nos` | which kurals were cited |
| `retrieval_mode` | dense / sparse / hybrid / **rerank** |
| `rewritten_query` | the rewritten query, if query rewriting was on |
| `prompt_variant` | concise / structured |
| `model`, `provider` | LLM used |
| `latency_s` | end-to-end answer latency |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | token usage |
| `feedback` | 👍 (`1`) / 👎 (`-1`) from the UI buttons |

## Architecture

```
  Streamlit UI ──▶ app/storage.py ──▶ app/db.py ──▶ Postgres ──▶ Grafana dashboard
   (👍/👎)          (backend seam)     (psycopg2)     (conversations)   (10 panels)
```

- **`app/storage.py`** is a **backend-selecting facade** exposing
  `log_interaction` / `log_feedback` / `load_interactions`. It picks a backend
  from the `MONITORING_DB` env var:
  - `postgres` — force Postgres.
  - `jsonl` — force the local `data/interactions.jsonl` file.
  - `auto` (default) — use Postgres if `POSTGRES_HOST` is set and reachable,
    otherwise fall back to JSONL. This means the app **always runs**, even with no
    database, and upgrades to full monitoring simply by pointing at a Postgres.
- **`app/db.py`** is the Postgres implementation (psycopg2). It creates the
  `conversations` table on first use (`CREATE TABLE IF NOT EXISTS`), so a fresh
  database — local or cloud — needs no manual migration. Each answer is one row;
  feedback updates that row's `feedback` column in place.
- **Grafana** is provisioned from code (no click-ops): a Postgres datasource
  (`monitoring/grafana/provisioning/datasources/postgres.yml`) and the dashboard
  (`monitoring/grafana/dashboards/valluvan.json`) are loaded automatically on
  startup.

## The dashboard (10 panels)

`Valluvan — Monitoring` includes well over the required 5 charts:

1. **Total conversations** (stat)
2. **Positive feedback %** (stat, thresholds red/yellow/green)
3. **Avg latency** (stat)
4. **Avg tokens / answer** (stat)
5. **Conversations over time** (bar time series)
6. **Token usage over time** (time series: total + avg)
7. **Average latency over time** (time series)
8. **Feedback breakdown** — positive / negative / none (donut)
9. **Retrieval mode distribution** (pie)
10. **Prompt variant usage** (bar) + **Recent questions** (table)

Together these cover **usage** (volume, modes, prompts), **cost/performance**
(latency, tokens), and **quality** (feedback) — the three things worth watching
for a RAG app.

## Run it locally

```bash
make monitoring-up   # start Postgres + Grafana (docker-compose)
make app             # run the UI; ask a few questions and rate them 👍/👎
make grafana         # prints the dashboard URL
```

Then open Grafana at <http://localhost:3000> (login `admin` / `admin`) and open
the **Valluvan — Monitoring** dashboard. For the UI to write to Postgres, make
sure `POSTGRES_HOST` is set in your `.env` (it is, by default, from
`.env.example`).

## Cloud deployment (Supabase / Neon)

Because the connection is entirely env-driven and SSL-capable, the **same code**
targets a managed cloud Postgres. To use **Supabase**:

1. Create a Supabase project and grab its database connection details.
2. In `.env` (or the host's secrets), set:
   ```
   POSTGRES_HOST=db.<project-ref>.supabase.co
   POSTGRES_PORT=5432
   POSTGRES_DB=postgres
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=<your-db-password>
   POSTGRES_SSLMODE=require
   ```
3. The `conversations` table is created automatically on the first logged answer.

For dashboards in the cloud, either run Grafana locally against Supabase, or use
**Grafana Cloud** (free tier) and add the Supabase Postgres as a data source —
the dashboard JSON in this repo can be imported as-is.

## Design notes

- **Feedback as an in-place column** (not an append-only event) keeps Grafana
  queries trivial; the JSONL fallback instead appends feedback events and folds
  them on read, matching the same public interface.
- **Graceful degradation** matters for a demo/peer review: a reviewer who just
  runs `make app` without Postgres still gets a working app (JSONL), while the
  full stack (`make up`) gives the complete monitoring experience.
