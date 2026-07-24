"""Postgres-backed storage for Valluvan interactions & feedback (Phase 9).

Implements the same interface as the JSONL fallback in app/storage.py
(`init`, `log_interaction`, `log_feedback`, `load_interactions`) so the Streamlit
UI is backend-agnostic. Every answered question is one row in `conversations`,
with a nullable `feedback` column updated in place by 👍/👎.

Connection is fully env-driven, so the SAME code targets a local docker Postgres
or a managed cloud Postgres (e.g. Supabase / Neon):

  POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
  POSTGRES_SSLMODE   # "prefer" locally; set "require" for Supabase/Neon

Tables are created on first use (CREATE TABLE IF NOT EXISTS), so pointing at a
fresh cloud database needs no manual migration step.
"""

import os
import uuid
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id               UUID PRIMARY KEY,
    ts               TIMESTAMPTZ NOT NULL,
    question         TEXT,
    answer           TEXT,
    kural_nos        INTEGER[],
    retrieval_mode   TEXT,
    rewritten_query  TEXT,
    prompt_variant   TEXT,
    model            TEXT,
    provider         TEXT,
    latency_s        REAL,
    prompt_tokens    INTEGER,
    completion_tokens INTEGER,
    total_tokens     INTEGER,
    feedback         SMALLINT,
    feedback_ts      TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS conversations_ts_idx ON conversations (ts);
"""


def _conn_kwargs() -> dict:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.getenv("POSTGRES_DB", "valluvan"),
        "user": os.getenv("POSTGRES_USER", "valluvan"),
        "password": os.getenv("POSTGRES_PASSWORD", "valluvan"),
        "sslmode": os.getenv("POSTGRES_SSLMODE", "prefer"),
        "connect_timeout": int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5")),
    }


def _connect():
    return psycopg2.connect(**_conn_kwargs())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def init() -> None:
    """Create tables if needed. Raises if the database is unreachable."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(_SCHEMA)


def log_interaction(question: str, result: dict) -> str:
    interaction_id = str(uuid.uuid4())
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversations (
                id, ts, question, answer, kural_nos, retrieval_mode,
                rewritten_query, prompt_variant, model, provider, latency_s,
                prompt_tokens, completion_tokens, total_tokens
            ) VALUES (
                %(id)s, %(ts)s, %(question)s, %(answer)s, %(kural_nos)s,
                %(retrieval_mode)s, %(rewritten_query)s, %(prompt_variant)s,
                %(model)s, %(provider)s, %(latency_s)s, %(prompt_tokens)s,
                %(completion_tokens)s, %(total_tokens)s
            )
            """,
            {
                "id": interaction_id,
                "ts": _utc_now(),
                "question": question,
                "answer": result.get("answer"),
                "kural_nos": [k["kural_no"] for k in result.get("kurals", [])],
                "retrieval_mode": result.get("retrieval_mode"),
                "rewritten_query": result.get("rewritten_query"),
                "prompt_variant": result.get("prompt_variant"),
                "model": result.get("model"),
                "provider": result.get("provider"),
                "latency_s": result.get("latency_s"),
                "prompt_tokens": result.get("prompt_tokens"),
                "completion_tokens": result.get("completion_tokens"),
                "total_tokens": result.get("total_tokens"),
            },
        )
    return interaction_id


def log_feedback(interaction_id: str, rating: int) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE conversations SET feedback = %s, feedback_ts = %s WHERE id = %s",
            (int(rating), _utc_now(), interaction_id),
        )


def load_interactions() -> list[dict]:
    with _connect() as conn, conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute("SELECT * FROM conversations ORDER BY ts")
        return [dict(row) for row in cur.fetchall()]
