"""Interaction & feedback storage for Valluvan.

This is the persistence *seam* used by the Streamlit app. Every answered
question is logged with its retrieval/LLM telemetry, and 👍/👎 feedback is
recorded against it. Records are needed by Phase 9 (Postgres + Grafana
monitoring).

For now this writes newline-delimited JSON to `data/interactions.jsonl`, which is
simple, dependency-free, and works without any database running. Phase 9 will add
a Postgres-backed implementation behind this same interface
(`log_interaction` / `log_feedback` / `load_interactions`) so the UI does not
change.
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_PATH = DATA_DIR / "interactions.jsonl"
_LOCK = threading.Lock()

# Telemetry fields copied from rag.answer()'s result dict.
_TELEMETRY_KEYS = (
    "retrieval_mode",
    "rewritten_query",
    "prompt_variant",
    "model",
    "provider",
    "latency_s",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_interaction(question: str, result: dict) -> str:
    """Persist one answered question. Returns a unique interaction id."""
    interaction_id = str(uuid.uuid4())
    record = {
        "id": interaction_id,
        "ts": _utc_now(),
        "question": question,
        "answer": result.get("answer"),
        "kural_nos": [k["kural_no"] for k in result.get("kurals", [])],
        "feedback": None,
    }
    for key in _TELEMETRY_KEYS:
        record[key] = result.get(key)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK, LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return interaction_id


def log_feedback(interaction_id: str, rating: int) -> None:
    """Attach 👍 (+1) / 👎 (-1) feedback to a previously logged interaction.

    Appended as a separate event so the JSONL stays append-only; the latest
    feedback event for an id wins when reading back.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "id": interaction_id,
        "ts": _utc_now(),
        "feedback": int(rating),
        "event": "feedback",
    }
    with _LOCK, LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_interactions() -> list[dict]:
    """Read all interactions back, applying the latest feedback per id."""
    if not LOG_PATH.exists():
        return []
    interactions: dict[str, dict] = {}
    with LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") == "feedback":
                if row["id"] in interactions:
                    interactions[row["id"]]["feedback"] = row["feedback"]
            else:
                interactions[row["id"]] = row
    return list(interactions.values())
