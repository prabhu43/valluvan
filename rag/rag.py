"""RAG answer generation for Valluvan.

Retrieves the most relevant kurals, builds a grounded prompt (Tamil verse +
English translation + explanation as context), and asks an LLM to answer the
user's life/ethics question while citing the kural numbers it used.

The LLM provider is swappable via env (LLM_PROVIDER): groq (default), openai,
or ollama. All three use the OpenAI-compatible client.
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from rag.search import search

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
# Best retrieval mode per eval/eval_retrieval.py: rerank (dense recall +
# cross-encoder re-rank) beats dense > hybrid > sparse on hit-rate and MRR.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "rerank")
# Optional best-practice: rewrite the user's raw question into a cleaner,
# theme-focused retrieval query before searching (see rag/query_rewrite.py).
# Evaluation showed it HURTS retrieval on specific questions, so it is OFF by
# default; enable only for very vague/messy free-text input.
REWRITE_QUERY = os.getenv("REWRITE_QUERY", "false").lower() in ("1", "true", "yes")

_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "openai": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "api_key_env": None,
    },
}

PROMPTS = {
    # Baseline: short and direct.
    "concise": (
        "You are Valluvan, a wise assistant grounded in the Thirukkural, the "
        "classic Tamil text of 1,330 couplets (kurals) by Thiruvalluvar. Answer "
        "the user's question using ONLY the kurals provided in the context. Give "
        "practical, compassionate guidance. For each point, cite the kural "
        "number(s) you rely on like (Kural 301). If the context does not address "
        "the question, say so honestly rather than inventing wisdom. Keep the "
        "answer concise and clear."
    ),
    # Structured: explicit sections + stronger grounding/citation discipline.
    "structured": (
        "You are Valluvan, a thoughtful guide whose wisdom comes entirely from "
        "the Thirukkural by Thiruvalluvar. You will receive a user's question and "
        "a set of retrieved kurals (couplets) as context.\n\n"
        "Rules:\n"
        "1. Use ONLY the provided kurals. Never invent verses, numbers, or "
        "wisdom not present in the context.\n"
        "2. Cite every claim with the kural number(s) it comes from, e.g. "
        "(Kural 301).\n"
        "3. If the context does not answer the question, say so honestly.\n\n"
        "Structure your reply as:\n"
        "- **Guidance:** 2-4 sentences of practical, compassionate advice.\n"
        "- **Grounded in:** a short bulleted list of the kurals you used, each as "
        "'Kural N — one-line gist'."
    ),
}
DEFAULT_PROMPT = os.getenv("PROMPT_VARIANT", "concise")


def _client() -> OpenAI:
    cfg = _PROVIDERS.get(LLM_PROVIDER)
    if cfg is None:
        raise ValueError(f"unknown LLM_PROVIDER {LLM_PROVIDER!r}")
    api_key = os.getenv(cfg["api_key_env"]) if cfg["api_key_env"] else "ollama"
    return OpenAI(api_key=api_key, base_url=cfg["base_url"])


def build_context(kurals: list[dict]) -> str:
    blocks = []
    for k in kurals:
        blocks.append(
            f"Kural {k['kural_no']} — Chapter: {k['adhigaram_en']} "
            f"({k['section_en']})\n"
            f"Tamil: {k['kural_ta']}\n"
            f"Translation: {k['translation_en']}\n"
            f"Meaning: {k['explanation_en']}"
        )
    return "\n\n".join(blocks)


def answer(
    question: str,
    mode: str = None,
    limit: int = 5,
    prompt: str = None,
    model: str = None,
    rewrite: bool = None,
) -> dict:
    """Return a grounded answer plus the retrieved kurals and telemetry."""
    mode = mode or RETRIEVAL_MODE
    prompt = prompt or DEFAULT_PROMPT
    model = model or LLM_MODEL
    rewrite = REWRITE_QUERY if rewrite is None else rewrite
    if prompt not in PROMPTS:
        raise ValueError(f"unknown prompt {prompt!r}; choose from {list(PROMPTS)}")

    search_query = question
    rewritten_query = None
    if rewrite:
        from rag.query_rewrite import rewrite_query

        rewritten_query = rewrite_query(question)
        search_query = rewritten_query
    kurals = search(search_query, mode=mode, limit=limit)
    context = build_context(kurals)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context (retrieved kurals):\n{context}\n\n"
        "Answer the question grounded in these kurals, citing kural numbers."
    )

    client = _client()
    start = time.time()
    resp = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PROMPTS[prompt]},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            break
        except Exception as e:  # noqa: BLE001 - retry transient/rate-limit errors
            if attempt == 4:
                raise
            wait = 2 ** attempt
            print(f"    LLM error ({e.__class__.__name__}); retry in {wait}s")
            time.sleep(wait)
    elapsed = time.time() - start

    usage = resp.usage
    return {
        "answer": resp.choices[0].message.content,
        "kurals": kurals,
        "context": context,
        "retrieval_mode": mode,
        "rewritten_query": rewritten_query,
        "prompt_variant": prompt,
        "model": model,
        "provider": LLM_PROVIDER,
        "latency_s": round(elapsed, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How can I control my anger?"
    out = answer(q)
    print(f"\nQ: {q}\n")
    print(out["answer"])
    print(
        f"\n[{out['provider']}/{out['model']} | {out['retrieval_mode']} | "
        f"prompt={out['prompt_variant']} | {out['latency_s']}s | "
        f"{out['total_tokens']} tokens]"
    )
    print("Retrieved:", ", ".join(f"#{k['kural_no']}" for k in out["kurals"]))
