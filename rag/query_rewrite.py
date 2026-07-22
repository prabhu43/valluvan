"""LLM-based user-query rewriting for Valluvan.

Real users ask messy, first-person, emotionally-loaded questions
("my ex still texts me even though we fought, why?"). Retrieval works better on
a clean, keyword-rich, thematic query ("anger and love in relationships;
forgiveness after conflict"). This module asks the LLM to rewrite the raw
question into a concise retrieval query — expanding the underlying *theme* the
Thirukkural would speak to, without answering it.

It reuses the same OpenAI-compatible provider config as rag.rag (Groq default).
On any failure it falls back to the original question, so retrieval never breaks.

See docs/retrieval-evaluation.md for the measured effect.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
# A tiny/cheap model is plenty for a one-line rewrite; default to the fast one.
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "llama-3.1-8b-instant")

_PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),
    "ollama": (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), None),
}

_SYSTEM = (
    "You rewrite a user's personal question into a short search query for "
    "retrieving relevant couplets from the Thirukkural, a classic Tamil text on "
    "virtue, wealth, and love. Identify the underlying ethical/emotional theme "
    "and express it as a concise, keyword-rich query (3-12 words). Do NOT answer "
    "the question, add commentary, or mention kural numbers. Return ONLY the "
    "rewritten query."
)


def _client() -> OpenAI:
    base_url, key_env = _PROVIDERS[LLM_PROVIDER]
    api_key = os.getenv(key_env) if key_env else "ollama"
    return OpenAI(api_key=api_key, base_url=base_url)


def rewrite_query(question: str, model: str = None) -> str:
    """Rewrite `question` into a retrieval-friendly query.

    Falls back to the original question on any error.
    """
    model = model or REWRITE_MODEL
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=40,
        )
        rewritten = resp.choices[0].message.content.strip().strip('"')
        return rewritten or question
    except Exception:  # noqa: BLE001 - never let rewriting break retrieval
        return question


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "my ex still texts me even though we fought, why?"
    print(f"original : {q}")
    print(f"rewritten: {rewrite_query(q)}")
