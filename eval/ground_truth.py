"""Generate a retrieval ground-truth dataset for Valluvan.

For each sampled kural we ask the LLM to invent a few realistic user questions
whose answer is that specific kural. This gives us (question -> kural_no) gold
pairs to measure retrieval quality (hit-rate, MRR) in eval/eval_retrieval.py.

The run is incremental and resumable: results are appended to
data/ground_truth.json and already-processed kurals are skipped. It is also
rate-limit tolerant (retries with backoff on Groq 429s).

Usage:
  python -m eval.ground_truth                # default sample
  SAMPLE_SIZE=300 QUESTIONS_PER_KURAL=3 python -m eval.ground_truth
  SAMPLE_SIZE=0 python -m eval.ground_truth  # 0 = ALL 1330 kurals
"""

import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KURALS_PATH = DATA_DIR / "thirukkural.json"
OUT_PATH = DATA_DIR / "ground_truth.json"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv("GROUND_TRUTH_MODEL", os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "200"))  # 0 == all kurals
QUESTIONS_PER_KURAL = int(os.getenv("QUESTIONS_PER_KURAL", "3"))
SEED = int(os.getenv("SEED", "42"))

_PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),
    "ollama": (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), None),
}

PROMPT = """You are helping build an evaluation set for a Thirukkural search engine.

Given ONE kural (couplet) and its meaning, write {n} short, natural questions a
real person might ask that this specific kural answers. Vary the phrasing: some
direct, some about a life situation. Do NOT mention the kural number or the word
"kural". Do NOT quote the verse. Return ONLY a JSON array of {n} strings.

Chapter: {chapter} ({section})
Verse (English): {translation}
Meaning: {explanation}"""


def _client() -> OpenAI:
    base_url, key_env = _PROVIDERS[LLM_PROVIDER]
    api_key = os.getenv(key_env) if key_env else "ollama"
    return OpenAI(api_key=api_key, base_url=base_url)


def _parse_questions(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    data = json.loads(text)
    return [str(q).strip() for q in data if str(q).strip()]


def generate_for_kural(client: OpenAI, k: dict) -> list[str]:
    prompt = PROMPT.format(
        n=QUESTIONS_PER_KURAL,
        chapter=k["adhigaram_en"],
        section=k["section_en"],
        translation=k["translation_en"].replace("\n", " "),
        explanation=k["explanation_en"],
    )
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return _parse_questions(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001 - retry on rate limit / transient
            wait = 2 ** attempt
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "timeout" in msg:
                print(f"    rate/transient error, retry in {wait}s ...")
                time.sleep(wait)
            else:
                print(f"    error: {e}")
                time.sleep(wait)
    return []


def load_existing() -> tuple[list[dict], set[int]]:
    if OUT_PATH.exists():
        rows = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        done = {r["kural_no"] for r in rows}
        return rows, done
    return [], set()


def main() -> None:
    kurals = json.loads(KURALS_PATH.read_text(encoding="utf-8"))
    if SAMPLE_SIZE and SAMPLE_SIZE < len(kurals):
        random.seed(SEED)
        kurals = random.sample(kurals, SAMPLE_SIZE)

    rows, done = load_existing()
    todo = [k for k in kurals if k["kural_no"] not in done]
    print(
        f"Ground truth: {len(kurals)} sampled kurals, {len(done)} already done, "
        f"{len(todo)} to generate ({QUESTIONS_PER_KURAL} q each) via "
        f"{LLM_PROVIDER}/{LLM_MODEL}"
    )

    client = _client()
    for i, k in enumerate(todo, 1):
        questions = generate_for_kural(client, k)
        for q in questions:
            rows.append({"question": q, "kural_no": k["kural_no"]})
        OUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        if i % 10 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] kural #{k['kural_no']}: +{len(questions)} q "
                  f"(total pairs: {len(rows)})")

    print(f"\nDone. {len(rows)} (question -> kural_no) pairs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
