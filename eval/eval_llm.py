"""LLM evaluation for Valluvan: compare prompt variants with an LLM-as-judge.

For a sample of user questions we generate an answer with each prompt variant
(rag.rag.PROMPTS), then ask a judge LLM to score each answer on:

  - relevance    : does the answer address the question?
                   NON_RELEVANT (0) / PARTLY_RELEVANT (0.5) / RELEVANT (1)
  - groundedness : is every claim supported by the retrieved kurals, with correct
                   citations, and no invented wisdom?  (yes = 1 / no = 0)

We then aggregate mean scores per variant and pick the best. This satisfies the
course criterion "multiple approaches (prompts) are evaluated, and the best one
is used".

Usage:
  python -m eval.eval_llm                 # default sample
  EVAL_LLM_SAMPLE=30 python -m eval.eval_llm
"""

import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

from rag.rag import PROMPTS, _client, answer

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GT_PATH = DATA_DIR / "ground_truth.json"
OUT_PATH = DATA_DIR / "eval_llm_results.json"

SAMPLE = int(os.getenv("EVAL_LLM_SAMPLE", "25"))
SEED = int(os.getenv("SEED", "42"))
JUDGE_MODEL = os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
VARIANTS = list(PROMPTS.keys())

JUDGE_PROMPT = """You are a strict evaluator of a Thirukkural Q&A assistant.

You are given a user QUESTION, the CONTEXT (kurals retrieved for it), and an
ANSWER produced by the assistant. Judge the ANSWER on two axes:

1. relevance: does it address the question?
   - "RELEVANT"       : fully addresses it
   - "PARTLY_RELEVANT": partially addresses it
   - "NON_RELEVANT"   : does not address it
2. groundedness: is every claim supported by the CONTEXT kurals, with correct
   kural citations, and NO invented verses or wisdom?
   - "yes" or "no"

Return ONLY a JSON object:
{{"relevance": "...", "groundedness": "...", "reason": "one short sentence"}}

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}"""

REL_SCORE = {"RELEVANT": 1.0, "PARTLY_RELEVANT": 0.5, "NON_RELEVANT": 0.0}


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


def judge(client, question: str, context: str, ans: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=ans)
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return _parse_json(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            time.sleep(2 ** attempt)
            if attempt == 4:
                print(f"    judge error: {e}")
    return {"relevance": "NON_RELEVANT", "groundedness": "no", "reason": "judge failed"}


def main() -> None:
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    random.seed(SEED)
    questions = [r["question"] for r in random.sample(gt, min(SAMPLE, len(gt)))]
    print(
        f"LLM eval: {len(questions)} questions x {len(VARIANTS)} prompt variants, "
        f"judge={JUDGE_MODEL}\n"
    )

    client = _client()
    per_variant = {v: {"relevance": [], "grounded": [], "latency": []} for v in VARIANTS}
    records = []

    for qi, q in enumerate(questions, 1):
        for v in VARIANTS:
            out = answer(q, prompt=v)
            verdict = judge(client, q, out["context"], out["answer"])
            rel = REL_SCORE.get(str(verdict.get("relevance")).upper(), 0.0)
            grounded = 1.0 if str(verdict.get("groundedness")).lower() == "yes" else 0.0
            per_variant[v]["relevance"].append(rel)
            per_variant[v]["grounded"].append(grounded)
            per_variant[v]["latency"].append(out["latency_s"])
            records.append(
                {
                    "question": q,
                    "variant": v,
                    "relevance": verdict.get("relevance"),
                    "groundedness": verdict.get("groundedness"),
                    "reason": verdict.get("reason"),
                }
            )
        if qi % 5 == 0 or qi == len(questions):
            print(f"  [{qi}/{len(questions)}] questions judged")

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    summary = []
    for v in VARIANTS:
        d = per_variant[v]
        summary.append(
            {
                "variant": v,
                "relevance": mean(d["relevance"]),
                "groundedness": mean(d["grounded"]),
                "avg_latency_s": mean(d["latency"]),
                "combined": round((mean(d["relevance"]) + mean(d["grounded"])) / 2, 4),
            }
        )

    print(
        f"\n{'variant':<12} {'relevance':<11} {'grounded':<10} "
        f"{'combined':<10} {'lat(s)':<8}"
    )
    print("-" * 52)
    for s in sorted(summary, key=lambda x: x["combined"], reverse=True):
        print(
            f"{s['variant']:<12} {s['relevance']:<11} {s['groundedness']:<10} "
            f"{s['combined']:<10} {s['avg_latency_s']:<8}"
        )

    best = max(summary, key=lambda x: x["combined"])
    print(f"\nBest prompt variant: {best['variant'].upper()}")

    OUT_PATH.write_text(
        json.dumps(
            {
                "judge_model": JUDGE_MODEL,
                "n_questions": len(questions),
                "summary": summary,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
