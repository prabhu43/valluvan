# LLM Evaluation

This document records how Valluvan's **answer-generation** step was evaluated,
the results, and the decision. It corresponds to **Phase 6** and the course
criterion *"multiple approaches (prompts) are evaluated, and the best one is
used"*.

While [retrieval-evaluation.md](retrieval-evaluation.md) asks *"did we fetch the
right kurals?"*, this document asks *"given the right kurals, did the LLM write a
good, grounded answer?"*.

## What we compared

Two system-prompt variants defined in `rag/rag.py` (`PROMPTS`):

- **`concise`** — a short, direct instruction: answer using only the provided
  kurals, cite kural numbers, be brief and clear.
- **`structured`** — a longer prompt with explicit rules and a required output
  layout (`**Guidance:**` paragraph + `**Grounded in:**` bulleted kural list).

Both use the same retrieval (`dense`, the Phase 5 winner) and the same model, so
the only variable is the prompt.

## Method: LLM-as-a-judge (`eval/eval_llm.py`)

1. Sample **25 questions** (seeded) from the ground-truth set.
2. For each question, generate an answer with **each** prompt variant.
3. A **judge LLM** scores every answer on two axes:
   - **relevance** — does it address the question?
     `RELEVANT` (1.0) / `PARTLY_RELEVANT` (0.5) / `NON_RELEVANT` (0.0)
   - **groundedness** — is every claim supported by the retrieved kurals, with
     correct citations and **no invented wisdom**? `yes` (1.0) / `no` (0.0)
4. Aggregate the mean score per variant; `combined = (relevance + grounded) / 2`.

The judge returns a strict JSON verdict with a one-line reason, run at
`temperature = 0` for consistency. Results are saved to
`data/eval_llm_results.json`.

## Results

25 questions per variant. Judge: `llama-3.1-8b-instant`.

| variant        | relevance | groundedness | combined | avg latency |
|----------------|:---------:|:------------:|:--------:|:-----------:|
| **concise** 🏆 | 0.74      | 0.32         | **0.53** | 9.4 s       |
| structured     | 0.68      | 0.20         | 0.44     | 10.3 s      |

(Reproduce with `make eval-llm`; raw verdicts in `data/eval_llm_results.json`.)

## Findings

1. **`concise` wins on both axes.** It is more relevant *and* more grounded than
   `structured`, and slightly faster.
2. **The `structured` prompt hurt groundedness.** Forcing a "Guidance" narrative
   section nudged the model to *elaborate* and add interpretation beyond the
   literal kurals — exactly what the judge penalizes. The simpler prompt stays
   closer to the source.
3. **Groundedness scores are conservative.** With `llama-3.1-8b-instant` as *both*
   generator and judge, the judge frequently flagged answers for "inventing
   interpretation." Representative judge reasons:
   - *"The answer invents a new interpretation of the kurals ... which is not
     present in the context."*
   - *"Accurately addresses the question and is supported by the provided kurals
     with correct citations."* (a `yes`)
   A stronger generator (e.g. `llama-3.3-70b-versatile`) produced fully grounded
   answers in spot checks, so absolute groundedness will rise with a larger model;
   the **relative ranking between prompts is what this experiment establishes.**

## Decision

**Use the `concise` prompt variant** as the default.

- Configured via the `PROMPT_VARIANT` environment variable (default: `concise`),
  consumed by `rag/rag.py`.
- Both variants remain available for future re-evaluation.

## Notes & reproducibility

- **Model quota:** the Groq free tier caps `llama-3.3-70b-versatile` at 100k
  tokens/day, which the ground-truth generation exhausts. The LLM evaluation
  therefore uses `llama-3.1-8b-instant` (much larger daily allowance) for both
  generation and judging. Override with `LLM_MODEL` / `JUDGE_MODEL` env vars.
- **Answer generation** retries with exponential backoff on transient/rate-limit
  errors (`rag/rag.py`).

## Future work

- Re-run the judge with a stronger model (e.g. GPT-4o-mini or Llama 70B) once
  quota allows, to get higher-fidelity absolute groundedness scores.
- Add a third prompt variant (e.g. few-shot with an example answer).
- Evaluate across multiple **generator models** (8B vs 70B) in addition to
  prompts.
