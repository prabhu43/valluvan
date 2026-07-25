"""Evaluate Valluvan's retrieval quality across modes.

Uses the (question -> kural_no) ground-truth pairs from data/ground_truth.json.
For each question we run each retrieval mode and check where the correct kural
appears in the top-k results, then report:

  - hit-rate@k : fraction of questions whose gold kural is in the top-k
  - MRR@k      : mean reciprocal rank of the gold kural (0 if not in top-k)

Modes evaluated (all local, no API cost):
  dense, sparse, hybrid, rerank (dense recall -> cross-encoder re-rank)

Optionally (EVAL_REWRITE=true) it also measures LLM query rewriting on a sample,
comparing dense/rerank on the raw vs rewritten question. Rewriting needs LLM
calls, so it is sampled and cached to data/rewrite_cache.json.

Usage:  python -m eval.eval_retrieval                 # k=5, retrieval modes only
        EVAL_K=10 python -m eval.eval_retrieval
        EVAL_REWRITE=true EVAL_REWRITE_SAMPLE=60 python -m eval.eval_retrieval
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from rag.search import SEARCHERS

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GT_PATH = DATA_DIR / "ground_truth.json"
OUT_PATH = DATA_DIR / "eval_retrieval_results.json"
REWRITE_CACHE = DATA_DIR / "rewrite_cache.json"
K = int(os.getenv("EVAL_K", "5"))
MODES = ["dense", "sparse", "hybrid", "rerank"]

EVAL_REWRITE = os.getenv("EVAL_REWRITE", "false").lower() in ("1", "true", "yes")
REWRITE_SAMPLE = int(os.getenv("EVAL_REWRITE_SAMPLE", "60"))


def _metrics(mode: str, rows: list[dict], k: int, kural_chapter: dict) -> dict:
    """Compute exact- and chapter-level hit-rate/MRR for `rows`.

    Each row must have a "query" (what to search) and "kural_no" (gold answer).
    Two granularities are reported:
      - exact  : the gold kural itself appears in top-k
      - chapter: any kural from the gold kural's adhigaram appears in top-k
                 (a sibling verse from the right chapter is still useful)
    """
    searcher = SEARCHERS[mode]
    hits = reciprocal_ranks = chapter_hits = chapter_rr = 0.0
    for row in rows:
        gold = row["kural_no"]
        gold_chapter = kural_chapter[gold]
        results = searcher(row["query"], limit=k)

        rank = next(
            (i for i, r in enumerate(results, start=1) if r.get("kural_no") == gold),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_ranks += 1.0 / rank

        ch_rank = next(
            (
                i
                for i, r in enumerate(results, start=1)
                if r.get("kural_no") in kural_chapter
                and kural_chapter[r["kural_no"]] == gold_chapter
            ),
            None,
        )
        if ch_rank is not None:
            chapter_hits += 1
            chapter_rr += 1.0 / ch_rank
    n = len(rows)
    return {
        "n": n,
        "hit_rate": round(hits / n, 4),
        "mrr": round(reciprocal_ranks / n, 4),
        "chapter_hit_rate": round(chapter_hits / n, 4),
        "chapter_mrr": round(chapter_rr / n, 4),
    }


def _print_table(title: str, rows: list[dict], k: int) -> None:
    print(f"\n{title}")
    print(
        f"{'variant':<18} {'hit@'+str(k):<9} {'MRR@'+str(k):<9} "
        f"{'chapHit@'+str(k):<11} {'chapMRR@'+str(k):<10}"
    )
    print("-" * 58)
    for r in sorted(rows, key=lambda x: x["mrr"], reverse=True):
        print(
            f"{r['variant']:<18} {r['hit_rate']:<9} {r['mrr']:<9} "
            f"{r['chapter_hit_rate']:<11} {r['chapter_mrr']:<10}"
        )


def _load_rewrites(sample: list[dict]) -> dict:
    """Rewrite the sampled questions (cached & resumable)."""
    from rag.query_rewrite import rewrite_query

    cache = {}
    if REWRITE_CACHE.exists():
        cache = json.loads(REWRITE_CACHE.read_text(encoding="utf-8"))
    changed = False
    for i, row in enumerate(sample, start=1):
        q = row["question"]
        if q not in cache:
            cache[q] = rewrite_query(q)
            changed = True
            if i % 10 == 0:
                print(f"  rewritten {i}/{len(sample)}")
                REWRITE_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    if changed:
        REWRITE_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache


def main() -> None:
    if not GT_PATH.exists():
        raise SystemExit(
            "Ground truth not found. Run `python -m eval.ground_truth` first."
        )
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    kurals = json.loads((DATA_DIR / "thirukkural.json").read_text(encoding="utf-8"))
    kural_chapter = {k["kural_no"]: k["adhigaram_no"] for k in kurals}
    print(f"Evaluating {len(MODES)} modes on {len(gt)} questions (k={K}) ...")

    rows = [{"query": r["question"], "kural_no": r["kural_no"]} for r in gt]
    retrieval = []
    for m in MODES:
        res = _metrics(m, rows, K, kural_chapter)
        res["variant"] = m
        retrieval.append(res)
    _print_table("Retrieval modes (full ground truth):", retrieval, K)
    best = max(retrieval, key=lambda x: x["mrr"])
    print(f"\nBest mode by MRR@{K}: {best['variant'].upper()}")

    payload = {"k": K, "n": len(gt), "results": retrieval}

    if EVAL_REWRITE:
        sample = gt[:REWRITE_SAMPLE]
        print(
            f"\nQuery rewriting on a {len(sample)}-question sample "
            f"(cached -> {REWRITE_CACHE.name}) ..."
        )
        rewrites = _load_rewrites(sample)
        raw_rows = [{"query": r["question"], "kural_no": r["kural_no"]} for r in sample]
        rw_rows = [
            {"query": rewrites[r["question"]], "kural_no": r["kural_no"]}
            for r in sample
        ]
        rewrite_results = []
        for base in ("dense", "rerank"):
            a = _metrics(base, raw_rows, K, kural_chapter)
            a["variant"] = f"{base} (raw)"
            b = _metrics(base, rw_rows, K, kural_chapter)
            b["variant"] = f"{base} (rewritten)"
            rewrite_results += [a, b]
        _print_table(
            f"Query rewriting (sample n={len(sample)}):", rewrite_results, K
        )
        payload["rewrite"] = {"sample": len(sample), "results": rewrite_results}

    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
