#!/usr/bin/env python3
"""
Token efficiency against FAIR baselines (not just "re-send everything").

The headline 66% in analyze.py is Korely's block vs a memory-less agent that
re-sends the FULL conversation every turn. That is a deliberately naive baseline
(README says so). A real memory-less agent would keep a recency WINDOW instead.

This script answers the honest follow-up: how much does Korely's selected block
save against an equal-budget recency window? It reuses analyze.py's exact prompt
template and tokenizer, so the only thing that changes is the memory strategy.

Deterministic, no API key, no LLM. Reads the same published transcripts + the
public LongMemEval dataset.

  python3 baselines.py --transcripts "data/*.jsonl"

Baselines compared (each is the {CONTEXT} a memory-less agent would carry):
  full history          every turn, chronological            (the published 66% baseline)
  window @2000 tok      most recent ~2000 tokens             (= Korely's soft budget)
  window @4000 tok      most recent ~4000 tokens             (2x Korely's budget)
  window @matched       most recent tokens == Korely's block (per-question equal budget)
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics

from analyze import ENC, ntok, prompt_tokens, full_history, load_dataset


def window_last_tokens(text: str, n: int) -> str:
    """The most recent n tokens of the chronological history (a recency window)."""
    toks = ENC.encode(text)
    if len(toks) <= n:
        return text
    return ENC.decode(toks[-n:])


def pooled_and_median(kor_tokens: list[int], base_tokens: list[int]) -> dict:
    pooled = (1 - sum(kor_tokens) / sum(base_tokens)) * 100
    per_q = [(1 - k / b) * 100 for k, b in zip(kor_tokens, base_tokens) if b]
    costs_more = sum(1 for k, b in zip(kor_tokens, base_tokens) if k >= b)
    return {
        "pooled_reduction_pct": round(pooled, 1),
        "median_reduction_pct": round(statistics.median(per_q), 1),
        "mean_reduction_pct": round(statistics.mean(per_q), 1),
        "korely_costs_more": f"{costs_more}/{len(kor_tokens)}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="data/*.jsonl")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    by_qid = load_dataset(args.dataset)

    kor, full, w2000, w4000, wmatched = [], [], [], [], []
    seen = set()
    for f in sorted(glob.glob(args.transcripts)):
        for line in open(f):
            r = json.loads(line)
            if r.get("system") != "korely" or not r.get("use_hint"):
                continue
            qid = r.get("qid")
            if qid in seen:
                continue
            item = by_qid.get(qid)
            ctx = r.get("retrieved_context") or ""
            if item is None or not ctx.strip():
                continue
            seen.add(qid)

            q = item["question"]
            hist = full_history(item)
            block_n = ntok(ctx)

            kor.append(prompt_tokens(ctx, q))
            full.append(prompt_tokens(hist, q))
            w2000.append(prompt_tokens(window_last_tokens(hist, 2000), q))
            w4000.append(prompt_tokens(window_last_tokens(hist, 4000), q))
            wmatched.append(prompt_tokens(window_last_tokens(hist, block_n), q))

    baselines = {
        "full history (the published 66% baseline)": full,
        "recency window @2000 tok (= Korely budget)": w2000,
        "recency window @4000 tok (2x budget)": w4000,
        "recency window @matched (per-q equal budget)": wmatched,
    }

    print(f"\nN = {len(kor)} questions · Korely block avg = {round(statistics.mean(kor))} prompt tokens\n")
    print(f"{'baseline':46} {'pooled':>8} {'median':>8} {'mean':>7} {'Korely costs >=':>16}")
    print("-" * 90)
    out = {}
    for name, base in baselines.items():
        m = pooled_and_median(kor, base)
        out[name] = m
        print(f"{name:46} {m['pooled_reduction_pct']:>7}% {m['median_reduction_pct']:>7}% "
              f"{m['mean_reduction_pct']:>6}% {m['korely_costs_more']:>16}")
    print("-" * 90)
    print("Reading: positive = Korely uses fewer tokens than that baseline.\n"
          "Against an EQUAL-budget recency window the token saving vanishes — the\n"
          "66% is 'bounded vs unbounded', not 'Korely vs a sensible memory-less agent'.\n"
          "The real question is whether the SELECTED block ANSWERS better at equal\n"
          "budget (accuracy), measured separately in accuracy.py.")

    import os
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "baselines.json"), "w") as fh:
        json.dump({"n": len(kor), "korely_block_avg_prompt_tokens": round(statistics.mean(kor)),
                   "baselines": out}, fh, indent=2)


if __name__ == "__main__":
    main()
