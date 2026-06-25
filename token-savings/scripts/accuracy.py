#!/usr/bin/env python3
"""
Accuracy at EQUAL token budget (the question the 66% leaves open).

The token benchmark shows Korely's block is bounded (~1900 tok) vs an unbounded
full history. But at an *equal* token budget a dumb recency window costs the same
(see baselines.py). So bounding context is not the contribution — *selecting* it
is. This script measures the thing that actually matters:

  For each LongMemEval question, a reader LLM answers TWICE, same prompt, same
  token budget, only the memory strategy differs:
    (a) Korely's get_context() block   (semantic selection)
    (b) the most-recent turns that fit in the SAME number of tokens (recency)
  A neutral judge (LongMemEval-style, question-type aware) scores each answer
  against the gold. We report accuracy(a) vs accuracy(b).

The reader and judge are the SAME model for both conditions, so any model bias
cancels in the DELTA — the comparison is robust to the exact judge.

  python3 accuracy.py --limit 10        # smoke
  python3 accuracy.py                    # full 178

No private harness: public dataset + the committed Korely blocks. Uses Gemini
(reads GEMINI_API_KEY / GOOGLE_API_KEY from env or .env).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import ntok, full_history, load_dataset, PROMPT_PRE, MIN_TURN_CHARS  # noqa: E402

import google.generativeai as genai  # noqa: E402


def _history_turns(item):
    """Chronological 'role: content' turns, same filtering as analyze.full_history."""
    sessions = item["haystack_sessions"]
    dates = item.get("haystack_dates") or [""] * len(sessions)
    turns = []
    for i in sorted(range(len(sessions)), key=lambda i: dates[i]):
        for t in sessions[i]:
            c = (t.get("content") or "").strip()
            if len(c) >= MIN_TURN_CHARS:
                turns.append(f"{t.get('role', 'user')}: {c}")
    return turns


def recent_turns_within(item, budget):
    """The fairest recency baseline: most recent COMPLETE turns that fit in the
    same token budget as Korely's block (never starts mid-turn)."""
    turns = _history_turns(item)
    out, used = [], 0
    for line in reversed(turns):
        n = ntok(line) + 1
        if out and used + n > budget:
            break
        out.insert(0, line)
        used += n
    return "\n".join(out)


def _load_key() -> str:
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(k):
            return os.environ[k]
    for env in (".env", "../.env", os.path.expanduser("~/Code/korely-graphrag/.env")):
        if os.path.exists(env):
            for line in open(env):
                m = re.match(r"\s*(GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*(.+)", line)
                if m:
                    return m.group(2).strip().strip('"').strip("'")
    sys.exit("Set GEMINI_API_KEY")


# gemini-2.5-flash is a thinking model: reasoning tokens count against
# max_output_tokens, so a tiny budget returns empty text. Give room for both.
READER_TEMP = {"temperature": 0.0, "max_output_tokens": 512}
JUDGE_TEMP = {"temperature": 0.0, "max_output_tokens": 512}

TYPE_LINE = {
    "knowledge-update": "- This fact changed over time; the answer must match the MOST RECENT value in the ground truth.",
    "temporal-reasoning": "- The answer must be temporally accurate (dates, durations, ordering).",
    "single-session-preference": "- This is a user preference; accept any answer that captures the preference, even loosely.",
}

JUDGE_NORMAL = """You are a strict grader. Decide if the MODEL ANSWER correctly answers the QUESTION, using the GROUND TRUTH as the reference.

QUESTION: {q}
GROUND TRUTH: {gold}
MODEL ANSWER: {ans}

Rules:
- CORRECT if the model answer conveys the ground-truth information, even if phrased differently or with extra correct detail.
- INCORRECT if it contradicts the ground truth, omits the key fact, gives a different value, or says it doesn't know.
{type_line}
Respond with exactly one word: CORRECT or INCORRECT."""

JUDGE_ABS = """You are a strict grader for an ABSTENTION question: the necessary information is NOT in the user's history, so the only correct behavior is to abstain.

QUESTION: {q}
GROUND TRUTH (why it is unanswerable): {gold}
MODEL ANSWER: {ans}

The MODEL ANSWER is CORRECT only if it abstains — says it doesn't know, the info isn't available, or there's no record. It is INCORRECT if it states a specific factual answer (a number, name, place, date) as if the info were available.
Respond with exactly one word: CORRECT or INCORRECT."""


def _call(model, prompt, cfg, retries=5):
    for i in range(retries):
        try:
            r = model.generate_content(prompt, generation_config=cfg)
            return (r.text or "").strip()
        except Exception as e:
            if i == retries - 1:
                return ""
            time.sleep(2 ** i + 0.5)
    return ""


def reader(model, context, question):
    return _call(model, PROMPT_PRE + context + f"\n\nQuestion: {question}\nAnswer:", READER_TEMP)


def judge(model, q, gold, ans, qtype, is_abs):
    if not ans.strip():
        return False  # empty/blocked answer is never correct
    if is_abs:
        prompt = JUDGE_ABS.format(q=q, gold=gold, ans=ans)
    else:
        prompt = JUDGE_NORMAL.format(q=q, gold=gold, ans=ans, type_line=TYPE_LINE.get(qtype, ""))
    v = _call(model, prompt, JUDGE_TEMP).upper()
    # take the FINAL verdict word (the model may reason before answering)
    matches = re.findall(r"\bINCORRECT\b|\bCORRECT\b", v)
    return bool(matches) and matches[-1] == "CORRECT"


def process(item, ctx, model_name):
    """One question, both conditions. Returns a result row."""
    rm = genai.GenerativeModel(model_name)
    q, gold, qtype = item["question"], item["_gold"], item["question_type"]
    is_abs = str(item["_qid"]).endswith("_abs")
    block_n = ntok(ctx)
    window = recent_turns_within(item, block_n)

    ans_k = reader(rm, ctx, q)
    ans_w = reader(rm, window, q)
    corr_k = judge(rm, q, gold, ans_k, qtype, is_abs)
    corr_w = judge(rm, q, gold, ans_w, qtype, is_abs)
    return {
        "qid": item["_qid"], "axis": qtype, "is_abstention": is_abs,
        "gold": gold, "korely_answer": ans_k, "window_answer": ans_w,
        "korely_correct": corr_k, "window_correct": corr_w,
        "budget_tokens": block_n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="data/*.jsonl")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    genai.configure(api_key=_load_key())
    by_qid = load_dataset(args.dataset)

    work = []
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
            item = dict(item, _qid=qid, _gold=r.get("gold"))
            work.append((item, ctx))
    if args.limit:
        work = work[:args.limit]

    print(f"Scoring {len(work)} questions x 2 conditions with {args.model} "
          f"({args.workers} workers)...", flush=True)
    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, it, ctx, args.model): it["_qid"] for it, ctx in work}
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 10 == 0 or n == len(work):
                print(f"  {n}/{len(work)}  ({time.time()-t0:.0f}s)", flush=True)

    # aggregate
    per_axis = defaultdict(lambda: {"n": 0, "k": 0, "w": 0})
    for r in rows:
        a = per_axis[r["axis"]]
        a["n"] += 1
        a["k"] += int(r["korely_correct"])
        a["w"] += int(r["window_correct"])
    tot = {"n": len(rows), "k": sum(r["korely_correct"] for r in rows),
           "w": sum(r["window_correct"] for r in rows)}

    print(f"\n{'axis':28} {'N':>4} {'Korely':>8} {'window':>8} {'delta':>7}")
    print("-" * 60)
    summ = {"model": args.model, "n": tot["n"], "axes": {}, "overall": {}}
    for axis in sorted(per_axis):
        a = per_axis[axis]
        ka, wa = 100*a["k"]/a["n"], 100*a["w"]/a["n"]
        summ["axes"][axis] = {"n": a["n"], "korely_acc": round(ka, 1),
                              "window_acc": round(wa, 1), "delta": round(ka-wa, 1)}
        print(f"{axis:28} {a['n']:>4} {ka:>7.1f}% {wa:>7.1f}% {ka-wa:>+6.1f}")
    ka, wa = 100*tot["k"]/tot["n"], 100*tot["w"]/tot["n"]
    summ["overall"] = {"n": tot["n"], "korely_acc": round(ka, 1),
                       "window_acc": round(wa, 1), "delta": round(ka-wa, 1)}
    print("-" * 60)
    print(f"{'ALL':28} {tot['n']:>4} {ka:>7.1f}% {wa:>7.1f}% {ka-wa:>+6.1f}")
    print(f"\nKorely block: {ka:.1f}% correct · equal-budget window: {wa:.1f}% · "
          f"delta {ka-wa:+.1f} pts at the SAME token budget.")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "accuracy.jsonl"), "w") as fh:
        for r in sorted(rows, key=lambda r: (r["axis"], r["qid"])):
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "accuracy_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=2)
    print(f"Wrote {args.out}/accuracy.jsonl + accuracy_summary.json")


if __name__ == "__main__":
    main()
