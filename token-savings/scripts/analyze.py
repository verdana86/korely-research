#!/usr/bin/env python3
"""
Token-efficiency analysis on LongMemEval (oracle split).

For every question, it compares the input tokens a reader LLM must process to
answer, in two conditions, over the IDENTICAL question + answer prompt template:

  full history : the whole conversation (every turn) is placed in the prompt
                 (what a memory-less agent carries).
  Korely       : only Korely's get_context() block is placed in the prompt
                 (the server-side, contradiction-resolved memory block).

It also recomputes recall@k (judge-free): did Korely's block still contain the
gold answer-evidence so the question stays answerable? Recall is measured with
the SAME definition the harness uses (a 48-char normalised prefix of any gold
evidence turn appears as a substring of the retrieved block).

Everything here is deterministic and needs NO API key and NO LLM call: it reads
the published run transcripts (the Korely block was logged at run time) and the
public LongMemEval dataset, and counts tokens with a standard tokenizer.

Tokenizer: tiktoken o200k_base (the GPT-4o family BPE). The reader used in the
runs is Llama-3.3-70B, whose tokenizer differs slightly per token, but the
ratio (the headline) is stable across tokenizers. We tokenize BOTH conditions
with the same tokenizer so the comparison is apples-to-apples.

Usage:
  pip install tiktoken            # + huggingface_hub if you let it fetch the dataset
  python analyze.py \
      --transcripts "data/*.jsonl" \
      --dataset <path-to longmemeval_oracle.json | auto> \
      --out results/

Outputs: results/per_question.jsonl and results/summary.json (+ a printed table).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict

import tiktoken

ENC = tiktoken.get_encoding("o200k_base")

# Reader prompt template, verbatim from the harness (bench/judge.py:306-311),
# with the recency hint included (use_hint=True variant). Only `context` and
# `question` change; everything else is identical between the two conditions.
PROMPT_PRE = (
    "Answer the question using ONLY the memory context below. Give a short, direct answer. "
    "If a fact changed over time, answer with the MOST RECENT value. "
    'If the answer is not in the context, reply exactly "I don\'t know".\n\n'
    "Memory context:\n"
)

MIN_TURN_CHARS = 12   # harness ingest noise floor (bench/dataset.py / recall.py)
SIG_CHARS = 48        # recall signature length (bench/recall.py)
_WS = re.compile(r"\s+")


def ntok(text: str) -> int:
    return len(ENC.encode(text))


def norm(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def prompt_tokens(context: str, question: str) -> int:
    return ntok(PROMPT_PRE + context + f"\n\nQuestion: {question}\nAnswer:")


def full_history(item: dict) -> str:
    """The whole conversation, chronological, as a memory-less agent would carry it.
    Mirrors bench/dataset.py turns_chrono: sessions ordered by haystack_dates,
    turns shorter than MIN_TURN_CHARS dropped (the harness never ingests them)."""
    sessions = item["haystack_sessions"]
    dates = item.get("haystack_dates") or [""] * len(sessions)
    lines = []
    for i in sorted(range(len(sessions)), key=lambda i: dates[i]):
        for t in sessions[i]:
            c = (t.get("content") or "").strip()
            if len(c) >= MIN_TURN_CHARS:
                lines.append(f"{t.get('role', 'user')}: {c}")
    return "\n".join(lines)


def gold_signatures(item: dict) -> list[str]:
    """48-char normalised prefixes of every gold answer-evidence turn
    (bench/recall.py build_item_index)."""
    ans = set(item.get("answer_session_ids") or [])
    sids = item.get("haystack_session_ids") or []
    sessions = item.get("haystack_sessions") or []
    sigs: list[str] = []
    for sid, sess in zip(sids, sessions):
        if sid not in ans:
            continue
        for t in sess:
            c = norm(t.get("content"))
            if len(c) >= MIN_TURN_CHARS:
                sigs.append(c[:SIG_CHARS])
    return sigs


def load_dataset(path: str | None) -> dict:
    """Return {question_id: item}. Uses --dataset if given, else the HF cache,
    else downloads the public LongMemEval mirror (no token needed)."""
    if path and os.path.exists(path):
        data = json.load(open(path))
    else:
        # Try the standard HF cache, else fetch (public, MIT mirror).
        cached = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval-cleaned/"
            "snapshots/*/longmemeval_oracle.json"))
        if cached:
            data = json.load(open(cached[0]))
        else:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download("xiaowu0162/longmemeval-cleaned",
                                "longmemeval_oracle.json", repo_type="dataset")
            data = json.load(open(p))
    return {it["question_id"]: it for it in data}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="data/*.jsonl",
                    help="glob of run transcripts (Korely native-mode jsonl)")
    ap.add_argument("--dataset", default=None,
                    help="path to longmemeval_oracle.json (auto-resolved if omitted)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    by_qid = load_dataset(args.dataset)

    # True per-axis sizes of the full LongMemEval oracle split (for the
    # dataset-reweighted overall reduction, so the question mix is explicit).
    DATASET_AXIS_N = {"knowledge-update": 78, "temporal-reasoning": 133,
                      "multi-session": 133, "single-session-user": 70,
                      "single-session-assistant": 56, "single-session-preference": 30}
    TOKEN_BUDGET = 2000

    per_axis = defaultdict(lambda: {"n": 0, "full": 0, "kor": 0, "present": 0,
                                    "retain": 0.0, "over": 0, "full_max": 0, "kor_max": 0})
    rows = []
    seen = set()

    files = sorted(glob.glob(args.transcripts))
    if not files:
        raise SystemExit(f"No transcripts matched {args.transcripts!r}")

    for f in files:
        for line in open(f):
            r = json.loads(line)
            # Only Korely, the hinted reader variant, one row per question.
            if r.get("system") != "korely" or not r.get("use_hint"):
                continue
            qid = r.get("qid")
            if qid in seen:
                continue
            item = by_qid.get(qid)
            if item is None:
                continue
            ctx = r.get("retrieved_context") or ""
            if not ctx.strip():
                continue  # a failed/empty cell: skip, never count as a win
            seen.add(qid)

            axis = item["question_type"]
            q = item["question"]
            full_t = prompt_tokens(full_history(item), q)
            kor_t = prompt_tokens(ctx, q)
            block_t = ntok(ctx)

            nctx = norm(ctx)
            sigs = gold_signatures(item)
            retained = sum(1 for s in sigs if s and s in nctx)
            # "evidence present" = the floor: at least one gold-evidence turn kept.
            # "retention" = fraction of gold-evidence turns kept (the honest detail).
            present = retained >= 1
            retention = retained / len(sigs) if sigs else 1.0

            rows.append({
                "qid": qid, "axis": axis,
                "full_tokens": full_t, "korely_tokens": kor_t, "block_tokens": block_t,
                "reduction_pct": round((1 - kor_t / full_t) * 100, 1) if full_t else 0.0,
                "evidence_present": present,
                "evidence_retained": retained, "evidence_total": len(sigs),
                "evidence_retention_pct": round(retention * 100, 1),
            })
            a = per_axis[axis]
            a["n"] += 1
            a["full"] += full_t
            a["kor"] += kor_t
            a["present"] += int(present)
            a["retain"] += retention
            a["over"] += int(block_t > TOKEN_BUDGET)
            a["full_max"] = max(a["full_max"], full_t)
            a["kor_max"] = max(a["kor_max"], kor_t)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "per_question.jsonl"), "w") as fh:
        for row in sorted(rows, key=lambda r: (r["axis"], r["qid"])):
            fh.write(json.dumps(row) + "\n")

    reductions = [r["reduction_pct"] for r in rows]
    retentions = [r["evidence_retention_pct"] for r in rows]
    costs_more = sum(1 for r in rows if r["korely_tokens"] >= r["full_tokens"])
    over_budget = sum(a["over"] for a in per_axis.values())

    summary = {"tokenizer": "tiktoken/o200k_base", "split": "longmemeval_oracle",
               "condition": "korely get_context (native, ~2000-token soft budget)",
               "axes": {}, "overall": {}}

    print(f"\n{'axis':26} {'N':>4} {'full avg':>9} {'korely avg':>11} "
          f"{'fewer':>6} {'evid.kept':>10} {'>budget':>8}")
    print("-" * 78)
    tot = {"n": 0, "full": 0, "kor": 0, "present": 0}
    reweight_num = reweight_den = 0.0
    for axis in sorted(per_axis):
        a = per_axis[axis]
        fa, ka = a["full"] / a["n"], a["kor"] / a["n"]
        red = (1 - ka / fa) * 100
        pres = 100 * a["present"] / a["n"]
        ret = 100 * a["retain"] / a["n"]
        summary["axes"][axis] = {
            "n": a["n"], "full_avg": round(fa, 1), "korely_avg": round(ka, 1),
            "reduction_pct": round(red, 1),
            "evidence_present_pct": round(pres, 1),
            "evidence_retention_avg_pct": round(ret, 1),
            "blocks_over_budget": a["over"],
            "is_full_census": a["n"] == DATASET_AXIS_N.get(axis),
            "full_max": a["full_max"], "korely_max": a["kor_max"],
        }
        for k in ("n", "full", "kor", "present"):
            tot[k] += a[k]
        if axis in DATASET_AXIS_N:
            reweight_num += red * DATASET_AXIS_N[axis]
            reweight_den += DATASET_AXIS_N[axis]
        print(f"{axis:26} {a['n']:>4} {fa:>9.0f} {ka:>11.0f} {red:>5.0f}% "
              f"{ret:>9.0f}% {a['over']:>8}")

    fa, ka = tot["full"] / tot["n"], tot["kor"] / tot["n"]
    red = (1 - ka / fa) * 100
    reweighted = reweight_num / reweight_den
    summary["overall"] = {
        "n": tot["n"], "full_avg": round(fa, 1), "korely_avg": round(ka, 1),
        "reduction_pct_pooled": round(red, 1),
        "reduction_pct_per_question_median": round(statistics.median(reductions), 1),
        "reduction_pct_per_question_mean": round(statistics.mean(reductions), 1),
        "reduction_pct_dataset_reweighted": round(reweighted, 1),
        "questions_where_korely_costs_more": costs_more,
        "evidence_present_pct": round(100 * tot["present"] / tot["n"], 1),
        "evidence_retention_avg_pct": round(statistics.mean(retentions), 1),
        "evidence_retention_median_pct": round(statistics.median(retentions), 1),
        "blocks_over_budget": over_budget,
        "saved_per_turn": round(fa - ka),
    }
    print("-" * 78)
    print(f"{'ALL (pooled)':26} {tot['n']:>4} {fa:>9.0f} {ka:>11.0f} {red:>5.0f}%")
    print(f"\n  pooled reduction (token-weighted) : {red:.0f}%")
    print(f"  per-question median / mean        : {statistics.median(reductions):.0f}% / {statistics.mean(reductions):.0f}%")
    print(f"  dataset-axis reweighted           : {reweighted:.0f}%")
    print(f"  questions where Korely costs MORE : {costs_more}/{len(rows)}")
    print(f"  evidence present (>=1 gold turn)  : {100*tot['present']/tot['n']:.0f}%  (floor, not a quality measure)")
    print(f"  evidence retention avg / median   : {statistics.mean(retentions):.0f}% / {statistics.median(retentions):.0f}%")
    print(f"  blocks over ~2000 budget          : {over_budget}/{len(rows)}")
    print(f"\nWrote {len(rows)} questions -> {args.out}/per_question.jsonl + summary.json")

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
