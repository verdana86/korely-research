#!/usr/bin/env python3
"""Full-history accuracy (send everything) — the ceiling for a given reader.

Completes the cost x quality picture and, with a strong reader, tells us whether
the gap to competitors is the READER (ceiling rises) or our RETRIEVAL (the block
trails full-context even with a strong reader). Reuses accuracy.py reader+judge.

  python3 fullhist_accuracy.py                                   # full 178, flash
  python3 fullhist_accuracy.py --model gemini-2.5-pro --per-axis 6
"""
from __future__ import annotations
import argparse, glob, json, os, sys, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import full_history, load_dataset
import accuracy as A
import google.generativeai as genai

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemini-2.5-flash")
ap.add_argument("--per-axis", type=int, default=0)
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()

genai.configure(api_key=A._load_key())
by_qid = load_dataset(None)

work, seen = [], set()
for f in sorted(glob.glob("data/*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r.get("system") != "korely" or not r.get("use_hint"):
            continue
        qid = r["qid"]
        if qid in seen:
            continue
        it = by_qid.get(qid)
        if it is None or not (r.get("retrieved_context") or "").strip():
            continue
        seen.add(qid)
        work.append(dict(it, _qid=qid, _gold=r.get("gold")))

if args.per_axis:
    byax = defaultdict(list)
    for it in work:
        byax[it["question_type"]].append(it)
    work = [x for ax in sorted(byax) for x in byax[ax][:args.per_axis]]


def proc(it):
    m = genai.GenerativeModel(args.model)
    q, gold, qt = it["question"], it["_gold"], it["question_type"]
    is_abs = str(it["_qid"]).endswith("_abs")
    ans = A.reader(m, full_history(it), q)
    correct = bool(A.judge(m, q, gold, ans, qt, is_abs))
    return {"qid": it["_qid"], "axis": qt, "gold": gold,
            "is_abstention": is_abs, "fullhist_answer": ans,
            "fullhist_correct": correct}


rows = []
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futs = [ex.submit(proc, it) for it in work]
    for n, fut in enumerate(as_completed(futs), 1):
        rows.append(fut.result())
        if n % 12 == 0:
            print(f"  {n}/{len(work)}", flush=True)

per = defaultdict(lambda: [0, 0])
for r in rows:
    per[r["axis"]][0] += int(r["fullhist_correct"])
    per[r["axis"]][1] += 1
tot = [sum(int(r["fullhist_correct"]) for r in rows), len(rows)]

# Persist the per-answer audit trail (same shape as accuracy.jsonl) + a summary,
# so the full-history column is reproducible/auditable like the other two.
os.makedirs("results", exist_ok=True)
with open("results/fullhist_accuracy.jsonl", "w") as fh:
    for r in sorted(rows, key=lambda r: (r["axis"], r["qid"])):
        fh.write(json.dumps(r) + "\n")
summary = {
    "model": args.model,
    "split": "longmemeval_oracle",
    "condition": "full history (send the entire conversation, ~5,500 tok)",
    "n": tot[1],
    "overall_pct": round(100 * tot[0] / tot[1], 1),
    "axes": {ax: {"n": per[ax][1], "correct": per[ax][0],
                  "pct": round(100 * per[ax][0] / per[ax][1], 1)} for ax in sorted(per)},
}
with open("results/fullhist_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"\nFULL-HISTORY accuracy ({args.model}, send everything ~5,500 tok):")
for ax in sorted(per):
    print(f"  {ax:28} {100*per[ax][0]/per[ax][1]:5.1f}%  (n={per[ax][1]})")
print(f"  {'ALL':28} {100*tot[0]/tot[1]:5.1f}%  (n={tot[1]})")
print("\nWrote results/fullhist_accuracy.jsonl + results/fullhist_summary.json")
