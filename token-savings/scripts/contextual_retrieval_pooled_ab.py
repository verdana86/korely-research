#!/usr/bin/env python3
"""
Anthropic Contextual Retrieval A/B — WITH DISTRACTORS (pooled).

The oracle-split A/B is degenerate: every turn is evidence, so retrieval is
trivial and contextual prefixes show no effect. This builds a realistic
needle-in-haystack: pool the chunks of N conversations into ONE corpus; for each
question, its "gold" chunks are its own evidence chunks, and it must retrieve
them out of the WHOLE pool (the other conversations are distractors). Now the
contextual prefix has a job: make the right conversation's chunks findable.

Only the embedded text changes (RAW chunk vs Anthropic-context + chunk); same
query, same embedding model. gen = gemini-2.5-flash-lite, embed = gemini-embedding-001.

  python3 contextual_retrieval_pooled_ab.py --per-axis 4
"""
from __future__ import annotations
import argparse, glob, json, os, sys, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load_dataset, gold_signatures, norm  # noqa: E402
from accuracy import _load_key  # noqa: E402
import contextual_retrieval_ab as AB  # noqa: E402
import google.generativeai as genai  # noqa: E402

KS = (5, 10, 20, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-axis", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    genai.configure(api_key=_load_key())
    by = load_dataset(None)

    qids, seen = [], set()
    for f in sorted(glob.glob("data/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("system") != "korely" or not r.get("use_hint"):
                continue
            q = r.get("qid")
            if q in seen or by.get(q) is None:
                continue
            seen.add(q)
            qids.append(q)
    byax = defaultdict(list)
    for q in qids:
        byax[by[q]["question_type"]].append(q)
    sample = [q for ax in sorted(byax) for q in byax[ax][:args.per_axis]]

    # build the pool: every chunk of every sampled conversation
    pool = []  # {qid, text, gold(for its own q), prefix}
    questions = {}  # qid -> question string
    for qid in sample:
        item = by[qid]
        questions[qid] = item["question"]
        chunks = AB.chunk_turns(item)
        sigs = [s for s in gold_signatures(item) if s]
        doc = AB.conversation_text(item, AB.DOC_CTX_TOKENS)
        for c in chunks:
            gold = any(s in norm(c) for s in sigs)
            pool.append({"qid": qid, "text": c, "gold": gold, "doc": doc, "prefix": ""})
    print(f"Pool: {len(pool)} chunks from {len(sample)} conversations "
          f"(distractors present). Generating prefixes + embeddings...", flush=True)

    gmodel = genai.GenerativeModel(AB.GEN_MODEL)

    # 1) contextual prefixes (Anthropic) for every pool chunk
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(AB.gen_prefix, gmodel, p["doc"], p["text"]): i for i, p in enumerate(pool)}
        for n, fut in enumerate(as_completed(futs), 1):
            pool[futs[fut]]["prefix"] = fut.result() or ""
            if n % 100 == 0:
                print(f"  prefixes {n}/{len(pool)}", flush=True)

    # 2) embed RAW + CTX for every pool chunk
    raw_emb = [None] * len(pool)
    ctx_emb = [None] * len(pool)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        jobs = {}
        for i, p in enumerate(pool):
            jobs[ex.submit(AB.embed, p["text"], "retrieval_document")] = ("raw", i)
            ct = (p["prefix"] + "\n\n" + p["text"]) if p["prefix"] else p["text"]
            jobs[ex.submit(AB.embed, ct, "retrieval_document")] = ("ctx", i)
        done = 0
        for fut in as_completed(jobs):
            kind, i = jobs[fut]
            (raw_emb if kind == "raw" else ctx_emb)[i] = fut.result()
            done += 1
            if done % 200 == 0:
                print(f"  embeds {done}/{2*len(pool)}", flush=True)

    # 3) per-question retrieval over the WHOLE pool
    qemb = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(AB.embed, questions[q], "retrieval_query"): q for q in sample}
        for fut in as_completed(futs):
            qemb[futs[fut]] = fut.result()

    agg = {f"{c}_rec@{k}": 0.0 for c in ("raw", "ctx") for k in KS}
    agg.update({f"{c}_hit@{k}": 0 for c in ("raw", "ctx") for k in KS})
    scored = 0
    for qid in sample:
        gold_idx = [i for i, p in enumerate(pool) if p["qid"] == qid and p["gold"]]
        if not gold_idx:
            continue
        scored += 1
        for cond, embs in (("raw", raw_emb), ("ctx", ctx_emb)):
            order = sorted(range(len(pool)), key=lambda i: AB.cosine(qemb[qid], embs[i]), reverse=True)
            for k in KS:
                topk = set(order[:k])
                hit = any(i in topk for i in gold_idx)
                rec = sum(1 for i in gold_idx if i in topk) / len(gold_idx)
                agg[f"{cond}_hit@{k}"] += int(hit)
                agg[f"{cond}_rec@{k}"] += rec

    print(f"\nScored {scored} questions over a {len(pool)}-chunk pool (with distractors):\n")
    print(f"{'metric':10} {'RAW':>8} {'CTX (Anthropic)':>18} {'delta':>8}")
    print("-" * 48)
    for k in KS:
        for m in ("hit", "rec"):
            raw = 100 * agg[f"raw_{m}@{k}"] / scored
            ctx = 100 * agg[f"ctx_{m}@{k}"] / scored
            print(f"{m}@{k:<7} {raw:>7.1f}% {ctx:>17.1f}% {ctx-raw:>+7.1f}")
    print("\n(retrieving each question's gold chunks out of the full multi-conversation pool)")


if __name__ == "__main__":
    main()
