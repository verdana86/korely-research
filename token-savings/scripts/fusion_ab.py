#!/usr/bin/env python3
"""
Multi-signal fusion A/B — does RRF(vector, lexical) beat vector-only retrieval?

This isolates the prod change in memory_engine.search() (Sprint 18): the agent
memory store retrieved VECTOR-ONLY; now it fuses pgvector cosine with a Postgres
FTS ranking via Reciprocal Rank Fusion (Cormack et al. 2009). reproduce.py writes
ONE memory per turn under a per-question user_id and retrieves within that
conversation, so this mirrors it exactly: per conversation, rank that
conversation's turns, recall the gold-evidence turns.

  RAW    : rank turns by cosine(query, turn).
  FUSION : RRF( cosine-rank , lexical-rank ), lexical = OR-of-terms weighted by
           IDF (a faithful Python proxy of to_tsquery('simple', 't1:*|t2:*') +
           ts_rank — match ANY salient term, rank by rarity/overlap).

Only the RANKING changes; same turns, same gold, same embeddings. Costs only
embeddings (no reader/judge), so it is cheap to run wide.

  python3 fusion_ab.py --per-axis 6
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load_dataset, gold_signatures, norm, MIN_TURN_CHARS  # noqa: E402
from accuracy import _load_key  # noqa: E402
import contextual_retrieval_ab as AB  # noqa: E402  (reuse embed() + cosine())
import google.generativeai as genai  # noqa: E402

KS = (5, 10, 20)
RRF_K = 60
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def turns_of(item: dict) -> list[str]:
    """One string per turn, date-ordered, harness noise floor — exactly the unit
    reproduce.py writes as a memory."""
    sessions = item["haystack_sessions"]
    dates = item.get("haystack_dates") or [""] * len(sessions)
    out = []
    for i in sorted(range(len(sessions)), key=lambda i: dates[i]):
        for t in sessions[i]:
            c = (t.get("content") or "").strip()
            if len(c) >= MIN_TURN_CHARS:
                out.append(f"{t.get('role', 'user')}: {c}")
    return out


def toks(s: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(s or "") if len(t) >= 2]


def lexical_rank(query: str, turns: list[str], mode: str = "or") -> list[int]:
    """Lexical ranking over turns. Returns turn indices, best first.

      or   : match ANY query term, score Σ IDF(shared) — proxy of to_tsquery
             OR + ts_rank. High recall, but in a multi-topic pool common words
             collide across topics and inject distractors.
      rare : keep only SALIENT query terms (IDF above the query's median, i.e.
             drop the words that are common in THIS corpus), require ≥1 such
             match, score Σ IDF. Precision-oriented: rescues verbatim-entity
             needles without firing on stopword/common-word collisions."""
    qset = set(toks(query))
    if not qset:
        return []
    n = len(turns)
    df: dict[str, int] = defaultdict(int)
    tset = [set(toks(t)) for t in turns]
    for ts in tset:
        for w in ts:
            df[w] += 1
    idf = {w: math.log(1 + n / df[w]) for w in qset if df.get(w)}
    if not idf:
        return []
    if mode == "rare":
        vals = sorted(idf.values())
        thresh = vals[len(vals) // 2]  # median IDF of the matched query terms
        salient = {w for w, v in idf.items() if v >= thresh}
    else:
        salient = set(idf)
    scored = []
    for i, ts in enumerate(tset):
        shared = salient & ts
        if not shared:
            continue
        scored.append((i, sum(idf[w] for w in shared)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    score: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            score[idx] += 1.0 / (k + rank)
    return [i for i, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)]


def process(item: dict, mode: str = "or"):
    turns = turns_of(item)
    if len(turns) < 4:
        return None
    sigs = [s for s in gold_signatures(item) if s]
    gold = [i for i, t in enumerate(turns) if any(s in norm(t) for s in sigs)]
    if not gold:
        return None

    q_emb = AB.embed(item["question"], "retrieval_query")
    if q_emb is None:
        return None
    embs = [None] * len(turns)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(AB.embed, t, "retrieval_document"): i for i, t in enumerate(turns)}
        for fut in as_completed(futs):
            embs[futs[fut]] = fut.result()

    vec = sorted(range(len(turns)), key=lambda i: AB.cosine(q_emb, embs[i]), reverse=True)
    lex = lexical_rank(item["question"], turns, mode)
    fused = rrf([vec, lex])
    # fused only contains turns that appeared in at least one ranking; vec covers
    # all, so fused == a reordering of all turns.

    out = {"gold": len(gold), "axis": item["question_type"]}
    gold_set = set(gold)
    for cond, order in (("raw", vec), ("fusion", fused)):
        for k in KS:
            topk = set(order[:k])
            out[f"{cond}_rec@{k}"] = len(gold_set & topk) / len(gold)
            out[f"{cond}_hit@{k}"] = int(bool(gold_set & topk))
    return out


def process_pooled(sample, by, workers, mode="or"):
    """Distractor setting: pool every sampled conversation's turns into ONE
    corpus; each question recalls its own gold turns out of the whole pool (the
    other conversations are distractors). This is what a real agent memory store
    looks like — one end_user, many topics — and where fusion earns its keep."""
    pool, questions = [], {}
    for qid in sample:
        item = by[qid]
        questions[qid] = item["question"]
        sigs = [s for s in gold_signatures(item) if s]
        for t in turns_of(item):
            pool.append({"qid": qid, "text": t,
                         "gold": any(s in norm(t) for s in sigs)})
    print(f"Pool: {len(pool)} turns from {len(sample)} conversations "
          f"(distractors present). Embedding...", flush=True)

    embs = [None] * len(pool)
    with ThreadPoolExecutor(max_workers=workers * 2) as ex:
        futs = {ex.submit(AB.embed, p["text"], "retrieval_document"): i for i, p in enumerate(pool)}
        for n, fut in enumerate(as_completed(futs), 1):
            embs[futs[fut]] = fut.result()
            if n % 200 == 0:
                print(f"  embeds {n}/{len(pool)}", flush=True)
    texts = [p["text"] for p in pool]
    qemb = {}
    with ThreadPoolExecutor(max_workers=workers * 2) as ex:
        futs = {ex.submit(AB.embed, questions[q], "retrieval_query"): q for q in sample}
        for fut in as_completed(futs):
            qemb[futs[fut]] = fut.result()

    rows = []
    for qid in sample:
        gold = {i for i, p in enumerate(pool) if p["qid"] == qid and p["gold"]}
        if not gold or qemb.get(qid) is None:
            continue
        vec = sorted(range(len(pool)), key=lambda i: AB.cosine(qemb[qid], embs[i]), reverse=True)
        lex = lexical_rank(questions[qid], texts, mode)
        fused = rrf([vec, lex])
        row = {"gold": len(gold), "axis": by[qid]["question_type"]}
        for cond, order in (("raw", vec), ("fusion", fused)):
            for k in KS:
                topk = set(order[:k])
                row[f"{cond}_rec@{k}"] = len(gold & topk) / len(gold)
                row[f"{cond}_hit@{k}"] = int(bool(gold & topk))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-axis", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pooled", action="store_true",
                    help="cross-conversation pool (distractors), like real usage")
    ap.add_argument("--lex-mode", default="or", choices=["or", "rare"],
                    help="lexical signal: 'or' (any term) or 'rare' (salient terms only)")
    args = ap.parse_args()
    genai.configure(api_key=_load_key())
    by = load_dataset(None)

    qids, seen = [], set()
    for f in sorted(glob.glob("data/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("system") != "korely" or not r.get("use_hint"):
                continue
            qid = r.get("qid")
            if qid in seen or by.get(qid) is None:
                continue
            seen.add(qid)
            qids.append(qid)
    byax = defaultdict(list)
    for qid in qids:
        byax[by[qid]["question_type"]].append(qid)
    sample = [q for ax in sorted(byax) for q in byax[ax][:args.per_axis]]

    if args.pooled:
        rows = process_pooled(sample, by, args.workers, args.lex_mode)
    else:
        print(f"Fusion A/B on {len(sample)} questions (per-conversation, per-turn, lex={args.lex_mode})...", flush=True)
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process, by[q], args.lex_mode): q for q in sample}
            for n, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if r:
                    rows.append(r)
                if n % 5 == 0:
                    print(f"  {n}/{len(sample)}", flush=True)

    if not rows:
        print("No scorable questions.")
        return
    print(f"\nScored {len(rows)} questions. Recall of gold-evidence turns:\n")
    print(f"{'metric':10} {'VECTOR':>8} {'FUSION (RRF)':>14} {'delta':>8}")
    print("-" * 44)
    for k in KS:
        for m in ("hit", "rec"):
            raw = 100 * sum(r[f"raw_{m}@{k}"] for r in rows) / len(rows)
            fus = 100 * sum(r[f"fusion_{m}@{k}"] for r in rows) / len(rows)
            print(f"{m}@{k:<7} {raw:>7.1f}% {fus:>13.1f}% {fus-raw:>+7.1f}")
    print("\n(per conversation: rank that conversation's turns, recall gold-evidence turns)")


if __name__ == "__main__":
    main()
