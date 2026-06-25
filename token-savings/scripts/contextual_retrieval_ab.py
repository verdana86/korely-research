#!/usr/bin/env python3
"""
Anthropic Contextual Retrieval — A/B on LongMemEval retrieval.

Paper: Anthropic, "Introducing Contextual Retrieval" (Sept 2024). For each chunk,
an LLM writes a short context situating the chunk within the WHOLE document; that
context is prepended before embedding. Anthropic reports ~35% (−49% with rerank)
fewer retrieval failures. Korely HAD this (services/contextual_retrieval.py) but
killswitched it 2026-06-02 because Qwen-on-Hetzner saturated the box. This script
tests, self-contained, whether the technique helps OUR retrieval on LongMemEval —
before re-enabling it in prod with a fast generator.

Method (clean A/B, only the embedding text changes):
  conversation -> turn-aware chunks (~400 tok).
  RAW   : embed the chunk text.
  CTX   : embed (Anthropic contextual prefix) + chunk text.
  query : embed the question (asymmetric retrieval_query task type).
  rank chunks by cosine; a chunk is "gold" if it contains a gold-evidence turn.
  report recall@k and hit@k for RAW vs CTX.

Generator = gemini-2.5-flash-lite (fast/cheap, the prod replacement for Qwen).
Embeddings = text-embedding-004 (held constant across conditions, so the A/B is
unbiased by model choice). No API key in the block math; this needs a Gemini key.

  python3 contextual_retrieval_ab.py --per-axis 6
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load_dataset, gold_signatures, norm, ntok, MIN_TURN_CHARS  # noqa: E402
from accuracy import _load_key, _call, READER_TEMP  # noqa: E402
import google.generativeai as genai  # noqa: E402

CHUNK_TOKENS = 400
DOC_CTX_TOKENS = 3000          # cap the document context fed to the prefix LLM (cost)
GEN_MODEL = "gemini-2.5-flash-lite"
EMBED_MODEL = "models/gemini-embedding-001"
KS = (1, 3, 5, 10)

# Anthropic's canonical contextual-retrieval prompt (adapted: document = conversation).
ANTHROPIC_PROMPT = """<document>
{doc}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


def conversation_text(item: dict, cap_tokens: int | None = None) -> str:
    sessions = item["haystack_sessions"]
    dates = item.get("haystack_dates") or [""] * len(sessions)
    lines = []
    for i in sorted(range(len(sessions)), key=lambda i: dates[i]):
        for t in sessions[i]:
            c = (t.get("content") or "").strip()
            if len(c) >= MIN_TURN_CHARS:
                lines.append(f"{t.get('role', 'user')}: {c}")
    text = "\n".join(lines)
    if cap_tokens:
        toks = text.split()
        # rough word->token; good enough for a context cap
        if len(toks) > cap_tokens:
            text = " ".join(toks[:cap_tokens])
    return text


def chunk_turns(item: dict) -> list[str]:
    """Turn-aware chunks (never split a turn), ~CHUNK_TOKENS each."""
    sessions = item["haystack_sessions"]
    dates = item.get("haystack_dates") or [""] * len(sessions)
    turns = []
    for i in sorted(range(len(sessions)), key=lambda i: dates[i]):
        for t in sessions[i]:
            c = (t.get("content") or "").strip()
            if len(c) >= MIN_TURN_CHARS:
                turns.append(f"{t.get('role', 'user')}: {c}")
    chunks, cur, cur_tok = [], [], 0
    for line in turns:
        n = ntok(line)
        if cur and cur_tok + n > CHUNK_TOKENS:
            chunks.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(line)
        cur_tok += n
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def embed(text: str, task: str, retries=4):
    for i in range(retries):
        try:
            r = genai.embed_content(model=EMBED_MODEL, content=text[:8000], task_type=task)
            return r["embedding"]
        except Exception:
            if i == retries - 1:
                return None
            import time
            time.sleep(2 ** i + 0.3)
    return None


def cosine(a, b):
    if a is None or b is None:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else -1.0


def gen_prefix(model, doc_capped: str, chunk: str) -> str:
    p = ANTHROPIC_PROMPT.format(doc=doc_capped, chunk=chunk)
    return _call(model, p, READER_TEMP)


def process_question(item: dict):
    q = item["question"]
    sigs = [s for s in gold_signatures(item) if s]
    if not sigs:
        return None
    chunks = chunk_turns(item)
    if len(chunks) < 2:
        return None  # nothing to retrieve among
    gold_flags = [any(s in norm(c) for s in sigs) for c in chunks]
    total_gold = sum(gold_flags)
    if total_gold == 0:
        return None  # gold turn shorter than MIN/sig — skip (can't score)

    doc_capped = conversation_text(item, DOC_CTX_TOKENS)
    gmodel = genai.GenerativeModel(GEN_MODEL)

    # generate contextual prefixes (one LLM call per chunk)
    prefixes = [""] * len(chunks)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(gen_prefix, gmodel, doc_capped, c): i for i, c in enumerate(chunks)}
        for fut in as_completed(futs):
            prefixes[futs[fut]] = fut.result() or ""

    # embed both conditions + the query
    q_emb = embed(q, "retrieval_query")
    raw_embs = [None] * len(chunks)
    ctx_embs = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=8) as ex:
        jobs = {}
        for i, c in enumerate(chunks):
            jobs[ex.submit(embed, c, "retrieval_document")] = ("raw", i)
            ctx_text = (prefixes[i] + "\n\n" + c) if prefixes[i] else c
            jobs[ex.submit(embed, ctx_text, "retrieval_document")] = ("ctx", i)
        for fut in as_completed(jobs):
            kind, i = jobs[fut]
            (raw_embs if kind == "raw" else ctx_embs)[i] = fut.result()

    def rank(embs):
        scored = sorted(range(len(chunks)), key=lambda i: cosine(q_emb, embs[i]), reverse=True)
        return scored

    out = {"total_gold": total_gold}
    for cond, embs in (("raw", raw_embs), ("ctx", ctx_embs)):
        order = rank(embs)
        for k in KS:
            topk = order[:k]
            hit = any(gold_flags[i] for i in topk)
            rec = sum(1 for i in topk if gold_flags[i]) / total_gold
            out[f"{cond}_hit@{k}"] = int(hit)
            out[f"{cond}_rec@{k}"] = rec
    out["axis"] = item["question_type"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="data/*.jsonl")
    ap.add_argument("--per-axis", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    genai.configure(api_key=_load_key())
    by_qid = load_dataset(None)

    import glob, json
    qids, seen = [], set()
    for f in sorted(glob.glob(args.transcripts)):
        for line in open(f):
            r = json.loads(line)
            if r.get("system") != "korely" or not r.get("use_hint"):
                continue
            qid = r.get("qid")
            if qid in seen or by_qid.get(qid) is None:
                continue
            seen.add(qid)
            qids.append(qid)

    byax = defaultdict(list)
    for qid in qids:
        byax[by_qid[qid]["question_type"]].append(qid)
    sample = [q for ax in sorted(byax) for q in byax[ax][:args.per_axis]]
    print(f"Contextual Retrieval A/B on {len(sample)} questions "
          f"(gen={GEN_MODEL}, embed={EMBED_MODEL})...", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_question, by_qid[qid]): qid for qid in sample}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if n % 5 == 0:
                print(f"  {n}/{len(sample)}", flush=True)

    if not rows:
        print("No scorable questions.")
        return
    print(f"\nScored {len(rows)} questions. Recall of gold-evidence chunks:\n")
    print(f"{'metric':10} {'RAW':>8} {'CTX (Anthropic)':>18} {'delta':>8}")
    print("-" * 48)
    for k in KS:
        for m in ("hit", "rec"):
            raw = 100 * sum(r[f"raw_{m}@{k}"] for r in rows) / len(rows)
            ctx = 100 * sum(r[f"ctx_{m}@{k}"] for r in rows) / len(rows)
            print(f"{m}@{k:<7} {raw:>7.1f}% {ctx:>17.1f}% {ctx-raw:>+7.1f}")
    print("\n(hit@k = any gold chunk in top-k; rec@k = fraction of gold chunks in top-k)")


if __name__ == "__main__":
    main()
