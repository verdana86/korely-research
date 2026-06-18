#!/usr/bin/env python3
"""
Self-contained end-to-end reproduction, on YOUR own Korely.

For each LongMemEval question it ingests the whole conversation into Korely
(one memory per turn, with the turn's date as the event time so bi-temporal
resolution works), waits until it is queryable, then calls get_context() and
logs a transcript that scripts/analyze.py can score.

No private harness: the public LongMemEval dataset in, your Korely memory out,
the same token + recall math. It uses your Korely write quota; the analysis
itself (analyze.py) stays free. Settings mirror the published runs
(content template, token_budget=2000).

  export KORELY_API_KEY=kor_live_...        # free key at korely.ai/agents
  python reproduce.py --axis knowledge-update --n 20
  python analyze.py --transcripts "results/repro-*.jsonl" --out results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

BASE = os.environ.get("KORELY_BASE_URL", "https://api.korely.ai")
CONTENT_TEMPLATE = "[{date}] {role}: {content}"   # same as the harness config
MAX_CONTENT_CHARS = 16000
MIN_TURN_CHARS = 12
TOKEN_BUDGET = 2000
AGENT_ID = "longmemeval-repro"

KEY = ""
_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})")


def korely_key() -> str:
    k = os.environ.get("KORELY_API_KEY")
    if k:
        return k
    for p in (os.environ.get("BENCH_ENV"),
              os.path.expanduser("~/Code/GordonPro/.env"), ".env"):
        if p and os.path.exists(p):
            for line in open(p):
                if line.startswith("KORELY_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("Set KORELY_API_KEY (get a free key at korely.ai/agents)")


def _req(method: str, url: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read() or "null")


def iso_event_time(date: str):
    m = _DATE.search(date or "")
    if not m:
        return None
    y, mo, d, h, mi = m.groups()
    return f"{y}-{mo}-{d}T{h}:{mi}:00"


def turns_chrono(item: dict):
    sessions = item["haystack_sessions"]
    dates = item.get("haystack_dates") or [""] * len(sessions)
    for i in sorted(range(len(sessions)), key=lambda i: dates[i]):
        for t in sessions[i]:
            c = (t.get("content") or "").strip()
            if len(c) >= MIN_TURN_CHARS:
                yield dates[i], t.get("role", "user"), c


def load_axis(axis: str, n: int) -> list:
    cached = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval-cleaned/"
        "snapshots/*/longmemeval_oracle.json"))
    if cached:
        data = json.load(open(cached[0]))
    else:
        from huggingface_hub import hf_hub_download
        data = json.load(open(hf_hub_download(
            "xiaowu0162/longmemeval-cleaned", "longmemeval_oracle.json",
            repo_type="dataset")))
    return [d for d in data if d.get("question_type") == axis][:n]


def get_context(uid: str, query: str) -> str:
    url = f"{BASE}/v1/context?" + urllib.parse.urlencode(
        {"query": query, "user_id": uid, "token_budget": TOKEN_BUDGET})
    _st, j = _req("GET", url)
    return (j or {}).get("context") or ""


def main() -> None:
    global KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="knowledge-update")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tag", default="repro")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    KEY = korely_key()

    items = load_axis(args.axis, args.n)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"repro-{args.tag}-{args.axis}.jsonl")
    print(f"Reproducing {len(items)} '{args.axis}' questions on your Korely -> {path}")

    with open(path, "w") as fh:
        for i, item in enumerate(items, 1):
            qid = item["question_id"]
            uid = f"lme-{args.tag}-{qid}"
            turns = list(turns_chrono(item))
            ok = 0
            for date, role, content in turns:
                body = {
                    "content": CONTENT_TEMPLATE.format(date=date, role=role, content=content)[:MAX_CONTENT_CHARS],
                    "user_id": uid, "agent_id": AGENT_ID,
                }
                ev = iso_event_time(date)
                if ev:
                    body["timestamp"] = ev
                try:
                    st, _ = _req("POST", f"{BASE}/v1/memories", body)
                    ok += int(st in (200, 201))
                except Exception as e:  # noqa: BLE001
                    print(f"    write error: {e}")
            ctx = ""
            for _ in range(8):  # settle: wait until queryable
                ctx = get_context(uid, item["question"])
                if len(ctx) > 40:
                    break
                time.sleep(3)
            fh.write(json.dumps({
                "run_id": f"repro-{args.tag}", "qid": qid, "system": "korely",
                "mode": "native", "question": item["question"],
                "gold": item.get("answer", ""), "retrieved_context": ctx,
                "n_expected": len(turns), "use_hint": True, "seed": None,
            }) + "\n")
            fh.flush()
            print(f"  [{i}/{len(items)}] {qid}: {ok}/{len(turns)} writes, context {len(ctx)} chars")

    print(f"\nDone -> {path}\nNow score it:\n  python analyze.py --transcripts \"{path}\" --out {args.out}")


if __name__ == "__main__":
    main()
