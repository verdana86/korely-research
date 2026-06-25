"""Patch questions.jsonl with human-reviewed related-question ground truth.

Reads related_ground_truth.json (produced by a human reviewer's pass over the
Gemini-generated candidates), strips all existing type=='related' entries
from questions.jsonl, and inserts the new human-reviewed ones.

Seeds with empty related sets are dropped from the benchmark (documented
in related_ground_truth.json._meta.dropped_reason).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


QUESTIONS_PATH = Path("/app/benchmark/questions.jsonl")
GROUND_TRUTH_PATH = Path("/app/benchmark/related_ground_truth.json")


def main():
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    seeds = ground_truth["seeds"]

    existing = [json.loads(l) for l in QUESTIONS_PATH.read_text().splitlines() if l.strip()]
    kept = [q for q in existing if q.get("type") != "related"]
    dropped_count = len(existing) - len(kept)
    print(f"[load] {len(existing)} existing questions, dropping {dropped_count} old 'related'", file=sys.stderr)

    max_id = max((q["id"] for q in kept), default=0)
    next_id = max_id + 1

    added = 0
    skipped_empty = 0
    for seed in seeds:
        related_ids = seed.get("related") or []
        if not related_ids:
            skipped_empty += 1
            print(f"  [skip] {seed['seed_title']}: empty ground truth", file=sys.stderr)
            continue
        kept.append({
            "id": next_id,
            "type": "related",
            "question": f"Which posts are thematically related to \"{seed['seed_title']}\"?",
            "ground_truth": related_ids,
            # ground_truth_titles is what lets cross-system benchmarks (e.g.
            # nano-graphrag, which hashes doc IDs differently) resolve the
            # truth set via a shared title bridge.
            "ground_truth_titles": seed.get("related_titles", []),
            "seed_item_id": seed["seed_id"],
            "seed_item_title": seed["seed_title"],
            "ground_truth_source": "human_reviewed_gemini_candidates",
        })
        next_id += 1
        added += 1

    QUESTIONS_PATH.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in kept) + "\n")

    by_type: dict[str, int] = {}
    for q in kept:
        by_type[q["type"]] = by_type.get(q["type"], 0) + 1

    print(f"\n[done] wrote {len(kept)} questions to {QUESTIONS_PATH}", file=sys.stderr)
    print(f"       added {added} human-reviewed 'related', skipped {skipped_empty} empty-truth seeds", file=sys.stderr)
    print(f"       final distribution by type: {by_type}", file=sys.stderr)


if __name__ == "__main__":
    main()
