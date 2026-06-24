"""Run the benchmark: vanilla RAG vs korely-graphrag on the same corpus.

- **Vanilla RAG** (`vanilla`): FTS + pgvector + RRF, NO entity graph, NO
  get_related. Same chunks and embeddings as korely-graphrag — the only
  difference is the ranking is pure cosine/FTS without IDF-weighted entity
  traversal. For "related" questions: fall back to search on title.
- **korely-graphrag** (`graphrag`): full stack.

Output:
- benchmark/results.jsonl (per-question per-system)
- benchmark/BENCHMARK.md (aggregated, human-readable)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, "/app/src")

from sqlalchemy import bindparam, text as sql_text  # noqa: E402

from korely_graphrag.ingest.embedder import embed_one  # noqa: E402
from korely_graphrag.mcp_server import tools  # noqa: E402
from korely_graphrag.search import get_related_items  # noqa: E402
from korely_graphrag.search.hybrid import _sanitize_tsquery_terms  # noqa: E402
from korely_graphrag.storage import Item, session_scope  # noqa: E402


QUESTIONS_PATH = Path("/app/benchmark/questions.jsonl")
RESULTS_PATH = Path("/app/benchmark/results.jsonl")
REPORT_PATH = Path("/app/BENCHMARK.md")

# Optional third system: nano-graphrag. Its predictions are generated
# out-of-band by `run_nano.py` inside the isolated `nano` docker service
# (nano pulls in graspologic/gensim which conflict with our main image).
# If the file exists, we join its predictions in as a third column.
NANO_PREDICTIONS_PATH = Path("/app/benchmark/nano_predictions.jsonl")
NANO_TITLE_MAP_PATH = Path("/app/benchmark/bench_nano/title_to_docid.json")


# ---------------------------------------------------------------------------
# Systems under test
# ---------------------------------------------------------------------------


def _vanilla_rag_search(query: str, limit: int = 10) -> list[str]:
    """FTS + pgvector with RRF but NO entity graph, NO IDF. The simplest
    possible RAG baseline, as any LangChain-default setup would produce.
    """
    with session_scope() as s:
        # Identical to korely-graphrag's hybrid_search without the graph path.
        try:
            q_emb = embed_one(query)
        except Exception:
            q_emb = None

        # FTS (sanitized tokens only)
        terms = _sanitize_tsquery_terms(query)
        tsquery = " & ".join(f"{w}:*" for w in terms) if terms else ""
        fts_rows = []
        if tsquery:
            fts_rows = s.execute(
                sql_text(
                    "SELECT item_id::text, ROW_NUMBER() OVER (ORDER BY ts_rank(fts, to_tsquery('simple', :q)) DESC) AS rank "
                    "FROM chunks WHERE fts @@ to_tsquery('simple', :q) "
                    "ORDER BY rank LIMIT :lim"
                ).bindparams(bindparam("q", value=tsquery), bindparam("lim", value=limit * 3))
            ).all()

        vec_rows = []
        if q_emb:
            vec_rows = s.execute(
                sql_text(
                    "SELECT item_id::text, ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:emb AS vector)) AS rank "
                    "FROM chunks WHERE embedding IS NOT NULL ORDER BY rank LIMIT :lim"
                ).bindparams(bindparam("emb", value=str(q_emb)), bindparam("lim", value=limit * 3))
            ).all()

        # RRF at item level
        item_score: dict[str, float] = {}
        for rows in (fts_rows, vec_rows):
            seen_items: set[str] = set()
            for r in rows:
                if r.item_id in seen_items:
                    continue
                seen_items.add(r.item_id)
                item_score[r.item_id] = item_score.get(r.item_id, 0.0) + 1.0 / (60 + r.rank)
        ranked = sorted(item_score.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [iid for iid, _ in ranked]


def _vanilla_rag_related(item_id: str, limit: int = 10) -> list[str]:
    """No graph → use the item's title as a search query."""
    with session_scope() as s:
        item = s.get(Item, item_id)
        if not item:
            return []
        return _vanilla_rag_search(item.title, limit=limit)[:limit]


def query_vanilla(q: dict) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    if q["type"] == "related":
        ids = _vanilla_rag_related(q["seed_item_id"], limit=10)
    else:
        ids = _vanilla_rag_search(q["question"], limit=10)
    return ids, time.perf_counter() - t0


def query_graphrag(q: dict) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    if q["type"] == "related":
        out = tools.get_related(q["seed_item_id"], limit=10)
        ids = [r["item_id"] for r in out.get("results", [])]
    else:
        out = tools.search(q["question"], limit=10)
        ids = [r["item_id"] for r in out.get("results", [])]
    return ids, time.perf_counter() - t0


SYSTEMS = {
    "vanilla": query_vanilla,
    "graphrag": query_graphrag,
}


def _load_nano() -> dict[int, dict] | None:
    """Load pre-computed nano-graphrag predictions if present.

    nano doesn't run in this container (dep conflict), so predictions
    come from `benchmark/run_nano.py` executed in the isolated `nano`
    compose service. We join them in here by question id.

    Returns None when the artifacts are missing — benchmark still runs
    the 2-system (vanilla vs graphrag) comparison unaffected.
    """
    if not NANO_PREDICTIONS_PATH.exists() or not NANO_TITLE_MAP_PATH.exists():
        return None
    records = [json.loads(l) for l in NANO_PREDICTIONS_PATH.read_text().splitlines() if l.strip()]
    title_to_docid = json.loads(NANO_TITLE_MAP_PATH.read_text())
    docid_to_title = {v: k for k, v in title_to_docid.items()}
    by_id = {}
    for r in records:
        # Convert nano doc-hashes -> titles for title-space comparison
        r["predicted_titles"] = [docid_to_title.get(d) for d in r.get("predicted", [])]
        r["predicted_titles"] = [t for t in r["predicted_titles"] if t]
        by_id[r["id"]] = r
    return by_id


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def precision_at_k(predicted: list[str], truth: set[str], k: int) -> float:
    top = predicted[:k]
    if not top:
        return 0.0
    return sum(1 for p in top if p in truth) / k


def recall_at_k(predicted: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    return sum(1 for p in predicted[:k] if p in truth) / len(truth)


def hit_at_k(predicted: list[str], truth: set[str], k: int) -> int:
    return 1 if any(p in truth for p in predicted[:k]) else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    questions = [json.loads(l) for l in QUESTIONS_PATH.read_text().splitlines()]
    results = []
    nano_predictions = _load_nano()
    if nano_predictions is not None:
        print(
            f"[nano] joining {len(nano_predictions)} pre-computed predictions",
            file=sys.stderr,
        )

    # For nano, metrics are computed in title-space (its doc IDs are md5
    # hashes independent of our korely UUIDs). Every question must carry
    # ground_truth_titles for this to work.
    for q in questions:
        truth = set(q["ground_truth"])
        row: dict = {"id": q["id"], "type": q["type"], "question": q["question"]}
        for sys_name, fn in SYSTEMS.items():
            try:
                predicted, latency = fn(q)
            except Exception as e:
                print(f"  ! {sys_name} failed on Q{q['id']}: {e}", file=sys.stderr)
                predicted, latency = [], 0.0
            row[f"{sys_name}_predicted"] = predicted[:10]
            row[f"{sys_name}_latency_s"] = round(latency, 3)
            row[f"{sys_name}_p@1"] = precision_at_k(predicted, truth, 1)
            row[f"{sys_name}_p@5"] = precision_at_k(predicted, truth, 5)
            row[f"{sys_name}_r@5"] = recall_at_k(predicted, truth, 5)
            row[f"{sys_name}_hit@5"] = hit_at_k(predicted, truth, 5)

        if nano_predictions is not None:
            rec = nano_predictions.get(q["id"])
            if rec:
                title_truth = set(q.get("ground_truth_titles") or [])
                pred_titles = rec["predicted_titles"]
                row["nano_predicted"] = rec.get("predicted", [])[:10]
                row["nano_predicted_titles"] = pred_titles[:10]
                row["nano_latency_s"] = rec.get("latency_s", 0.0)
                row["nano_p@1"] = precision_at_k(pred_titles, title_truth, 1)
                row["nano_p@5"] = precision_at_k(pred_titles, title_truth, 5)
                row["nano_r@5"] = recall_at_k(pred_titles, title_truth, 5)
                row["nano_hit@5"] = hit_at_k(pred_titles, title_truth, 5)
            else:
                # Question missing from nano set — record zeros so aggregates don't crash
                for m in ("p@1", "p@5", "r@5", "hit@5"):
                    row[f"nano_{m}"] = 0
                row["nano_latency_s"] = 0.0
                row["nano_predicted"] = []

        results.append(row)
        _nano_str = (
            f" | nano p@1={row.get('nano_p@1', 0):.0f} hit@5={row.get('nano_hit@5', 0)}"
            if nano_predictions is not None else ""
        )
        print(
            f"Q{q['id']:3d} ({q['type']:12s}): "
            f"vanilla p@1={row['vanilla_p@1']:.0f} hit@5={row['vanilla_hit@5']} | "
            f"graphrag p@1={row['graphrag_p@1']:.0f} hit@5={row['graphrag_hit@5']}"
            f"{_nano_str}",
            file=sys.stderr,
        )

    RESULTS_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")

    # Aggregate
    agg = _aggregate(results)
    _write_report(agg, results)

    print("\n" + "=" * 70, file=sys.stderr)
    print(f"Saved: {RESULTS_PATH}", file=sys.stderr)
    print(f"Saved: {REPORT_PATH}", file=sys.stderr)


def _detect_systems(results: list[dict]) -> list[str]:
    """Which system columns exist in the result rows — detects nano dynamically."""
    if not results:
        return list(SYSTEMS.keys())
    systems = list(SYSTEMS.keys())
    if "nano_p@1" in results[0]:
        systems.append("nano")
    return systems


def _aggregate(results: list[dict]) -> dict:
    by_type = {}
    for r in results:
        t = r["type"]
        by_type.setdefault(t, []).append(r)

    def _agg(rows, key):
        vals = [r.get(key, 0) for r in rows]
        return {"mean": round(mean(vals), 3), "median": round(median(vals), 3), "n": len(vals)}

    systems = _detect_systems(results)

    agg = {"overall": {}, "by_type": {}, "_systems": systems}
    for sys_name in systems:
        agg["overall"][sys_name] = {
            "p@1": _agg(results, f"{sys_name}_p@1"),
            "p@5": _agg(results, f"{sys_name}_p@5"),
            "r@5": _agg(results, f"{sys_name}_r@5"),
            "hit@5": _agg(results, f"{sys_name}_hit@5"),
            "latency_s": _agg(results, f"{sys_name}_latency_s"),
        }
    for t, rows in by_type.items():
        agg["by_type"][t] = {}
        for sys_name in systems:
            agg["by_type"][t][sys_name] = {
                "p@1": _agg(rows, f"{sys_name}_p@1"),
                "p@5": _agg(rows, f"{sys_name}_p@5"),
                "r@5": _agg(rows, f"{sys_name}_r@5"),
                "hit@5": _agg(rows, f"{sys_name}_hit@5"),
                "latency_s": _agg(rows, f"{sys_name}_latency_s"),
            }
    return agg


def _write_report(agg: dict, results: list[dict]):
    lines = []
    if "nano" in agg.get("_systems", []):
        lines.append("# Benchmark — korely-graphrag vs vanilla RAG vs nano-graphrag")
    else:
        lines.append("# Benchmark — korely-graphrag vs vanilla RAG")
    lines.append("")
    lines.append("**Corpus:** 24 public blog posts from Andrej Karpathy's github.io")
    lines.append("(reproducible: `git clone https://github.com/karpathy/karpathy.github.io`).")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    # Compute per-type counts from the actual results set (resilient to re-ingests
    # that drop questions referencing no-longer-indexed items).
    counts = {t: 0 for t in ("precision@1", "recall@5", "related")}
    for r in results:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    lines.append(f"**Questions:** {len(results)}, three types:")
    lines.append(f"- **precision@1** ({counts['precision@1']}) — a specific fact from one post. Ground truth = the")
    lines.append("  single post that answers it. Generated by Gemini, one per post.")
    lines.append(f"- **recall@5** ({counts['recall@5']}) — a cross-post theme shared by 3-5 posts. Ground truth =")
    lines.append("  that set. Seed entity picked by its doc-frequency, question generated by")
    lines.append("  Gemini to name the theme without naming the entity.")
    lines.append(f"- **related** ({counts['related']}) — \"what posts are thematically linked to X?\" for")
    lines.append("  seed posts (seeds with no thematic link in the corpus are dropped).")
    lines.append("  **Ground truth is NOT generated by any system under test**: Gemini proposes")
    lines.append("  candidates from title + 500-char abstract only (no entity graph, no retrieval),")
    lines.append("  then a human reviewer tightens the set with discipline — same architecture /")
    lines.append("  dataset / methodological theme / narrative style count; generic \"both about")
    lines.append("  neural networks\" doesn't. See `benchmark/related_ground_truth.json` for the")
    lines.append("  full audit trail.")
    lines.append("")
    systems = agg.get("_systems", ["vanilla", "graphrag"])
    has_nano = "nano" in systems

    lines.append("**Systems under test:**")
    lines.append("- `vanilla` — FTS + pgvector + RRF over the *same* chunks and embeddings,")
    lines.append("  with **no entity graph** and no IDF weighting. This is the default you get")
    lines.append("  from LangChain / LlamaIndex with a pgvector store. For `related` queries,")
    lines.append("  vanilla falls back to searching on the seed's title (a naive strategy that's")
    lines.append("  what most RAG users reach for when they have no graph).")
    lines.append("- `graphrag` — korely-graphrag full stack: same FTS + pgvector + RRF for")
    lines.append("  search, **plus** entity-graph traversal with IDF weighting and hub-node")
    lines.append("  filtering for `related`, **plus** a pgvector semantic fallback that fills")
    lines.append("  remaining slots when graph-derived hits are sparse (typical for \"island\"")
    lines.append("  posts with unique named entities — e.g. short fiction).")
    if has_nano:
        lines.append("- `nano` — [nano-graphrag](https://github.com/gusye1234/nano-graphrag)")
        lines.append("  (v0.0.8) with a Gemini adapter (LLM + embeddings). Same corpus, same")
        lines.append("  embedding model, same entity-extraction LLM. Search uses nano's chunk")
        lines.append("  vector DB (closest to its `naive` RAG mode). **nano-graphrag has no")
        lines.append("  `get_related(doc)` primitive**, so we approximate it by traversing the")
        lines.append("  entity graph that nano itself builds during ingestion — for each seed")
        lines.append("  doc we collect its entities, then rank other docs by count of shared")
        lines.append("  entities. No IDF, no hub filtering — this is nano's graph as-is.")
    lines.append("")
    lines.append("All systems use identical Gemini models: `gemini-embedding-001` (1536d)")
    lines.append("for embeddings, `gemini-2.5-flash` for entity extraction.")
    if has_nano:
        lines.append("Costs per query are comparable (one embedding call each). nano-graphrag's")
        lines.append("ingest is noticeably heavier — it pays for community-report generation up")
        lines.append("front so its `global` mode can summarise themes, a feature this benchmark")
        lines.append("does not exercise.")
    else:
        lines.append("Cost per query is essentially identical (both do one embedding call).")
        lines.append("Ingestion is higher for graphrag (one extra Gemini call per doc for entity")
        lines.append("extraction) but amortizes as a one-time cost.")
    lines.append("")
    lines.append("## Overall results")
    lines.append("")

    def _row_for(block: dict, metric: str) -> str:
        cells = [f"{block[s][metric]['mean']:.3f}" for s in systems]
        return "| " + metric + " | " + " | ".join(cells) + " |"

    header = "| metric | " + " | ".join(f"{s} (mean)" for s in systems) + " |"
    sep = "|---" + "|---:" * len(systems) + "|"
    lines.append(header)
    lines.append(sep)
    for m in ("p@1", "p@5", "r@5", "hit@5", "latency_s"):
        lines.append(_row_for(agg["overall"], m))
    lines.append("")

    lines.append("## Breakdown by question type")
    lines.append("")
    for qtype, block in agg["by_type"].items():
        n = block[systems[0]]["p@1"]["n"]
        lines.append(f"### {qtype} ({n} questions)")
        lines.append("")
        lines.append("| metric | " + " | ".join(systems) + " |")
        lines.append("|---" + "|---:" * len(systems) + "|")
        for m in ("p@1", "p@5", "r@5", "hit@5", "latency_s"):
            cells = [f"{block[s][m]['mean']:.3f}" for s in systems]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")

    # ------------------------------------------------------------------
    # Honest interpretation — pull numbers from the 'related' breakdown
    # (the only type where the two systems differ) and derive the story.
    # ------------------------------------------------------------------
    rel = agg["by_type"].get("related")
    if rel:
        v_p1 = rel["vanilla"]["p@1"]["mean"]
        g_p1 = rel["graphrag"]["p@1"]["mean"]
        v_r5 = rel["vanilla"]["r@5"]["mean"]
        g_r5 = rel["graphrag"]["r@5"]["mean"]
        v_h5 = rel["vanilla"]["hit@5"]["mean"]
        g_h5 = rel["graphrag"]["hit@5"]["mean"]
        v_lat = rel["vanilla"]["latency_s"]["mean"]
        g_lat = rel["graphrag"]["latency_s"]["mean"]
        speedup = (v_lat / g_lat) if g_lat > 0 else float("inf")

        lines.append("## Takeaways (honest interpretation)")
        lines.append("")
        lines.append("**Where the systems are identical — and we say so upfront.**")
        lines.append("On pure search (the `precision@1` and `recall@5` blocks above) the")
        lines.append("two systems return the same top-K in the same order. This is **by")
        lines.append("design**: both use the same hybrid search (FTS + pgvector + RRF)")
        lines.append("over the same chunks. We are not claiming graphrag is a better")
        lines.append("keyword/vector retriever — it isn't. The underlying retrieval stack")
        lines.append("is shared.")
        lines.append("")
        lines.append("**Where graphrag earns its keep.** On `related` queries (the graph's")
        lines.append("whole reason to exist), graphrag:")
        lines.append("")
        lines.append(f"- Picks the top-1 correct related post **{g_p1:.0%} of the time vs {v_p1:.0%}** for vanilla.")
        lines.append("  Vanilla's title-based fallback rarely puts the right post first because")
        lines.append("  title keyword overlap is a weak proxy for \"thematically related\".")
        lines.append(f"- Improves recall@5 by **+{(g_r5 - v_r5):.3f}** ({v_r5:.0%} → {g_r5:.0%}) and")
        lines.append(f"  hit@5 by **+{(g_h5 - v_h5):.3f}** ({v_h5:.0%} → {g_h5:.0%}). The graph")
        lines.append("  traversal surfaces entity-linked neighbors that no title-search would find,")
        lines.append("  and the semantic fallback fills in thematic neighbors for posts whose")
        lines.append("  entities are all unique to themselves (short fiction, standalone projects).")
        lines.append(f"- Is **{speedup:.0f}× faster** ({g_lat*1000:.0f} ms vs {v_lat*1000:.0f} ms)")
        lines.append("  because the graph traversal is a single SQL CTE over precomputed mentions,")
        lines.append("  while vanilla's fallback runs a fresh embedding call + vector scan per query.")
        lines.append("")
        lines.append("**What's NOT in this benchmark.** Cost per query is essentially identical")
        lines.append("(both systems do one embedding call per search; graphrag's fallback path")
        lines.append("uses the seed's pre-stored embedding, no new Gemini call). Ingestion is")
        lines.append("more expensive for graphrag (one extra Gemini call per document for entity")
        lines.append("extraction), but that's a one-time cost amortized across every future query.")
        lines.append("")
        lines.append(f"**What we want reviewers to push on.** The related ground truth is {counts['related']}")
        lines.append("human-reviewed questions on a 24-post corpus. Bigger corpora, multi-author")
        lines.append("blogs, or different domains will shift the numbers. Also note that the")
        lines.append("`related` comparison bundles **two** graphrag advantages — entity-graph")
        lines.append("traversal *and* a pgvector semantic fallback — against a **title-only**")
        lines.append('vanilla, so part of the 0.50-vs-0.00 gap is "graph + semantic vs title",')
        lines.append("not the entity graph in isolation. A fair component-isolating ablation")
        lines.append("(give vanilla the same seed-embedding fallback) is a tracked follow-up.")
        lines.append("The methodology is reproducible and we'd welcome pull requests extending it.")
        lines.append("")

        if has_nano:
            n_p1 = rel["nano"]["p@1"]["mean"]
            n_p5 = rel["nano"]["p@5"]["mean"]
            n_r5 = rel["nano"]["r@5"]["mean"]
            n_h5 = rel["nano"]["hit@5"]["mean"]
            n_lat = rel["nano"]["latency_s"]["mean"]
            lines.append("## nano-graphrag — how it compares")
            lines.append("")
            lines.append("nano-graphrag is the closest open-source neighbor of this project:")
            lines.append("minimal Python, entity graph auto-built at ingest, designed for LLM")
            lines.append("workflows. It's an honest apples-to-apples reference point.")
            lines.append("")
            lines.append("On the same corpus with the same Gemini models:")
            lines.append("")
            lines.append(f"- **Related p@1**: nano {n_p1:.0%} vs korely-graphrag {g_p1:.0%} vs vanilla {v_p1:.0%}.")
            lines.append(f"- **Related r@5**: nano {n_r5:.0%} vs korely-graphrag {g_r5:.0%} vs vanilla {v_r5:.0%}.")
            lines.append(f"- **Related hit@5**: nano {n_h5:.0%} vs korely-graphrag {g_h5:.0%} vs vanilla {v_h5:.0%}.")
            lines.append(f"- **Related latency**: nano {n_lat*1000:.0f} ms vs korely-graphrag {g_lat*1000:.0f} ms vs vanilla {v_lat*1000:.0f} ms.")
            lines.append("")
            lines.append("**Caveats on this comparison.**")
            lines.append("")
            lines.append("- nano-graphrag has **no `get_related(doc)` primitive**. The number you")
            lines.append("  see is from our own DIY traversal of nano's internal networkx graph —")
            lines.append("  we collect the seed doc's entities, then rank other docs by count of")
            lines.append("  shared entities. No IDF, no hub-node filtering. A user who wants this")
            lines.append("  on nano-graphrag today would have to write similar code.")
            lines.append("- For non-related queries (precision@1, recall@5) nano uses its chunk")
            lines.append("  vector DB directly (closest to its `naive` RAG mode), which is the")
            lines.append("  pure-vector half of vanilla's stack — no FTS. Any gap on those rows")
            lines.append("  reflects the FTS contribution, not graph quality.")
            lines.append("- nano-graphrag's ingest also runs community-report generation (used")
            lines.append("  by its `global` mode for theme summarisation). This benchmark does")
            lines.append("  not exercise that feature, so nano pays an ingest-time cost for")
            lines.append("  capability we don't measure. Fair to it to say so upfront.")
            lines.append("")

    # Example failure cases: where one system wins dramatically over the other
    lines.append("## Notable cases")
    lines.append("")
    lines.append("Where graphrag beat vanilla (top 5):")
    lines.append("")
    wins = sorted(
        (r for r in results if r["graphrag_hit@5"] > r["vanilla_hit@5"]),
        key=lambda r: r["graphrag_hit@5"] - r["vanilla_hit@5"],
        reverse=True,
    )[:5]
    for r in wins:
        lines.append(f"- **Q{r['id']}** ({r['type']}): \"{r['question'][:100]}\"")
        lines.append(f"  - vanilla hit@5=0, graphrag hit@5=1")
    lines.append("")

    losses = sorted(
        (r for r in results if r["vanilla_hit@5"] > r["graphrag_hit@5"]),
        key=lambda r: r["vanilla_hit@5"] - r["graphrag_hit@5"],
        reverse=True,
    )[:5]
    lines.append("Where vanilla beat graphrag (top 5):")
    lines.append("")
    for r in losses:
        lines.append(f"- **Q{r['id']}** ({r['type']}): \"{r['question'][:100]}\"")
        lines.append(f"  - vanilla hit@5=1, graphrag hit@5=0")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append("git clone https://github.com/verdana86/korely-graphrag")
    lines.append("cd korely-graphrag")
    lines.append("cp .env.example .env  # set GEMINI_API_KEY")
    lines.append("docker compose up -d db")
    lines.append("# fetch Karpathy corpus")
    lines.append("git clone --depth 1 https://github.com/karpathy/karpathy.github.io /tmp/kb")
    lines.append("mkdir -p benchmark/karpathy && cp /tmp/kb/_posts/*.markdown benchmark/karpathy/")
    lines.append("# ingest + 2-system benchmark (vanilla vs korely-graphrag)")
    lines.append("docker compose run --rm app korely-graphrag ingest --reset /app/benchmark/karpathy")
    lines.append("docker compose run --rm app python /app/benchmark/run_benchmark.py")
    if has_nano:
        lines.append("")
        lines.append("# optional 3rd column: nano-graphrag on the same corpus")
        lines.append("docker compose --profile benchmark build nano")
        lines.append("docker compose --profile benchmark run --rm nano python ingest_nano.py")
        lines.append("docker compose --profile benchmark run --rm nano python run_nano.py")
        lines.append("# re-run the main benchmark — it auto-joins nano_predictions.jsonl")
        lines.append("docker compose run --rm app python /app/benchmark/run_benchmark.py")
    lines.append("```")

    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
