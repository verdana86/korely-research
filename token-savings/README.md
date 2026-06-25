# Memory on LongMemEval: selection beats recency at equal budget

**At an equal token budget, does a memory layer's *selected* context answer better than just keeping the most recent turns?**

On the public [LongMemEval](https://arxiv.org/abs/2410.10813) benchmark (oracle split, 178 questions, 6 types), we give a reader LLM Korely's `get_context()` block versus a recency window **of the same token size**, and judge both answers against the gold:

> ## 76% correct vs 42% — same token budget, 1.8× the answers right.

Two measurements live here, and the order is deliberate:

1. **Accuracy at equal budget — the result that matters.** Same budget, only the memory strategy differs: Korely's selection answers **76.4%** of questions correctly vs **42.1%** for a recency window (**+34 pts**), biggest where the evidence isn't recent (temporal +70, multi-session +45). Needs any LLM (~$0.30 to reproduce).
2. **Token efficiency — the supporting number.** Korely's block is ~1,900 tokens. Versus re-sending the **entire** history that's **66% fewer**; but versus an equal-budget recency window it's a **wash (−0.7%)**. Bounding context is cheap — any truncation does it. Deterministic, no API key.

**The honest takeaway:** the win is not *fewer* tokens — a recency window ties on that. It's *which* tokens: the **selected** block answers far more often at the same cost. The 66% is real but it's "bounded vs unbounded"; the 76%-vs-42% is the part that's actually hard.

---

## Accuracy at equal budget

Same reader, same prompt, same per-question token budget; only the memory block differs. A neutral, question-type-aware judge (LongMemEval-style) scores each answer against the gold. **The same reader and the same judge grade both conditions, so any model or judge bias cancels in the delta.**

| question type | N | full history (~5,500 tok) | Korely block (~1,900) | recency window (~1,900) |
|---|---:|---:|---:|---:|
| temporal-reasoning | 20 | 90% | **80%** | 10% |
| multi-session | 20 | 60% | **50%** | 5% |
| single-session-user | 20 | 90% | **70%** | 25% |
| knowledge-update | 78 | 92% | **87%** | 58% |
| single-session-preference | 20 | 40% | **40%** | 15% |
| single-session-assistant | 20 | 100% | 100% | 95% |
| **all** | **178** | **83.1%** | **76.4%** | **42.1%** |

Read across the three conditions: **at one-third the tokens, Korely keeps ~92% of the accuracy of sending the entire conversation (76.4% vs 83.1%), and nearly doubles a same-size recency window (42.1%).** The honest trade is real — Korely gives up ~7 points versus dumping everything, biggest on single-session-user and the temporal / multi-session axes (the hardest to compress) — but it does so at a third of the cost and far above the obvious cheap alternative. The window can't reach evidence in **old** sessions (temporal spans, cross-session counts); it answers "I don't know" or guesses. On short single-session chats all three tie — the whole conversation already fits the budget, so there's nothing to select. Reported, not hidden.

**Honest scope.** This is judge-based, so unlike the token math it is not bit-deterministic: the reader and judge were both `gemini-2.5-flash` at temperature 0. The *delta* is robust to the judge because both conditions are graded identically. The recency baseline is the fairest we could build — the most recent **complete** turns that fit the same token budget as Korely's block (never mid-turn; 0 empty contexts across 178). Six of the 178 are abstention questions (no answer exists in the history); on those, an answer is scored correct only when the reader correctly declines ("I don't know"), identically for both conditions. Every per-answer judgement ships in [`results/accuracy.jsonl`](results/accuracy.jsonl) (Korely + recency window) and [`results/fullhist_accuracy.jsonl`](results/fullhist_accuracy.jsonl) (full history), so you can audit or re-grade all three columns.

```bash
cd token-savings
pip3 install google-generativeai tiktoken huggingface_hub
export GEMINI_API_KEY=...          # YOUR OWN Gemini key — free tier works; ~$0.30 for the full 178
python3 scripts/accuracy.py             # Korely block vs recency window  -> the 76.4 / 42.1 columns
python3 scripts/fullhist_accuracy.py    # full-history ceiling            -> the 83.1 column
```

> Reproducing the accuracy number uses **your own** Gemini key (the repo ships no key). The token-efficiency number above needs none. The harness pins the legacy `google-generativeai` SDK to match the published transcripts; the engine itself uses the newer `google-genai`.

---

## Token efficiency

This is the supporting number (the headline above is accuracy). It measures **input tokens only**, is fully **deterministic** and needs **no API key and no LLM call**: it reads the published run transcripts (Korely's block was logged at run time) and the public dataset, and counts tokens with a standard tokenizer. Run [`scripts/analyze.py`](scripts/analyze.py).

It is measured against a deliberately naive baseline: an agent that re-sends the entire conversation every turn. A real memory-less agent would keep a recency window instead — and against an **equal-budget** window the token saving disappears (see [Against a fair baseline](#against-a-fair-baseline-equal-token-budget) below). On its own it does **not** measure answer accuracy — that is the section above.

## Contents

- [**Accuracy at equal budget** — the result that matters](#accuracy-at-equal-budget)
- [**Token efficiency** — the supporting number](#token-efficiency)
- [Dashboard](#dashboard)
- [Result (token table)](#result)
- [Against a fair baseline (equal token budget)](#against-a-fair-baseline-equal-token-budget)
- [What "evidence retained" means (and what it does not)](#what-evidence-retained-means-and-what-it-does-not)
- [What is being measured](#what-is-being-measured)
- [Methodology and honest scope](#methodology--honest-scope)
- [Reproduce it](#reproduce-it)
- [Cost](#cost)
- [Citation](#citation)

## Introduction

An agent with no memory layer answers each turn by carrying its whole history in the prompt. That history grows every session, and you pay for it on every input token. A memory layer is supposed to replace the growing transcript with a small, relevant block. This page measures exactly one thing, as objectively as we can: **how many input tokens that swap saves.**

**The question.** For a fixed set of real questions over long, multi-session chat histories, how many input tokens does an agent spend to answer (a) by re-sending the full conversation, versus (b) by reading Korely's `get_context()` block instead?

**Why a judge-free token number too.** Answer accuracy (the headline above) needs an LLM judge, and a judge is a moving, arguable part. Token counts are not: same prompt template, same tokenizer, count both sides, report the ratio. Anyone can rerun the arithmetic from the published transcripts with no API key and no model. So we keep both — the judged accuracy result that matters, and this judge-free token number nobody has to take on trust.

**The choices we made, and why:**

- **Public dataset, hardest-for-us split.** [LongMemEval](https://arxiv.org/abs/2410.10813), `oracle` split. Oracle keeps only the evidence sessions and drops the distractors, so the full history is as *small* as it gets. That makes our reduction a conservative lower bound: on the full long-context split the gap is larger, not smaller.
- **A deliberately naive baseline.** "Full history" means re-sending every turn. A real memory-less agent would window or truncate, so 66% is "versus an agent that re-sends everything", stated plainly, not "versus every possible alternative".
- **Only one thing changes.** The reader prompt is identical in both conditions; only the knowledge block differs. So the token delta is attributable to the memory layer and nothing else.
- **We report where it loses, too.** On very short conversations the ~2,000-token block is bigger than the whole history, so memory costs more. That row (`single-session-assistant`, -9%) is in the table, not hidden.

The rest of this page is the result, an interactive dashboard, the exact method, and two ways to reproduce it: for free from our published data, or against your own Korely.

## Dashboard

![Token efficiency on LongMemEval](dashboards/dashboard.gif)

Interactive version: open [`dashboards/index.html`](dashboards/index.html) in any browser (self-contained, no server), or [view it rendered here](https://htmlpreview.github.io/?https://github.com/verdana86/korely-graphrag/blob/main/token-savings/dashboards/index.html). Full still: [`dashboards/dashboard.png`](dashboards/dashboard.png). It regenerates from the results with `python scripts/build_dashboard.py`.

**Illustrative walk-through (a demo, not the benchmark).** [`dashboards/video.html`](dashboards/video.html) animates the same swap on a single hand-built 26-turn coding session: re-sending the full history every turn (18,323 input tokens) vs reading Korely's `get_context()` block instead (6,725) — 63% fewer on that session. It's an intuition pump for the mechanism; the measured, reproducible result is the LongMemEval table below, not this number.

---

## Result

`tiktoken o200k_base` tokenizer · LongMemEval `oracle` split · Korely `get_context()` (native, ~2000-token soft budget) · N = 178.

| question type | N | full history (avg tok) | Korely (avg tok) | fewer | evidence retained |
|---|---:|---:|---:|---:|---:|
| multi-session | 20 | 10,483 | 1,878 | **82%** | 30% |
| temporal-reasoning | 20 | 7,374 | 1,830 | **75%** | 37% |
| knowledge-update *(full 78/78 census)* | 78 | 5,976 | 2,134 | **64%** | 31% |
| single-session-preference | 20 | 4,164 | 1,910 | **54%** | 62% |
| single-session-user | 20 | 2,960 | 1,808 | **39%** | 74% |
| single-session-assistant | 20 | 1,081 | 1,177 | **−9%** | 97% |
| **all (pooled)** | **178** | **5,547** | **1,902** | **66%** | **47%** |

Headline reductions, three ways, so the question mix isn't doing silent work: **pooled (token-weighted) 66%**, **per-question median 62% / mean 54%**, **dataset-axis re-weighted 60%**. They cluster in the 60s; the saving is real and grows with history length. Per-question rows: [`results/per_question.jsonl`](results/per_question.jsonl); aggregate: [`results/summary.json`](results/summary.json).

**The −9% row is reported, not hidden.** On `single-session-assistant` the whole conversation is ~1,080 tokens, already smaller than the ~2,000-token block, so memory adds overhead instead of saving. In total **20 of 178** questions cost more with Korely (19 of them single-session-assistant). The saving appears, and grows, where memory is meant to help: long histories (multi-session: 82%).

## Against a fair baseline (equal token budget)

The 66% above is against re-sending the **entire** history. A real memory-less agent would keep a recency window. So the honest question: at an *equal* token budget, how much does Korely's block actually save versus keeping the most recent turns? Run [`scripts/baselines.py`](scripts/baselines.py):

| baseline (the context a memory-less agent carries) | Korely token reduction |
|---|---:|
| full history (every turn) — *the 66% baseline* | **66%** |
| recency window, 2× Korely's budget (~4000 tok) | 47% |
| recency window, **equal budget** (matched per question) | **−0.7%** — a wash |

At equal budget the token saving is gone (Korely uses the same, slightly more on 176/178). That is honest and expected: **the 66% is "a bounded block vs unbounded history", which any truncation achieves.** Bounding context is not the contribution — *selecting* it is. Which is why the headline of this page is the [accuracy result](#accuracy-at-equal-budget), not this one.

## What "evidence retained" means (and what it does not)

For every question, Korely's block keeps **at least one** of the question's gold answer-evidence turns (100% on all 178 — call it the floor). But on average it keeps only **47% of the gold-evidence turns** (median 38%): the block is a **compression** of the history, not a copy. That is the whole point of a memory layer.

Two honest consequences:
- **This is not a recall@k leaderboard number.** Korely's block embeds verbatim conversation turns under a "Relevant memories" header, so a substring "did a gold turn survive" test passes easily by construction. We report it as an evidence-retention floor, not as a discriminating retrieval score.
- **It says nothing about answer correctness by itself.** Whether the 47%-retained, compressed block is *sufficient* to answer is the accuracy question — measured directly in [Accuracy at equal budget](#accuracy-at-equal-budget) above (76% vs 42%). Retention here is only a floor, not the quality measure.

## What is being measured

For each LongMemEval question the reader LLM is given the **same** answer prompt, and only the knowledge block inside it changes:

```
Answer the question using ONLY the memory context below. ...
Memory context:
{CONTEXT}            <-- the only thing that differs

Question: {QUESTION}
Answer:
```

- **full history**: `{CONTEXT}` = every turn of the conversation, chronological. This is a **naive baseline**: a real memory-less agent would window, truncate, or cache instead of re-sending everything, so 66% is "vs an agent that re-sends the whole history", not "vs every possible alternative".
- **Korely**: `{CONTEXT}` = Korely's `GET /v1/context` block, the server-side memory for that question.

We tokenize **both** prompts with the same tokenizer and report the reduction.

## Methodology & honest scope

- **Dataset.** [LongMemEval](https://arxiv.org/abs/2410.10813) (Wu et al., 2024), public MIT mirror [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned), file `longmemeval_oracle.json`. The **oracle** split has only the answer-evidence sessions (no distractors), so the full history is as small as it gets and the reduction here is a **conservative lower bound**; on the full long-context split (`longmemeval_s`), histories run past the context window while the block stays small, so the reduction is far larger (a tracked follow-up).
- **Question mix.** `knowledge-update` is a full census (78/78). The other five axes are 20-question subsamples of a larger pool, and the `multi-session` 20 skew toward larger histories (which lifts that axis's 82%). The **overall** number is robust to this: pooled 66%, per-question median 62%, dataset-re-weighted 60%.
- **Tokenizer.** `tiktoken o200k_base` for both conditions, so the **ratio** is apples-to-apples and reader-agnostic: token counts depend on the tokenizer, not on which model reads the prompt. Treat exact percentages as ±a few points across tokenizers. (The separate *accuracy* result above is judged by `gemini-2.5-flash` — see that section.)
- **Soft budget.** The block targets ~2,000 tokens but is not hard-capped: **65 of 178** blocks exceed it (largest block 2,488 tokens; 2,566 once the prompt template is added). It is a soft target; the reduction holds regardless. A larger budget would trivially raise evidence retention and lower the reduction, so the 66% / 47% pair is one point on a budget tradeoff, not a tuned sweet spot.
- **Objective, no judge (this section).** Token counts and evidence retention need no LLM. **Answer accuracy** (% correct, judged) is the separate, judge-dependent metric measured in [Accuracy at equal budget](#accuracy-at-equal-budget) above.
- **Source data.** `data/korely_longmemeval_oracle.jsonl` is one row per question; `retrieved_context` is Korely's actual `get_context()` output, logged verbatim.

## Reproduce it

**A. Verify the numbers ($0, no API key, no model)** straight from the published data + public dataset:

```bash
cd token-savings
pip3 install tiktoken huggingface_hub
# First run fetches the public LongMemEval dataset (~15 MB) from HuggingFace to
# rebuild the full-history baseline: needs network (but no API key, no model).
# After that it is cached and the arithmetic runs offline in under a second.
python3 scripts/analyze.py --transcripts "data/*.jsonl" --out results
```

**B. Regenerate on your own Korely.** Free key at [korely.ai/agents](https://korely.ai/agents):

```bash
cd token-savings
export KORELY_API_KEY=kor_live_...
python3 scripts/reproduce.py --axis knowledge-update --n 20
python3 scripts/analyze.py --transcripts "results/repro-*.jsonl" --out results
```

Self-contained (no private harness): public dataset in, your Korely memory out, same math. Uses your write quota; the analysis stays free.

## Cost

Pooled, the full-history condition averages 5,547 input tokens/question vs 1,902 with Korely, i.e. **3,645 fewer input tokens per turn**. Multiply by your model's input price and your turn volume. Only the input shrinks (output is unaffected); the reduction is the same across providers, only the per-token price changes.

## Research scratch (negative results)

Three exploratory A/B scripts ship here for transparency — they were run and **not** adopted, because the honest answer was "no". They import only the committed helpers + dataset in this folder, so they reproduce on a fresh clone (with a Gemini key):

- [`scripts/fusion_ab.py`](scripts/fusion_ab.py) — does fusing a lexical (full-text-search) signal with vector retrieval via Reciprocal Rank Fusion beat vector-only? **No.** On a realistic multi-topic pool it *hurt* recall (−3 to −5 pts): in LongMemEval, keyword overlap correlates with the distractors, not the answer. Vector-only kept.
- [`scripts/contextual_retrieval_ab.py`](scripts/contextual_retrieval_ab.py) + [`scripts/contextual_retrieval_pooled_ab.py`](scripts/contextual_retrieval_pooled_ab.py) — does an Anthropic-style context prefix before embedding help? On the oracle split it is degenerate (every chunk is already on-topic, nothing to disambiguate); a small positive (+1–3 pts recall) appears only once distractors are added.

Kept as a record of what we tried, not as part of the headline pipeline.

## Citation

> Wu, Di, et al. *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.* 2024. [arXiv:2410.10813](https://arxiv.org/abs/2410.10813).

Dataset mirror: [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) (MIT).
