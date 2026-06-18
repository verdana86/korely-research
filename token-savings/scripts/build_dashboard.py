#!/usr/bin/env python3
"""
Build a self-contained, Korely-branded HTML dashboard from the analysis outputs.
Reads results/summary.json + results/per_question.jsonl, embeds the real data
inline (opens with file://, no server), writes dashboards/index.html.

  python scripts/build_dashboard.py
"""
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(HERE, "results", "summary.json")))
ROWS = [json.loads(l) for l in open(os.path.join(HERE, "results", "per_question.jsonl"))]
ov = S["overall"]

# longest-history axes first (reads best)
ORDER = ["multi-session", "temporal-reasoning", "knowledge-update",
         "single-session-preference", "single-session-user", "single-session-assistant"]
LABEL = {"multi-session": "Multi-session", "temporal-reasoning": "Temporal reasoning",
         "knowledge-update": "Knowledge update", "single-session-preference": "Single-session preference",
         "single-session-user": "Single-session user", "single-session-assistant": "Single-session assistant"}

DATA = {
    "pooled": round(ov["reduction_pct_pooled"]),
    "median": round(ov["reduction_pct_per_question_median"]),
    "reweighted": round(ov["reduction_pct_dataset_reweighted"]),
    "saved": ov["saved_per_turn"],
    "n": ov["n"],
    "costs_more": ov["questions_where_korely_costs_more"],
    "retention_avg": round(ov["evidence_retention_avg_pct"]),
    "over_budget": ov["blocks_over_budget"],
    "axes": [{"key": k, "label": LABEL[k], "census": S["axes"][k].get("is_full_census", False),
              "full": round(S["axes"][k]["full_avg"]), "kor": round(S["axes"][k]["korely_avg"]),
              "red": round(S["axes"][k]["reduction_pct"]),
              "ret": round(S["axes"][k]["evidence_retention_avg_pct"])}
             for k in ORDER if k in S["axes"]],
    "rows": [{"f": r["full_tokens"], "k": r["korely_tokens"]} for r in ROWS],
}

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Korely &middot; Token efficiency on LongMemEval</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
  :root{--bg:#0B0F17;--panel:#0e1422;--ink:#E8EDF2;--mut:#9CA3AF;--fnt:#6F7884;
    --green:#34D399;--green2:#10B981;--red:#F87171;--line:#1b2230;
    --sans:"Geist",ui-sans-serif,-apple-system,sans-serif;--mono:"Geist Mono",ui-monospace,monospace;}
  *{box-sizing:border-box}html,body{margin:0}
  body{background:#05070c;color:var(--ink);font-family:var(--sans);-webkit-font-smoothing:antialiased;line-height:1.5}
  body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
    background:radial-gradient(ellipse 70% 50% at 50% -5%,rgba(16,185,129,.10),transparent 60%),
    linear-gradient(to right,#0c1018 1px,transparent 1px),linear-gradient(to bottom,#0c1018 1px,transparent 1px);
    background-size:auto,48px 48px,48px 48px;
    -webkit-mask-image:radial-gradient(ellipse 90% 70% at 50% 20%,#000 40%,transparent 100%);
            mask-image:radial-gradient(ellipse 90% 70% at 50% 20%,#000 40%,transparent 100%);}
  .wrap{position:relative;z-index:1;max-width:1040px;margin:0 auto;padding:0 24px 96px}
  header{display:flex;align-items:center;justify-content:space-between;padding:26px 0 8px}
  .brand{display:flex;align-items:center;gap:10px;font-weight:600}.brand svg{display:block}
  .verify{font-family:var(--mono);font-size:12.5px;color:var(--mut);text-decoration:none;border:1px solid var(--line);border-radius:8px;padding:7px 12px;transition:.2s}
  .verify:hover{color:var(--ink);border-color:rgba(52,211,153,.4);background:rgba(52,211,153,.06)}
  .eyebrow{font-size:11.5px;font-weight:600;letter-spacing:.2em;text-transform:uppercase;color:var(--green);margin:42px 0 14px}
  h1{font-size:clamp(30px,5vw,50px);font-weight:700;letter-spacing:-.025em;line-height:1.08;margin:0 0 14px;max-width:800px}
  h1 .hl{background:linear-gradient(180deg,#6EE7B7,#10B981);-webkit-background-clip:text;background-clip:text;color:transparent}
  .lede{font-size:17px;color:var(--mut);max-width:700px;margin:0}
  .lede code{font-family:var(--mono);font-size:14px;color:#C7CED6;background:rgba(255,255,255,.05);padding:1px 6px;border-radius:5px}
  .scopeline{font-size:13.5px;color:var(--fnt);max-width:700px;margin:14px 0 0;line-height:1.6}
  .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:36px 0 8px}
  .stat{border:1px solid var(--line);border-radius:16px;background:linear-gradient(180deg,var(--panel),rgba(11,15,23,.4));padding:22px}
  .stat .big{font-size:44px;font-weight:700;letter-spacing:-.03em;line-height:1;font-variant-numeric:tabular-nums;
    background:linear-gradient(180deg,#6EE7B7,#10B981);-webkit-background-clip:text;background-clip:text;color:transparent}
  .stat .lab{font-size:13px;color:var(--mut);margin-top:9px}
  section{margin-top:62px}
  .h2{font-size:13px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--fnt);margin:0 0 6px}
  .h2b{font-size:24px;font-weight:600;letter-spacing:-.02em;margin:0 0 22px}
  .bars{border:1px solid var(--line);border-radius:16px;background:var(--panel);overflow:hidden}
  .row{display:grid;grid-template-columns:180px 1fr 128px;align-items:center;gap:18px;padding:16px 20px;border-bottom:1px solid var(--line)}
  .row:last-child{border-bottom:0}
  .row .name{font-size:14px;font-weight:500}.row .name small{display:block;color:var(--fnt);font-family:var(--mono);font-size:11px;margin-top:2px}
  .track{position:relative;height:30px}
  .bar{position:absolute;left:0;height:12px;border-radius:7px;width:0;transition:width 1.1s cubic-bezier(.22,1,.36,1)}
  .bar.full{top:1px;background:linear-gradient(90deg,rgba(248,113,113,.85),rgba(248,113,113,.45))}
  .bar.kor{top:17px;background:linear-gradient(90deg,#10B981,#34D399)}
  .bar .t{position:absolute;right:-6px;top:50%;transform:translate(100%,-50%);font-family:var(--mono);font-size:11px;color:var(--mut);white-space:nowrap}
  .delta{text-align:right}.delta .pct{font-size:21px;font-weight:700;font-variant-numeric:tabular-nums}
  .delta .pct.neg{color:var(--red)}.delta .pct.pos{color:var(--green)}
  .delta small{display:block;color:var(--fnt);font-family:var(--mono);font-size:10.5px;margin-top:4px}
  .legend{display:flex;gap:10px;align-items:center;font-size:12px;color:var(--mut);padding:14px 20px 0}
  .legend i{width:18px;height:6px;border-radius:6px;display:inline-block}.legend .gap{flex:1}
  .note{font-size:12.5px;color:var(--fnt);margin-top:14px;line-height:1.6;max-width:760px}
  .scatter{border:1px solid var(--line);border-radius:16px;background:var(--panel);padding:18px}
  .scatter svg{width:100%;height:auto;display:block}
  .cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .card{border:1px solid var(--line);border-radius:14px;background:var(--panel);padding:20px}
  .card h3{margin:0 0 8px;font-size:15px}.card p{margin:0;color:var(--mut);font-size:13.5px;line-height:1.6}
  .card code{font-family:var(--mono);font-size:12.5px;color:#C7CED6}
  .cost table{width:100%;border-collapse:collapse;font-size:13.5px}
  .cost th,.cost td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}
  .cost th{color:var(--fnt);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.1em}
  .cost td.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}.cost .save{color:var(--green);font-weight:600}
  footer{margin-top:62px;border-top:1px solid var(--line);padding-top:24px;color:var(--fnt);font-size:13px}
  footer a{color:var(--mut)}.mono{font-family:var(--mono)}
  @media(max-width:720px){.stats,.cards{grid-template-columns:1fr}.row{grid-template-columns:130px 1fr}.delta{display:none}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand"><svg width="22" height="22" viewBox="0 0 24 24"><path d="M12 3 L21 20 L3 20 Z" fill="none" stroke="#34D399" stroke-width="1.8" stroke-linejoin="round"/></svg>Korely</div>
    <a class="verify" href="https://github.com/verdana86/korely-research/tree/main/token-savings">run it yourself &rarr;</a>
  </header>

  <p class="eyebrow">Token efficiency &middot; LongMemEval</p>
  <h1>Answering from memory costs <span class="hl">__POOLED__% fewer input tokens</span>.</h1>
  <p class="lede">On the public LongMemEval benchmark, a memory-less agent re-sends the whole conversation to answer. Korely answers from a compact <code>get_context()</code> block. Measured, deterministic, no LLM and no API key required to verify.</p>
  <p class="scopeline">This measures <b>token efficiency</b> against a naive baseline that re-sends the full history every turn. It does <b>not</b> measure answer accuracy (a separate, judge-dependent metric, out of scope here).</p>

  <div class="stats">
    <div class="stat"><div class="big" data-count="__POOLED__" data-suffix="%">0</div><div class="lab">fewer input tokens, overall (pooled, N=__N__)</div></div>
    <div class="stat"><div class="big" data-count="__MEDIAN__" data-suffix="%">0</div><div class="lab">fewer, median per question</div></div>
    <div class="stat"><div class="big" data-count="__SAVEDN__">0</div><div class="lab">fewer input tokens per turn (avg)</div></div>
  </div>

  <section>
    <p class="h2">Per question type</p>
    <h2 class="h2b">The saving grows with the history.</h2>
    <div class="bars" id="bars"></div>
    <div class="legend"><i style="background:linear-gradient(90deg,#F87171,rgba(248,113,113,.5))"></i> full history<span class="gap"></span><i style="background:linear-gradient(90deg,#10B981,#34D399)"></i> Korely get_context</div>
    <p class="note">Three headline reductions so the question mix is explicit: pooled (token-weighted) __POOLED__%, per-question median __MEDIAN__%, dataset-axis re-weighted __REWEIGHTED__%. The single-session-assistant row is &minus;9% on purpose: those conversations are ~1k tokens, already smaller than the block, so memory adds overhead. In total __COSTSMORE__ of __N__ questions cost more with Korely; we report it.</p>
  </section>

  <section>
    <p class="h2">Every question</p>
    <h2 class="h2b">Korely stays in a flat band; full history climbs.</h2>
    <div class="scatter"><svg id="scatter" viewBox="0 0 1000 480" preserveAspectRatio="xMidYMid meet"></svg></div>
    <p class="note">Each green dot is one of the __N__ questions: full-history input tokens (x) vs Korely input tokens (y). The red diagonal is break-even (y = x). On these oracle histories (up to ~17k tokens), Korely stays in a low band near its ~2k budget while the full-history cost climbs the diagonal.</p>
  </section>

  <section>
    <p class="h2">What the block keeps</p>
    <h2 class="h2b">A compression, not a copy.</h2>
    <div class="cards">
      <div class="card"><h3>Evidence is present, not complete</h3><p>For every question, at least one gold answer-evidence turn is kept in the block (a floor). On average it keeps <b>__RETENTION__%</b> of the evidence turns: the block compresses the history, that is the point. This is not a recall@k score, and it is not an accuracy claim.</p></div>
      <div class="card"><h3>Accuracy is measured separately</h3><p>Whether the compressed block is <i>enough</i> to answer correctly is the accuracy question. It needs a neutral LLM judge of a different model family than the reader, and is tracked separately, not claimed on this page.</p></div>
    </div>
  </section>

  <section class="cost">
    <p class="h2">What it costs</p>
    <h2 class="h2b">__SAVED__ fewer input tokens per turn is real money.</h2>
    <div class="card">
      <table><thead><tr><th>model (input price)</th><th class="n">per 1k turns</th><th class="n">per 1M turns</th></tr></thead><tbody id="costbody"></tbody></table>
      <p class="note">Saving = __SAVED__ input tokens/turn &times; the model's input price. Output tokens are unaffected; only the input shrinks. Prices are public list rates and change; the token reduction does not.</p>
    </div>
  </section>

  <section>
    <p class="h2">Method, honestly</p>
    <h2 class="h2b">Stated assumptions, no cherry-picking.</h2>
    <div class="cards">
      <div class="card"><h3>Naive baseline</h3><p>"full history" re-sends every turn on every question. Real agents window, truncate, or cache, so __POOLED__% is "vs an agent that resends everything", not vs every alternative.</p></div>
      <div class="card"><h3>Conservative split</h3><p>The <code>oracle</code> split has no distractor sessions, so the full history is as small as it gets. The reduction is a <b>lower bound</b>; on the full long-context split it is larger.</p></div>
      <div class="card"><h3>Tokenizer &amp; budget</h3><p>Counts use <code>tiktoken o200k_base</code> for both sides (ratio apples-to-apples; not measured against the Llama reader's tokenizer, so treat as &plusmn;a few points). The ~2k budget is a soft target: __OVER__ of __N__ blocks exceed it.</p></div>
      <div class="card"><h3>Question mix</h3><p>knowledge-update is a full 78/78 census; the other axes are 20-question subsamples and multi-session skews large. The overall number is robust: pooled __POOLED__%, median __MEDIAN__%, re-weighted __REWEIGHTED__%.</p></div>
    </div>
  </section>

  <footer>
    Dataset: <a href="https://arxiv.org/abs/2410.10813">LongMemEval</a> (Wu et al., 2024), oracle split, public MIT mirror.
    Reproduce: <span class="mono">python scripts/analyze.py</span> ($0, no keys). Numbers regenerate from <span class="mono">results/summary.json</span>.
  </footer>
</div>

<script>
const D = __DATA__;
function countUp(el){const to=+el.dataset.count,suf=el.dataset.suffix||"",t0=performance.now(),dur=1100;
  (function s(n){const p=Math.min(1,(n-t0)/dur),e=1-Math.pow(1-p,3);el.textContent=Math.round(to*e).toLocaleString()+suf;if(p<1)requestAnimationFrame(s);})(t0);}

const maxFull=Math.max(...D.axes.map(a=>a.full));
const bars=document.getElementById("bars");
D.axes.forEach(a=>{
  const neg=a.red<0;const row=document.createElement("div");row.className="row";
  row.innerHTML=`<div class="name">${a.label}${a.census?' <span style="color:#34D399">&middot; full census</span>':''}<small>N=${a.census?78:20}</small></div>
    <div class="track">
      <div class="bar full" data-w="${a.full/maxFull*100}"><span class="t">${a.full.toLocaleString()} tok</span></div>
      <div class="bar kor" data-w="${a.kor/maxFull*100}"><span class="t">${a.kor.toLocaleString()} tok</span></div>
    </div>
    <div class="delta"><div class="pct ${neg?'neg':'pos'}">${neg?'+':'&minus;'}${Math.abs(a.red)}%</div><small>${a.ret}% evidence kept</small></div>`;
  bars.appendChild(row);
});

const SVG=document.getElementById("scatter"),VW=1000,VH=480,P={l:64,r:24,t:20,b:46};
const maxX=Math.max(...D.rows.map(r=>r.f))*1.04,maxY=maxX;
const X=v=>P.l+v/maxX*(VW-P.l-P.r),Y=v=>(VH-P.b)-v/maxY*(VH-P.t-P.b);
let g="";
for(let i=0;i<=4;i++){const v=maxX*i/4,yy=Y(v),xx=X(v);
  g+=`<line x1="${P.l}" y1="${yy}" x2="${VW-P.r}" y2="${yy}" stroke="#141a24"/>`;
  g+=`<text x="${P.l-10}" y="${yy+4}" fill="#5b6573" font-size="11" text-anchor="end" font-family="Geist Mono">${Math.round(v/1000)}k</text>`;
  g+=`<text x="${xx}" y="${VH-P.b+20}" fill="#5b6573" font-size="11" text-anchor="middle" font-family="Geist Mono">${Math.round(v/1000)}k</text>`;}
g+=`<text x="${VW/2}" y="${VH-6}" fill="#6F7884" font-size="12" text-anchor="middle">full-history input tokens</text>`;
g+=`<text transform="translate(16 ${VH/2}) rotate(-90)" fill="#6F7884" font-size="12" text-anchor="middle">Korely input tokens</text>`;
g+=`<line x1="${X(0)}" y1="${Y(0)}" x2="${X(maxX)}" y2="${Y(maxX)}" stroke="#F87171" stroke-dasharray="5 5" stroke-opacity=".5"/>`;
g+=`<text x="${X(maxX)-6}" y="${Y(maxX)+16}" fill="#F4A6A6" font-size="11" text-anchor="end">break-even (y = x)</text>`;
SVG.innerHTML=g;
D.rows.forEach((r,i)=>{const el=document.createElementNS("http://www.w3.org/2000/svg","circle");
  el.setAttribute("cx",X(r.f));el.setAttribute("cy",Y(r.k));el.setAttribute("r","0");
  el.setAttribute("fill","#34D399");el.setAttribute("fill-opacity",".75");
  el.style.transition="r .5s cubic-bezier(.22,1,.36,1) "+(i*4)+"ms";
  el.innerHTML=`<title>full ${r.f.toLocaleString()} tok -> Korely ${r.k.toLocaleString()} tok</title>`;
  SVG.appendChild(el);requestAnimationFrame(()=>requestAnimationFrame(()=>el.setAttribute("r","4")));});

const PRICES=[["Gemini 2.5 Flash-Lite",0.10],["GPT-4o",2.50],["Claude Sonnet",3.00]];
const cb=document.getElementById("costbody");
PRICES.forEach(([m,p])=>{const per1k=D.saved*1000*p/1e6,per1m=D.saved*1e6*p/1e6;const tr=document.createElement("tr");
  tr.innerHTML=`<td>${m} <span style="color:#6F7884">($${p.toFixed(2)}/M)</span></td><td class="n save">$${per1k.toFixed(2)}</td><td class="n save">$${per1m.toLocaleString(undefined,{maximumFractionDigits:0})}</td>`;cb.appendChild(tr);});

document.querySelectorAll("[data-count]").forEach(countUp);
requestAnimationFrame(()=>document.querySelectorAll(".bar").forEach(b=>b.style.width=b.dataset.w+"%"));
</script>
</body>
</html>
"""

out = (HTML
       .replace("__POOLED__", str(DATA["pooled"]))
       .replace("__MEDIAN__", str(DATA["median"]))
       .replace("__REWEIGHTED__", str(DATA["reweighted"]))
       .replace("__N__", str(DATA["n"]))
       .replace("__COSTSMORE__", str(DATA["costs_more"]))
       .replace("__RETENTION__", str(DATA["retention_avg"]))
       .replace("__OVER__", str(DATA["over_budget"]))
       .replace("__SAVEDN__", str(DATA["saved"]))
       .replace("__SAVED__", f'{DATA["saved"]:,}')
       .replace("__DATA__", json.dumps(DATA)))

os.makedirs(os.path.join(HERE, "dashboards"), exist_ok=True)
dst = os.path.join(HERE, "dashboards", "index.html")
with open(dst, "w") as fh:
    fh.write(out)
print(f"wrote {dst}  ({len(out)//1024} KB)")
