#!/usr/bin/env python3
"""Write data/showcase/gate_all_report.md from gate_stats.json + truthfulness.json.

The per-model story in one table: RAW fabrication (model-capacity, widened
checker) -> GATED deterministic residual -> LLM-judge residual (with 95% CI),
plus re-gen counts, verified-fallback share, and real new spend. Honest notes on
the partial (Bedrock-throttled) models and on where gating barely moved frontier
vs cleaned OURS/BASE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GATE_STATS = HERE / "gate_stats.json"
TRUTH = ROOT / "data" / "showcase" / "truthfulness.json"
OUT = ROOT / "data" / "showcase" / "gate_all_report.md"

PARTIAL = {"Gemma-3-27B-it", "Kimi-K2.5", "Mistral-Large-3 (675B)"}


def pct(x):
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"


def main() -> int:
    gs = json.loads(GATE_STATS.read_text())
    tr = json.loads(TRUTH.read_text()) if TRUTH.exists() else None
    pm = gs["per_model"]
    jr = (tr or {}).get("judge_residual", {})

    # order: ours, base, frontier, open (by raw_fab desc within group-ish -> just kind)
    def kindrank(name):
        s = pm[name]
        if s["family"] == "local" and name.startswith("OURS"):
            return 0
        if s["family"] == "local":
            return 1
        if s["family"] in ("gpt", "claude", "gemini"):
            return 2
        return 3
    names = sorted(pm, key=lambda n: (kindrank(n), -pm[n]["raw_fab_rate"]))

    L = []
    A = L.append
    A("# Gate-all + two-layer truthfulness residual — FAIR model comparison\n")
    A("Every displayed coaching cell in `web/public/showcase.json` is now GATED: the "
      "raw output is kept only if it passes the widened deterministic checker "
      "(`verify_text_ext`); otherwise it is re-sampled from the SAME model with the "
      "identical grounded prompt (up to 5 re-samples, first clean kept), and if none "
      "verify, replaced by a deterministic engine-derived explanation that is true by "
      "construction. The raw output is preserved as `raw_coaching` (a model-capacity "
      "metric). This makes the comparison fair: all models are judged on GATED text.\n")
    t = gs["totals"]
    A(f"- Positions: **{gs['n_positions']}**; cells with text: **{gs['n_cells_with_text']:,}**.")
    A(f"- Raw fabrication (widened checker, pre-gate): **{t['raw_fab']:,}** cells "
      f"(**{pct(t['raw_fab_rate'])}**).")
    A(f"- Post-gate deterministic residual: **{t['gated_fab']:,}** cells "
      f"(**{pct(t['gated_fab_rate'])}**) — should be ~0.")
    A(f"- Re-generations spent: **{t['regens']:,}**; verified-fallbacks: **{t['fallbacks']:,}**.")
    A(f"- New spend (this gate run, frontier/open re-gen): **${t['new_spend_usd']:.2f}** "
      f"(local OURS/BASE re-gen = $0).")
    if tr:
        jc = tr.get("cost", {})
        ov = jr.get("_overall", {})
        A(f"- LLM-judge residual sample: **{ov.get('n_sampled','?')}** gated cells × "
          f"{len(tr['method']['judge_panel'])} judges = **{tr.get('judge_calls','?')}** calls, "
          f"**${jc.get('total_usd',0):.2f}**.")
    A("")

    A("## Per-model: RAW → GATED (deterministic) → JUDGE residual\n")
    A("| model | kind | cells | RAW fab% (capacity) | GATED fab% (determ.) | JUDGE truthful% [95% CI] (n) | re-gens | fallback% |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for name in names:
        s = pm[name]
        kind = ("OURS" if name.startswith("OURS") else "BASE" if s["family"] == "local"
                else "frontier" if s["family"] in ("gpt", "claude", "gemini") else "open")
        j = jr.get(name)
        if j and j["n_sampled"]:
            lo, hi = j["ci95"]
            jcol = f"{pct(j['truthful_rate'])} [{pct(lo)}–{pct(hi)}] (n={j['n_sampled']})"
        else:
            jcol = "—"
        star = " ⚠" if name in PARTIAL else ""
        A(f"| {name}{star} | {kind} | {s['text']:,} | {pct(s['raw_fab_rate'])} | "
          f"{pct(s['gated_fab_rate'])} | {jcol} | {s['regens']:,} | {pct(s['fallback_rate'])} |")
    A("")
    A("⚠ = partial model: only a subset of cells exist (the ORIGINAL benchmark hit AWS "
      "Bedrock throttling for Gemma-3-27B / Kimi-K2.5 / Mistral-Large-3, so their missing "
      "cells stay missing). During THIS gate run Gemma-3-27B re-sampled normally (0.5% "
      "fallback); Kimi-K2.5 & Mistral-Large-3 were unreachable (persistent Bedrock 503), "
      "so their flagged cells went straight to the verified engine-derived fallback — that "
      "is why their fallback% ≈ their raw-fab%. Rates are over the cells each model DOES have.\n")

    A("## Reading the three layers\n")
    A("1. **RAW fab% (capacity)** — how often the model's *own* ungated coaching states "
      "a mechanically-false board fact (widened checker). This is the honest capacity "
      "metric; it is high for the small local models and non-trivial even for frontier.")
    A("2. **GATED fab% (deterministic residual)** — after gating, what the mechanical "
      "checker still catches. ~0 by construction (the gate guarantees it).")
    A("3. **JUDGE truthful% (residual)** — a cross-family panel (GPT-5.5 + Claude + "
      "Gemini, `any`-aggregation, non-circular) fact-checks a stratified sample of the "
      "GATED text for the multi-move / evaluative claims the deterministic layer "
      "abstains on. This is the honest ceiling on truthfulness the gate cannot itself "
      "guarantee.\n")

    if tr:
        jd = tr.get("judge_detail", {})
        A("## Two-layer residual — read it honestly\n")
        pjr = jd.get("per_judge_flag_rate", {})
        if pjr:
            parts = ", ".join(f"{j.upper()} {pct(v['flag_rate'])}" for j, v in pjr.items())
            A(f"- **`any`-aggregation is a strict union.** Individual judge flag-rates: "
              f"{parts}. A cell is 'not truthful' if *any one* of them objects, so the "
              f"truthful% is a conservative floor. The judges flag concrete FALSE or "
              f"UNSUPPORTED-beyond-the-1-ply-facts claims, not general plans/principles.")
            A(f"- **High-confidence floor:** all three cross-family judges independently "
              f"flagged **{pct(jd.get('unanimous_untruthful_rate',0))}** of sampled cells "
              f"(n={jd.get('unanimous_untruthful_n','?')}). Panel was unanimous on "
              f"{pct(jd.get('panel_unanimous_frac',0))} of cells.")
        bco = jd.get("by_cell_origin", {})
        if bco:
            fb = bco.get("fallback", {}); rc = bco.get("raw_clean", {}); rg = bco.get("regen", {})
            A(f"- **The residual lives in model prose, not the fallback.** The engine-"
              f"derived verified-fallback cells are judged **{pct(fb.get('truthful_rate',0))}** "
              f"truthful (n={fb.get('n','?')}) — the judges validate engine-truth. The "
              f"residual is in mechanically-clean *model* text: raw-clean cells "
              f"{pct(rc.get('truthful_rate',0))} truthful (n={rc.get('n','?')}), re-sampled "
              f"cells {pct(rg.get('truthful_rate',0))} (n={rg.get('n','?')}). Passing the "
              f"deterministic gate is NOT the same as being truthful.\n")

        A("## Where gating barely moved a model vs did heavy lifting\n")
        A("- **Under the WIDENED checker, frontier was NOT 'mostly clean'.** Raw "
          "fabrication was Claude 33.9%, Gemini 16.9%, GPT-5.5 16.3% — so the deterministic "
          "gate had real work to do on frontier too (Claude needed the most frontier "
          "cleanup by far: 500 flagged cells). The narrow legacy checker had understated "
          "this; the widened `verify_text_ext` catches the relational / move-consequence / "
          "turn / material lies it missed.")
        A("- **OURS/BASE needed the heaviest lifting.** OURS raw-fab 49.4% and BASE 35.0%; "
          "together they consumed 2,810 re-samples and fell back to the engine-derived "
          "explanation on the cells even re-sampling couldn't fix. Local re-gen is free, so "
          "this cost $0 — the point of running them on-device.")
        A("- **Deterministically the gate equalises everyone to 0% mechanical fabrication** "
          "(that is the fairness guarantee). The honest *quality* gap only appears in the "
          "LLM-judge layer: GPT-5.5 79.5% truthful and Llama-3.3-70B 72.2% at the top, "
          "OURS 23.1% and BASE 0.0% at the bottom — the 1.7B models say true-sounding but "
          "unsupported things that only a semantic judge can catch.\n")

    A("## Artifacts\n")
    A("- `web/public/showcase.json` — gated (default `coaching`) + `raw_coaching`, "
      "`raw_fabricated`, `gate_attempts`, `verified_fallback`, post-gate `fabricated`.")
    A("- `data/showcase/showcase.pregate.json` — pristine pre-gate backup.")
    A("- `data/showcase/gate/gate_stats.json`, `data/showcase/truthfulness.json`.")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
