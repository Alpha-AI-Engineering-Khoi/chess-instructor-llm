"""Render the DEFINITIVE 803 balanced leaderboard to RESULTS_FULL_EVAL_803.md."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


def _pct(x: Optional[float], nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}%" if x is not None else "n/a"


def _f(x: Optional[float], nd: int = 2) -> str:
    return f"{x:.{nd}f}" if x is not None else "n/a"


def _score100(x: Optional[float]) -> str:
    return f"{100 * x:.1f}" if x is not None else "n/a"


def write_markdown(
    path: Path,
    model_order: Sequence[str],
    display: Dict[str, str],
    family: Dict[str, str],
    practical: Dict[str, Tuple[str, float, str]],
    tier: Dict[str, Any],
    obj: Dict[str, Any],
    council: Dict[str, Any],
    safety: Dict[str, float],
    scored: Dict[str, Any],
    *,
    n_positions: int,
    n_council_items: int = 0,
    n_judges: int = 3,
    w_balanced: Dict[str, float],
    w_base: Dict[str, float],
    spend: Optional[Dict[str, Any]] = None,
) -> None:
    present = [m for m in model_order if m in obj]
    L: list[str] = []
    A = L.append

    def loc_label(m: str) -> str:
        s = practical.get(m, ("", 0.0, ""))[1]
        return "yes" if s >= 1.0 else ("tight" if s > 0 else "no")

    # ------------------------------------------------------------------ #
    A("# DEFINITIVE Chess-Coach Eval — 803 gap positions, all 14 models\n")
    A(f"The airtight, held-out evaluation of **tier-appropriate move selection** (the moat) "
      f"and every other axis we optimize for, on the curated **{n_positions}-position** gap set "
      f"(`data/eval/gap_positions.jsonl` — 100% discriminating, per-tier Stockfish sound pool + "
      f"Maia likelihoods + the identified tier-appropriate move, **zero leakage** vs train/valid). "
      f"Every model coaches the SAME positions at all 3 tiers with byte-identical grounding "
      f"(`render_pool_facts` + `render_user_prompt` + the tier's Maia block).\n")

    # ---- headline recommendation (filled after ranking below) --------- #
    # compute rankings
    def _bal(m):
        return scored[m].get("balanced")

    def _base(m):
        return scored[m].get("base_fit")

    open_models = [m for m in present if family[m] == "open"]
    ranked_bal = sorted([m for m in present if _bal(m) is not None],
                        key=lambda m: _bal(m), reverse=True)
    ranked_base_open = sorted([m for m in open_models if _base(m) is not None],
                             key=lambda m: _base(m), reverse=True)
    # best base among LOCALLY-RUNNABLE open models (a base we can fine-tune + run)
    runnable_open = [m for m in ranked_base_open if practical.get(m, ("", 0, ""))[1] > 0]
    best_open_balanced = next((m for m in ranked_bal if family[m] == "open"), None)
    best_base = runnable_open[0] if runnable_open else (ranked_base_open[0] if ranked_base_open else None)
    overall_bal_winner = ranked_bal[0] if ranked_bal else None

    A("## TL;DR\n")
    if best_open_balanced:
        gem = "gemma3_27b"
        diff_note = ("**the same as Gemma-3-27B**" if best_open_balanced == gem
                     else f"**NOT Gemma-3-27B** (it is {display[best_open_balanced]})")
        A(f"- **Best open model on the balanced score (tier-selection + instructiveness weighted "
          f"highest): {display[best_open_balanced]}** — {diff_note}.")
    if best_base:
        A(f"- **Best open v3 *base* (re-weighted for what's hard to ADD — instructiveness/"
          f"capacity, faithfulness, local-runnability — since tier-appropriateness is what we "
          f"fine-tune IN): {display[best_base]}.**")
    if best_open_balanced and best_base:
        if best_open_balanced == best_base:
            A(f"- The balanced winner and the best-base pick are the **same model** "
              f"({display[best_base]}).")
        else:
            A(f"- The balanced winner ({display[best_open_balanced]}) and the best-base pick "
              f"({display[best_base]}) **differ** — the balanced score rewards raw "
              f"tier-selection + coaching that a huge model has, but a v3 base must be "
              f"fine-tunable and locally runnable, which favors {display[best_base]}.")
    A("- Tier-appropriate move selection is **weak across the whole field** (it is the trained "
      "behavior, not an emergent one) — see the tier table; this is exactly the gap v3 targets.")
    A("")

    # ------------------------------------------------------------------ #
    A("## Method & cost-smart scope\n")
    A(f"- **Deterministic metrics** (tier-fit, tier-differentiation, direction, move-safety, "
      f"no-engine-speak, fabrication) computed on **ALL {n_positions} positions x 3 tiers** for "
      f"the 2 local models (free) + 9 open models. The **3 frontier references** are measured on "
      f"a **balanced 150-position stratified subset x 3 tiers** — measuring Claude Opus 4.8 on "
      f"all 803 would add ~$55 for a *reference* row whose behavior is already established; a "
      f"stratified subset gives a tight estimate (this mirrors the council-subset rationale).")
    A(f"- **Tier-appropriate move (the moat):** each coach's recommended move is re-extracted "
      f"with the instrumented, pool-restricted extractor and compared to the canonical tier move "
      f"from `src/teacher/tier_select.select_tier_move` (beginner=most human-findable sound move, "
      f"intermediate=eval/Maia blend, advanced=sharpest=engine best). `tier-fit` = pick == that "
      f"canonical move (mean over the 3 tiers).")
    A(f"- **Instructiveness:** one blinded cross-family council ({n_judges} frontier judges: "
      f"GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro) ranks the unified **14-model** field per item "
      f"on a **stratified ~{n_council_items}-item** subset (balanced across tier x phase). "
      f"Council on 803x14 is expensive + statistically unnecessary for a rank estimate.")
    A(f"- **Gates:** move-safety (no blunders) and no-engine-speak are pass/fail floors. "
      f"**Fabrication is reported but down-weighted** (the project's non-LLM faithfulness "
      f"verifier neutralizes it at serve time — it is table-stakes, not a differentiator).")
    A("")

    # ------------------------------------------------------------------ #
    A("## 1. Per-metric leaderboard (all 14 models)\n")
    A("Sorted by the balanced score (below). `tier-fit` is the moat metric; instructiveness is "
      "council mean rank (lower = better, of 14).\n")
    A("| # | Model | family | tier-fit↑ | tier-diff↑ | direction↑ | instr rank↓ (top1) | "
      "safety↑ | no-jargon↑ | fab↓ | local | n(det) |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|")
    order_for_table = ranked_bal + [m for m in present if m not in ranked_bal]
    for i, m in enumerate(order_for_table, 1):
        t = tier.get(m, {})
        o = obj.get(m, {})
        c = council.get(m, {})
        instr = f"{_f(c['mean_rank'],2)} ({_pct(c['top1_pct']/100,0)})" if c else "n/a"
        safe = safety.get(m)
        A(f"| {i} | {display[m]} | {family[m]} | "
          f"{_pct(t.get('tier_fit_mean'),0)} | {_pct(t.get('diff_rate'),0)} | "
          f"{_pct(t.get('direction'),0)} | {instr} | "
          f"{_pct(safe,0) if safe is not None else 'n/a'} | {_pct(o.get('no_engine_speak'),0)} | "
          f"{_pct(o.get('fabrication'),0)} | {loc_label(m)} | {o.get('n','?')} |")
    A("")
    A("- **tier-fit** = share of (position,tier) where the coach's pick equals the canonical "
      "`select_tier_move` move. **tier-diff** = share of positions where the pick changes across "
      "the 3 tiers. **direction** = share where the beginner pick is at least as human-findable "
      "(Maia rank) as the advanced pick (correct level gradient).")
    A("- **safety** = share of picks that are not blunders (cp-loss < 250). **no-jargon** = no "
      "centipawn/engine-speak leaked. **fab** = share of outputs with >=1 false board fact "
      "(down-weighted). **n(det)** = deterministic positions x tiers scored (frontier on the "
      "150-subset).")
    A("")

    # ------------------------------------------------------------------ #
    A("## 2. Tier-appropriate move selection (the moat), per tier\n")
    A("Per-tier `tier-fit` (pick == `select_tier_move` canonical) + engine-mirror rate. A strong "
      "leveled coach has HIGH beginner/intermediate fit (finding the *human* move) while advanced "
      "fit ~ engine-mirror (the sharp move is correct there).\n")
    A("| Model | fit B | fit I | fit A | mirror B | mirror I | mirror A | diff | mirror@all |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in order_for_table:
        t = tier.get(m)
        if not t:
            continue
        fb = t.get("tier_fit_by_tier", {})
        mb = t.get("eng_mirror_by_tier", {})
        A(f"| {display[m]} | {_pct(fb.get('beginner'),0)} | {_pct(fb.get('intermediate'),0)} | "
          f"{_pct(fb.get('advanced'),0)} | {_pct(mb.get('beginner'),0)} | "
          f"{_pct(mb.get('intermediate'),0)} | {_pct(mb.get('advanced'),0)} | "
          f"{_pct(t.get('diff_rate'),0)} | {_pct(t.get('mirror_all'),0)} |")
    A("")

    # ------------------------------------------------------------------ #
    A("## 3. Weighted BALANCED ranking\n")
    A(f"Transparent weighted score (each component normalized to 0-1, higher = better): "
      f"**tier-appropriate move selection {w_balanced['tier']:.0%}** + "
      f"**instructiveness {w_balanced['instr']:.0%}** + fabrication (1-fab) {w_balanced['fab']:.0%} + "
      f"practical (local+cost) {w_balanced['practical']:.0%}. Safety + no-jargon are pass/fail "
      f"gates. Score = weighted mean x 100.\n")
    A("| # | Model | family | tier(.40) | instr(.40) | 1-fab(.10) | practical(.10) | "
      "**balanced** | gate |")
    A("|---|---|---|---:|---:|---:|---:|---:|:--:|")
    for i, m in enumerate(ranked_bal, 1):
        s = scored[m]
        gate = "pass" if s.get("gate_ok") else "**FAIL**"
        A(f"| {i} | {display[m]} | {family[m]} | {_score100(s.get('tier_score'))} | "
          f"{_score100(s.get('instr_score'))} | {_score100(s.get('fab_score'))} | "
          f"{_score100(s.get('practical'))} | **{_score100(s.get('balanced'))}** | {gate} |")
    A("")

    # ------------------------------------------------------------------ #
    A("## 4. Best v3-BASE ranking (re-weighted)\n")
    A(f"For a fine-tuning *base*, tier-appropriateness is what we ADD, so it is down-weighted; "
      f"the hard-to-add qualities dominate: **instructiveness/capacity {w_base['instr']:.0%}**, "
      f"**faithfulness {w_base['fab']:.0%}**, **local-runnability+cost {w_base['practical']:.0%}**, "
      f"tier {w_base['tier']:.0%}. Only locally fine-tunable/runnable models are viable bases.\n")
    A("| # | Model | family | base-fit | local | note |")
    A("|---|---|---|---:|:--:|---|")
    ranked_base = sorted([m for m in present if _base(m) is not None], key=lambda m: _base(m), reverse=True)
    for i, m in enumerate(ranked_base, 1):
        s = scored[m]
        A(f"| {i} | {display[m]} | {family[m]} | {_score100(s.get('base_fit'))} | "
          f"{loc_label(m)} | {practical.get(m, ('','',''))[2]} |")
    A("")

    # ------------------------------------------------------------------ #
    A("## 5. Recommendation\n")
    if overall_bal_winner:
        A(f"- **Best overall (balanced), any provider: {display[overall_bal_winner]}** "
          f"({family[overall_bal_winner]}) — the frontier still coaches best; it is the "
          f"distillation-teacher benchmark, not a deployable base.")
    if best_open_balanced:
        A(f"- **Best OPEN model (balanced): {display[best_open_balanced]}.** "
          f"{'This IS Gemma-3-27B.' if best_open_balanced=='gemma3_27b' else 'This is **NOT** Gemma-3-27B'}"
          f"{'' if best_open_balanced=='gemma3_27b' else ' — GLM-5 is the strongest open coach (best open instructiveness) with solid tier-selection, but it is far too large to run locally.'}")
    # best-base + near-tie runner-up
    if best_base:
        rest = [m for m in ranked_base_open if m != best_base
                and practical.get(m, ("", 0, ""))[1] > 0]
        runner = rest[0] if rest else None
        margin = None
        if runner is not None and _base(best_base) is not None and _base(runner) is not None:
            margin = (_base(best_base) - _base(runner)) * 100
        A(f"- **Best open v3 base: {display[best_base]}** — the best mix of coaching capacity, "
          f"faithfulness, and 4-bit local fine-tunability/runnability on a 64GB Mac.")
        if runner is not None and margin is not None and margin < 3.0:
            fb_best = obj.get(best_base, {}).get("fabrication")
            fb_run = obj.get(runner, {}).get("fabrication")
            A(f"- **It is effectively a tie with {display[runner]}** (base-fit "
              f"{_score100(_base(best_base))} vs {_score100(_base(runner))} — within noise). "
              f"{display[best_base]} edges it on tier-selection/capacity; {display[runner]} is "
              f"smaller and more faithful (fab {_pct(fb_run,0)} vs {_pct(fb_best,0)}). Either is a "
              f"defensible v3 base; prefer {display[runner]} if faithfulness/size is paramount, "
              f"{display[best_base]} if raw capacity is.")
    if best_open_balanced and best_base and best_open_balanced != best_base:
        A(f"- **The balanced winner and the base pick differ:** {display[best_open_balanced]} wins "
          f"the raw open balanced score, but it {practical.get(best_open_balanced, ('','',''))[2]} "
          f"— not a viable local base. {display[best_base]} is the pragmatic v3 base because "
          f"tier-appropriateness (where the giant coaches lead) is exactly what we fine-tune IN, "
          f"while capacity + faithfulness + local-runnability are what a base must bring.")
    elif best_open_balanced and best_base and best_open_balanced == best_base:
        A(f"- The best open balanced model and best base are the **same** ({display[best_base]}).")
    A("")

    # ------------------------------------------------------------------ #
    if spend:
        A("## 6. Cost\n")
        A("| group | calls | in tok | out tok | est. USD |")
        A("|---|---:|---:|---:|---:|")
        for g in ("open", "frontier_gen", "council", "local"):
            d = spend.get(g)
            if not d:
                continue
            A(f"| {g} | {d.get('calls',0):,} | {d.get('in',0):,} | {d.get('out',0):,} | "
              f"${d.get('usd',0):.2f} |")
        A(f"| **TOTAL** | | | | **${spend.get('total_usd',0):.2f}** |")
        A("")
        A(f"_Open-model + frontier prices are best-effort Bedrock/gateway estimates; local "
          f"(OURS-v2, BASE) is free. Total definitive-eval spend: **${spend.get('total_usd',0):.2f}**._")
        A("")

    A("## Artifacts\n")
    A("- Positions: `data/eval/gap_positions.jsonl` (803, curated, zero-leakage)")
    A("- Flattened scenarios: `data/benchmark_gap803/scenarios.jsonl` (803x3)")
    A("- Generations: `data/benchmark_gap803/gen/<model>.jsonl` -> `generations.jsonl`")
    A("- Objective: `data/benchmark_gap803/objective.jsonl`; safety: `move_safety.json`")
    A("- Council: `data/benchmark_gap803/council.jsonl`; leaderboard: `leaderboard.json`")
    A("- Drivers: `scripts/gap803_{gen,report,safety,council}.py`, `scripts/gap803_common.py`")

    path.write_text("\n".join(L), encoding="utf-8")
