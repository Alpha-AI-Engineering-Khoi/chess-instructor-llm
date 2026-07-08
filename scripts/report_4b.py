#!/usr/bin/env python3
"""4B-iteration headline for the HONEST eval — computed from the SAME artifacts the
shared ``scripts.honest_eval`` harness produces (gen/*.jsonl, council.jsonl, the
val slice), using the harness's OWN helper functions (no rebuild).

The shared ``honest_eval report`` headline is hard-wired to the 1.7B pair, so this
script emits the 4B deliverable the loop needs:

  A. base_4b -> ours_4b delta (tier-fit, instructiveness rank, 6-dim rubric)
  B. litmus: does the best prompt-engineered 4B base match/beat ours_4b?
  C. distance to frontier (best frontier council rank vs ours_4b)
  D. deterministic gate pass-rates vs the 100% targets
  E. distinct-moves-per-level on DIFFERENTIATING positions (beginner != advanced)

Writes ``RESULTS_HONEST_EVAL_4B.md`` + prints a compact console + JSON block.

    ~/.venvs/mlx/bin/python -m scripts.report_4b
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import gap803_council_stats as CS  # noqa: E402
from scripts.honest_eval import (  # noqa: E402
    COUNCIL_PATH, FIELD, FRONTIER_KEYS, GEN_DIR, HONEST_MODELS, VAL_IDS,
    _gate_stats, _load_all_scenarios, _read_jsonl, _rec_by_model_pos_tier,
    _slice_scenarios, _tier_fit,
)
from src.eval.honest import rubric as R  # noqa: E402

OUT_MD = Path(_ROOT / "RESULTS_HONEST_EVAL_4B.md")
OUT_JSON = Path(_ROOT / "data" / "benchmark_honest" / "report_4b.json")

TUNED, BASE, PBASE = "ours_4b", "base_4b", "pbase_4b"
R_SIX = ("move_purpose", "transferable_principle", "board_specific_reason",
         "how_to_find", "level_calibration", "grounded_concise")


def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else round(a - b, 4)


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{float(x):.3f}" if abs(x) < 10 else f"{float(x):.2f}"
    return str(x)


def main() -> int:
    scns = _slice_scenarios(VAL_IDS, _load_all_scenarios())
    scns_by_id = {s["id"]: s for s in scns}
    field = [m for m in FIELD if (GEN_DIR / f"{m}.jsonl").exists()]
    council = _read_jsonl(COUNCIL_PATH)

    rank_stats = CS.compute_council_stats(council, field=field, frontier=FRONTIER_KEYS)
    rankmap = rank_stats.get("models", {})
    dims = R.dim_means(council, field)
    tier = _tier_fit(field, scns_by_id)
    gates = _gate_stats(field)
    rec = _rec_by_model_pos_tier(field, scns_by_id)
    coh = R.tier_coherence(rec, scns_by_id)

    def rank(mk: str) -> Optional[float]:
        m = rankmap.get(mk)
        return m.get("mean_rank") if m else None

    def instr(mk: str) -> Optional[float]:
        return (dims.get(mk) or {}).get("sum_0_12")

    def tf(mk: str) -> Optional[float]:
        return (tier.get(mk) or {}).get("tier_fit_mean")

    def ms(mk: str) -> Optional[float]:
        return (tier.get(mk) or {}).get("move_sound")

    # ---- E. distinct-moves-per-level on DIFFERENTIATING positions --------- #
    canon: Dict[str, Dict[str, Optional[str]]] = {}
    for s in scns:
        canon.setdefault(s["pos_id"], {})[s["tier"]] = s.get("canonical_uci")

    def distinct(mk: str) -> Dict[str, Any]:
        picks = rec.get(mk, {})
        n = d = 0
        for pid, tp in picks.items():
            cb, ca = canon.get(pid, {}).get("beginner"), canon.get(pid, {}).get("advanced")
            mb, ma = tp.get("beginner"), tp.get("advanced")
            if cb and ca and cb != ca and mb and ma:   # differentiating + both present
                n += 1
                if mb != ma:
                    d += 1
        return {"differentiating_n": n,
                "distinct_rate": (round(d / n, 4) if n else None),
                "collapsed_BA": n - d}

    def well_formed(mk: str) -> Optional[float]:
        rows = [r for r in _read_jsonl(GEN_DIR / f"{mk}.jsonl") if not r.get("reused_ungated")]
        if not rows:
            return None
        return round(sum(1 for r in rows if r.get("rec_uci")) / len(rows), 4)

    # ---- headline --------------------------------------------------------- #
    best_fr = None
    for mk in FRONTIER_KEYS:
        r = rank(mk)
        if r is not None and (best_fr is None or r < best_fr[1]):
            best_fr = (mk, r)

    litmus_matches = None
    if None not in (rank(PBASE), rank(TUNED), tf(PBASE), tf(TUNED)):
        litmus_matches = bool(rank(PBASE) <= rank(TUNED) and tf(PBASE) >= tf(TUNED))

    headline = {
        "A_base_vs_tuned": {
            "tier_fit": {BASE: tf(BASE), TUNED: tf(TUNED), "delta": _sub(tf(TUNED), tf(BASE))},
            "instr_rank_lower_better": {BASE: rank(BASE), TUNED: rank(TUNED),
                                        "delta_ours_minus_base": _sub(rank(TUNED), rank(BASE))},
            "instr_0_12": {BASE: instr(BASE), TUNED: instr(TUNED), "delta": _sub(instr(TUNED), instr(BASE))},
        },
        "B_litmus_prompt_vs_tune": {
            "prompt_base": PBASE, "tune": TUNED,
            "instr_rank": {PBASE: rank(PBASE), TUNED: rank(TUNED), "delta": _sub(rank(PBASE), rank(TUNED))},
            "tier_fit": {PBASE: tf(PBASE), TUNED: tf(TUNED), "delta": _sub(tf(PBASE), tf(TUNED))},
            "instr_0_12": {PBASE: instr(PBASE), TUNED: instr(TUNED), "delta": _sub(instr(PBASE), instr(TUNED))},
            "prompt_matches_or_beats_tune": litmus_matches,
        },
        "C_distance_to_frontier": {
            "best_frontier": best_fr, "ours_4b_rank": rank(TUNED),
            "gap_ours4b_minus_bestfrontier": _sub(rank(TUNED), best_fr[1] if best_fr else None),
            "ours_v3_rank_ref": rank("ours_v3"),
        },
        "D_gates_vs_100pct": {
            mk: {"move_sound": ms(mk), "no_engine_speak": (gates.get(mk) or {}).get("no_jargon"),
                 "well_formed": well_formed(mk),
                 "gate_fallback_rate": (gates.get(mk) or {}).get("fallback_rate"),
                 "mean_attempts": (gates.get(mk) or {}).get("mean_attempts")}
            for mk in (TUNED, BASE, PBASE)},
        "E_distinct_moves_per_level": {
            mk: {**distinct(mk),
                 "zigzag_BA_ne_I_rate": (coh.get(mk) or {}).get("zigzag_rate"),
                 "flat_rate": (coh.get(mk) or {}).get("flat_rate"),
                 "coherence_violation_rate": (coh.get(mk) or {}).get("violation_rate")}
            for mk in field},
    }

    report = {
        "n_val_positions": len(scns) // 3,
        "council": {"n_items": rank_stats.get("n_items"), "n_judges": rank_stats.get("n_judges"),
                    "n_rankings": rank_stats.get("n_rankings")},
        "field": field, "headline": headline,
        "per_model": {mk: {"mean_rank": rank(mk), "tier_fit": tf(mk), "move_sound": ms(mk),
                           "instr_0_12": instr(mk), "dims": (dims.get(mk) or {}).get("dims"),
                           "coherence": coh.get(mk), "gate": gates.get(mk),
                           "distinct": distinct(mk) if mk in rec else None}
                      for mk in field},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(report, rankmap, dims, tier, gates, coh, field)
    _print_console(headline, report)
    return 0


def _disp(mk: str) -> str:
    m = HONEST_MODELS.get(mk)
    return m.display if m else mk


def _write_md(rep, rankmap, dims, tier, gates, coh, field) -> None:
    h = rep["headline"]
    L: List[str] = []
    L.append("# HONEST 4B base-vs-tuned eval — Qwen3-4B chess coach (iter1)\n")
    L.append("Every gated 4B contender coaches the SAME held-out positions through the **identical "
             "shipped pipeline** (grounding + `src.teacher.coach_gate.run_gate`), so `base_4b` vs "
             "`ours_4b` differ ONLY in the LoRA weights and `pbase_4b` differs ONLY in its system "
             "prompt. `ours_4b` = `mlx-community/Qwen3-4B-Instruct-2507-4bit` + our iter1 LoRA fused "
             "into the identical MLX base. Frontier (GPT-5.5 / Claude Opus 4.8 / Gemini 3.1 Pro via "
             "TrueFoundry) + `ours_v3` (our 32B tuned) rows are REUSED ungated references.\n")
    L.append(f"- **Validation slice:** {rep['n_val_positions']} positions x 3 tiers; council "
             f"n_items={rep['council']['n_items']}, judges={rep['council']['n_judges']}, "
             f"rankings={rep['council']['n_rankings']}.\n")

    a = h["A_base_vs_tuned"]
    L.append("## Headline\n")
    L.append("**A. Training as the only variable (4B, identical gated pipeline):**")
    L.append(f"- tier-fit (canonical tier move): ours_4b {_fmt(a['tier_fit'][TUNED])} vs base_4b "
             f"{_fmt(a['tier_fit'][BASE])} (**Δ {_fmt(a['tier_fit']['delta'])}**).")
    L.append(f"- instructiveness council mean rank (lower=better): ours_4b "
             f"{_fmt(a['instr_rank_lower_better'][TUNED])} vs base_4b "
             f"{_fmt(a['instr_rank_lower_better'][BASE])} "
             f"(**Δ {_fmt(a['instr_rank_lower_better']['delta_ours_minus_base'])}**).")
    L.append(f"- instructiveness rubric sum (0-12): ours_4b {_fmt(a['instr_0_12'][TUNED])} vs base_4b "
             f"{_fmt(a['instr_0_12'][BASE])} (**Δ {_fmt(a['instr_0_12']['delta'])}**).\n")

    b = h["B_litmus_prompt_vs_tune"]
    verdict = ("PROMPT MATCHES/BEATS TUNE" if b["prompt_matches_or_beats_tune"]
               else "tune still wins" if b["prompt_matches_or_beats_tune"] is False else "n/a")
    L.append(f"**B. Litmus — can the best prompt-engineered 4B base match the tune?** **{verdict}**")
    L.append(f"- instr rank: pbase_4b {_fmt(b['instr_rank'][PBASE])} vs ours_4b "
             f"{_fmt(b['instr_rank'][TUNED])} (Δ {_fmt(b['instr_rank']['delta'])}); "
             f"tier-fit Δ {_fmt(b['tier_fit']['delta'])}; 6-dim Δ {_fmt(b['instr_0_12']['delta'])}.\n")

    c = h["C_distance_to_frontier"]
    bf = c["best_frontier"]
    L.append(f"**C. Distance to frontier:** best frontier = {bf[0] if bf else '—'} "
             f"(rank {_fmt(bf[1] if bf else None)}); ours_4b rank {_fmt(c['ours_4b_rank'])}; "
             f"gap ours_4b−bestfrontier = {_fmt(c['gap_ours4b_minus_bestfrontier'])} rank positions "
             f"(ref: ours_v3 32B rank {_fmt(c['ours_v3_rank_ref'])}).\n")

    L.append("**D. Deterministic gate pass-rates (targets = 100% / 0% fabrication):**")
    for mk in (TUNED, BASE, PBASE):
        g = h["D_gates_vs_100pct"][mk]
        L.append(f"- {mk}: move-sound {_fmt(g['move_sound'])}, no-engine-speak {_fmt(g['no_engine_speak'])}, "
                 f"well-formed {_fmt(g['well_formed'])} (gate fallback {_fmt(g['gate_fallback_rate'])}, "
                 f"mean attempts {_fmt(g['mean_attempts'])}). Post-gate fabrication = 0 by gate design.")
    L.append("")

    L.append("**E. Distinct-moves-per-level on DIFFERENTIATING positions "
             "(canonical beginner≠advanced; target ≥95%):**")
    for mk in (TUNED, BASE, PBASE):
        e = h["E_distinct_moves_per_level"].get(mk, {})
        L.append(f"- {mk}: {_fmt(e.get('distinct_rate'))} distinct over {e.get('differentiating_n')} "
                 f"differentiating positions ({e.get('collapsed_BA')} B==A collapses); "
                 f"zigzag(B==A≠I) {_fmt(e.get('zigzag_BA_ne_I_rate'))}, flat {_fmt(e.get('flat_rate'))}.")
    L.append("")

    order = sorted(field, key=lambda m: (rankmap.get(m, {}).get("mean_rank", 99)))
    L.append("## Leaderboard (validation field)\n")
    L.append("| Model | gated | tier-fit↑ | instr rank↓ | 6-dim/12↑ | move-sound↑ | distinct↑ | coh-viol↓ |")
    L.append("|---|:--:|---:|---:|---:|---:|---:|---:|")
    hd = rep["headline"]["E_distinct_moves_per_level"]
    for mk in order:
        r = rankmap.get(mk, {})
        dist = (hd.get(mk) or {}).get("distinct_rate")
        L.append(f"| {_disp(mk)} | {'yes' if (gates.get(mk,{}) or {}).get('gated') else 'reuse'} | "
                 f"{_fmt((tier.get(mk,{}) or {}).get('tier_fit_mean'))} | {_fmt(r.get('mean_rank'))} | "
                 f"{_fmt((dims.get(mk,{}) or {}).get('sum_0_12'))} | "
                 f"{_fmt((tier.get(mk,{}) or {}).get('move_sound'))} | {_fmt(dist)} | "
                 f"{_fmt((coh.get(mk,{}) or {}).get('violation_rate'))} |")
    L.append("")

    L.append("## Instructiveness rubric — six dimensions (mean 0/1/2)\n")
    L.append("| Model | " + " | ".join(d.replace('_', ' ') for d in R_SIX) + " |")
    L.append("|---" + "|---:" * len(R_SIX) + "|")
    for mk in order:
        dd = (dims.get(mk, {}) or {}).get("dims", {}) or {}
        L.append(f"| {_disp(mk)} | " + " | ".join(_fmt(dd.get(d)) for d in R_SIX) + " |")
    L.append("")

    L.append("## Gate telemetry (gated contenders)\n")
    L.append("| Model | mean attempts | fallback rate | no-engine-speak |")
    L.append("|---|---:|---:|---:|")
    for mk in order:
        g = gates.get(mk, {}) or {}
        if g.get("gated"):
            L.append(f"| {_disp(mk)} | {_fmt(g.get('mean_attempts'))} | {_fmt(g.get('fallback_rate'))} "
                     f"| {_fmt(g.get('no_jargon'))} |")
    L.append("")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def _print_console(h, rep) -> None:
    print("\n=== 4B HONEST EVAL HEADLINE ===")
    print(f"val positions={rep['n_val_positions']}  council items={rep['council']['n_items']} "
          f"judges={rep['council']['n_judges']} rankings={rep['council']['n_rankings']}")
    print(json.dumps(h, ensure_ascii=False, indent=2))
    print(f"\nwrote -> {OUT_MD}\nwrote -> {OUT_JSON}")


if __name__ == "__main__":
    raise SystemExit(main())
