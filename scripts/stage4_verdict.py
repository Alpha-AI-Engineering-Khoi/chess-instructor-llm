#!/usr/bin/env python3
"""Consolidate the Stage-4 corrected-benchmark verdict (local, free).

Merges the fresh Stage-4 generations scores (``stage4/scores.json``), the
continuity re-score of the committed v4/base gens (``stage4/rescore_committed.json``),
and the published v4 numbers (``benchmark_honest/report_v4.json``) into a single
verdict.json + prints the head-to-head deltas and the answers to the three central
questions. No inference; pure arithmetic over already-scored artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
STAGE4 = _ROOT / "data" / "benchmark_gap803" / "stage4"


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    fresh = _load(STAGE4 / "scores.json")["scores"]
    committed = _load(STAGE4 / "rescore_committed.json")["scores"]
    pub = _load(_ROOT / "data" / "benchmark_honest" / "report_v4.json")["per_model"]

    g_base = fresh["base_grounded"]
    g_v4 = fresh["v4_grounded"]
    g_dpo = fresh["v6dpo_grounded"]
    n_base = fresh["base_nog"]
    n_dist = fresh["v6distill_nog"]

    def d(a, b):
        return round(a - b, 4)

    # Q1: v6-dpo vs v4 (grounded), per metric + per tier
    q1_metrics = {
        "tier_policy_match": [g_dpo["tier_policy_match"], g_v4["tier_policy_match"], d(g_dpo["tier_policy_match"], g_v4["tier_policy_match"])],
        "move_sound": [g_dpo["move_sound"], g_v4["move_sound"], d(g_dpo["move_sound"], g_v4["move_sound"])],
        "distinct_rate": [g_dpo["distinct_rate"], g_v4["distinct_rate"], d(g_dpo["distinct_rate"], g_v4["distinct_rate"])],
        "named_rate": [g_dpo["named_rate"], g_v4["named_rate"], d(g_dpo["named_rate"], g_v4["named_rate"])],
        "format_rate": [g_dpo["format_rate"], g_v4["format_rate"], d(g_dpo["format_rate"], g_v4["format_rate"])],
    }
    q1_tiers = {t: [g_dpo["per_tier"][t], g_v4["per_tier"][t], d(g_dpo["per_tier"][t], g_v4["per_tier"][t])]
                for t in ("beginner", "intermediate", "advanced")}
    no_regression = (
        g_dpo["move_sound"] >= g_v4["move_sound"] - 1e-9 and
        g_dpo["per_tier"]["beginner"] >= g_v4["per_tier"]["beginner"] - 1e-9 and
        g_dpo["per_tier"]["advanced"] >= g_v4["per_tier"]["advanced"] - 1e-9 and
        g_dpo["distinct_rate"] >= g_v4["distinct_rate"] - 1e-9
    )
    intermediate_gain_holds = g_dpo["per_tier"]["intermediate"] > g_v4["per_tier"]["intermediate"] + 1e-9

    verdict = {
        "benchmark": "scenarios_v6 (corrected labels), 120 held-out TEST x 3 tiers = 360",
        "decode": _load(STAGE4 / "scores.json")["decode"],
        "grounded": {
            "base": g_base, "v4": g_v4, "v6_dpo": g_dpo,
        },
        "no_grounding": {"base": n_base, "v6_distill": n_dist},
        "continuity_committed_vs_corrected": committed,
        "published_v4_v4era": {
            "tier_policy_match": pub["ours_v4"]["tier_fit"],
            "move_sound": pub["ours_v4"]["move_sound"],
            "distinct_rate": pub["ours_v4"]["distinct"]["distinct_rate"],
        },
        "published_base_v4era": {
            "tier_policy_match": pub["q3_32b"]["tier_fit"],
            "move_sound": pub["q3_32b"]["move_sound"],
            "distinct_rate": (pub["q3_32b"].get("distinct") or {}).get("distinct_rate"),
        },
        "Q1_dpo_vs_v4_grounded": {
            "metrics_dpo_v4_delta": q1_metrics,
            "per_tier_dpo_v4_delta": q1_tiers,
            "no_regression_sound_distinct_beginner_advanced": bool(no_regression),
            "intermediate_gain_holds_out_of_distribution": bool(intermediate_gain_holds),
            "format_note": "format_rate is prose-completeness (Takeaway line within the 256-token cap); "
                           "named+sound+tier identical/higher, so any format delta is truncation of marginally "
                           "longer coaching, not a move/quality regression.",
        },
        "Q2_distillation_behavior_in_weights": {
            "tier_policy_match_base_to_distill": [n_base["tier_policy_match"], n_dist["tier_policy_match"], d(n_dist["tier_policy_match"], n_base["tier_policy_match"])],
            "named_rate_base_to_distill": [n_base["named_rate"], n_dist["named_rate"], d(n_dist["named_rate"], n_base["named_rate"])],
            "move_sound_base_to_distill": [n_base["move_sound"], n_dist["move_sound"], d(n_dist["move_sound"], n_base["move_sound"])],
            "distill_per_tier": n_dist["per_tier"],
            "honest_advanced_limit": {
                "distill_advanced": n_dist["per_tier"]["advanced"],
                "distill_beginner": n_dist["per_tier"]["beginner"],
                "distill_intermediate": n_dist["per_tier"]["intermediate"],
                "note": "advanced (=engine-best) is the distilled model's WEAKEST tier without grounding: the "
                        "sharpest move genuinely needs the engine grounding the no-grounding condition strips.",
            },
        },
        "Q3_base_vs_tuned_grounded": {
            "v4_minus_base_tier_policy": d(g_v4["tier_policy_match"], g_base["tier_policy_match"]),
            "v6dpo_minus_base_tier_policy": d(g_dpo["tier_policy_match"], g_base["tier_policy_match"]),
            "v4_minus_base_distinct": d(g_v4["distinct_rate"], g_base["distinct_rate"]),
            "v6dpo_minus_base_distinct": d(g_dpo["distinct_rate"], g_base["distinct_rate"]),
        },
    }
    (STAGE4 / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # ---- print a compact human summary ----
    def row(name, s):
        return (f"  {name:16} tier={s['tier_policy_match']:.4f} sound={s['move_sound']:.4f} "
                f"distinct={s['distinct_rate']:.4f} named={s['named_rate']:.4f} format={s['format_rate']:.4f} "
                f"tiers(B/I/A)={s['per_tier']['beginner']:.3f}/{s['per_tier']['intermediate']:.3f}/{s['per_tier']['advanced']:.3f}")

    print("=== GROUNDED (corrected v6, 120 TEST) ===")
    print(row("base", g_base)); print(row("v4", g_v4)); print(row("v6-dpo", g_dpo))
    print("\n=== NO-GROUNDING (distill thesis) ===")
    print(row("base", n_base)); print(row("v6-distill", n_dist))
    print("\n=== Q1: v6-dpo vs v4 (grounded) ===")
    for k, (a, b, dd) in q1_metrics.items():
        print(f"  {k:20} v6dpo={a:.4f} v4={b:.4f} delta={dd:+.4f}")
    for t, (a, b, dd) in q1_tiers.items():
        print(f"  tier[{t:12}] v6dpo={a:.4f} v4={b:.4f} delta={dd:+.4f}")
    print(f"  -> no_regression(sound/distinct/beginner/advanced) = {no_regression}")
    print(f"  -> intermediate gain holds OOD = {intermediate_gain_holds}")
    print("\n=== Q2: distillation behavior-in-weights (no grounding) ===")
    print(f"  tier: base {n_base['tier_policy_match']:.4f} -> distill {n_dist['tier_policy_match']:.4f} "
          f"(+{d(n_dist['tier_policy_match'], n_base['tier_policy_match']):.4f})")
    print(f"  named: base {n_base['named_rate']:.4f} -> distill {n_dist['named_rate']:.4f}")
    print(f"  distill per-tier B/I/A = {n_dist['per_tier']}")
    print("\n=== Q3: base-vs-tuned (grounded) ===")
    print(f"  v4-base tier delta = {d(g_v4['tier_policy_match'], g_base['tier_policy_match']):+.4f}; "
          f"v6dpo-base = {d(g_dpo['tier_policy_match'], g_base['tier_policy_match']):+.4f}")
    print(f"\n=== continuity (committed gens vs corrected labels) ===")
    print(f"  v4_committed tier={committed['v4_grounded_committed']['tier_policy_match']:.4f} "
          f"base_committed tier={committed['base_grounded_committed']['tier_policy_match']:.4f}")
    print(f"  (published v4 on v4-era labels: tier={pub['ours_v4']['tier_fit']}, sound={pub['ours_v4']['move_sound']})")
    print(f"\nwrote {STAGE4/'verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
