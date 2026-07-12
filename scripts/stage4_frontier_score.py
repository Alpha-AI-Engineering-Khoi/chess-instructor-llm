#!/usr/bin/env python3
"""Score the 120-TEST MATCHED-GROUNDING field: v6-dpo2 / v6-dpo / v4 / base + 3 frontier.

Deterministic, local scoring with the SAME vendored extractor + tier-policy logic the
shipped v4 report uses (``src.eval.evaluate.extract_recommended_move``, which
``scripts/reproduce_v4.py`` asserts against). The ``score_condition`` /``_extract``
here are byte-identical to ``scripts/stage4_eval_v6dpo2.score_condition`` — the exact
function that produced v6-dpo2's committed 0.892 — so the frontier rows land on the
same ruler as ours.

Reads generations for each model from disk and scores on the 120 held-out TEST x 3
tiers (``data/benchmark_gap803/stage4_eval_inputs.jsonl``):

    base   -> stage4/base_grounded.jsonl        (fresh Stage-4 grounding)
    v4     -> stage4/v4_grounded.jsonl           (fresh Stage-4 grounding)
    v6_dpo -> stage4/v6dpo_grounded.jsonl        (fresh Stage-4 grounding)
    v6_dpo2-> stage4_v6dpo2/v6dpo2_grounded.jsonl (fresh Stage-4 grounding)
    gpt/claude/gemini -> stage4_frontier/{k}.jsonl (SAME fresh grounding; this stage)

It SELF-VALIDATES that the four OURS rows reproduce the committed
``stage4/scores.json`` + ``stage4_v6dpo2/scores.json`` numbers before writing the
combined ``data/benchmark_gap803/stage4_frontier/scores.json``. Frontier rows are
written even if incomplete (with an explicit ``n`` and coverage note) so a partial
run is visible, never silently averaged over a short set.

Run::

    python scripts/stage4_frontier_score.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TIERS: Tuple[str, ...] = ("beginner", "intermediate", "advanced")

GAP = _ROOT / "data" / "benchmark_gap803"
INPUTS = GAP / "stage4_eval_inputs.jsonl"
OUT = GAP / "stage4_frontier" / "scores.json"

# model key -> generation file. OURS from the committed Stage-4 fresh-grounding runs,
# frontier from this stage's matched-grounding run.
GEN_FILES: Dict[str, Path] = {
    "base": GAP / "stage4" / "base_grounded.jsonl",
    "v4": GAP / "stage4" / "v4_grounded.jsonl",
    "v6_dpo": GAP / "stage4" / "v6dpo_grounded.jsonl",
    "v6_dpo2": GAP / "stage4_v6dpo2" / "v6dpo2_grounded.jsonl",
    "gpt": GAP / "stage4_frontier" / "gpt.jsonl",
    "claude": GAP / "stage4_frontier" / "claude.jsonl",
    "gemini": GAP / "stage4_frontier" / "gemini.jsonl",
}
FIELD_ORDER: Tuple[str, ...] = ("v6_dpo2", "v6_dpo", "v4", "base", "gpt", "claude", "gemini")

DISPLAY: Dict[str, str] = {
    "base": "BASE (Qwen3-32B untuned)",
    "v4": "OURS-v4 (shipped)",
    "v6_dpo": "OURS-v6-dpo",
    "v6_dpo2": "OURS-v6-dpo2",
    "gpt": "GPT-5.5",
    "claude": "Claude Opus 4.8",
    "gemini": "Gemini 3.1 Pro",
}

TOL = 1e-3  # committed numbers are 4dp; an exact recompute matches within this


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# --------------------------------------------------------------------------- #
# Byte-identical to scripts/stage4_eval_v6dpo2.py::{_extract, score_condition}
# --------------------------------------------------------------------------- #
def _extract(output: str, fen: str, student_uci: str) -> Optional[str]:
    from src.eval.evaluate import extract_recommended_move
    _san, uci = extract_recommended_move(output or "", fen, student_uci or "")
    return uci


def score_condition(outputs: List[dict], inputs: List[dict]) -> Dict[str, Any]:
    by_id = {r["id"]: r for r in inputs}
    by_tier: Dict[str, List[int]] = {t: [0, 0] for t in TIERS}
    sound = [0, 0]
    named = [0, 0]
    fmt = [0, 0]
    preds_by_pos: Dict[str, Dict[str, Optional[str]]] = {}
    canon_by_pos: Dict[str, Dict[str, Optional[str]]] = {}
    for o in outputs:
        r = by_id.get(o["id"])
        if r is None:
            continue
        tier = r["tier"]
        uci = _extract(o["output"], r["fen"], r.get("student_uci") or "")
        if tier in by_tier:
            by_tier[tier][1] += 1
            if uci and uci == r.get("canonical_uci"):
                by_tier[tier][0] += 1
        sound[1] += 1
        if uci and uci in set(r.get("sound_ucis", [])):
            sound[0] += 1
        named[1] += 1
        if uci:
            named[0] += 1
        fmt[1] += 1
        text = (o["output"] or "")
        if uci and ("I'd play" in text or "I\u2019d play" in text) and "Takeaway:" in text:
            fmt[0] += 1
        preds_by_pos.setdefault(r["pos_id"], {})[tier] = uci
        canon_by_pos.setdefault(r["pos_id"], {})["beginner"] = r.get("canonical_beginner_uci")
        canon_by_pos[r["pos_id"]]["advanced"] = r.get("canonical_advanced_uci")

    per_tier = {t: (by_tier[t][0] / by_tier[t][1]) for t in TIERS if by_tier[t][1]}
    diff = dist = 0
    for pid, cd in canon_by_pos.items():
        cb, ca = cd.get("beginner"), cd.get("advanced")
        if not (cb and ca and cb != ca):
            continue
        diff += 1
        mb = preds_by_pos.get(pid, {}).get("beginner")
        ma = preds_by_pos.get(pid, {}).get("advanced")
        if mb and ma and mb != ma:
            dist += 1
    return {
        "tier_policy_match": round(mean(per_tier.values()), 4) if per_tier else 0.0,
        "per_tier": {t: round(v, 4) for t, v in per_tier.items()},
        "per_tier_counts": {t: by_tier[t] for t in TIERS if by_tier[t][1]},
        "move_sound": round(sound[0] / sound[1], 4) if sound[1] else 0.0,
        "named_rate": round(named[0] / named[1], 4) if named[1] else 0.0,
        "format_rate": round(fmt[0] / fmt[1], 4) if fmt[1] else 0.0,
        "distinct_rate": round(dist / diff, 4) if diff else 0.0,
        "distinct_counts": [dist, diff],
        "n": len(outputs),
    }


# --------------------------------------------------------------------------- #
# Self-validation: OURS rows must reproduce the committed Stage-4 numbers
# --------------------------------------------------------------------------- #
def _validate_ours(scores: Dict[str, Any]) -> None:
    committed: Dict[str, float] = {}
    s4 = json.loads((GAP / "stage4" / "scores.json").read_text())["scores"]
    committed["base"] = s4["base_grounded"]["tier_policy_match"]
    committed["v4"] = s4["v4_grounded"]["tier_policy_match"]
    committed["v6_dpo"] = s4["v6dpo_grounded"]["tier_policy_match"]
    s4d2 = json.loads((GAP / "stage4_v6dpo2" / "scores.json").read_text())["scores"]
    committed["v6_dpo2"] = s4d2["v6dpo2_grounded"]["tier_policy_match"]

    print("Self-validation — recomputed OURS tier-policy vs committed Stage-4:")
    for k, want in committed.items():
        got = scores[k]["tier_policy_match"]
        delta = abs(got - want)
        status = "OK" if delta <= TOL else "MISMATCH"
        print(f"  {k:8} recomputed={got:.4f}  committed={want:.4f}  Δ={delta:.5f}  [{status}]")
        assert delta <= TOL, (
            f"{k}: recomputed {got:.4f} != committed {want:.4f}. The vendored scorer is "
            f"NOT byte-identical to the run that produced v6-dpo2 — refusing to publish.")
    print("  PASS — scorer reproduces the committed OURS numbers exactly.\n")


def main() -> int:
    inputs = _load_jsonl(INPUTS)
    n_expected = len(inputs)
    print(f"120-TEST matched-grounding scoring: {n_expected} scenarios "
          f"({len({r['pos_id'] for r in inputs})} positions x 3 tiers)\n")

    scores: Dict[str, Any] = {}
    coverage: Dict[str, int] = {}
    for key in FIELD_ORDER:
        p = GEN_FILES[key]
        if not p.exists():
            print(f"  [{key}] MISSING gen file {p} — skipping (run stage4_frontier_gen.py)")
            continue
        gens = _load_jsonl(p)
        # de-dup by id keeping the last occurrence (resumable append safety)
        gens = list({g["id"]: g for g in gens if g.get("id")}.values())
        s = score_condition(gens, inputs)
        s["display"] = DISPLAY[key]
        scores[key] = s
        coverage[key] = s["n"]

    _validate_ours(scores)

    report = {
        "benchmark": "scenarios_v6 (corrected labels), 120 held-out TEST x 3 tiers",
        "scope": "MATCHED FRESH GROUNDING — every model (ours + frontier) scored on the "
                 "SAME stage4_eval_inputs.jsonl grounded prompt (Stockfish sound-pool + "
                 "Maia + verify-gate facts) that produced v6-dpo2's 0.892.",
        "n_scenarios": n_expected,
        "extractor": "src.eval.evaluate.extract_recommended_move (== reproduce_v4.py)",
        "frontier_gen": "TrueFoundry gateway (src.eval.benchmark.backends.TFYChat), "
                        "GEN_MAX_TOKENS_TFY, per-model reasoning_effort; NOT Modal.",
        "field_order": list(FIELD_ORDER),
        "scores": scores,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== 120-TEST matched-grounding tier-policy match (overall + per tier) ===")
    print(f"{'model':26} {'n':>4} {'tier_fit':>9} {'beg':>7} {'int':>7} {'adv':>7} "
          f"{'sound':>7} {'named':>7}")
    for key in FIELD_ORDER:
        if key not in scores:
            continue
        s = scores[key]
        pt = s["per_tier"]
        flag = "" if s["n"] == n_expected else f"  <-- PARTIAL ({s['n']}/{n_expected})"
        print(f"{DISPLAY[key]:26} {s['n']:>4} {s['tier_policy_match']:>9.4f} "
              f"{pt.get('beginner',0):>7.4f} {pt.get('intermediate',0):>7.4f} "
              f"{pt.get('advanced',0):>7.4f} {s['move_sound']:>7.4f} {s['named_rate']:>7.4f}{flag}")
    print(f"\nwrote -> {OUT}")

    incomplete = [k for k in scores if scores[k]["n"] != n_expected]
    if incomplete:
        print(f"\nNOTE: incomplete field(s): {incomplete} — re-run stage4_frontier_gen.py to finish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
