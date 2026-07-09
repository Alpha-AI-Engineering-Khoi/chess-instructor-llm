#!/usr/bin/env python3
"""Reproduce the headline v4 result from committed files — no GPU, no network.

This is the **LEVEL-1** reproduction: it *re-scores the published v4 generations
against the committed ground truth* and asserts the deterministic headline
numbers. It does **not** re-derive the ground truth (Stockfish sound pool / Maia
policy / ``select_tier_move`` canonical move) and does **not** re-run model
inference. Everything it reads is tracked in the repo, so a clean clone
reproduces the headline exactly with only ``python-chess`` installed.

What it checks
--------------
On the 120 held-out VAL positions x 3 tiers (360 scenarios), for OURS-v4:

- **tier-policy exact match** — agreement with the preregistered
  ``select_tier_move`` canonical move (``canonical_uci``), averaged over tiers.
- **distinct-moves-per-level** — of the positions whose canonical beginner and
  advanced moves differ, the fraction where v4's beginner and advanced picks
  also differ.
- **move soundness** — fraction of picks that land in the engine sound pool
  (``sound_uci``).

Inputs (all committed / tracked)
--------------------------------
- ``data/benchmark_honest/gen/ours_v4.jsonl``   — v4's 360 published generations
- ``data/benchmark_gap803/scenarios.jsonl``     — ground truth (canonical_uci / sound_uci / pool_policy)
- ``data/benchmark_honest/val_ids.txt``          — the 120 VAL position ids
- ``data/benchmark_honest/report_v4.json``       — the published numbers to assert against

Method fidelity
---------------
The recommended move is re-extracted from each generation's ``output`` with the
SAME strict, any-legal extractor the report uses
(:func:`src.eval.evaluate.extract_recommended_move`, which delegates to
:func:`src.teacher.coach_gate.pick_recommendation` with ``accept`` = any legal
move). It deliberately does NOT trust the stored ``rec_uci`` field, so the score
is derived from the raw published prose exactly as the report derives it. This
file is standalone and does NOT import ``scripts/honest_v4.py`` (that module
crashes on missing 4B generations and needs the full v4 gen pipeline).

Run
---
    python -m scripts.reproduce_v4

Exits 0 and prints the metrics when they match ``report_v4.json`` within
tolerance; raises AssertionError (nonzero exit) otherwise.
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

# The single canonical strict extractor the report uses (any-legal move).
from src.eval.evaluate import extract_recommended_move  # noqa: E402

TIERS: Tuple[str, ...] = ("beginner", "intermediate", "advanced")
#: report values are rounded to 4 dp; an exact recompute matches to 4 dp, so this
#: tolerance is comfortably tight while surviving float rounding.
TOL: float = 1e-3
EXPECTED_VAL: int = 120

HB = _ROOT / "data" / "benchmark_honest"
VAL_IDS = HB / "val_ids.txt"
OURS_V4_GEN = HB / "gen" / "ours_v4.jsonl"
REPORT_JSON = HB / "report_v4.json"
SCENARIOS = _ROOT / "data" / "benchmark_gap803" / "scenarios.jsonl"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _val_scenarios() -> Dict[str, Dict[str, Any]]:
    """Scenarios for the VAL slice, keyed by scenario id (``{pos_id}#{tier}``)."""
    keep = set(VAL_IDS.read_text(encoding="utf-8").split())
    return {s["id"]: s for s in _read_jsonl(SCENARIOS) if s.get("pos_id") in keep}


def _recommended_moves(
    gens: List[Dict[str, Any]], by_id: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Optional[str]]]:
    """pos_id -> tier -> recommended UCI, re-extracted from raw ``output``."""
    rec: Dict[str, Dict[str, Optional[str]]] = {}
    for row in gens:
        scn = by_id.get(row["scenario_id"])
        if scn is None:
            continue
        _san, uci = extract_recommended_move(
            row.get("output", ""), scn["fen"], scn["student_move"].get("uci") or ""
        )
        rec.setdefault(scn["pos_id"], {})[scn["tier"]] = uci
    return rec


def _tier_policy_and_soundness(
    rec: Dict[str, Dict[str, Optional[str]]], by_id: Dict[str, Dict[str, Any]]
) -> Tuple[float, float, Dict[str, float]]:
    """(tier-policy exact-match mean, move-sound rate, per-tier match rates)."""
    by_tier: Dict[str, List[int]] = {t: [0, 0] for t in TIERS}
    sound = [0, 0]
    for pos_id, picks in rec.items():
        for tier, uci in picks.items():
            scn = by_id.get(f"{pos_id}#{tier}")
            if scn is None:
                continue
            by_tier[tier][1] += 1
            if uci and uci == scn.get("canonical_uci"):
                by_tier[tier][0] += 1
            sound[1] += 1
            if uci and uci in set(scn.get("sound_uci", [])):
                sound[0] += 1
    per_tier = {t: (by_tier[t][0] / by_tier[t][1]) for t in TIERS if by_tier[t][1]}
    tier_policy_match = mean(per_tier.values()) if per_tier else 0.0
    move_sound = sound[0] / sound[1] if sound[1] else 0.0
    return tier_policy_match, move_sound, per_tier


def _distinct_rate(
    rec: Dict[str, Dict[str, Optional[str]]], by_id: Dict[str, Dict[str, Any]]
) -> Tuple[float, int, int]:
    """Distinct-moves-per-level over positions whose canonical beginner!=advanced."""
    canon: Dict[str, Dict[str, Optional[str]]] = {}
    for scn in by_id.values():
        canon.setdefault(scn["pos_id"], {})[scn["tier"]] = scn.get("canonical_uci")
    differentiating = distinct = 0
    for pos_id, picks in rec.items():
        cb, ca = canon.get(pos_id, {}).get("beginner"), canon.get(pos_id, {}).get("advanced")
        mb, ma = picks.get("beginner"), picks.get("advanced")
        if cb and ca and cb != ca and mb and ma:
            differentiating += 1
            if mb != ma:
                distinct += 1
    rate = distinct / differentiating if differentiating else 0.0
    return rate, differentiating, distinct


def _assert_close(name: str, got: float, want: float) -> None:
    delta = abs(got - want)
    status = "OK" if delta <= TOL else "MISMATCH"
    print(f"  {name:24} computed={got:.4f}  published={want:.4f}  Δ={delta:.5f}  [{status}]")
    assert delta <= TOL, (
        f"{name}: recomputed {got:.4f} != published {want:.4f} "
        f"(Δ={delta:.5f} > tol {TOL}). The headline did NOT reproduce from committed files."
    )


def main() -> int:
    by_id = _val_scenarios()
    n_val = len({scn["pos_id"] for scn in by_id.values()})
    gens = _read_jsonl(OURS_V4_GEN)
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    pub = report["per_model"]["ours_v4"]

    print("Reproduce v4 headline (LEVEL-1: re-score published generations vs committed ground truth)")
    print(f"  val positions : {n_val}   scenarios: {len(by_id)}   ours_v4 gens: {len(gens)}")

    assert n_val == EXPECTED_VAL, (
        f"val_ids.txt has {n_val} positions, expected {EXPECTED_VAL}. "
        "It looks STALE — commit the 120-line val_ids.txt so the headline reproduces."
    )
    assert len(by_id) == EXPECTED_VAL * 3, f"expected {EXPECTED_VAL * 3} val scenarios, got {len(by_id)}"

    rec = _recommended_moves(gens, by_id)
    tier_policy_match, move_sound, per_tier = _tier_policy_and_soundness(rec, by_id)
    distinct_rate, diff_n, diff_d = _distinct_rate(rec, by_id)

    print(f"  per-tier match: " + ", ".join(f"{t}={per_tier.get(t, 0.0):.4f}" for t in TIERS))
    print(f"  distinct       : {diff_d}/{diff_n} differentiating positions\n")

    print("Asserting recomputed metrics equal report_v4.json:")
    _assert_close("tier-policy match", tier_policy_match, float(pub["tier_fit"]))
    _assert_close("distinct-per-level", distinct_rate, float(pub["distinct"]["distinct_rate"]))
    _assert_close("move soundness", move_sound, float(pub["move_sound"]))

    print("\nPASS — headline v4 reproduces exactly from committed files (no GPU, no network).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
