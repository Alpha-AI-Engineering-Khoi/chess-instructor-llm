"""Assemble ``web/public/showcase_interim.json`` — the interim Showcase slice.

This is the array contract the web loader reads *between* the full
``showcase.json`` (owned by the big worker; NOT touched here) and the shipped
``showdown.json`` fallback. It joins, per held-out position:

* every FRONTIER / open / BASE rival exactly as ``showdown.json`` already scored
  them (their single benchmarked tier) — reused verbatim, never re-run;
* OURS (chess-coach-v2) re-scored at ALL THREE tiers by
  ``scripts/rescore_ours_3tier.py`` (``data/ours_3tier/ours_cells.jsonl``).

From those it derives, honestly:

* ``ours_tier_differentiates`` — OURS gives a move that is sound AND tier-fit at
  every tier AND not identical across all three (the platform's MOAT lens). This
  is exactly the fallback rule in ``web/src/lib/showcase.ts`` (deriveTierDifferentiates),
  computed here from the real 3-tier moves so all-same-move positions are excluded.
* ``ours_wins`` / ``ours_loses`` — a SYMMETRIC head-to-head vs the three frontier
  models at the position's benchmarked tier, on the two general-coaching axes the
  benchmark measures (tier-fit + raw faithfulness):
     wins  <=> OURS beats >=1 frontier: sound+tier-fit where it isn't, OR
               sound+faithful where it fabricates.
     loses <=> a frontier beats OURS the same way (the mirror).
  ``ours_wins`` keeps the exact meaning shipped in build_showdown.py; ``ours_loses``
  is now its mirror against the SAME frontier set, fixing the old asymmetry where
  loses was "OURS unclean while ANY of 13 rivals was clean" (an inconsistent
  opponent set). The count barely moves — it was never the inflation source — but
  the two tallies are now the same kind of measurement.

Honesty notes printed at the end so the numbers can be reported truthfully.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

SHOWDOWN = ROOT / "web" / "public" / "showdown.json"
OURS_CELLS = ROOT / "data" / "ours_3tier" / "ours_cells.jsonl"
OUT = ROOT / "web" / "public" / "showcase_interim.json"

TIERS = ("beginner", "intermediate", "advanced")
FRONTIER = ("gpt", "claude", "gemini")


def load_ours() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """key -> {tier -> ours cell row}."""
    by_key: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for line in OURS_CELLS.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        by_key.setdefault(r["key"], {})[r["tier"]] = r
    return by_key


def ours_cell(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "move": row["rec_san"],
        "sound": bool(row["sound"]),
        "tier_fit": bool(row["tier_fit"]),
        "fabricated": bool(row["fabricated"]),
        "coaching": row["coaching"],
        "council_move": None,
        "council_instr": None,
        "n_violations": int(row.get("n_violations") or 0),
        "violations": row.get("violations") or [],
    }


def rival_cell(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "move": m.get("rec_san"),
        "sound": bool(m.get("sound")),
        "tier_fit": bool(m.get("tier_appropriate")),
        "fabricated": bool(m.get("fabricated")),
        "coaching": m.get("coaching", ""),
        "council_move": None,
        "council_instr": None,
        "n_violations": int(m.get("n_violations") or 0),
        "violations": m.get("violations") or [],
    }


def derive_differentiates(ours_by_tier: Dict[str, Optional[Dict[str, Any]]]) -> bool:
    """Exactly web/src/lib/showcase.ts deriveTierDifferentiates."""
    cells = [ours_by_tier.get(t) for t in TIERS]
    if any(c is None or not c.get("move") for c in cells):
        return False
    if not all(c["tier_fit"] and c["sound"] for c in cells):
        return False
    moves = {c["move"] for c in cells}
    return len(moves) >= 2


def main() -> int:
    doc = json.loads(SHOWDOWN.read_text(encoding="utf-8"))
    model_meta = doc["meta"]["model_meta"]
    ours = load_ours()

    positions_out: List[Dict[str, Any]] = []
    stats = Counter()
    per_tier = {t: Counter() for t in TIERS}
    same_move = 0
    tierfit_fail = 0
    diff_count = 0
    wins = loses = shine = 0
    old_wins = old_loses_any13 = 0
    native_repro = Counter()

    for P in doc["positions"]:
        key = P["key"]
        o3 = ours.get(key)
        if not o3 or any(t not in o3 for t in TIERS):
            print(f"  ! missing OURS 3-tier for {key}; skipping row")
            continue
        native = P["tier"]

        # --- models: rivals verbatim (native tier only) + OURS at 3 tiers ------
        models_out: List[Dict[str, Any]] = []
        for m in P["models"]:
            mk = m["key"]
            meta = model_meta.get(mk, {"name": mk, "family": "open", "kind": "open"})
            if mk == "ours":
                by_tier = {t: ours_cell(o3[t]) for t in TIERS}
                models_out.append({
                    "name": meta["name"], "family": "ours", "local": True, "byTier": by_tier,
                })
            else:
                by_tier = {native: rival_cell(m)}
                models_out.append({
                    "name": meta["name"],
                    "family": meta.get("family", "open"),
                    "local": meta.get("kind") != "frontier",
                    "byTier": by_tier,
                })
        # If showdown had no explicit ours row, add ours from the 3-tier re-score.
        if not any(mm["family"] == "ours" for mm in models_out):
            models_out.insert(0, {
                "name": model_meta.get("ours", {}).get("name", "OURS"),
                "family": "ours", "local": True,
                "byTier": {t: ours_cell(o3[t]) for t in TIERS},
            })

        ours_by_tier = {t: ours_cell(o3[t]) for t in TIERS}

        # --- MOAT: tier-differentiation ---------------------------------------
        differentiates = derive_differentiates(ours_by_tier)
        diff_count += int(differentiates)
        moves = {ours_by_tier[t]["move"] for t in TIERS}
        if len(moves) < 2:
            same_move += 1
        elif not all(ours_by_tier[t]["tier_fit"] and ours_by_tier[t]["sound"] for t in TIERS):
            tierfit_fail += 1

        # --- honest, symmetric wins/loses vs the frontier at the native tier ---
        on = ours_by_tier[native]
        frontier_cells = {m["key"]: m for m in P["models"] if m["key"] in FRONTIER}
        w_tier = w_faith = l_tier = l_faith = False
        for fk, f in frontier_cells.items():
            f_tier = bool(f.get("tier_appropriate"))
            f_sound = bool(f.get("sound"))
            f_fab = bool(f.get("fabricated"))
            if on["tier_fit"] and not f_tier:
                w_tier = True
            if (on["sound"] and not on["fabricated"]) and f_fab:
                w_faith = True
            if f_tier and not on["tier_fit"]:
                l_tier = True
            if (f_sound and not f_fab) and on["fabricated"]:
                l_faith = True
        ours_wins = w_tier or w_faith
        ours_loses = l_tier or l_faith
        wins += int(ours_wins)
        loses += int(ours_loses)
        shine += int(ours_wins)  # keep shine == wins (the "standout" lens), recomputed on fresh data

        # reference: the numbers the shipped UI shows today (for the before/after report)
        old_wins += int(bool(P.get("ours_wins")))
        ours_clean_native = on["tier_fit"] and not on["fabricated"]
        any13_clean = any(
            (mm["key"] != "ours" and mm.get("tier_appropriate") and not mm.get("fabricated"))
            for mm in P["models"]
        )
        old_loses_any13 += int((not ours_clean_native) and any13_clean)

        # per-tier OURS stats (over all 3 tiers)
        for t in TIERS:
            c = ours_by_tier[t]
            per_tier[t]["n"] += 1
            per_tier[t]["sound"] += int(c["sound"])
            per_tier[t]["tier_fit"] += int(c["tier_fit"])
            per_tier[t]["fabricated"] += int(c["fabricated"])

        # native-cell reproduction check vs the old showdown OURS cell
        old_ours = next((m for m in P["models"] if m["key"] == "ours"), None)
        if old_ours is not None:
            native_repro["n"] += 1
            native_repro["same_move"] += int(old_ours.get("rec_uci") == o3[native]["rec_uci"])

        tier_targets = {t: o3[t].get("tier_target_san") for t in TIERS}
        sm = P.get("student_move") or None
        positions_out.append({
            "id": key,
            "fen": P["fen"],
            "phase": P["phase"],
            "split": "test",
            "benchmark": P.get("benchmark"),
            "severity": P.get("severity"),
            "student_move": (
                {"san": sm.get("san"), "uci": sm.get("uci"), "severity": sm.get("severity")}
                if sm else None
            ),
            "tier_targets": tier_targets,
            "models": models_out,
            "ours_wins": ours_wins,
            "ours_loses": ours_loses,
            "shine": bool(ours_wins),
            "ours_tier_differentiates": differentiates,
        })
        stats["positions"] += 1

    OUT.write_text(json.dumps(positions_out, ensure_ascii=False), encoding="utf-8")

    n = stats["positions"]
    print("\n================= showcase_interim.json =================")
    print(f"positions written: {n}  ->  {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print("\n-- OURS per-tier (over all 3 tiers, n per tier = %d) --" % n)
    for t in TIERS:
        c = per_tier[t]
        nn = max(1, c["n"])
        print(f"  {t:12s}: sound {c['sound']:3d}/{c['n']}  tier_fit {c['tier_fit']:3d}/{c['n']}  "
              f"fabricated(raw) {c['fabricated']:3d}/{c['n']}")
    print("\n-- MOAT: tier differentiation --")
    print(f"  differentiates (sound+tier-fit all 3 AND move varies): {diff_count}/{n}")
    print(f"  excluded: all-three-same-move {same_move}   not-all-tier-fit/sound {tierfit_fail}")
    print("\n-- General-quality tally vs frontier (native tier) --")
    print(f"  OURS wins (beats >=1 frontier on tier|faithful):  {wins}/{n}")
    print(f"  OURS loses (a frontier beats OURS, mirror):       {loses}/{n}")
    print(f"  shine (== wins):                                  {shine}/{n}")
    print("\n-- before/after (the shipped showdown numbers) --")
    print(f"  wins:  showdown-shipped {old_wins}  ->  interim {wins}")
    print(f"  loses: showdown-shipped(any-of-13) {old_loses_any13}  ->  interim(mirror-vs-frontier) {loses}")
    if native_repro["n"]:
        print(f"\n-- native-tier reproduction vs old showdown OURS cell --")
        print(f"  same recommended move: {native_repro['same_move']}/{native_repro['n']} "
              f"({100*native_repro['same_move']/native_repro['n']:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
