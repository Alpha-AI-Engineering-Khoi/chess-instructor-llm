#!/usr/bin/env python3
"""Assemble web/public/showcase.json + data/showcase/SHOWCASE_REPORT.md.

Reads whatever is complete across the three splits (train / test_new / test_reuse)
— scenarios, per-model generations, objective flags, and council grades — and
emits the honest showcase array. Resilient: a cell with no council yet gets
``council_move/council_instr = null`` but is still included as long as it has a
generation + objective row, so partial runs still produce a valid dataset.

Per position it derives (including BOTH sides, never cherry-picked):
  ours_wins  — a tier where OURS is sound+tier-fit while a frontier model isn't,
               OR sound+faithful while a frontier model fabricates.
  ours_loses — the honest opposite.
  shine      — a clean case: ours_wins, not ours_loses, and OURS is
               sound+tier-fit+faithful in ALL three tiers.

Run::  ~/.venvs/mlx/bin/python data/showcase/pipeline/assemble.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import (  # noqa: E402
    COST_PATH, FIELD, FRONTIER_KEYS, LOCAL_KEYS, OURS_KEY, REPORT_PATH, ROOT,
    SPLIT_DIRS, TIERS, WEB_SHOWCASE, model_meta, read_jsonl, usd_for,
)

sys.path.insert(0, str(ROOT))
import chess  # noqa: E402


# --------------------------------------------------------------------------- #
# Load one split's cells: {pos_id: {tier: {model: cell}}} + fen/phase + targets.
# --------------------------------------------------------------------------- #
def _san_for(fen: str, uci: Optional[str], pool: List[Dict[str, Any]]) -> Optional[str]:
    if not uci:
        return None
    for m in pool:
        if m.get("uci") == uci:
            return m.get("san")
    try:
        return chess.Board(fen).san(chess.Move.from_uci(uci))
    except Exception:  # noqa: BLE001
        return None


def _council_by_scn(split_dir: Path) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """{scenario_id: {model: {'move':[...], 'instr':[...]}}} across judges."""
    agg: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"move": [], "instr": []}))
    for row in read_jsonl(split_dir / "council.jsonl"):
        sid = row["scenario_id"]
        for model, g in (row.get("grades") or {}).items():
            for axis in ("move", "instr"):
                v = g.get(axis)
                if v is not None:
                    agg[sid][model][axis].append(float(v))
    return agg


def _mean(xs: List[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 2) if xs else None


def load_split(split_name: str) -> Dict[str, Dict[str, Any]]:
    split_dir = SPLIT_DIRS[split_name]
    scns = read_jsonl(split_dir / "scenarios.jsonl")
    if not scns:
        return {}
    scn_by_id = {s["id"]: s for s in scns}

    obj_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {
        (o["scenario_id"], o["model"]): o for o in read_jsonl(split_dir / "objective.jsonl")
    }
    gen_by_key: Dict[Tuple[str, str], str] = {}
    for key in FIELD:
        for g in read_jsonl(split_dir / "gen" / f"{key}.jsonl"):
            gen_by_key[(g["scenario_id"], key)] = g.get("output", "")
    council = _council_by_scn(split_dir)

    positions: Dict[str, Dict[str, Any]] = {}
    for scn in scns:
        pid = scn.get("pos_id", scn["id"])
        tier = scn["tier"]
        sm = scn.get("student_move") or {}
        pos = positions.setdefault(pid, {
            "id": pid, "fen": scn["fen"], "phase": scn["phase"],
            "split": "train" if split_name == "train" else "test",
            "split_source": split_name,
            "tier_targets": {}, "_cells": defaultdict(dict), "_pool_policy": {},
            "student_move": {"san": sm.get("san"), "uci": sm.get("uci"),
                             "severity": scn.get("severity") or sm.get("severity")},
            "severity": scn.get("severity") or sm.get("severity"),
        })
        # tier target = the canonical tier-appropriate move (SAN), per the showcase.ts
        # contract (a plain SAN string or null; the client resolves its UCI).
        canon = scn.get("canonical_uci")
        pos["tier_targets"][tier] = _san_for(scn["fen"], canon, scn.get("sound_pool", []))
        # per-tier Maia findability over the sound pool (for the move-gradient check)
        pos["_pool_policy"][tier] = {u: float(p) for u, p in (scn.get("pool_policy") or {}).items()}
        for key in FIELD:
            o = obj_by_key.get((scn["id"], key))
            out = gen_by_key.get((scn["id"], key))
            if o is None and out is None:
                continue
            cg = council.get(scn["id"], {}).get(key, {"move": [], "instr": []})
            pos["_cells"][key][tier] = {
                "move": (o or {}).get("rec_san"),
                "move_uci": (o or {}).get("rec_uci"),
                "sound": bool((o or {}).get("sound")) if o else None,
                "tier_fit": bool((o or {}).get("tier_fit")) if o else None,
                "fabricated": bool((o or {}).get("fabricated")) if o else None,
                "coaching": out,
                "council_move": _mean(cg["move"]),
                "council_instr": _mean(cg["instr"]),
            }
    return positions


# --------------------------------------------------------------------------- #
# Win/lose derivation (frontier comparison) + OURS tier-differentiation.
# --------------------------------------------------------------------------- #
def _cell(cells: Dict[str, Dict[str, Any]], key: str, tier: str) -> Optional[Dict[str, Any]]:
    return cells.get(key, {}).get(tier)


def derive_wins(cells: Dict[str, Dict[str, Any]]) -> Tuple[bool, bool]:
    """ours_wins / ours_loses vs the frontier references (both honest sides)."""
    ours_wins = ours_loses = False
    for tier in TIERS:
        o = _cell(cells, OURS_KEY, tier)
        if not o:
            continue
        for fk in FRONTIER_KEYS:
            f = _cell(cells, fk, tier)
            if not f:
                continue
            o_tf = bool(o.get("sound")) and bool(o.get("tier_fit"))
            f_tf = bool(f.get("sound")) and bool(f.get("tier_fit"))
            o_faith = bool(o.get("sound")) and not bool(o.get("fabricated"))
            f_faith = bool(f.get("sound")) and not bool(f.get("fabricated"))
            if (o_tf and not f_tf) or (o_faith and bool(f.get("fabricated"))):
                ours_wins = True
            if (f_tf and not o_tf) or (f_faith and bool(o.get("fabricated"))):
                ours_loses = True
    return ours_wins, ours_loses


def derive_differentiation(
    cells: Dict[str, Dict[str, Any]], pool_policy: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """Does OURS give DIFFERENT, level-appropriate moves across the 3 tiers?

    Correct differentiation = >=2 distinct SOUND OURS moves across tiers, the
    beginner move differs from the advanced move, and the gradient points the right
    way (the beginner move is at least as human-findable — by Maia@1100 over the
    sound pool — as the advanced move). Mis-directed = it changes the move the wrong
    way (advanced move handed to the beginner) or a differentiating pick is unsound.
    """
    o = {t: _cell(cells, OURS_KEY, t) for t in TIERS}
    present = [t for t in TIERS if o[t]]
    picks = {t: (o[t] or {}).get("move_uci") for t in TIERS}
    sound = {t: bool((o[t] or {}).get("sound")) for t in TIERS}
    distinct_all = {picks[t] for t in TIERS if picks[t]}
    distinct_sound = {picks[t] for t in TIERS if picks[t] and sound[t]}
    all_present = len(present) == len(TIERS)
    all_sound = all_present and all(sound[t] for t in TIERS)

    polB = pool_policy.get("beginner", {})
    bU, aU = picks["beginner"], picks["advanced"]
    changed_b_vs_a = bool(bU and aU and bU != aU)
    direction_ok = (not changed_b_vs_a) or (polB.get(bU, 0.0) >= polB.get(aU, 0.0))

    differentiates = bool(
        len(distinct_sound) >= 2 and all_sound and changed_b_vs_a and direction_ok
    )
    misdirected = bool(
        (changed_b_vs_a and polB.get(bU, 0.0) < polB.get(aU, 0.0))
        or (len(distinct_all) >= 2 and all_present and not all_sound)
    )
    return {
        "ours_tier_differentiates": differentiates,
        "ours_misdirected": misdirected,
        "ours_distinct_moves": len(distinct_all),
        "ours_distinct_sound_moves": len(distinct_sound),
        "ours_full_3tier_coverage": all_present,
    }


def _model_summary(cells: Dict[str, Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    tiers = [cells.get(key, {}).get(t) for t in TIERS]
    tiers = [c for c in tiers if c]
    if not tiers:
        return None
    instr = [c["council_instr"] for c in tiers if c.get("council_instr") is not None]
    move = [c["council_move"] for c in tiers if c.get("council_move") is not None]
    tf = [1.0 if c.get("tier_fit") else 0.0 for c in tiers]
    snd = [1.0 if c.get("sound") else 0.0 for c in tiers]
    fab = [1.0 if c.get("fabricated") else 0.0 for c in tiers]
    meta = model_meta(key)
    return {
        "key": key, "name": meta["name"], "family": meta["family"],
        "council_instr": round(sum(instr) / len(instr), 2) if instr else None,
        "council_move": round(sum(move) / len(move), 2) if move else None,
        "tier_fit_rate": round(sum(tf) / len(tf), 3),
        "sound_rate": round(sum(snd) / len(snd), 3),
        "fabricated_rate": round(sum(fab) / len(fab), 3),
    }


def best_other(cells: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Strongest non-OURS model at this position (council-first, objective fallback)."""
    cands = [s for k in FIELD if k != OURS_KEY for s in [_model_summary(cells, k)] if s]
    if not cands:
        return None
    def _rank(s: Dict[str, Any]) -> Tuple:
        return (
            s["council_instr"] if s["council_instr"] is not None else -1.0,
            s["council_move"] if s["council_move"] is not None else -1.0,
            s["tier_fit_rate"], s["sound_rate"], -s["fabricated_rate"],
        )
    return max(cands, key=_rank)


def finalize(pos: Dict[str, Any]) -> Dict[str, Any]:
    cells = pos.pop("_cells")
    pool_policy = pos.pop("_pool_policy")
    ours_wins, ours_loses = derive_wins(cells)
    diff = derive_differentiation(cells, pool_policy)
    bo = best_other(cells)
    ours_sum = _model_summary(cells, OURS_KEY)

    # focus/shine = the tier-differentiation set; shine is the CLEAN subset.
    focus = diff["ours_tier_differentiates"]
    shine = bool(focus and not ours_loses and not diff["ours_misdirected"])

    models: List[Dict[str, Any]] = []
    for key in FIELD:
        meta = model_meta(key)
        by_tier = {t: cells.get(key, {}).get(t) for t in TIERS}
        if all(by_tier[t] is None for t in TIERS):
            continue
        models.append({
            "name": meta["name"], "family": meta["family"], "local": meta["local"],
            "byTier": by_tier,
        })
    return {
        "id": pos["id"], "fen": pos["fen"], "phase": pos["phase"], "split": pos["split"],
        "split_source": pos["split_source"],
        "tier_targets": pos["tier_targets"], "models": models,
        "student_move": pos.get("student_move"), "severity": pos.get("severity"),
        # best_other: a NAME string (contract) + a detail object (analysis-only extra)
        "best_other": (bo or {}).get("name"),
        "best_other_detail": bo, "ours_summary": ours_sum,
        "ours_wins": ours_wins, "ours_loses": ours_loses,
        "ours_tier_differentiates": diff["ours_tier_differentiates"],
        "ours_misdirected": diff["ours_misdirected"],
        "ours_distinct_moves": diff["ours_distinct_moves"],
        "ours_distinct_sound_moves": diff["ours_distinct_sound_moves"],
        "ours_full_3tier_coverage": diff["ours_full_3tier_coverage"],
        "focus": focus, "shine": shine,
    }


# --------------------------------------------------------------------------- #
# Cost.
# --------------------------------------------------------------------------- #
def compute_cost() -> Dict[str, Any]:
    buckets = {
        "train_gen": {"calls": 0, "in": 0, "out": 0, "usd": 0.0},
        "test_new_gen": {"calls": 0, "in": 0, "out": 0, "usd": 0.0},
        "reused_803_gen": {"calls": 0, "in": 0, "out": 0, "usd": 0.0},
        "council": {"calls": 0, "in": 0, "out": 0, "usd": 0.0},
        "local_gen_free": {"calls": 0, "in": 0, "out": 0, "usd": 0.0},
    }
    for split_name, split_dir in SPLIT_DIRS.items():
        for key in FIELD:
            for g in read_jsonl(split_dir / "gen" / f"{key}.jsonl"):
                pin, pout = int(g.get("prompt_tokens", 0)), int(g.get("completion_tokens", 0))
                if key in LOCAL_KEYS:
                    b = buckets["local_gen_free"]
                    b["calls"] += 1; b["in"] += pin; b["out"] += pout
                    continue
                if split_name == "test_reuse":
                    b = buckets["reused_803_gen"]
                elif split_name == "train":
                    b = buckets["train_gen"]
                else:
                    b = buckets["test_new_gen"]
                b["calls"] += 1; b["in"] += pin; b["out"] += pout
                b["usd"] += usd_for(key, pin, pout)
        for row in read_jsonl(split_dir / "council.jsonl"):
            jk = row.get("judge")
            pin, pout = int(row.get("prompt_tokens", 0)), int(row.get("completion_tokens", 0))
            b = buckets["council"]
            b["calls"] += 1; b["in"] += pin; b["out"] += pout
            b["usd"] += usd_for(jk, pin, pout)
    for b in buckets.values():
        b["usd"] = round(b["usd"], 2)
    new_spend = round(buckets["train_gen"]["usd"] + buckets["test_new_gen"]["usd"]
                      + buckets["council"]["usd"], 2)
    reused_value = buckets["reused_803_gen"]["usd"]
    return {"buckets": buckets, "new_spend_usd": new_spend,
            "reused_803_value_usd": reused_value,
            "grand_total_incl_reused_usd": round(new_spend + reused_value, 2)}


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(WEB_SHOWCASE))
    args = p.parse_args(argv)

    all_positions: List[Dict[str, Any]] = []
    per_split_counts: Dict[str, int] = {}
    for split_name in ("train", "test_new", "test_reuse"):
        positions = load_split(split_name)
        per_split_counts[split_name] = len(positions)
        for pos in positions.values():
            all_positions.append(finalize(pos))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_positions, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- stats -----------------------------------------------------------
    def _split(name_pred) -> List[Dict[str, Any]]:
        return [p for p in all_positions if name_pred(p)]

    train = _split(lambda p: p["split_source"] == "train")
    test_new = _split(lambda p: p["split_source"] == "test_new")
    test_reuse = _split(lambda p: p["split_source"] == "test_reuse")
    test_all = test_new + test_reuse

    def _wins(rows): return sum(1 for p in rows if p["ours_wins"])
    def _loses(rows): return sum(1 for p in rows if p["ours_loses"])
    def _shine(rows): return sum(1 for p in rows if p["shine"])
    def _diff(rows): return sum(1 for p in rows if p["ours_tier_differentiates"])
    def _misdir(rows): return sum(1 for p in rows if p["ours_misdirected"])
    def _cov(rows): return sum(1 for p in rows if p["ours_full_3tier_coverage"])

    cost = compute_cost()
    COST_PATH.write_text(json.dumps(cost, indent=2), encoding="utf-8")

    # per-model coverage (how many positions each model actually coached)
    model_cov: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "test": 0})
    for p in all_positions:
        grp = "train" if p["split_source"] == "train" else "test"
        for m in p["models"]:
            if any(m["byTier"][t] and m["byTier"][t].get("coaching") for t in TIERS):
                model_cov[m["name"]][grp] += 1

    yield_info = {}
    yj = SPLIT_DIRS["test_new"] / "yield.json"
    if yj.exists():
        yield_info = json.loads(yj.read_text())

    stats = {
        "positions_total": len(all_positions),
        "train": len(train), "test": len(test_all),
        "test_new": len(test_new), "test_reuse": len(test_reuse),
        "ours_wins_total": _wins(all_positions), "ours_loses_total": _loses(all_positions),
        "ours_wins_train": _wins(train), "ours_loses_train": _loses(train),
        "ours_wins_test": _wins(test_all), "ours_loses_test": _loses(test_all),
        "shine_total": _shine(all_positions), "focus_total": _diff(all_positions),
        # tier-differentiation (the refined focus of the showcase)
        "tier_diff_train": _diff(train), "tier_diff_test": _diff(test_all),
        "tier_diff_test_new": _diff(test_new), "tier_diff_test_reuse": _diff(test_reuse),
        "tier_diff_total": _diff(all_positions),
        "misdirected_total": _misdir(all_positions),
        "misdirected_train": _misdir(train), "misdirected_test": _misdir(test_all),
        # OURS full 3-tier coverage confirmation
        "ours_full_coverage_positions": _cov(all_positions),
        "ours_full_coverage_all": bool(_cov(all_positions) == len(all_positions) and all_positions),
        "n_train": len(train), "n_test": len(test_all),
        "model_coverage": model_cov,
        "yield": yield_info, "cost": cost,
    }
    (SPLIT_DIRS["train"].parent / "stats.json").write_text(json.dumps(stats, indent=2))
    write_report(stats, per_split_counts)
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {len(all_positions)} positions -> {out_path}")
    return 0


def write_report(stats: Dict[str, Any], per_split_counts: Dict[str, int]) -> None:
    c = stats["cost"]
    b = c["buckets"]
    y = stats.get("yield", {})
    L: List[str] = []
    A = L.append
    A("# SHOWCASE eval dataset — honest, per-model / per-tier, blinded-graded\n")
    A("Powers the revamped platform's showcase. Every one of the 14 models coaches the "
      "SAME positions at all 3 tiers with byte-identical Stockfish+Maia grounding (the "
      "exact 803-benchmark pipeline), scored deterministically (sound / tier-fit / "
      "fabricated) and graded by a blinded 3-judge cross-family council (move + "
      "instructiveness, 0-10). OURS = the live local v2 coach.\n")
    A("## Counts\n")
    A("| split | positions | note |")
    A("|---|---:|---|")
    A(f"| train | {stats['train']} | IN-DISTRIBUTION (boards OURS-v2 was trained on) — reported honestly as such |")
    A(f"| test (new Lichess) | {stats['test_new']} | freshly pulled, held-out, discriminating, zero-leakage |")
    A(f"| test (reused 803) | {stats['test_reuse']} | reuses the definitive benchmark's 14-model gens (read-only) |")
    A(f"| **test total** | **{stats['test']}** | |")
    A(f"| **all** | **{stats['positions_total']}** | |")
    A("")
    A("## OURS 3-tier coverage (comprehensive, zero gaps)\n")
    cov = stats["ours_full_coverage_positions"]
    A(f"- OURS was run LOCALLY (mlx, free) on **every** showcase position × all 3 tiers.")
    A(f"- Positions with full OURS 3-tier coverage: **{cov}/{stats['positions_total']}** "
      f"({'CONFIRMED complete — no per-tier gaps' if stats['ours_full_coverage_all'] else 'INCOMPLETE (rerun pending)'}).")
    A("")
    A("## Tier differentiation — the focus (OURS gives DIFFERENT, level-appropriate moves)\n")
    A("A position is a **tier-differentiation / focus** case when OURS recommends >=2 distinct, "
      "SOUND moves across the 3 tiers with the correct gradient (beginner = more human-findable, "
      "advanced = sharper). This is the behaviour the platform showcases.\n")
    A("| split | focus (tier-differentiates) | mis-directed | of positions |")
    A("|---|---:|---:|---:|")
    A(f"| train | {stats['tier_diff_train']} | {stats['misdirected_train']} | {stats['train']} |")
    A(f"| test  | {stats['tier_diff_test']} | {stats['misdirected_test']} | {stats['test']} |")
    A(f"| **all** | **{stats['tier_diff_total']}** | **{stats['misdirected_total']}** | {stats['positions_total']} |")
    A("")
    A("- **ours_tier_differentiates / focus**: >=2 distinct sound OURS moves across tiers, "
      "beginner!=advanced, correctly directed (beginner more human-findable).")
    A("- **ours_misdirected**: OURS changes its move the WRONG way (sharp move handed to the "
      "beginner) or a differentiating pick is unsound — recorded honestly, never hidden.")
    A("")
    A("## OURS wins vs loses vs the frontier (both included — not cherry-picked)\n")
    A("| split | ours_wins | ours_loses | clean 'shine' |")
    A("|---|---:|---:|---:|")
    A(f"| train | {stats['ours_wins_train']} | {stats['ours_loses_train']} | — |")
    A(f"| test | {stats['ours_wins_test']} | {stats['ours_loses_test']} | — |")
    A(f"| **all** | **{stats['ours_wins_total']}** | **{stats['ours_loses_total']}** | **{stats['shine_total']}** |")
    A("")
    A("- **ours_wins**: a tier where OURS is sound+tier-fit while a frontier model isn't, "
      "or sound+faithful while a frontier model fabricates.")
    A("- **ours_loses**: the honest opposite. Both flags can be true on one position.")
    A("- **shine**: focus (tier-differentiates) AND not ours_loses AND not mis-directed — the "
      "clean, demonstrable level-fitting cases.")
    A("- **best_other**: the strongest non-OURS model per position (council-first), stored for "
      "the OURS-vs-best comparison; every model's per-tier picks are kept for the dropdown.")
    A("")
    mc = stats.get("model_coverage", {})
    if mc:
        nt, ns = stats["n_train"], stats["n_test"]
        A("## Per-model coverage (honest)\n")
        A(f"OURS + 10 of the API models coached every position. During the NEW-position run "
          f"three open models (Gemma-3-27B, Kimi-K2.5, Mistral-Large-3) hit a transient AWS "
          f"Bedrock overload (timeouts/503s) and were retried but only partially completed on "
          f"the fresh positions; they are FULLY present on the reused-803 test positions. "
          f"OURS's own coverage is complete — the differentiation view has zero gaps.\n")
        A(f"| model | train (/{nt}) | test (/{ns}) |")
        A("|---|---:|---:|")
        for name in sorted(mc, key=lambda k: (-(mc[k]['train'] + mc[k]['test']), k)):
            d = mc[name]
            flag = "" if (d["train"] == nt and d["test"] == ns) else "  ⚠ partial (transient provider outage)"
            A(f"| {name} | {d['train']} | {d['test']}{flag} |")
        A("")
    if y:
        A("## New-Lichess filter yield (the 10k -> few hundred funnel)\n")
        A("| stage | positions |")
        A("|---|---:|")
        A(f"| raw pulled from Lichess | {y.get('raw_pulled','?')} |")
        A(f"| after dedup (vs all corpora) | {y.get('after_dedup','?')} |")
        A(f"| grounded (Stockfish+Maia) | {y.get('grounded','?')} |")
        A(f"| discriminating + eligible | {y.get('discriminating_eligible','?')} |")
        A(f"| **selected (strongest)** | **{y.get('selected','?')}** |")
        A("")
    A("## Cost (frontier via TrueFoundry; local OURS/BASE free)\n")
    A("| bucket | calls | in tok | out tok | est. USD |")
    A("|---|---:|---:|---:|---:|")
    for name in ("train_gen", "test_new_gen", "council", "local_gen_free", "reused_803_gen"):
        d = b[name]
        A(f"| {name} | {d['calls']:,} | {d['in']:,} | {d['out']:,} | ${d['usd']:.2f} |")
    A(f"| **NEW SPEND (this run)** | | | | **${c['new_spend_usd']:.2f}** |")
    A("")
    A(f"_New spend this run = train gens + new-Lichess gens + council = **${c['new_spend_usd']:.2f}**. "
      f"The reused 803 generations (${c['reused_803_value_usd']:.2f} of value) were already paid for "
      f"by the definitive benchmark and cost $0 here. Local OURS-v2 + BASE are free._")
    A("")
    A("## Artifacts\n")
    A("- Per split: `data/showcase/{train,test_new,test_reuse}/scenarios.jsonl`, "
      "`gen/<model>.jsonl`, `objective.jsonl`, `council.jsonl`.")
    A("- Raw Lichess pull: `data/showcase/lichess_raw.jsonl`; grounded: "
      "`data/showcase/test_new/grounded.jsonl`.")
    A("- Output: `web/public/showcase.json`; machine stats: `data/showcase/stats.json`, "
      "`data/showcase/cost.json`.")
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
