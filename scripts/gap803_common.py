"""Shared helpers for the DEFINITIVE 803-position gap eval (all 14 models).

The eval reuses the existing benchmark pipeline (``src/eval/benchmark`` generate /
objective / council) by *flattening* each curated gap position (``data/eval/
gap_positions.jsonl``) into three benchmark "scenarios" — one per tier — with a
composite id ``"<pos_id>#<tier>"``. The per-tier scenario carries byte-identical
grounding (the shared Stockfish sound pool + that tier's Maia block), so a coach
is asked the SAME position at each tier and we can measure whether its move
changes with level.

Tier-appropriateness is scored against the project's own deterministic rule
:func:`src.teacher.tier_select.select_tier_move` (beginner -> most human-findable
sound move, advanced -> sharpest = engine best, intermediate -> blend). That is
the canonical "tier-appropriate move"; the 803 set's own ``tier_move`` field is
the pure-Maia move (findability only) and is kept alongside for reference.
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.teacher.tier_select import select_tier_move  # noqa: E402

TIERS: Tuple[str, ...] = ("beginner", "intermediate", "advanced")


def load_positions(path: Path) -> List[Dict[str, Any]]:
    import json
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def canonical_tier_uci(tier: str, sound_pool: List[Dict[str, Any]],
                       pool_policy: Dict[str, float]) -> Optional[str]:
    """The tier-appropriate move per ``select_tier_move`` (uci), or None."""
    if not sound_pool:
        return None
    pick = select_tier_move(tier, sound_pool, pool_policy)
    return pick.uci


def to_scenario(pos: Dict[str, Any], tier: str) -> Dict[str, Any]:
    """Flatten one (position, tier) into a benchmark-compatible scenario dict.

    Carries the fields the reused generate/objective/council modules need
    (``id, tier, phase, severity, fen, student_move, sound_pool, sound_uci,
    maia``) plus extras used only by the 803 report (``pos_id, source_tier,
    primary_motif, engine_best_uci, pool_policy, pool_order, canonical_uci,
    tier_move_uci, discriminating, n_sound``).
    """
    mt = pos["maia_by_tier"][tier]
    pool = [
        {"uci": m["uci"], "san": m["san"], "cp": int(m["cp"]), "pv": list(m.get("pv") or [])}
        for m in pos["sound_pool"]
    ]
    pool_policy = {u: float(p) for u, p in mt["pool_policy"].items()}
    pm = pos["played_move"]
    student = {
        "san": pm.get("san", "(none)"),
        "uci": pm.get("uci", ""),
        "cp_loss": int(pm.get("cp_loss", 0)),
        "severity": pm.get("severity", "none"),
    }
    tier_move = mt.get("tier_move") or {}
    return {
        "id": f"{pos['id']}#{tier}",
        "pos_id": pos["id"],
        "tier": tier,
        "phase": pos["phase"],
        "source_tier": pos.get("source_tier"),
        "primary_motif": pos.get("primary_motif"),
        "severity": student["severity"],
        "fen": pos["fen"],
        "student_move": student,
        "sound_pool": pool,
        "sound_uci": [m["uci"] for m in pool],
        "maia": [
            {"uci": m["uci"], "san": m["san"], "policy": float(m["policy"])}
            for m in mt.get("top", [])
        ],
        "engine_best_uci": pos["engine_best"]["uci"],
        "engine_best_san": pos["engine_best"]["san"],
        "pool_policy": pool_policy,
        "pool_order": mt.get("pool_order", []),
        "canonical_uci": canonical_tier_uci(tier, pool, pool_policy),
        "tier_move_uci": tier_move.get("uci"),
        "discriminating": bool(mt.get("discriminating")),
        "n_sound": pos["n_sound"],
    }


def flatten(positions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """All (position x tier) scenarios in a stable order."""
    out: List[Dict[str, Any]] = []
    for pos in positions:
        for tier in TIERS:
            out.append(to_scenario(pos, tier))
    return out


def stratified_positions(positions: Sequence[Dict[str, Any]], n: int,
                         seed: int = 3407) -> List[Dict[str, Any]]:
    """Round-robin over (source_tier x phase) buckets for a balanced subset."""
    if n >= len(positions):
        return list(positions)
    rng = random.Random(seed)
    buckets: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for p in positions:
        buckets[(p.get("source_tier"), p.get("phase"))].append(p)
    for b in buckets.values():
        rng.shuffle(b)
    order = sorted(buckets.keys(), key=lambda k: (str(k[0]), str(k[1])))
    picked: List[Dict[str, Any]] = []
    i = 0
    while len(picked) < n and any(buckets[k] for k in order):
        k = order[i % len(order)]
        if buckets[k]:
            picked.append(buckets[k].pop())
        i += 1
    return picked


def stratified_scenarios(scenarios: Sequence[Dict[str, Any]], n: int,
                         seed: int = 3407) -> List[Dict[str, Any]]:
    """Round-robin over (tier x phase) buckets of (pos,tier) scenarios."""
    if n >= len(scenarios):
        return list(scenarios)
    rng = random.Random(seed)
    buckets: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for s in scenarios:
        buckets[(s["tier"], s["phase"])].append(s)
    for b in buckets.values():
        rng.shuffle(b)
    order = sorted(buckets.keys(), key=lambda k: (str(k[0]), str(k[1])))
    picked: List[Dict[str, Any]] = []
    i = 0
    while len(picked) < n and any(buckets[k] for k in order):
        k = order[i % len(order)]
        if buckets[k]:
            picked.append(buckets[k].pop())
        i += 1
    return picked
