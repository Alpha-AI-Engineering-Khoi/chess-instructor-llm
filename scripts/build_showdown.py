"""Build ``web/public/showdown.json`` — the "Model Showdown" data slice.

Real data only. For every held-out benchmark position this joins, per model:

* the recommended move (SAN/UCI) and the deterministic objective flags already
  scored by the project's own pipeline (``objective.jsonl``):
    - ``sound``       — the pick is in this position's Stockfish sound pool.
    - ``fabricated``  — the faithfulness verifier found >=1 false board fact.
* whether the pick is **tier-appropriate**, computed with the project's *own*
  canonical rule :func:`src.teacher.tier_select.select_tier_move` — i.e. the pick
  equals THE human-findable sound move for that position's tier (beginner ->
  highest-Maia sound move, advanced -> sharpest sound move, intermediate -> the
  blend). Reusing that function means "tier-appropriate" here is identical to the
  target the tuned teacher was built to hit; nothing is re-invented.
* the model's verbatim coaching text (``generations.jsonl``) + any flagged
  false sentences (so the UI can show the receipts, not just a flag).

From those it derives an honest **"OURS wins"** flag per position:

    OURS wins  <=>  OURS beats at least one FRONTIER model (GPT-5.5 / Claude /
                    Gemini) on the SAME grounded input, either by
      (a) tier-appropriateness: OURS is sound + tier-appropriate where that
          frontier model is NOT, or
      (b) faithfulness: OURS gives a sound, non-fabricated answer where that
          frontier model fabricates a board fact.

Nothing is inflated: OURS must itself be genuinely good on the axis it wins on,
and it only counts as a win against a frontier model that is actually worse on
that same axis. The two benchmarks (``benchmark_v2`` = 5 models with the
grounded product condition; ``benchmark_open`` = the same treatment extended to
9 bigger open models) are kept as separate, tagged rows — they are genuinely
different held-out sets (only 12 FENs coincide, and not always at the same tier),
so merging them would be a false join.

Everyone is compared on the **grounded** condition (identical VERIFIED-FACTS +
Stockfish sound pool + Maia handed to every model), which is the real product
condition and the only one the open models were run on.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine.position_facts import render_pool_facts  # noqa: E402
from src.teacher.tier_select import TIER_HUMAN_WEIGHT, select_tier_move  # noqa: E402

COND = "grounded"
FRONTIER = ("gpt", "claude", "gemini")
BENCHES = (
    ("v2", ROOT / "data" / "benchmark_v2"),
    ("open", ROOT / "data" / "benchmark_open"),
)
OUT = ROOT / "web" / "public" / "showdown.json"

# Display metadata for every model key that appears in either benchmark.
MODEL_META: Dict[str, Dict[str, str]] = {
    "ours": {"name": "OURS · chess-coach-v2 (1.7B)", "short": "OURS", "kind": "ours", "family": "ours"},
    "base": {"name": "BASE · Qwen3-1.7B (untuned)", "short": "BASE", "kind": "base", "family": "base"},
    "gpt": {"name": "GPT-5.5", "short": "GPT-5.5", "kind": "frontier", "family": "frontier"},
    "claude": {"name": "Claude Opus 4.8", "short": "Claude", "kind": "frontier", "family": "frontier"},
    "gemini": {"name": "Gemini 3.1 Pro", "short": "Gemini", "kind": "frontier", "family": "frontier"},
    "dsr1": {"name": "DeepSeek-R1", "short": "DeepSeek-R1", "kind": "open", "family": "open"},
    "dsv32": {"name": "DeepSeek-V3.2", "short": "DeepSeek-V3.2", "kind": "open", "family": "open"},
    "gemma3_27b": {"name": "Gemma-3-27B-it", "short": "Gemma-3-27B", "kind": "open", "family": "open"},
    "glm5": {"name": "GLM-5", "short": "GLM-5", "kind": "open", "family": "open"},
    "kimi25": {"name": "Kimi-K2.5", "short": "Kimi-K2.5", "kind": "open", "family": "open"},
    "llama33_70b": {"name": "Llama-3.3-70B", "short": "Llama-3.3-70B", "kind": "open", "family": "open"},
    "mistral3": {"name": "Mistral-Large-3 (675B)", "short": "Mistral-Large-3", "kind": "open", "family": "open"},
    "q3_32b": {"name": "Qwen3-32B", "short": "Qwen3-32B", "kind": "open", "family": "open"},
    "q3_next80b": {"name": "Qwen3-Next-80B-A3B", "short": "Qwen3-Next-80B", "kind": "open", "family": "open"},
}

TIER_RANK = {"beginner": 0, "intermediate": 1, "advanced": 2}


def norm_ws(s: str) -> str:
    """Collapse all whitespace to single spaces (prose reads fine; matching robust)."""
    return re.sub(r"\s+", " ", s or "").strip()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def model_sort_key(key: str) -> Tuple[int, str]:
    """OURS first, then the three frontier, then BASE, then open (alpha)."""
    order = {"ours": 0, "gpt": 1, "claude": 2, "gemini": 3, "base": 4}
    if key in order:
        return (order[key], "")
    return (5, MODEL_META.get(key, {}).get("short", key).lower())


def tier_target(scn: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """THE tier-appropriate sound move for this scenario, via the canonical rule."""
    pool = scn.get("sound_pool") or []
    if not pool:
        return None
    maia = {str(m["uci"]): float(m["policy"]) for m in scn.get("maia", [])}
    try:
        pick = select_tier_move(scn["tier"], pool, maia)
    except ValueError:
        return None
    return {
        "uci": pick.uci,
        "san": pick.san,
        "pool_rank": pick.pool_rank,
        "is_engine_best": pick.is_engine_best,
        "policy": pick.policy,
        "weight": pick.weight,
    }


def build_bench(bench: str, root: Path) -> List[Dict[str, Any]]:
    scenarios = {s["id"]: s for s in load_jsonl(root / "scenarios.jsonl")}

    obj: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for o in load_jsonl(root / "objective.jsonl"):
        if o.get("condition") == COND:
            obj[(o["scenario_id"], o["model"])] = o

    gen: Dict[Tuple[str, str], str] = {}
    for g in load_jsonl(root / "generations.jsonl"):
        if g.get("condition") == COND:
            gen[(g["scenario_id"], g["model"])] = g.get("output", "")

    models_here = sorted({m for (_sid, m) in obj}, key=model_sort_key)

    positions: List[Dict[str, Any]] = []
    for sid, scn in scenarios.items():
        board = chess.Board(scn["fen"])
        tgt = tier_target(scn)
        tgt_uci = tgt["uci"] if tgt else None

        facts = render_pool_facts(
            scn["fen"],
            [{"uci": m["uci"], "san": m["san"], "cp": int(m["cp"])} for m in scn["sound_pool"]],
        )

        models: List[Dict[str, Any]] = []
        flags: Dict[str, Dict[str, bool]] = {}
        for key in models_here:
            o = obj.get((sid, key))
            if o is None:
                continue
            meta = MODEL_META.get(key, {"name": key, "short": key, "kind": "open", "family": "open"})
            sound = bool(o.get("move_sound"))
            parseable = bool(o.get("move_parseable"))
            fabricated = bool(o.get("fabricated"))
            rec_uci = o.get("rec_uci")
            tier_ok = sound and tgt_uci is not None and rec_uci == tgt_uci
            flags[key] = {
                "sound": sound,
                "tier": tier_ok,
                "fab": fabricated,
                "parseable": parseable,
            }
            models.append(
                {
                    "key": key,
                    "name": meta["name"],
                    "short": meta["short"],
                    "kind": meta["kind"],
                    "rec_san": o.get("rec_san"),
                    "rec_uci": rec_uci,
                    "parseable": parseable,
                    "sound": sound,
                    "tier_appropriate": tier_ok,
                    "fabricated": fabricated,
                    "n_violations": int(o.get("n_violations") or 0),
                    "violations": [
                        {"sentence": norm_ws(v.get("sentence", "")), "reason": v.get("reason", "")}
                        for v in (o.get("violations") or [])
                    ],
                    "coaching": norm_ws(gen.get((sid, key), "")),
                }
            )

        # --- Honest "OURS wins" derivation (only vs the frontier) ------------
        ours = flags.get("ours")
        beats: List[Dict[str, Any]] = []
        wins_tier = wins_faithful = False
        if ours is not None:
            for f in FRONTIER:
                ff = flags.get(f)
                if ff is None:
                    continue
                on: List[str] = []
                # (a) tier-appropriateness win
                if ours["tier"] and not ff["tier"]:
                    on.append("tier")
                # (b) faithfulness win — OURS must give a sound, non-fabricated answer
                if (not ours["fab"] and ours["sound"]) and ff["fab"]:
                    on.append("faithful")
                if on:
                    beats.append({"model": f, "name": MODEL_META[f]["short"], "on": on})
            wins_tier = any("tier" in b["on"] for b in beats)
            wins_faithful = any("faithful" in b["on"] for b in beats)
        ours_wins = wins_tier or wins_faithful

        sm = scn.get("student_move") or {}
        positions.append(
            {
                "key": f"{bench}:{sid}",
                "benchmark": bench,
                "scenario_id": sid,
                "fen": scn["fen"],
                "tier": scn["tier"],
                "phase": scn["phase"],
                "severity": scn["severity"],
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "student_move": {
                    "san": sm.get("san"),
                    "uci": sm.get("uci"),
                    "cp_loss": sm.get("cp_loss"),
                    "severity": sm.get("severity"),
                }
                if sm
                else None,
                "best_san": scn.get("best_san"),
                "sound_sans": [m["san"] for m in scn["sound_pool"]],
                "tier_target": tgt,
                "maia_top": [
                    {"san": m["san"], "uci": m["uci"], "policy": round(float(m["policy"]), 4)}
                    for m in (scn.get("maia") or [])[:3]
                ],
                "facts": facts,
                "ours_wins": ours_wins,
                "ours_wins_tier": wins_tier,
                "ours_wins_faithful": wins_faithful,
                "beats": beats,
                "n_beats": len(beats),
                "models": models,
            }
        )
    return positions


def main() -> int:
    all_positions: List[Dict[str, Any]] = []
    per_bench: Dict[str, Dict[str, int]] = {}
    for bench, root in BENCHES:
        if not (root / "scenarios.jsonl").exists():
            print(f"skip {bench}: {root} missing")
            continue
        pos = build_bench(bench, root)
        wins = sum(1 for p in pos if p["ours_wins"])
        per_bench[bench] = {
            "positions": len(pos),
            "ours_wins": wins,
            "ours_wins_tier": sum(1 for p in pos if p["ours_wins_tier"]),
            "ours_wins_faithful": sum(1 for p in pos if p["ours_wins_faithful"]),
        }
        all_positions.extend(pos)

    # Surface the OURS-wins positions first: wins before non-wins, then by how many
    # frontier models were beaten (desc), then a stable tier/benchmark/id order.
    all_positions.sort(
        key=lambda p: (
            0 if p["ours_wins"] else 1,
            -p["n_beats"],
            TIER_RANK.get(p["tier"], 9),
            p["benchmark"],
            p["scenario_id"],
        )
    )

    totals = {
        "positions": len(all_positions),
        "ours_wins": sum(1 for p in all_positions if p["ours_wins"]),
        "ours_wins_tier": sum(1 for p in all_positions if p["ours_wins_tier"]),
        "ours_wins_faithful": sum(1 for p in all_positions if p["ours_wins_faithful"]),
        "by_benchmark": per_bench,
    }

    doc = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "condition": COND,
            "frontier": list(FRONTIER),
            "tier_weight": TIER_HUMAN_WEIGHT,
            "model_meta": MODEL_META,
            "benchmarks": {
                "v2": "5 models (OURS · BASE · GPT-5.5 · Claude · Gemini), grounded product condition, 100 held-out positions.",
                "open": "Same grounded treatment extended to 9 bigger open models + the v2 five, 100 held-out positions.",
            },
            "definitions": {
                "sound": "Recommended move is in this position's Stockfish sound pool (non-blunder).",
                "tier_appropriate": "Pick equals THE human-findable sound move for the tier (src/teacher/tier_select.select_tier_move): beginner=highest-Maia sound move, advanced=sharpest sound move, intermediate=blend.",
                "fabricated": "Faithfulness verifier found >=1 false board fact in the coaching text.",
                "ours_wins": "OURS beats >=1 frontier model on the same grounded input: (a) OURS sound+tier-appropriate where the frontier model is not, or (b) OURS sound+faithful where the frontier model fabricates.",
            },
            "totals": totals,
        },
        "positions": all_positions,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} · {size_kb:.0f} KB")
    print(f"positions total: {totals['positions']}  ours_wins: {totals['ours_wins']} "
          f"(tier={totals['ours_wins_tier']}, faithful={totals['ours_wins_faithful']})")
    for bench, st in per_bench.items():
        print(f"  {bench:4s}: {st['positions']:3d} positions · ours_wins {st['ours_wins']:3d} "
              f"(tier={st['ours_wins_tier']}, faithful={st['ours_wins_faithful']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
