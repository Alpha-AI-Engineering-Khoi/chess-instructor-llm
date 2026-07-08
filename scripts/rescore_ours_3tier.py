"""Re-run OURS (chess-coach-v2) at ALL THREE tiers over the showcase library.

The platform library is the 200 held-out benchmark positions (100 in
``data/benchmark_v2`` + 100 in ``data/benchmark_open``) that the UI reads via
``web/public/showdown.json``. Those rows already carry every FRONTIER / open model
scored at each position's *native* tier — we DO NOT touch those. This script only
re-runs the LOCAL tuned coach, and it does so at all three coaching tiers so the
platform can show OURS's move + verdict per level and compute the real
"tier-differentiates" signal.

Apples-to-apples by construction: it reuses the benchmark's OWN pieces —
``prompts.build_grounded_user`` (the identical VERIFIED-FACTS + sound-pool + Maia
prompt every model saw), the greedy ``MLXBackend`` the benchmark used for local
models, ``objective.score_one`` (the deterministic move/soundness/fabrication
scorer), and ``tier_select.select_tier_move`` (the canonical tier target). The
only change vs the benchmark is: OURS, and all three tiers per position.

Output: ``data/ours_3tier/ours_cells.jsonl`` — one row per (position, tier),
resumable (skips rows already written). Nothing here writes to ``web/public`` or
to the benchmark artifacts; the interim UI file is assembled separately by
``scripts/build_showcase_interim.py``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.eval.evaluate import MLXBackend  # noqa: E402
from src.eval.benchmark.prompts import build_grounded_user, load_system_prompt  # noqa: E402
from src.eval.benchmark.objective import score_one  # noqa: E402
from src.teacher.tier_select import select_tier_move  # noqa: E402

TIERS = ("beginner", "intermediate", "advanced")
BENCHES = (
    ("v2", ROOT / "data" / "benchmark_v2"),
    ("open", ROOT / "data" / "benchmark_open"),
)
OURS_PATH = str(settings.MODELS / "mlx" / "chess-coach-v2")
OUT_DIR = ROOT / "data" / "ours_3tier"
OUT_PATH = OUT_DIR / "ours_cells.jsonl"

_MAIA_CACHE: Dict[tuple, List[Dict[str, Any]]] = {}


def load_scenarios() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bench, root in BENCHES:
        p = root / "scenarios.jsonl"
        if not p.exists():
            print(f"  ! skip {bench}: {p} missing")
            continue
        for line in p.open(encoding="utf-8"):
            if line.strip():
                scn = json.loads(line)
                scn["_bench"] = bench
                scn["_key"] = f"{bench}:{scn['id']}"
                rows.append(scn)
    return rows


def maia_for(fen: str, tier: str) -> List[Dict[str, Any]]:
    """Per-tier Maia top-6 (matches the benchmark's maia_top_k=6). Empty if lc0 down."""
    key = (fen, tier)
    if key in _MAIA_CACHE:
        return _MAIA_CACHE[key]
    try:
        from src.engine import maia_engine
        moves = maia_engine.human_moves(fen, tier, top_k=6)["moves"]
        out = [{"uci": m["uci"], "san": m["san"], "policy": float(m["policy"])} for m in moves]
    except Exception as exc:  # noqa: BLE001 - degrade to the stored native-tier maia
        print(f"    maia unavailable for {tier} ({type(exc).__name__}); using stored maia")
        out = []
    _MAIA_CACHE[key] = out
    return out


def done_keys() -> set:
    if not OUT_PATH.exists():
        return set()
    seen = set()
    for line in OUT_PATH.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            seen.add((r["key"], r["tier"]))
    return seen


def main() -> int:
    scenarios = load_scenarios()
    print(f"loaded {len(scenarios)} scenarios "
          f"({sum(1 for s in scenarios if s['_bench']=='v2')} v2, "
          f"{sum(1 for s in scenarios if s['_bench']=='open')} open)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    already = done_keys()
    if already:
        print(f"resuming: {len(already)} (position,tier) cells already done")

    print(f"loading OURS: {OURS_PATH}")
    t0 = time.time()
    backend = MLXBackend(OURS_PATH, max_tokens=400)  # GEN_MAX_TOKENS_LOCAL, greedy
    system = load_system_prompt()
    print(f"  model ready in {time.time()-t0:.1f}s")

    total = len(scenarios) * len(TIERS)
    done = len(already)
    t_start = time.time()
    with OUT_PATH.open("a", encoding="utf-8") as fh:
        for scn in scenarios:
            for tier in TIERS:
                if (scn["_key"], tier) in already:
                    continue
                # Per-tier grounding: this tier's Maia (fall back to stored native maia).
                maia = maia_for(scn["fen"], tier) or scn.get("maia", [])
                variant = dict(scn)
                variant["tier"] = tier
                variant["maia"] = maia
                maia_map = {str(m["uci"]): float(m["policy"]) for m in maia}
                try:
                    tgt = select_tier_move(tier, scn["sound_pool"], maia_map)
                    tgt_uci, tgt_san = tgt.uci, tgt.san
                except Exception:  # noqa: BLE001 - empty pool shouldn't happen here
                    tgt_uci = tgt_san = None
                try:
                    out = backend.generate(system, build_grounded_user(variant))
                except Exception as exc:  # noqa: BLE001 - one item must not abort the run
                    print(f"  ! gen failed {scn['_key']}/{tier}: {exc}")
                    continue
                sc = score_one(variant, out)
                tier_fit = bool(sc["move_sound"]) and sc["rec_uci"] is not None and sc["rec_uci"] == tgt_uci
                row = {
                    "key": scn["_key"],
                    "bench": scn["_bench"],
                    "scenario_id": scn["id"],
                    "fen": scn["fen"],
                    "native_tier": scn["tier"],
                    "tier": tier,
                    "rec_san": sc["rec_san"],
                    "rec_uci": sc["rec_uci"],
                    "move_parseable": sc["move_parseable"],
                    "sound": sc["move_sound"],
                    "tier_fit": tier_fit,
                    "tier_target_uci": tgt_uci,
                    "tier_target_san": tgt_san,
                    "fabricated": sc["fabricated"],
                    "n_violations": sc["n_violations"],
                    "violations": sc["violations"],
                    "coaching": out,
                    "maia_source": "per_tier" if maia is not scn.get("maia", []) else "stored_native",
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done += 1
                if done % 15 == 0 or done == total:
                    rate = (time.time() - t_start) / max(1, done - len(already))
                    eta = rate * (total - done) / 60.0
                    print(f"  {done}/{total} cells  (~{rate:.1f}s/cell, ETA {eta:.1f} min)")
    print(f"DONE: {done}/{total} cells -> {OUT_PATH} in {(time.time()-t_start)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
