#!/usr/bin/env python3
"""Move-safety (blunder-only) for the 803 gap eval — deterministic, local, free.

A pick is SAFE unless it is a BLUNDER (cp-loss >= BLUNDER_CP, 250). Picks already
inside the position's Stockfish sound pool are safe by construction (cp-loss <=
150 < 250). Only *non-sound, parseable* picks need a fresh Stockfish evaluation;
unparseable/empty picks recommend no usable move and count as NOT safe (same
denominator as the objective's n). Mirrors ``scripts/rescore_move_safety.py`` but
runs over ``data/benchmark_gap803`` and all 14 models.

Run::  ~/.venvs/mlx/bin/python -m scripts.gap803_safety
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import BLUNDER_CP, STOCKFISH_BIN  # noqa: E402
from scripts.rescore_move_safety import Engine  # noqa: E402 (reuse the UCI wrapper)

BENCH = Path(os.environ.get("BENCH_DIR", str(_ROOT / "data" / "benchmark_gap803")))
SCN_PATH = BENCH / "scenarios.jsonl"
OBJ_PATH = BENCH / "objective.jsonl"
OUT_PATH = BENCH / "move_safety.json"


def _read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    scn = {s["id"]: s for s in _read_jsonl(SCN_PATH)}
    picks = _read_jsonl(OBJ_PATH)

    base_fens: set = set()
    after: set = set()
    for r in picks:
        if r.get("move_sound"):
            continue
        rec = r.get("rec_uci")
        if r.get("move_parseable") and rec:
            s = scn.get(r["scenario_id"])
            if not s:
                continue
            base_fens.add(s["fen"])
            after.add((s["fen"], rec))

    print(f"safety: {len(picks)} picks; {len(base_fens)} base fens + "
          f"{len(after)} (fen,move) to Stockfish-eval (movetime 500ms) ...", file=sys.stderr)

    eng = Engine(STOCKFISH_BIN)
    best_cp = {fen: eng.eval_cp(fen, []) for fen in sorted(base_fens)}
    rec_after = {}
    for i, (fen, uci) in enumerate(sorted(after), 1):
        rec_after[(fen, uci)] = eng.eval_cp(fen, [uci])
        if i % 100 == 0:
            print(f"   evaluated {i}/{len(after)}", file=sys.stderr)
    eng.close()

    def cp_loss(fen: str, uci: str) -> int:
        return best_cp[fen] - (-rec_after[(fen, uci)])

    agg = defaultdict(lambda: {"safe": 0, "blunder": 0, "n": 0, "unparseable": 0})
    worst = []
    for r in picks:
        m = r["model"]
        a = agg[m]
        a["n"] += 1
        rec = r.get("rec_uci")
        if r.get("move_sound"):
            a["safe"] += 1
            continue
        if not (r.get("move_parseable") and rec):
            a["unparseable"] += 1
            continue
        s = scn.get(r["scenario_id"])
        if not s:
            continue
        loss = cp_loss(s["fen"], rec)
        if loss >= BLUNDER_CP:
            a["blunder"] += 1
            worst.append((loss, m, r["scenario_id"], r.get("rec_san")))
        else:
            a["safe"] += 1

    out = {"move_safe": {}, "blunder_rate": {}, "detail": {}}
    print(f"\n{'model':16} {'safe%':>7} {'blunders':>9} {'unparse':>8} {'n':>6}", file=sys.stderr)
    for m, a in sorted(agg.items()):
        safe_rate = a["safe"] / a["n"] if a["n"] else 0.0
        out["move_safe"][m] = round(safe_rate, 4)
        out["blunder_rate"][m] = round(a["blunder"] / a["n"], 4) if a["n"] else 0.0
        out["detail"][m] = a
        print(f"{m:16} {safe_rate*100:>6.1f}% {a['blunder']:>9} {a['unparseable']:>8} {a['n']:>6}",
              file=sys.stderr)

    print("\nworst recommended blunders (cp-loss):", file=sys.stderr)
    for loss, m, sid, san in sorted(worst, reverse=True)[:12]:
        print(f"  {loss:>7}  {m:<14} {san}  ({sid})", file=sys.stderr)

    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
