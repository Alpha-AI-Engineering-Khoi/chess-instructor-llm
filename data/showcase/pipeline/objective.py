#!/usr/bin/env python3
"""Deterministic objective scoring for a showcase split (free, resumable).

For every (scenario x model) generation it records the recommended move plus the
three flags the showcase reports:

* ``sound``      — the recommended move is in this position's Stockfish sound pool
* ``tier_fit``   — the recommended move == the canonical ``select_tier_move`` move
                   for this tier (the moat)
* ``fabricated`` — the non-LLM faithfulness verifier found >=1 false board fact

Reuses ``src.eval.benchmark.objective.score_one`` verbatim (same extractor, same
verifier) so the numbers match the definitive benchmark. Keyed by
(scenario_id, model); re-running only scores what is new.

Run::  ~/.venvs/mlx/bin/python data/showcase/pipeline/objective.py --split train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from common import (  # noqa: E402
    FIELD, ROOT, SPLIT_DIRS, append_jsonl, done_keys, read_jsonl,
)

sys.path.insert(0, str(ROOT))
from src.eval.benchmark.objective import score_one  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True, choices=list(SPLIT_DIRS))
    args = p.parse_args(argv)

    split_dir = SPLIT_DIRS[args.split]
    by_id: Dict[str, Dict[str, Any]] = {s["id"]: s for s in read_jsonl(split_dir / "scenarios.jsonl")}
    if not by_id:
        print(f"BLOCKED: no scenarios in {split_dir}", file=sys.stderr)
        return 1

    out = split_dir / "objective.jsonl"
    done = done_keys(out, ["scenario_id", "model"])
    scored = skipped = 0

    for key in FIELD:
        gen_path = split_dir / "gen" / f"{key}.jsonl"
        for gen in read_jsonl(gen_path):
            sid = gen.get("scenario_id")
            if (sid, key) in done:
                continue
            scn = by_id.get(sid)
            if scn is None:
                skipped += 1
                continue
            s = score_one(scn, gen.get("output", ""))
            canonical = scn.get("canonical_uci")
            row = {
                "scenario_id": sid,
                "model": key,
                "tier": scn["tier"],
                "phase": scn["phase"],
                "pos_id": scn.get("pos_id", sid),
                "rec_san": s["rec_san"],
                "rec_uci": s["rec_uci"],
                "sound": bool(s["move_sound"]),
                "tier_fit": bool(s["rec_uci"] is not None and s["rec_uci"] == canonical),
                "fabricated": bool(s["fabricated"]),
                "n_violations": int(s["n_violations"]),
                "no_engine_speak": bool(s["no_engine_speak"]),
                "canonical_uci": canonical,
                "engine_best_uci": scn.get("engine_best_uci"),
            }
            append_jsonl(out, row)
            done.add((sid, key))
            scored += 1

    print(f"[objective/{args.split}] scored={scored} skipped(no-scn)={skipped} -> {out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
