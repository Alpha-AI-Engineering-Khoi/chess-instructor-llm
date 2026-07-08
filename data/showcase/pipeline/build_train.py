#!/usr/bin/env python3
"""Build the TRAINING showcase sample (honest, in-distribution).

Samples ~132 positions that the LIVE OURS-v2 coach was actually trained on: the
board (placement + side-to-move) must appear in ``data/dataset/train_v2.jsonl``
(the v2 SFT set), drawn from the v2 teacher-distillation candidate pool
(``data/generated/candidates_v2.jsonl``). Each sampled position is RE-GROUNDED
with the same Stockfish+Maia analysis as the 803 benchmark (so the grounding and
the canonical tier-move are identical), then flattened into (pos x tier)
scenarios labelled ``split="train"``.

This is deliberately IN-DISTRIBUTION and is reported as such — it measures how the
field (and OURS) coaches on positions the tuned model has seen, not a held-out
generalisation test.

Run::  ~/.venvs/mlx/bin/python data/showcase/pipeline/build_train.py --target 132
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

from common import (  # noqa: E402
    ROOT, TRAIN_DIR, ensure_dirs, read_jsonl, write_jsonl,
)

sys.path.insert(0, str(ROOT))
from config import settings  # noqa: E402
from scripts.divergence_analysis import build_heldin_keys, pos_key  # noqa: E402
from scripts.gap803_common import flatten, stratified_positions  # noqa: E402
from ground import ground_records  # noqa: E402


def _raw_from_candidates(train_keys: set) -> List[Dict[str, Any]]:
    """Unique in-distribution raw records (Lichess-sampler schema) from candidates_v2."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for row in read_jsonl(settings.GENERATED / "candidates_v2.jsonl"):
        ti = row.get("teacher_input") or {}
        fen = ti.get("fen")
        if not fen:
            continue
        key = pos_key(fen)
        if key not in train_keys or key in seen:
            continue
        seen.add(key)
        sm = ti.get("student_move") or {}
        meta = row.get("meta") or {}
        raw_id = meta.get("base_id") or row.get("id") or key
        clean_id = str(raw_id).split("#")[0]  # drop any '#ctr-<tier>' contrastive suffix
        out.append({
            "id": f"train_{clean_id}",
            "fen": fen,
            "tier": ti.get("tier") or row.get("tier"),
            "played_move_uci": sm.get("uci") or "",
            "mover_rating": meta.get("mover_rating"),
            "time_control": meta.get("time_control"),
        })
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int, default=132, help="Final positions (120-150).")
    p.add_argument("--ground-cap", type=int, default=260,
                   help="How many in-distribution candidates to ground (superset).")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=20260707)
    args = p.parse_args(argv)

    ensure_dirs()

    # Positions OURS-v2 was TRAINED on (train split only; valid excluded).
    train_keys = build_heldin_keys(settings.DATASET / "train_v2.jsonl", Path("/nonexistent"))
    print(f"train_v2 board keys: {len(train_keys)}", file=sys.stderr)

    raw = _raw_from_candidates(train_keys)
    print(f"in-distribution unique candidate boards: {len(raw)}", file=sys.stderr)
    if not raw:
        print("BLOCKED: no in-distribution candidates found.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(raw)
    superset = raw[: args.ground_cap]

    grounded_path = TRAIN_DIR / "grounded.jsonl"
    ground_records(superset, grounded_path, workers=args.workers, resume=True)

    grounded = read_jsonl(grounded_path)
    # Keep genuinely coachable positions (a real choice exists, non-trivial).
    eligible = [r for r in grounded if r.get("n_sound", 0) >= 2 and not r.get("trivial")]
    print(f"grounded={len(grounded)} eligible={len(eligible)}", file=sys.stderr)

    picked = stratified_positions(eligible, args.target, seed=args.seed)
    print(f"picked {len(picked)} positions (target {args.target})", file=sys.stderr)

    scenarios = flatten(picked)
    for s in scenarios:
        s["split"] = "train"
    write_jsonl(TRAIN_DIR / "scenarios.jsonl", scenarios)
    write_jsonl(TRAIN_DIR / "positions.jsonl", picked)

    import collections
    dist = collections.Counter((r.get("source_tier"), r.get("phase")) for r in picked)
    disc = sum(1 for r in picked if r.get("discriminating"))
    print(f"TRAIN scenarios: {len(scenarios)} from {len(picked)} positions "
          f"({disc} discriminating). tier x phase: {dict(dist)}", file=sys.stderr)
    print(f"wrote -> {TRAIN_DIR / 'scenarios.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
