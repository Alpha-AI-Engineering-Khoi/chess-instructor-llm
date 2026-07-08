#!/usr/bin/env python3
"""Build the NEW held-out TEST sample from freshly-pulled Lichess positions.

Pipeline: raw Lichess decision positions (from ``src/ingest/lichess_sampler.py``)
-> dedup against EVERY existing corpus (train_v2/valid_v2, train_v3/valid_v3, the
candidate pools, all benchmark scenario sets, and the showcase TRAIN split) ->
Stockfish+Maia grounding (same ``analyze_one`` as the 803 set) -> keep only the
truly discriminating, non-trivial, legal positions (tier-appropriate move != engine
#1 for >=1 tier) -> take the STRONGEST few hundred (balanced across phase) ->
flatten to (pos x tier) scenarios labelled ``split="test"``.

Reports the 10k -> filtered yield for the SHOWCASE_REPORT.

Run::
  # 1) pull (existing sampler, writes under data/showcase/):
  ~/.venvs/mlx/bin/python src/ingest/lichess_sampler.py --count 10000 \
      --out data/showcase/lichess_raw.jsonl --max-requests 4000 --sleep 0.25 \
      --games-per-user 25 --positions-per-game 4 --no-progress
  # 2) ground + filter:
  ~/.venvs/mlx/bin/python data/showcase/pipeline/build_test_new.py --target 210
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any, Dict, List

from common import (  # noqa: E402
    ROOT, SHOWCASE_DIR, TEST_NEW_DIR, TRAIN_DIR, ensure_dirs, read_jsonl, write_jsonl,
)

sys.path.insert(0, str(ROOT))
from scripts.divergence_analysis import pos_key  # noqa: E402
from scripts.gap803_common import flatten  # noqa: E402
from ground import dedup_keys, ground_records  # noqa: E402

RAW_PATH = SHOWCASE_DIR / "lichess_raw.jsonl"


def _strength(r: Dict[str, Any]) -> tuple:
    """Rank key for 'strongest' discriminating positions (higher = better)."""
    return (
        int(r.get("n_strong_tiers", 0)),
        int(r.get("n_distinct_tier_moves", 0)),
        int(r.get("n_discriminating_tiers", 0)),
        float(r.get("max_policy_gap", 0.0)),
    )


def _balanced_strong(rows: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    """Take the strongest ``target`` rows, round-robin across phase for variety."""
    buckets: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        buckets[r.get("phase", "middlegame")].append(r)
    for b in buckets.values():
        b.sort(key=_strength, reverse=True)
    order = sorted(buckets.keys())
    picked: List[Dict[str, Any]] = []
    i = 0
    while len(picked) < target and any(buckets[k] for k in order):
        k = order[i % len(order)]
        if buckets[k]:
            picked.append(buckets[k].pop(0))
        i += 1
    return picked


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default=str(RAW_PATH))
    p.add_argument("--target", type=int, default=210, help="Final test positions (few hundred).")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)

    ensure_dirs()
    raw_path = Path(args.raw)
    raw = read_jsonl(raw_path)
    n_raw = len(raw)
    if not raw:
        print(f"BLOCKED: no raw positions at {raw_path}. Run the lichess sampler first.",
              file=sys.stderr)
        return 1
    print(f"raw Lichess positions pulled: {n_raw}", file=sys.stderr)

    # Dedup against EVERY existing corpus + the showcase TRAIN split.
    avoid = dedup_keys(extra_fen_files=[TRAIN_DIR / "positions.jsonl"])
    print(f"dedup avoid-set (board keys): {len(avoid)}", file=sys.stderr)

    seen: set = set()
    fresh: List[Dict[str, Any]] = []
    n_dupe_corpus = n_dupe_self = 0
    for r in raw:
        fen = r.get("fen")
        if not fen:
            continue
        k = pos_key(fen)
        if k in avoid:
            n_dupe_corpus += 1
            continue
        if k in seen:
            n_dupe_self += 1
            continue
        seen.add(k)
        fresh.append(r)
    print(f"after dedup: {len(fresh)} fresh (dropped {n_dupe_corpus} corpus-overlap, "
          f"{n_dupe_self} self-dupe)", file=sys.stderr)

    grounded_path = TEST_NEW_DIR / "grounded.jsonl"
    ground_records(fresh, grounded_path, workers=args.workers, resume=True)
    grounded = read_jsonl(grounded_path)

    # Defensive re-dedup + the quality gate: legal (analyzed), non-trivial,
    # >=2 sound, discriminating for >=1 tier (tier-move != engine #1).
    eligible: List[Dict[str, Any]] = []
    seen2: set = set()
    for r in grounded:
        k = r.get("board_key") or pos_key(r["fen"])
        if k in avoid or k in seen2:
            continue
        seen2.add(k)
        if r.get("eligible"):  # non-trivial, >=2 sound, >=1 discriminating tier
            eligible.append(r)
    n_grounded = len(grounded)
    n_eligible = len(eligible)
    print(f"grounded={n_grounded} discriminating-eligible={n_eligible}", file=sys.stderr)
    if not eligible:
        print("BLOCKED: no discriminating eligible positions after grounding.", file=sys.stderr)
        return 1

    picked = _balanced_strong(eligible, args.target)
    scenarios = flatten(picked)
    for s in scenarios:
        s["split"] = "test"
    write_jsonl(TEST_NEW_DIR / "scenarios.jsonl", scenarios)
    write_jsonl(TEST_NEW_DIR / "positions.jsonl", picked)

    # Yield readout for the report.
    yield_info = {
        "raw_pulled": n_raw,
        "after_dedup": len(fresh),
        "grounded": n_grounded,
        "discriminating_eligible": n_eligible,
        "selected": len(picked),
        "dupe_corpus": n_dupe_corpus,
        "dupe_self": n_dupe_self,
    }
    write_jsonl(TEST_NEW_DIR / "yield.json.jsonl", [yield_info])
    (TEST_NEW_DIR / "yield.json").write_text(__import__("json").dumps(yield_info, indent=2))

    dist = collections.Counter((r.get("source_tier"), r.get("phase")) for r in picked)
    print(f"TEST-new: {len(scenarios)} scenarios from {len(picked)} positions. "
          f"yield {n_raw}->{len(picked)}. tier x phase: {dict(dist)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
