#!/usr/bin/env python3
"""Build the REUSE part of the TEST sample from the definitive 803 benchmark.

The 803 gap eval already ran all 14 models on the ``frontier_ids`` subset (the
positions where every one of the 14 has a grounded generation). Those are the
held-out, zero-leakage positions the task says to REUSE. This driver:

* copies those flattened (pos x tier) scenarios (``split="test"``) into
  ``data/showcase/test_reuse/scenarios.jsonl`` (read-only from benchmark_gap803);
* reuses the existing per-model generations verbatim into
  ``data/showcase/test_reuse/gen/<model>.jsonl`` — so NO new coaching calls are
  spent for this part (that is the whole point of reuse).

The unified showcase council (absolute move + instructiveness grades) is then run
over these items too, so every showcase cell is graded on one consistent scale.
benchmark_gap803 is only ever READ.

Run::  ~/.venvs/mlx/bin/python data/showcase/pipeline/build_test_reuse.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from common import (  # noqa: E402
    FIELD, ROOT, TEST_REUSE_DIR, ensure_dirs, read_jsonl, write_jsonl,
)

sys.path.insert(0, str(ROOT))
from config import settings  # noqa: E402

GAP = settings.DATA / "benchmark_gap803"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-positions", type=int, default=0,
                   help="Optional cap on reused positions (0 = all frontier_ids).")
    args = p.parse_args(argv)

    ensure_dirs()
    fids_path = GAP / "frontier_ids.txt"
    scn_path = GAP / "scenarios.jsonl"
    if not fids_path.exists() or not scn_path.exists():
        print(f"BLOCKED: missing {fids_path} or {scn_path}.", file=sys.stderr)
        return 1

    frontier_ids = set(fids_path.read_text(encoding="utf-8").split())
    if args.max_positions:
        frontier_ids = set(sorted(frontier_ids)[: args.max_positions])
    print(f"frontier positions to reuse: {len(frontier_ids)}", file=sys.stderr)

    scns = [s for s in read_jsonl(scn_path) if s.get("pos_id") in frontier_ids]
    for s in scns:
        s["split"] = "test"
    keep_ids = {s["id"] for s in scns}
    write_jsonl(TEST_REUSE_DIR / "scenarios.jsonl", scns)
    print(f"reuse scenarios: {len(scns)} (from {len(frontier_ids)} positions)", file=sys.stderr)

    # Reuse each model's generations verbatim (filtered to the reuse scenarios).
    coverage: Dict[str, int] = {}
    for key in FIELD:
        src = GAP / "gen" / f"{key}.jsonl"
        if not src.exists():
            print(f"  [warn] no gap803 gen for {key}", file=sys.stderr)
            coverage[key] = 0
            continue
        rows = [g for g in read_jsonl(src) if g.get("scenario_id") in keep_ids]
        write_jsonl(TEST_REUSE_DIR / "gen" / f"{key}.jsonl", rows)
        coverage[key] = len({g["scenario_id"] for g in rows})

    print("reuse gen coverage (unique scenarios per model):", file=sys.stderr)
    for key in FIELD:
        flag = "" if coverage[key] >= len(keep_ids) else "  <-- INCOMPLETE"
        print(f"  {key:12s} {coverage[key]}/{len(keep_ids)}{flag}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
