#!/usr/bin/env python3
"""Council instructiveness for the v4 803 eval — reuse the benchmark council.

Identical to ``scripts/gap803_council.py`` (same blinded, cross-family rubric,
same 3 frontier judges, ``run_council`` reused verbatim) but:

* ranks a **16-model** field that ADDS ``ours_v4`` next to ``ours_v3`` so the two
  are ranked head-to-head under one blinded rubric, and
* reads/writes the ISOLATED v4 benchmark dir ``data/benchmark_v4/`` so it never
  collides with the shared ``data/benchmark_gap803`` council another worker owns.

NOTE: this is the *existing* stratified-~120 council path. If the separate
full-council harness upgrade is available, prefer it for the final v4-vs-v3
instructiveness number; this gives the apples-to-apples signal in the meantime.

Run::  ~/.venvs/mlx/bin/python -m scripts.gap803_v4_council --items 120
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("BENCH_DIR", str(_ROOT / "data" / "benchmark_v4"))

from dotenv import load_dotenv  # noqa: E402

from config import settings  # noqa: E402
from src.eval.benchmark import config as bcfg  # noqa: E402
from scripts.gap803_common import stratified_scenarios  # noqa: E402

BENCH = Path(os.environ["BENCH_DIR"])
SCN_PATH = BENCH / "scenarios.jsonl"
FRONTIER_IDS = BENCH / "frontier_ids.txt"

#: v4 field: ours_v4 + ours_v3 + the same 14 the v3 council ranked = 16 models.
FIELD = (
    "ours_v4", "ours_v3", "ours", "base",
    "gemma3_27b", "q3_32b", "q3_next80b", "llama33_70b",
    "dsv32", "glm5", "mistral3", "kimi25", "dsr1",
    "gpt", "claude", "gemini",
)


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv=None) -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items", type=int, default=120)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--min-interval", type=float, default=0.06)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--judge-max-tokens", type=int, default=4000)
    args = p.parse_args(argv)

    load_dotenv(settings.ROOT / ".env")

    bcfg.MODEL_ORDER = tuple(FIELD)
    bcfg.ANON_LABELS = bcfg.labels_for(len(FIELD))
    bcfg.JUDGE_MAX_TOKENS = args.judge_max_tokens

    scns = _read_jsonl(SCN_PATH)
    keep = set(FRONTIER_IDS.read_text(encoding="utf-8").split())
    eligible = [s for s in scns if s["pos_id"] in keep]
    subset = stratified_scenarios(eligible, args.items, seed=args.seed)

    import collections
    dist = collections.Counter((s["tier"], s["phase"]) for s in subset)
    print(f"v4 council: {len(subset)} (pos,tier) items from {len(keep)} frontier positions; "
          f"field={len(FIELD)}, judges={list(bcfg.JUDGE_KEYS)}", file=sys.stderr)
    print(f"   tier x phase: {dict(dist)}", file=sys.stderr)

    from src.eval.benchmark.council import run_council
    res = run_council(
        subset, ["grounded"], list(bcfg.JUDGE_KEYS),
        concurrency=args.concurrency, min_interval=args.min_interval,
        timeout=args.timeout, max_retries=args.max_retries,
    )
    print(f"v4 council done: {res}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
