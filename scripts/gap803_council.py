#!/usr/bin/env python3
"""Council instructiveness for the 803 gap eval — reuse the benchmark council.

Runs ONE blinded, cross-family council that ranks a UNIFIED 14-model field per
item on a REPRESENTATIVE stratified subset of (position, tier) items (balanced
across tier x phase), drawn from the frontier-subset positions so every one of
the 14 models has a generation for each judged item. Council on all 803 x 14 is
expensive and statistically unnecessary for a rank estimate — a stratified ~120
gives a tight instructiveness signal.

Reuses ``src.eval.benchmark.council.run_council`` verbatim (same blinded rubric,
same 3 frontier judges) with ``MODEL_ORDER``/``ANON_LABELS`` widened to 14.

Run::  ~/.venvs/mlx/bin/python -m scripts.gap803_council --items 120
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
os.environ.setdefault("BENCH_DIR", str(_ROOT / "data" / "benchmark_gap803"))

from dotenv import load_dotenv  # noqa: E402

from config import settings  # noqa: E402
from src.eval.benchmark import config as bcfg  # noqa: E402
from scripts.gap803_common import stratified_scenarios  # noqa: E402

BENCH = Path(os.environ["BENCH_DIR"])
SCN_PATH = BENCH / "scenarios.jsonl"
FRONTIER_IDS = BENCH / "frontier_ids.txt"

#: The definitive 14-model field (all evaluated models get an instructiveness rank).
FIELD = (
    "ours", "base",
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

    # Widen the field to all 14 models.
    bcfg.MODEL_ORDER = tuple(FIELD)
    bcfg.ANON_LABELS = bcfg.labels_for(len(FIELD))
    bcfg.JUDGE_MAX_TOKENS = args.judge_max_tokens

    scns = _read_jsonl(SCN_PATH)
    keep = set(FRONTIER_IDS.read_text(encoding="utf-8").split())
    eligible = [s for s in scns if s["pos_id"] in keep]  # all 14 have gens here
    subset = stratified_scenarios(eligible, args.items, seed=args.seed)

    import collections
    dist = collections.Counter((s["tier"], s["phase"]) for s in subset)
    print(f"council: {len(subset)} (pos,tier) items from {len(keep)} frontier positions; "
          f"field={len(FIELD)}, judges={list(bcfg.JUDGE_KEYS)}", file=sys.stderr)
    print(f"   tier x phase: {dict(dist)}", file=sys.stderr)

    from src.eval.benchmark.council import run_council
    res = run_council(
        subset, ["grounded"], list(bcfg.JUDGE_KEYS),
        concurrency=args.concurrency, min_interval=args.min_interval,
        timeout=args.timeout, max_retries=args.max_retries,
    )
    print(f"council done: {res}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
