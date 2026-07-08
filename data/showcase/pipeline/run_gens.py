#!/usr/bin/env python3
"""Orchestrate all showcase generations with bounded parallelism (resumable).

Groups:
* ``local`` — OURS run on ALL three splits x 3 tiers (comprehensive, zero gaps,
  free) + BASE on train/test_new. Sequential (they share the Metal GPU).
* ``tfy``   — the 12 gateway models on train + test_new, up to ``--parallel``
  models at a time; each model does its two splits sequentially. test_reuse
  reuses the 803 generations, so no gateway calls are spent there.
* ``all``   — local then tfy.

Every unit is just ``gen.py`` (resumable + retrying), so this orchestrator is
safe to re-run: finished (scenario_id) rows are skipped.

Run::
  ~/.venvs/mlx/bin/python data/showcase/pipeline/run_gens.py --group local
  ~/.venvs/mlx/bin/python data/showcase/pipeline/run_gens.py --group tfy --parallel 4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
PY = sys.executable
GEN = str(HERE / "gen.py")

TFY_MODELS = ("gpt", "claude", "gemini", "q3_32b", "q3_next80b", "gemma3_27b",
              "llama33_70b", "dsv32", "glm5", "mistral3", "kimi25", "dsr1")


def _run(split: str, model: str, concurrency: int, max_retries: int = 6,
         timeout: float = 240.0) -> Tuple[str, str, int]:
    t0 = time.time()
    p = subprocess.run(
        [PY, GEN, "--split", split, "--model", model, "--concurrency", str(concurrency),
         "--max-retries", str(max_retries), "--timeout", str(timeout)],
        capture_output=True, text=True,
    )
    tail = (p.stderr or p.stdout or "").strip().splitlines()
    msg = tail[-1] if tail else ""
    print(f"[{split}/{model}] rc={p.returncode} {time.time()-t0:.0f}s :: {msg}", flush=True)
    return split, model, p.returncode


def run_local() -> None:
    for split in ("train", "test_new", "test_reuse"):
        _run(split, "ours", concurrency=1)
    for split in ("train", "test_new"):
        _run(split, "base", concurrency=1)


def run_tfy(parallel: int, concurrency: int, max_retries: int, timeout: float) -> None:
    def model_job(model: str) -> None:
        for split in ("train", "test_new"):
            _run(split, model, concurrency=concurrency, max_retries=max_retries, timeout=timeout)

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = {ex.submit(model_job, m): m for m in TFY_MODELS}
        for fut in as_completed(futs):
            m = futs[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[tfy/{m}] ORCHESTRATION ERROR: {exc}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", choices=("local", "tfy", "all"), required=True)
    p.add_argument("--parallel", type=int, default=4, help="TFY models in parallel.")
    p.add_argument("--concurrency", type=int, default=6, help="Per-model request concurrency.")
    p.add_argument("--max-retries", type=int, default=6, help="Per-request transient retries.")
    p.add_argument("--timeout", type=float, default=240.0, help="Per-request timeout (s).")
    args = p.parse_args(argv)

    t0 = time.time()
    if args.group in ("local", "all"):
        print("=== LOCAL group (OURS x3 splits, BASE x2) ===", flush=True)
        run_local()
    if args.group in ("tfy", "all"):
        print(f"=== TFY group ({len(TFY_MODELS)} models, parallel={args.parallel}) ===", flush=True)
        run_tfy(args.parallel, args.concurrency, args.max_retries, args.timeout)
    print(f"=== run_gens DONE ({args.group}) in {time.time()-t0:.0f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
