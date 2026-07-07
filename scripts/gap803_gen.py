#!/usr/bin/env python3
"""Generate grounded coaching for the DEFINITIVE 803 gap eval — per model, resumable.

Writes ONE file per model (``data/benchmark_gap803/gen/<model>.jsonl``) in the
benchmark generation schema, so several models can run as independent background
jobs with zero shared-file write races. Merge them into ``generations.jsonl``
afterwards (``gap803_report.py merge``) and reuse the benchmark objective/council.

Grounding + system + format are byte-identical to the v2 benchmark (reuses
``src.eval.benchmark.prompts`` + ``backends``), so all 14 models are comparable.

Sub-commands::

    ~/.venvs/mlx/bin/python -m scripts.gap803_gen seed --src data/eval/gap_positions.jsonl
    ~/.venvs/mlx/bin/python -m scripts.gap803_gen run --model gemma3_27b
    ~/.venvs/mlx/bin/python -m scripts.gap803_gen run --model ours          # local MLX (free)
    ~/.venvs/mlx/bin/python -m scripts.gap803_gen run --model gpt --subset-ids data/benchmark_gap803/frontier_ids.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("BENCH_DIR", str(_ROOT / "data" / "benchmark_gap803"))

from config import settings  # noqa: E402
from src.eval.benchmark import config as bcfg  # noqa: E402
from src.eval.benchmark.prompts import build_user_prompt, load_system_prompt  # noqa: E402
from scripts.gap803_common import flatten, load_positions  # noqa: E402

BENCH = Path(os.environ["BENCH_DIR"])
SCN_PATH = BENCH / "scenarios.jsonl"
GEN_DIR = BENCH / "gen"


def _abs(x: str) -> Path:
    p = Path(x)
    return p if p.is_absolute() else _ROOT / p


def _load_scn() -> List[Dict[str, Any]]:
    if not SCN_PATH.exists():
        raise SystemExit(f"missing {SCN_PATH}; run `seed` first.")
    return [json.loads(l) for l in SCN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def cmd_seed(a: argparse.Namespace) -> int:
    pos = load_positions(_abs(a.src))
    scns = flatten(pos)
    SCN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCN_PATH.open("w", encoding="utf-8") as fh:
        for s in scns:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"seed: {len(pos)} positions -> {len(scns)} (pos,tier) scenarios -> {SCN_PATH}")
    return 0


def _persist(fh, scn: Dict[str, Any], model: str, text: str, usage: Dict[str, int]) -> None:
    fh.write(json.dumps({
        "scenario_id": scn["id"], "model": model, "condition": "grounded",
        "tier": scn["tier"], "phase": scn["phase"], "severity": scn["severity"],
        "pos_id": scn["pos_id"],
        "output": text,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "ts": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False) + "\n")
    fh.flush()


def _done_ids(out: Path) -> set:
    done: set = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    done.add(json.loads(l)["scenario_id"])
                except Exception:  # noqa: BLE001
                    continue
    return done


def cmd_run(a: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    load_dotenv(settings.ROOT / ".env")

    from src.eval.benchmark.backends import (  # local import so `seed` needs no mlx/openai
        MLXLocal, RateLimiter, TFYChat, make_tfy_client,
    )

    scns = _load_scn()
    if a.subset_ids:
        keep = set(_abs(a.subset_ids).read_text(encoding="utf-8").split())
        scns = [s for s in scns if s["pos_id"] in keep or s["id"] in keep]
    if a.limit:
        scns = scns[: a.limit]

    model = a.model
    if model not in bcfg.MODELS:
        raise SystemExit(f"unknown model {model!r}; choices: {list(bcfg.MODELS)}")
    m = bcfg.MODELS[model]
    # OURS = the shipped v2 coach (config default ident is v1); point it at v2.
    if model == "ours":
        from dataclasses import replace as _replace
        ours_path = os.environ.get("BENCH_OURS_MODEL", str(settings.MODELS / "mlx" / "chess-coach-v2"))
        m = _replace(m, ident=ours_path, display="OURS (chess-coach-v2, 1.7B tuned)")

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out = _abs(a.out) if a.out else (GEN_DIR / f"{model}.jsonl")
    done = _done_ids(out)
    todo = [s for s in scns if s["id"] not in done]
    print(f"{model} ({m.kind}): {len(todo)} pending of {len(scns)} ({len(done)} done)", file=sys.stderr)
    if not todo:
        print(f"DONE {model}: nothing to do", file=sys.stderr)
        return 0

    system = load_system_prompt()
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        if m.kind == "mlx":
            backend = MLXLocal(m.ident, max_tokens=a.max_tokens or bcfg.GEN_MAX_TOKENS_LOCAL)
            for i, scn in enumerate(todo, 1):
                try:
                    text, usage = backend.complete(system, build_user_prompt(scn, "grounded"))
                    _persist(fh, scn, model, text, usage)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {scn['id']}: {e}", file=sys.stderr)
                if i % 50 == 0 or i == len(todo):
                    dt = time.time() - t0
                    print(f"  {model} {i}/{len(todo)} ({dt / i:.2f}s/it, eta {dt/i*(len(todo)-i)/60:.0f}m)",
                          file=sys.stderr)
        else:
            client = make_tfy_client(a.timeout)
            lim = RateLimiter(a.min_interval)
            chat = TFYChat(client, model_id=m.ident, max_tokens=bcfg.GEN_MAX_TOKENS_TFY,
                           max_retries=a.max_retries, limiter=lim,
                           reasoning_effort=m.reasoning_effort)
            lock = threading.Lock()
            n = [0]

            def task(scn: Dict[str, Any]):
                text, usage = chat.complete(system, build_user_prompt(scn, "grounded"))
                return scn, text, usage

            with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
                futs = {ex.submit(task, s): s for s in todo}
                for fut in as_completed(futs):
                    scn = futs[fut]
                    try:
                        scn, text, usage = fut.result()
                        with lock:
                            _persist(fh, scn, model, text, usage)
                    except Exception as e:  # noqa: BLE001
                        print(f"  ! {scn['id']}: {e}", file=sys.stderr)
                    n[0] += 1
                    if n[0] % 100 == 0 or n[0] == len(todo):
                        dt = time.time() - t0
                        print(f"  {model} {n[0]}/{len(todo)} ({dt / n[0]:.2f}s/it, "
                              f"eta {dt/n[0]*(len(todo)-n[0])/60:.0f}m)", file=sys.stderr)
    print(f"DONE {model}: {len(todo)} gens in {time.time() - t0:.0f}s -> {out}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("seed")
    ps.add_argument("--src", default="data/eval/gap_positions.jsonl")
    ps.set_defaults(func=cmd_seed)
    pr = sub.add_parser("run")
    pr.add_argument("--model", required=True)
    pr.add_argument("--out", default="")
    pr.add_argument("--subset-ids", dest="subset_ids", default="")
    pr.add_argument("--limit", type=int, default=0)
    pr.add_argument("--max-tokens", dest="max_tokens", type=int, default=0)
    pr.add_argument("--concurrency", type=int, default=8)
    pr.add_argument("--min-interval", dest="min_interval", type=float, default=0.05)
    pr.add_argument("--timeout", type=float, default=240.0)
    pr.add_argument("--max-retries", dest="max_retries", type=int, default=6)
    pr.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
