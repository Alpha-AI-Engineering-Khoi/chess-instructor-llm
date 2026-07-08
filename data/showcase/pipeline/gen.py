#!/usr/bin/env python3
"""Run ONE model's grounded coaching for a showcase split — resumable + costed.

Writes ``data/showcase/<split>/gen/<model>.jsonl`` in the benchmark generation
schema, so different models run as independent background jobs with no shared-file
write races. Grounding + system prompt + format are byte-identical to the 803
benchmark (reuses ``src.eval.benchmark.prompts`` + ``backends``), so all 14 models
are directly comparable.

Transient gateway failures (billing/timeout/rate-limit/empty) are retried with the
project's proven backoff (``TFYChat`` + ``--max-retries``). Re-running resumes from
the last checkpoint.

Run::
  ~/.venvs/mlx/bin/python data/showcase/pipeline/gen.py --split train --model ours
  ~/.venvs/mlx/bin/python data/showcase/pipeline/gen.py --split test_new --model claude
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from common import (  # noqa: E402
    FIELD, LOCAL_KEYS, ROOT, SPLIT_DIRS, append_jsonl, read_jsonl, resolved_ident,
)

sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

from config import settings  # noqa: E402
from src.eval.benchmark import config as bcfg  # noqa: E402
from src.eval.benchmark.prompts import build_user_prompt, load_system_prompt  # noqa: E402


def _done(out: Path) -> set:
    return {r["scenario_id"] for r in read_jsonl(out) if "scenario_id" in r}


def _persist(out: Path, scn: Dict[str, Any], model: str, text: str, usage: Dict[str, int]) -> None:
    append_jsonl(out, {
        "scenario_id": scn["id"], "model": model, "condition": "grounded",
        "tier": scn["tier"], "phase": scn["phase"], "severity": scn.get("severity", "none"),
        "pos_id": scn.get("pos_id", scn["id"]),
        "output": text,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "ts": datetime.now(timezone.utc).isoformat(),
    })


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True, choices=list(SPLIT_DIRS))
    p.add_argument("--model", required=True, choices=list(FIELD))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--min-interval", type=float, default=0.05)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--max-retries", type=int, default=8)
    args = p.parse_args(argv)

    load_dotenv(settings.ROOT / ".env")
    from src.eval.benchmark.backends import (  # local import: seed/build steps need no mlx
        MLXLocal, RateLimiter, TFYChat, make_tfy_client,
    )

    split_dir = SPLIT_DIRS[args.split]
    scns = read_jsonl(split_dir / "scenarios.jsonl")
    if not scns:
        print(f"BLOCKED: no scenarios at {split_dir/'scenarios.jsonl'}", file=sys.stderr)
        return 1
    if args.limit:
        scns = scns[: args.limit]

    key = args.model
    m = bcfg.MODELS[key]
    ident = resolved_ident(key)
    out = split_dir / "gen" / f"{key}.jsonl"
    done = _done(out)
    todo = [s for s in scns if s["id"] not in done]
    print(f"[{args.split}/{key}] ({m.kind}) {len(todo)} pending of {len(scns)} "
          f"({len(done)} done) ident={ident}", file=sys.stderr)
    if not todo:
        print(f"DONE {args.split}/{key}: nothing to do", file=sys.stderr)
        return 0

    system = load_system_prompt()
    t0 = time.time()

    if m.kind == "mlx":
        backend = MLXLocal(ident, max_tokens=args.max_tokens or bcfg.GEN_MAX_TOKENS_LOCAL)
        for i, scn in enumerate(todo, 1):
            try:
                text, usage = backend.complete(system, build_user_prompt(scn, "grounded"))
                _persist(out, scn, key, text, usage)
            except Exception as e:  # noqa: BLE001
                print(f"  ! {scn['id']}: {e}", file=sys.stderr)
            if i % 50 == 0 or i == len(todo):
                dt = time.time() - t0
                print(f"  {key} {i}/{len(todo)} ({dt/i:.2f}s/it, "
                      f"eta {dt/i*(len(todo)-i)/60:.0f}m)", file=sys.stderr)
    else:
        client = make_tfy_client(args.timeout)
        lim = RateLimiter(args.min_interval)
        chat = TFYChat(client, model_id=ident, max_tokens=bcfg.GEN_MAX_TOKENS_TFY,
                       max_retries=args.max_retries, limiter=lim,
                       reasoning_effort=m.reasoning_effort)
        lock = threading.Lock()
        n = [0]

        def task(scn: Dict[str, Any]):
            text, usage = chat.complete(system, build_user_prompt(scn, "grounded"))
            return scn, text, usage

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(task, s): s for s in todo}
            for fut in as_completed(futs):
                scn = futs[fut]
                try:
                    scn, text, usage = fut.result()
                    with lock:
                        _persist(out, scn, key, text, usage)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {scn['id']}: {e}", file=sys.stderr)
                n[0] += 1
                if n[0] % 100 == 0 or n[0] == len(todo):
                    dt = time.time() - t0
                    print(f"  {key} {n[0]}/{len(todo)} ({dt/n[0]:.2f}s/it, "
                          f"eta {dt/n[0]*(len(todo)-n[0])/60:.0f}m)", file=sys.stderr)

    print(f"DONE {args.split}/{key}: {len(todo)} gens in {time.time()-t0:.0f}s -> {out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
