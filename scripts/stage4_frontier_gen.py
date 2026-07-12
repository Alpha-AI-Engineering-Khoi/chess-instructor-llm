#!/usr/bin/env python3
"""Stage-4 FRONTIER generation on the 120 held-out TEST x 3 tiers (MATCHED grounding).

Generates the three frontier coaches (GPT-5.5 / Claude Opus 4.8 / Gemini 3.1 Pro)
on the EXACT SAME grounded prompt v6-dpo2 / v4 / base were run on
(``data/benchmark_gap803/stage4_eval_inputs.jsonl`` -> ``grounded_system`` /
``grounded_user``), via the TrueFoundry gateway (``src.eval.benchmark.backends.TFYChat``).

This is the apples-to-apples fix: the OLD cached frontier gens in
``data/benchmark_honest/gen`` were produced under a different/older grounding, so
reusing them would be an UNFAIR cross-scope comparison. Here every model — ours and
frontier — sees the SAME fresh grounding (Stockfish sound-pool + Maia block +
verify-gate facts) on the SAME 360 scenarios.

Resumable: each completed generation is appended immediately; a re-run skips ids
already present. Writes ``data/benchmark_gap803/stage4_frontier/{gpt,claude,gemini}.jsonl``
in the ``{"i","id","output"}`` schema ``scripts/stage4_eval_v6dpo2.score_condition``
reads, plus per-row token usage for the cost readout.

Run (TFY key from .env; NO Modal credits used)::

    python scripts/stage4_frontier_gen.py --limit 2                 # smoke (2 scenarios x 3 models)
    python scripts/stage4_frontier_gen.py                           # full 360 x 3 = 1080 calls
    python scripts/stage4_frontier_gen.py --models gpt              # one model
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

from config import settings  # noqa: E402
from src.eval.benchmark import config as bcfg  # noqa: E402
from src.eval.benchmark.backends import TFYChat, make_tfy_client  # noqa: E402
from src.teacher.generate import RateLimiter  # noqa: E402

log = logging.getLogger("stage4.frontier")

INPUTS = _ROOT / "data" / "benchmark_gap803" / "stage4_eval_inputs.jsonl"
OUT_DIR = _ROOT / "data" / "benchmark_gap803" / "stage4_frontier"
FRONTIER: Tuple[str, ...] = ("gpt", "claude", "gemini")

#: Outer resilience loop on top of TFYChat's own transient retries. Billing /
#: "unpaid invoice" gateway hiccups are transient (auto-paid within seconds), so we
#: retry generously before leaving an id unwritten (a re-run then resumes it).
OUTER_ATTEMPTS: int = 12
OUTER_BACKOFF_S: float = 8.0


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _done_ids(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    out: Dict[str, dict] = {}
    for r in _load_jsonl(path):
        rid = r.get("id")
        # only treat a row as done if it carries a non-empty output
        if rid and (r.get("output") or "").strip():
            out[rid] = r
    return out


def _gen_one(client_chat: TFYChat, row: dict) -> Tuple[str, Dict[str, int]]:
    """Generate one grounded coaching response for one scenario (with outer retries)."""
    system = row["grounded_system"]
    user = row["grounded_user"]
    last: Optional[BaseException] = None
    for attempt in range(OUTER_ATTEMPTS):
        try:
            text, usage = client_chat.complete(system, user)
            if (text or "").strip():
                return text, usage
            raise ValueError("empty after inner retries")
        except BaseException as exc:  # noqa: BLE001 — treat everything as transient here
            last = exc
            msg = str(exc).lower()
            transient = any(k in msg for k in (
                "invoice", "billing", "payment", "402", "429", "rate",
                "timeout", "connection", "temporarily", "overloaded", "unavailable",
                "500", "502", "503", "504", "empty",
            ))
            delay = OUTER_BACKOFF_S * (1.5 ** min(attempt, 6))
            log.warning("[%s] attempt %d/%d failed (%s: %s); retry in %.0fs%s",
                        client_chat.model_id, attempt + 1, OUTER_ATTEMPTS,
                        type(exc).__name__, str(exc)[:140], delay,
                        "" if transient else " [non-transient, still retrying]")
            time.sleep(delay)
    raise last if last is not None else RuntimeError("generation failed")


def run_model(key: str, rows: List[dict], concurrency: int, timeout: float,
              max_retries: int) -> Dict[str, int]:
    m = bcfg.MODELS[key]
    if m.kind != "tfy":
        raise SystemExit(f"BLOCKED: model {key} is not a TFY frontier model")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{key}.jsonl"
    done = _done_ids(out_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"[{key}] {m.display} ({m.ident}) — {len(done)} done, {len(todo)} to generate",
          file=sys.stderr, flush=True)
    if not todo:
        return {"key": key, "done": len(done), "generated": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "failed": 0}

    client = make_tfy_client(timeout=timeout)
    limiter = RateLimiter(min_interval=0.0)
    chat = TFYChat(client, model_id=m.ident, max_tokens=bcfg.GEN_MAX_TOKENS_TFY,
                   max_retries=max_retries, limiter=limiter,
                   reasoning_effort=m.reasoning_effort)

    lock = threading.Lock()
    fh = out_path.open("a", encoding="utf-8")
    counters = {"generated": 0, "prompt_tokens": 0, "completion_tokens": 0, "failed": 0}
    t0 = time.time()

    def work(row: dict) -> Optional[dict]:
        try:
            text, usage = _gen_one(chat, row)
        except BaseException as exc:  # noqa: BLE001
            log.error("[%s] GAVE UP on %s: %s", key, row["id"], str(exc)[:160])
            with lock:
                counters["failed"] += 1
            return None
        return {"i": row["i"], "id": row["id"], "output": text,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0))}

    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futs = {ex.submit(work, r): r for r in todo}
            for fut in as_completed(futs):
                res = fut.result()
                if res is None:
                    continue
                with lock:
                    fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fh.flush()
                    counters["generated"] += 1
                    counters["prompt_tokens"] += res["prompt_tokens"]
                    counters["completion_tokens"] += res["completion_tokens"]
                    n = counters["generated"]
                    if n % 20 == 0 or n == len(todo):
                        dt = time.time() - t0
                        print(f"[{key}] {n}/{len(todo)} ({dt:.0f}s, "
                              f"{dt/max(1,n):.2f}s/gen)", file=sys.stderr, flush=True)
    finally:
        fh.close()

    # tidy: rewrite the file sorted by scenario index for stable diffs
    final = _done_ids(out_path)
    ordered = sorted(final.values(), key=lambda r: r.get("i", 0))
    with out_path.open("w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counters.update({"key": key, "done": len(final)})
    print(f"[{key}] DONE — {len(final)}/{len(rows)} total, {counters['failed']} failed this run, "
          f"tokens in={counters['prompt_tokens']:,} out={counters['completion_tokens']:,}",
          file=sys.stderr, flush=True)
    return counters


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=",".join(FRONTIER),
                   help="comma-separated frontier keys (default: gpt,claude,gemini)")
    p.add_argument("--limit", type=int, default=0, help="only first N scenarios (smoke)")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--max-retries", type=int, default=6)
    args = p.parse_args(argv)

    load_dotenv(settings.ROOT / ".env")
    rows = _load_jsonl(INPUTS)
    for idx, r in enumerate(rows):
        r.setdefault("i", idx)
    if args.limit:
        rows = rows[:args.limit]
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    print(f"=== Stage-4 frontier grounded gen: {len(rows)} scenarios x {len(keys)} models "
          f"({len(rows)*len(keys)} calls) ===", file=sys.stderr, flush=True)

    summary = []
    for k in keys:
        summary.append(run_model(k, rows, args.concurrency, args.timeout, args.max_retries))

    print("\n=== frontier generation summary ===", file=sys.stderr)
    any_incomplete = False
    for s in summary:
        print(f"  {s['key']:8} done={s.get('done',0)}/{len(rows)}  gen_this_run={s.get('generated',0)}  "
              f"failed={s.get('failed',0)}  tok_out={s.get('completion_tokens',0):,}", file=sys.stderr)
        if s.get("done", 0) < len(rows):
            any_incomplete = True
    if any_incomplete:
        print("  NOTE: some models incomplete — re-run to resume the remaining ids.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
