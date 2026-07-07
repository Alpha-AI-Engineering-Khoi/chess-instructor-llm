#!/usr/bin/env python3
"""Measure TIER-APPROPRIATE MOVE SELECTION (the moat) for the 9 open models.

This is the axis most likely missing at the *model* level. It runs the SAME
grounded harness as :mod:`scripts.frontier_gap` — byte-identical grounding
(``render_pool_facts`` + ``render_user_prompt`` + the tier's Maia block), the
same move extraction, the same findability lens — on the **exact same 50
held-out positions x 3 tiers**, but for the nine reachable open-source models on
the TrueFoundry gateway. Because it reuses ``frontier_gap.analyze_position``
verbatim, the open numbers are directly comparable to the frontier GAP_REPORT
(tier-differentiation 22.7%).

Positions are pinned to the ids already in ``data/analysis/frontier_gap.jsonl``
(a balanced phase x severity subsample of ``data/analysis/divergence.jsonl``),
so the open models see byte-identical input to what the frontier models saw.

Output: ``data/benchmark_open/tier_gap.jsonl`` — one rich row per position; each
row's ``models`` map holds all requested open models plus ``v1-tuned`` (reused
free from ``divergence.jsonl``). Aggregate it with ``scripts.open_gap_report``
(or the balanced-leaderboard builder) using ``frontier_gap_report.compute_model``.

Run (from repo root; secrets come from ROOT/.env)::

    ~/.venvs/mlx/bin/python -m scripts.open_gap --resume
    ~/.venvs/mlx/bin/python -m scripts.open_gap --limit 1 --only-model gemma3_27b   # smoke

It makes NEW TrueFoundry calls (9 models x 50 positions x 3 tiers) but touches
neither the running servers, the local MLX models, nor the viz/HF Space. It only
READS ``divergence.jsonl`` / ``frontier_gap.jsonl`` and WRITES ``tier_gap.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import settings  # noqa: E402
from src.eval.benchmark import config as bcfg  # noqa: E402
from src.engine import maia_engine  # noqa: E402

# Reuse the frontier harness verbatim (import-safe: no MLX / server side effects).
from scripts.frontier_gap import (  # noqa: E402
    FrontierClient,
    _load_done_ids,
    _load_jsonl,
    analyze_position,
)

#: The nine reachable open-source models: key -> TFY gateway id, taken straight
#: from the benchmark registry so the ids match what the objective/council run
#: already validated as reachable.
OPEN_MODELS: Dict[str, str] = {k: bcfg.MODELS[k].ident for k in bcfg.OPEN_MODEL_ORDER}


def _abs(x: str) -> Path:
    pp = Path(x)
    return pp if pp.is_absolute() else _ROOT / pp


def _pin_positions(div_path: Path, fg_path: Path) -> List[Dict[str, Any]]:
    """Return the divergence rows for the EXACT ids the frontier gap ran on."""
    fg_ids = [r["id"] for r in _load_jsonl(fg_path)]
    div_by_id = {r["id"]: r for r in _load_jsonl(div_path)}
    missing = [i for i in fg_ids if i not in div_by_id]
    if missing:
        print(f"WARN: {len(missing)} frontier-gap ids not found in divergence: {missing[:5]}",
              file=sys.stderr)
    return [div_by_id[i] for i in fg_ids if i in div_by_id]


def main(argv: Optional[Sequence[str]] = None) -> int:
    from dotenv import load_dotenv

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--divergence", default="data/analysis/divergence.jsonl",
                   help="v1 held-out records (identical-grounding position source).")
    p.add_argument("--frontier-gap", default="data/analysis/frontier_gap.jsonl",
                   help="Pin to these exact position ids (what the frontier ran on).")
    p.add_argument("--out", default="data/benchmark_open/tier_gap.jsonl")
    p.add_argument("--limit", type=int, default=0, help="Smoke cap on positions (0 = all).")
    p.add_argument("--models", default="", help="Comma keys (default: all 9 open).")
    p.add_argument("--only-model", default="", help="Smoke: restrict to one model key.")
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--min-interval", type=float, default=0.08)
    p.add_argument("--resume", action="store_true", help="Skip ids already in --out.")
    args = p.parse_args(argv)

    load_dotenv(settings.ROOT / ".env")

    div_path, fg_path, out_path = _abs(args.divergence), _abs(args.frontier_gap), _abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not div_path.exists() or not fg_path.exists():
        print(f"missing input: {div_path if not div_path.exists() else fg_path}", file=sys.stderr)
        return 1

    models = dict(OPEN_MODELS)
    if args.models:
        want = {m.strip() for m in args.models.split(",") if m.strip()}
        models = {k: v for k, v in OPEN_MODELS.items() if k in want}
    if args.only_model:
        models = {k: v for k, v in OPEN_MODELS.items() if k == args.only_model}
    if not models:
        print(f"no models selected; choices: {list(OPEN_MODELS)}", file=sys.stderr)
        return 1

    sample = _pin_positions(div_path, fg_path)
    if args.limit:
        sample = sample[: args.limit]
    print(f"[1/2] {len(sample)} pinned positions x {len(('beginner','intermediate','advanced'))} "
          f"tiers x {len(models)} open models: {list(models)}", file=sys.stderr)

    done = _load_done_ids(out_path) if args.resume else set()
    if args.resume and done:
        print(f"      resuming: {len(done)} already done", file=sys.stderr)

    client = FrontierClient(
        timeout=args.timeout, max_retries=args.max_retries, min_interval=args.min_interval
    )

    print(f"[2/2] Coaching ...", file=sys.stderr)
    mode_open = "a" if (args.resume and done) else "w"
    t0 = time.time()
    n_done = 0
    with out_path.open(mode_open, encoding="utf-8") as fh:
        for i, drow in enumerate(sample, 1):
            if drow.get("id") in done:
                continue
            ts = time.time()
            try:
                row = analyze_position(drow, client, models)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! [{i}/{len(sample)}] {drow.get('id')} FAILED: {exc}", file=sys.stderr)
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += 1
            sfb = (row["stockfish_best"] or {}).get("san")
            picks = " ".join(
                f"{mn[:6]}:{row['models'][mn]['beginner']['rec_san']}" for mn in list(models)[:4]
            )
            print(f"  + [{i}/{len(sample)}] {drow.get('id')} [{row['phase'][:3]}/{row['severity'][:4]}] "
                  f"SFbest:{sfb} B[{picks}...]  ({time.time() - ts:.1f}s)", file=sys.stderr)

    dt = time.time() - t0
    print(f"DONE — wrote {n_done} rows to {out_path} in {dt:.0f}s "
          f"({dt / max(1, n_done):.1f}s/pos); tfy calls={client.calls} "
          f"tokens(in/out)={client.prompt_tokens:,}/{client.completion_tokens:,}", file=sys.stderr)
    try:
        maia_engine.close_all()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
