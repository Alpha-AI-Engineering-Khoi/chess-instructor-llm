#!/usr/bin/env python3
"""Cross-family LLM-judge truthfulness residual for one model's 803-eval gens.

Weak spot #3: the deterministic verifier only catches the claims a one-ply board
computation can decide; the *residual* (multi-move tactical claims, assessments,
unsupported concrete claims) is measured by an independent panel of cross-family
judges — reusing ``src.eval.truthfulness.judge`` verbatim. This runs that panel
over a stratified subset of a model's coaching outputs and reports the fraction
flagged (NOT truthful) under both aggregation modes.

Identical subset + panel for every model, so the v3→v4 delta is apples-to-apples.
Resumable (append per (scenario_id) to a per-model file) and costed.

Run (from repo root, .env sourced)::
    ~/.venvs/mlx/bin/python -m scripts.truthfulness_eval --model ours_v3 --items 90
    ~/.venvs/mlx/bin/python -m scripts.truthfulness_eval --model ours_v4 --items 90
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

from config import settings  # noqa: E402
from src.engine.position_facts import render_pool_facts  # noqa: E402
from src.eval.evaluate import extract_recommended_move  # noqa: E402
from src.eval.truthfulness.judge import TruthfulnessJudge, default_panel  # noqa: E402
from scripts.gap803_common import stratified_scenarios  # noqa: E402

BENCH = Path(os.environ.get("BENCH_DIR", str(_ROOT / "data" / "benchmark_v4")))


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _facts_text(scn: Dict[str, Any]) -> str:
    pool = [{"uci": m["uci"], "san": m["san"], "cp": int(m["cp"]), "pv": list(m.get("pv") or [])}
            for m in scn["sound_pool"]]
    facts = render_pool_facts(scn["fen"], pool)
    sound = ", ".join(m["san"] for m in scn["sound_pool"])
    return f"{facts}\n- Engine-sound moves (any is acceptable): {sound}."


def main(argv=None) -> int:
    import logging
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="gen file stem in BENCH/gen (e.g. ours_v4).")
    p.add_argument("--items", type=int, default=90)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--judges", default="gpt,claude,gemini",
                   help="comma keys; ours_* outputs are Qwen so all 3 frontier judges are cross-family.")
    p.add_argument("--aggregation", default="any", choices=("any", "majority"))
    args = p.parse_args(argv)
    load_dotenv(settings.ROOT / ".env")

    scns = {s["id"]: s for s in _read_jsonl(BENCH / "scenarios.jsonl")}
    keep = set((BENCH / "frontier_ids.txt").read_text(encoding="utf-8").split())
    eligible = [s for s in scns.values() if s["pos_id"] in keep]
    subset = stratified_scenarios(eligible, args.items, seed=args.seed)
    subset_ids = {s["id"] for s in subset}

    gens = {g["scenario_id"]: g for g in _read_jsonl(BENCH / "gen" / f"{args.model}.jsonl")
            if g["scenario_id"] in subset_ids}
    missing = subset_ids - set(gens)
    if missing:
        print(f"WARN: {len(missing)} subset items missing a {args.model} gen", file=sys.stderr)

    out_path = BENCH / f"truthfulness_{args.model}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["scenario_id"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [s for s in subset if s["id"] in gens and s["id"] not in done]
    print(f"truthfulness[{args.model}]: {len(todo)} pending of {len(subset)} "
          f"(judges={args.judges}, agg={args.aggregation})", file=sys.stderr)

    panel = default_panel(judge_keys=args.judges.split(","))
    judge = TruthfulnessJudge(panel, aggregation=args.aggregation, concurrency=len(panel))

    t0 = time.time()
    tin = tout = 0
    with out_path.open("a", encoding="utf-8") as out:
        def work(scn):
            g = gens[scn["id"]]
            output = g.get("output", "")
            rec_san, _ = extract_recommended_move(output, scn["fen"], scn["student_move"]["uci"])
            res = judge.assess(output, scn["fen"], rec_san, _facts_text(scn))
            return scn, res

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(work, s): s for s in todo}
            n = 0
            for fut in as_completed(futs):
                scn, res = fut.result()
                d = res.to_dict()
                out.write(json.dumps({
                    "scenario_id": scn["id"], "tier": scn["tier"], "phase": scn["phase"],
                    "truthful": d["truthful"], "n_flagged": len(d["flagged"]),
                    "flagged": d["flagged"][:4], "n_judges": d["n_judges"],
                    "agreement": d["agreement"], "usage": d["usage"],
                }, ensure_ascii=False) + "\n")
                out.flush()
                tin += d["usage"].get("prompt_tokens", 0)
                tout += d["usage"].get("completion_tokens", 0)
                n += 1
                if n % 20 == 0:
                    print(f"  {n}/{len(todo)} ({time.time()-t0:.0f}s)", file=sys.stderr)

    _summarize(out_path, args.model)
    print(f"tokens: in={tin} out={tout}", file=sys.stderr)
    return 0


def _summarize(out_path: Path, model: str) -> None:
    rows = _read_jsonl(out_path)
    if not rows:
        print("no rows to summarize"); return
    n = len(rows)
    flagged = sum(1 for r in rows if not r["truthful"])
    by_tier_n: Counter = Counter()
    by_tier_flag: Counter = Counter()
    for r in rows:
        by_tier_n[r["tier"]] += 1
        if not r["truthful"]:
            by_tier_flag[r["tier"]] += 1
    print(f"\n=== truthfulness residual [{model}] (n={n}) ===")
    print(f"residual (NOT truthful, strict any): {flagged}/{n} = {100*flagged/n:.1f}%")
    for t in ("beginner", "intermediate", "advanced"):
        tn = by_tier_n.get(t, 0)
        if tn:
            print(f"  {t:<12} {by_tier_flag.get(t,0)}/{tn} = {100*by_tier_flag.get(t,0)/tn:.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
