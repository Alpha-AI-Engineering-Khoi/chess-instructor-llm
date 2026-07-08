#!/usr/bin/env python3
"""LLM-judge the WIDE-flagged v4 training labels to confirm real fabrications.

The WIDE deterministic checker (``verify_text_ext``) over-fires on coaching that
describes the position AFTER the recommended move, so it can't be a hard reject.
But it IS high-recall: the rows it flags are the best CANDIDATE POOL for real
fabrications. This script sends those labels to the context-aware cross-family
LLM truthfulness judge (Claude + Gemini — both cross-family to the GPT-5.5
teacher that wrote the labels) and confirms which are genuinely
false/unsupported, writing the confirmed candidate ids so
``build_v4_dataset build --exclude-ids`` can drop them for a v4b retrain.

`recon` judges a small sample to ESTIMATE the real-fabrication rate cheaply
before committing to the full pass.

Run (repo root, .env sourced)::
    ~/.venvs/mlx/bin/python -m scripts.judge_labels_v4 recon --n 100
    ~/.venvs/mlx/bin/python -m scripts.judge_labels_v4 run --agg both
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

from config import settings  # noqa: E402
from src.engine.position_facts import render_pool_facts  # noqa: E402
from src.eval.truthfulness.judge import (  # noqa: E402
    TruthfulnessJudge, build_system_prompt, build_user_prompt, default_panel,
)

CANDS = settings.GENERATED / "candidates_v3.jsonl"
WIDE = settings.GENERATED / "v4_wide_flagged.jsonl"
OUT = settings.GENERATED / "v4_label_judge.jsonl"
CONFIRMED = settings.GENERATED / "v4_judge_fab_ids.txt"


def _load_wide_ids() -> set:
    return {json.loads(l)["id"] for l in WIDE.read_text(encoding="utf-8").splitlines() if l.strip()}


def _index_candidates(ids: set) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with CANDS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("id") in ids:
                out[d["id"]] = d
    return out


def _facts(ti: Dict[str, Any]) -> str:
    pool = [{"uci": m["uci"], "san": m["san"], "cp": int(m["cp"]), "pv": list(m.get("pv") or [])}
            for m in ti["sound_pool"]]
    facts = render_pool_facts(ti["fen"], pool)
    sound = ", ".join(m["san"] for m in ti["sound_pool"])
    return f"{facts}\n- Engine-sound moves (any is acceptable): {sound}."


def _run(ids: List[str], recs: Dict[str, Dict[str, Any]], judges: str, concurrency: int):
    from config import schema
    panel = default_panel(judge_keys=judges.split(","))
    judge = TruthfulnessJudge(panel, aggregation="any", concurrency=len(panel))

    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text().splitlines() if l.strip()}
    todo = [i for i in ids if i not in done]
    print(f"judge-labels: {len(todo)} pending of {len(ids)} (judges={judges})", file=sys.stderr)

    t0 = time.time(); tin = tout = 0
    with OUT.open("a", encoding="utf-8") as out:
        def work(cid):
            c = recs[cid]; ti = c["teacher_input"]; to = c["teacher_output"]
            target = schema.render_assistant_target_v2(to)
            res = judge.assess(target, ti["fen"], to.get("recommended_move_san"), _facts(ti))
            return cid, to.get("recommended_move_san"), res

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {pool.submit(work, i): i for i in todo}
            n = 0
            for fut in as_completed(futs):
                cid, rec, res = fut.result()
                d = res.to_dict()
                # per-judge flags: "both" = every judge flagged (unanimous, high precision)
                verdicts = d.get("judge_verdicts", {})
                per = {jn: (not v.get("truthful", True)) for jn, v in verdicts.items()}
                out.write(json.dumps({
                    "id": cid, "rec": rec, "any_flagged": not d["truthful"],
                    "both_flagged": bool(per) and all(per.values()),
                    "per_judge": per, "flagged": d["flagged"][:3], "usage": d["usage"],
                }, ensure_ascii=False) + "\n")
                out.flush()
                tin += d["usage"].get("prompt_tokens", 0); tout += d["usage"].get("completion_tokens", 0)
                n += 1
                if n % 25 == 0:
                    print(f"  {n}/{len(todo)} ({time.time()-t0:.0f}s)", file=sys.stderr)
    print(f"tokens in={tin} out={tout}", file=sys.stderr)


def _summarize(agg: str) -> None:
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    if not rows:
        print("no rows"); return
    n = len(rows)
    any_f = sum(1 for r in rows if r["any_flagged"])
    both_f = sum(1 for r in rows if r["both_flagged"])
    print(f"\n=== label-judge summary (n={n}) ===")
    print(f"ANY judge flagged (residual proxy):  {any_f}/{n} = {100*any_f/n:.1f}%")
    print(f"BOTH judges flagged (high-precision): {both_f}/{n} = {100*both_f/n:.1f}%")
    key = "both_flagged" if agg == "both" else "any_flagged"
    confirmed = [r["id"] for r in rows if r[key]]
    CONFIRMED.write_text("\n".join(confirmed) + "\n", encoding="utf-8")
    print(f"wrote {len(confirmed)} confirmed-fabrication ids ({agg}) -> {CONFIRMED}")


def main(argv=None) -> int:
    import logging
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("recon", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--n", type=int, default=(100 if name == "recon" else 0),
                        help="sample size (0 = all, run only).")
        sp.add_argument("--judges", default="claude,gemini")
        sp.add_argument("--concurrency", type=int, default=8)
        sp.add_argument("--agg", default="both", choices=("any", "both"),
                        help="which confirmation to write as excludable ids (run).")
        sp.add_argument("--seed", type=int, default=3407)
    args = p.parse_args(argv)
    load_dotenv(settings.ROOT / ".env")

    wide_ids = sorted(_load_wide_ids())
    if args.cmd == "recon":
        rng = random.Random(args.seed); rng.shuffle(wide_ids)
        wide_ids = wide_ids[: args.n]
    elif args.n:
        rng = random.Random(args.seed); rng.shuffle(wide_ids)
        wide_ids = wide_ids[: args.n]

    recs = _index_candidates(set(wide_ids))
    _run([i for i in wide_ids if i in recs], recs, args.judges, args.concurrency)
    _summarize(args.agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
