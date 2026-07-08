#!/usr/bin/env python3
"""Phase B: two-layer truthfulness residual over the GATED showcase.

Layer 1 (deterministic, free): verify_text_ext fabrication rate over ALL gated
cells, per model — the post-gate residual the mechanical checker still sees
(expected ~0 by construction of the gate).

Layer 2 (LLM-judge, paid): a cross-family panel (GPT-5.5 + Claude + Gemini via
TFY, `any`-aggregation, non-circular) fact-checks a STRATIFIED REPRESENTATIVE
sample of gated cells (~300-360, balanced across models x tiers, over-sampling
OURS + GPT/Claude/Gemini + Qwen3-32B + Gemma-3-27B). Reports per-model truthful
rate with a 95% Wilson CI, total judge calls, and real cost. This catches the
multi-move / evaluative claims the deterministic layer deliberately abstains on.

Uses src/eval/truthfulness/judge.py's PUBLIC API (build_system_prompt /
build_user_prompt / parse_judge_reply / aggregate / default_panel) — never edits
it. Resumable: per-(sample,judge) rows cached in judge_raw.jsonl.

Run:  ~/.venvs/mlx/bin/python data/showcase/gate/phaseB.py --per-priority 30 --per-other 18
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gate_lib as G  # noqa: E402
from common import price_for, usd_for  # noqa: E402
from src.engine.faithfulness_ext import verify_text_ext  # noqa: E402
from src.engine.position_facts import render_pool_facts  # noqa: E402
from src.eval.truthfulness import judge as J  # noqa: E402

WEB_SHOWCASE = G.ROOT / "web" / "public" / "showcase.json"
JUDGE_RAW = HERE / "judge_raw.jsonl"
OUT = G.ROOT / "data" / "showcase" / "truthfulness.json"
JUDGE_KEYS = ("gpt", "claude", "gemini")
SEED = 20260707

# Models to over-sample for the judge (highest-signal comparisons).
PRIORITY_NAMES = {
    "OURS-v2 (1.7B tuned)", "GPT-5.5", "Claude Opus 4.8", "Gemini 3.1 Pro",
    "Qwen3-32B", "Gemma-3-27B-it",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def collect_gated(positions, name_to_key, scn_index):
    """All gated cells with text -> flat list of sample dicts."""
    cells = []
    for pi, pos in enumerate(positions):
        fen = pos["fen"]
        pos_id = pos["id"]
        for m in pos.get("models", []):
            name = m["name"]
            key = name_to_key.get(name, name)
            for tier in G.TIERS:
                cell = (m.get("byTier") or {}).get(tier)
                if not cell:
                    continue
                coaching = cell.get("coaching")
                if not coaching or not str(coaching).strip():
                    continue
                scn = scn_index.get((pos_id, tier))
                cells.append({
                    "sid": f"{pi}:{key}:{tier}", "pi": pi, "pos_id": pos_id,
                    "name": name, "key": key, "tier": tier, "fen": fen,
                    "coaching": str(coaching), "move": cell.get("move"),
                    "move_uci": cell.get("move_uci"),
                    "fabricated": bool(cell.get("fabricated")),
                    "scn": scn,
                })
    return cells


def deterministic_residual(cells) -> Dict[str, Any]:
    """Layer 1: post-gate verify_text_ext fabrication rate per model (free)."""
    per = defaultdict(lambda: {"text": 0, "fab": 0})
    for c in cells:
        p = per[c["name"]]
        p["text"] += 1
        p["fab"] += int(c["fabricated"])
    out = {}
    tot_t = tot_f = 0
    for name, p in sorted(per.items()):
        out[name] = {"n": p["text"], "fabricated": p["fab"],
                     "fab_rate": round(p["fab"] / p["text"], 5) if p["text"] else 0.0}
        tot_t += p["text"]; tot_f += p["fab"]
    out["_overall"] = {"n": tot_t, "fabricated": tot_f,
                       "fab_rate": round(tot_f / tot_t, 5) if tot_t else 0.0}
    return out


def build_sample(cells, per_priority: int, per_other: int) -> List[Dict[str, Any]]:
    """Stratified sample: per (model, tier), balanced; over-sample priority models."""
    rng = random.Random(SEED)
    by_mt: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for c in cells:
        if c["scn"] is None:
            continue  # need the scenario to build engine facts
        by_mt[(c["name"], c["tier"])].append(c)
    sample: List[Dict[str, Any]] = []
    for (name, tier), group in by_mt.items():
        quota = per_priority if name in PRIORITY_NAMES else per_other
        rng.shuffle(group)
        sample.extend(group[:quota])
    rng.shuffle(sample)
    return sample


def load_done() -> Dict[Tuple[str, str], Dict[str, Any]]:
    done = {}
    if JUDGE_RAW.exists():
        for r in G.read_jsonl(JUDGE_RAW):
            done[(r["sid"], r["judge"])] = r
    return done


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-priority", type=int, default=30)
    p.add_argument("--per-other", type=int, default=18)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=6)
    args = p.parse_args(argv)

    positions = json.loads(WEB_SHOWCASE.read_text(encoding="utf-8"))
    name_to_key = G.name_to_key_map()
    scn_index = G.build_scn_index()
    cells = collect_gated(positions, name_to_key, scn_index)
    print(f"[phaseB] gated cells with text: {len(cells)}", file=sys.stderr)

    det = deterministic_residual(cells)
    print(f"[phaseB] deterministic residual overall fab_rate = "
          f"{det['_overall']['fab_rate']} ({det['_overall']['fabricated']}/{det['_overall']['n']})",
          file=sys.stderr)

    sample = build_sample(cells, args.per_priority, args.per_other)
    print(f"[phaseB] judge sample size = {len(sample)} cells "
          f"(x{len(JUDGE_KEYS)} judges = {len(sample)*len(JUDGE_KEYS)} calls)", file=sys.stderr)

    # ---- build the cross-family panel (uses judge.py public factory) ----
    panel = J.default_panel(JUDGE_KEYS, timeout=args.timeout, max_retries=args.max_retries)
    by_name = {jc.name: jc for jc in panel}
    system = J.build_system_prompt()

    done = load_done()
    fh = JUDGE_RAW.open("a", encoding="utf-8")
    write_lock = threading.Lock()
    usage_by_judge = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    ucount = defaultdict(int)

    # seed usage from already-done rows so cost is cumulative & correct on resume
    for (sid, jk), r in done.items():
        usage_by_judge[jk]["prompt_tokens"] += int(r.get("prompt_tokens", 0))
        usage_by_judge[jk]["completion_tokens"] += int(r.get("completion_tokens", 0))

    tasks = []
    for c in sample:
        facts = render_pool_facts(c["fen"], c["scn"]["sound_pool"])
        user = J.build_user_prompt(c["coaching"], c["fen"], c["move"], facts)
        for jk in JUDGE_KEYS:
            if (c["sid"], jk) in done:
                continue
            tasks.append((c, jk, user))
    print(f"[phaseB] judge calls to run: {len(tasks)} ({len(done)} cached)", file=sys.stderr)

    def run(task):
        c, jk, user = task
        text, usage = by_name[jk].client.complete(system, user)
        parsed = J.parse_judge_reply(text)
        return c, jk, parsed, usage

    n = 0
    if tasks:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(run, t): t for t in tasks}
            for fut in as_completed(futs):
                t = futs[fut]
                c, jk, _u = t[0], t[1], None
                try:
                    c, jk, parsed, usage = fut.result()
                    row = {"sid": c["sid"], "judge": jk, "name": c["name"],
                           "tier": c["tier"], "truthful": parsed["truthful"],
                           "flagged": parsed["flagged"],
                           "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                           "completion_tokens": int(usage.get("completion_tokens", 0))}
                    with write_lock:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
                        done[(c["sid"], jk)] = row
                        usage_by_judge[jk]["prompt_tokens"] += row["prompt_tokens"]
                        usage_by_judge[jk]["completion_tokens"] += row["completion_tokens"]
                        ucount[jk] += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! judge {jk} {c['sid']}: {type(exc).__name__}: {exc}", file=sys.stderr)
                n += 1
                if n % 100 == 0 or n == len(tasks):
                    print(f"  [phaseB] {n}/{len(tasks)} judge calls", file=sys.stderr)
    fh.close()

    # ---- aggregate per sampled cell (mode='any') ----
    verdicts_by_sid: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    meta_by_sid: Dict[str, Dict[str, Any]] = {}
    for c in sample:
        meta_by_sid[c["sid"]] = {"name": c["name"], "tier": c["tier"]}
    for (sid, jk), r in done.items():
        if sid in meta_by_sid:
            verdicts_by_sid[sid].append((jk, {"truthful": r["truthful"], "flagged": r["flagged"]}))

    per_model = defaultdict(lambda: {"n": 0, "truthful": 0, "n_judges_sum": 0,
                                     "agreement_sum": 0.0, "flagged_examples": []})
    for sid, pj in verdicts_by_sid.items():
        agg = J.aggregate(pj, mode="any")
        name = meta_by_sid[sid]["name"]
        pm = per_model[name]
        pm["n"] += 1
        pm["truthful"] += int(agg["truthful"])
        pm["n_judges_sum"] += agg["n_judges"]
        pm["agreement_sum"] += agg["agreement"]
        if not agg["truthful"] and len(pm["flagged_examples"]) < 3:
            fl = agg["flagged"][0] if agg["flagged"] else {}
            pm["flagged_examples"].append({"sid": sid, "claim": fl.get("claim", ""),
                                           "reason": fl.get("reason", ""), "judge": fl.get("judge", "")})

    judge_residual = {}
    tot_n = tot_t = 0
    for name, pm in sorted(per_model.items()):
        n_ = pm["n"]; t_ = pm["truthful"]
        lo, hi = wilson_ci(t_, n_)
        judge_residual[name] = {
            "n_sampled": n_, "truthful": t_,
            "truthful_rate": round(t_ / n_, 4) if n_ else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)],
            "untruthful_rate": round(1 - t_ / n_, 4) if n_ else 0.0,
            "avg_judges": round(pm["n_judges_sum"] / n_, 2) if n_ else 0.0,
            "avg_agreement": round(pm["agreement_sum"] / n_, 3) if n_ else 0.0,
            "flagged_examples": pm["flagged_examples"],
        }
        tot_n += n_; tot_t += t_
    lo, hi = wilson_ci(tot_t, tot_n)
    judge_residual["_overall"] = {
        "n_sampled": tot_n, "truthful": tot_t,
        "truthful_rate": round(tot_t / tot_n, 4) if tot_n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
    }

    # ---- cost ----
    total_calls = sum(ucount.values()) if ucount else len(done)
    cost = {"per_judge": {}, "total_usd": 0.0, "total_calls": len(done)}
    for jk in JUDGE_KEYS:
        u = usage_by_judge[jk]
        usd = usd_for(jk, u["prompt_tokens"], u["completion_tokens"])
        cost["per_judge"][jk] = {"calls": sum(1 for (s, j) in done if j == jk),
                                 "prompt_tokens": u["prompt_tokens"],
                                 "completion_tokens": u["completion_tokens"],
                                 "usd": round(usd, 4)}
        cost["total_usd"] += usd
    cost["total_usd"] = round(cost["total_usd"], 4)

    doc = {
        "method": {
            "deterministic_checker": "src.engine.faithfulness_ext.verify_text_ext",
            "judge_panel": list(JUDGE_KEYS),
            "aggregation": "any (a single judge's objection => not truthful)",
            "sample": {"per_priority_per_tier": args.per_priority,
                       "per_other_per_tier": args.per_other,
                       "priority_models": sorted(PRIORITY_NAMES),
                       "seed": SEED},
        },
        "deterministic_residual": det,
        "judge_residual": judge_residual,
        "judge_calls": cost["total_calls"],
        "cost": cost,
    }
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phaseB] wrote {OUT}")
    print(json.dumps({"overall_judge_truthful_rate": judge_residual["_overall"],
                      "judge_calls": cost["total_calls"],
                      "judge_cost_usd": cost["total_usd"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
