#!/usr/bin/env python3
"""Stage-4 blinded cross-family instructiveness council (SECONDARY, on TFY).

Reuses the shipped blinded council verbatim (``src.eval.benchmark.council.run_council``
+ the same 3 frontier judges GPT-5.5 / Claude / Gemini and the same instructiveness
rubric) on a stratified sample of the 120 held-out TEST positions, ranking a compact
6-model field on the deployable GROUNDED condition:

    v6_dpo, v4, base   (our fresh Stage-4 grounded generations)
    gpt, claude, gemini (the committed frontier grounded generations)

so we can answer the one secondary question the deterministic eval cannot: does
preference-tuning (v6-dpo) REGRESS coaching instructiveness vs shipped v4? Judges
grade blinded (labels shuffled per item); we report each model's mean rank with a
bootstrap 95% CI and the v6-dpo-vs-v4 head-to-head win rate.

This is a REPRESENTATIVE sample by design (cost): default 96 items x 3 judges = 288
gateway calls, NOT the full 7,200-call council. Resumable + costed (token usage is
recorded per row by the council).

Run (TFY key from .env; no Modal credits used)::

    python scripts/stage4_council.py --items 96
    python scripts/stage4_council.py --aggregate-only     # just recompute the table
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BENCH_DIR = _ROOT / "data" / "benchmark_gap803" / "stage4_council"
os.environ["BENCH_DIR"] = str(BENCH_DIR)

from dotenv import load_dotenv  # noqa: E402
from config import settings  # noqa: E402
from src.eval.benchmark import config as bcfg  # noqa: E402
from scripts.gap803_common import stratified_scenarios  # noqa: E402

STAGE4_GEN = _ROOT / "data" / "benchmark_gap803" / "stage4"
HONEST_GEN = _ROOT / "data" / "benchmark_honest" / "gen"
SCEN_V6 = _ROOT / "data" / "benchmark_gap803" / "scenarios_v6.jsonl"
SCEN_V4ERA = _ROOT / "data" / "benchmark_gap803" / "scenarios.jsonl"

# ranked field (judges are the frontier 3). map: our fresh gen file -> model key
FIELD: Tuple[str, ...] = ("v6_dpo", "v4", "base", "gpt", "claude", "gemini")
FRESH_MAP = {"v6_dpo": "v6dpo_grounded", "v4": "v4_grounded", "base": "base_grounded"}
FRONTIER = ("gpt", "claude", "gemini")


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _build_scenarios() -> List[dict]:
    """Council scenarios for the 120 TEST positions: corrected v6 sound pool +
    severity/pv enrichment from the v4-era scenarios (stable FENs, same ids)."""
    v6 = {s["id"]: s for s in _load_jsonl(SCEN_V6) if s.get("is_val")}
    v4 = {s["id"]: s for s in _load_jsonl(SCEN_V4ERA)}
    out: List[dict] = []
    for sid, s in v6.items():
        prev = v4.get(sid, {})
        prev_pool = {p["uci"]: p for p in prev.get("sound_pool", [])}
        sm = s["student_move"]
        prev_sm = prev.get("student_move") or {}
        out.append({
            "id": sid, "pos_id": s["pos_id"], "tier": s["tier"], "phase": s.get("phase"),
            "fen": s["fen"],
            "severity": prev.get("severity") or prev_sm.get("severity") or "none",
            "student_move": {
                "san": sm.get("san"), "uci": sm.get("uci"),
                "cp_loss": int(prev_sm.get("cp_loss") or 0),
                "severity": prev.get("severity") or prev_sm.get("severity") or "none",
            },
            "sound_pool": [
                {"uci": p["uci"], "san": p["san"], "cp": int(p.get("cp") or 0),
                 "pv": list(prev_pool.get(p["uci"], {}).get("pv") or [])}
                for p in s.get("sound_pool", [])
            ],
            "maia": [],
        })
    return out


def _write_generations(scenarios_by_id: Dict[str, dict]) -> int:
    """Assemble generations.jsonl in the council schema for the ranked field."""
    rows: List[dict] = []
    # fresh Stage-4 grounded gens
    for mk, fname in FRESH_MAP.items():
        p = STAGE4_GEN / f"{fname}.jsonl"
        if not p.exists():
            raise SystemExit(f"BLOCKED: missing fresh gen {p} (run scripts/stage4_eval.py first)")
        for g in _load_jsonl(p):
            sid = g.get("id") or g.get("scenario_id")
            if sid in scenarios_by_id:
                rows.append({"scenario_id": sid, "model": mk, "condition": "grounded",
                             "output": g.get("output", "")})
    # committed frontier grounded gens
    for mk in FRONTIER:
        for g in _load_jsonl(HONEST_GEN / f"{mk}.jsonl"):
            sid = g.get("scenario_id")
            if sid in scenarios_by_id and g.get("condition") == "grounded":
                rows.append({"scenario_id": sid, "model": mk, "condition": "grounded",
                             "output": g.get("output", "")})
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    with (BENCH_DIR / "generations.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


# --------------------------------------------------------------------------- #
# Aggregation: mean rank (bootstrap 95% CI) + v6_dpo-vs-v4 head-to-head
# --------------------------------------------------------------------------- #
def _bootstrap_ci(values: List[float], n_boot: int = 5000, seed: int = 3407
                  ) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    m = len(values)
    for _ in range(n_boot):
        s = sum(values[rng.randrange(m)] for _ in range(m)) / m
        means.append(s)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (round(lo, 3), round(hi, 3))


def aggregate() -> Dict[str, Any]:
    council = _load_jsonl(BENCH_DIR / "council.jsonl")
    # per (item) collect each model's rank position averaged over judges
    # ranks: 1 = best. label_to_model maps blinded label -> model key.
    per_item_ranks: Dict[str, Dict[str, List[int]]] = {}
    n_judgements = 0
    for row in council:
        sid = row["scenario_id"]
        mapping = row["label_to_model"]
        ranking = row["ranking"]  # list of labels best->worst
        n_judgements += 1
        pos = {lab: i + 1 for i, lab in enumerate(ranking)}
        d = per_item_ranks.setdefault(sid, {})
        for lab, mk in mapping.items():
            d.setdefault(mk, []).append(pos.get(lab, len(ranking)))
    # per-item mean rank per model (averaged across judges)
    item_mean: Dict[str, Dict[str, float]] = {
        sid: {mk: statistics.mean(rs) for mk, rs in d.items()}
        for sid, d in per_item_ranks.items()
    }
    models = FIELD
    table: Dict[str, Any] = {}
    for mk in models:
        vals = [im[mk] for im in item_mean.values() if mk in im]
        if not vals:
            continue
        lo, hi = _bootstrap_ci(vals)
        table[mk] = {"mean_rank": round(statistics.mean(vals), 3),
                     "ci95": [lo, hi], "n_items": len(vals)}
    # v6_dpo vs v4 head-to-head (per item, lower mean rank wins)
    wins = losses = ties = 0
    diffs: List[float] = []
    for sid, im in item_mean.items():
        if "v6_dpo" in im and "v4" in im:
            d = im["v4"] - im["v6_dpo"]  # positive => dpo ranked better
            diffs.append(d)
            if im["v6_dpo"] < im["v4"]:
                wins += 1
            elif im["v6_dpo"] > im["v4"]:
                losses += 1
            else:
                ties += 1
    n = wins + losses + ties
    dlo, dhi = _bootstrap_ci(diffs)
    h2h = {
        "v6dpo_better": wins, "v4_better": losses, "ties": ties, "n": n,
        "v6dpo_winrate_excl_ties": round(wins / (wins + losses), 3) if (wins + losses) else None,
        "mean_rank_advantage_dpo_minus_v4": round(statistics.mean(diffs), 3) if diffs else 0.0,
        "mean_rank_advantage_ci95": [dlo, dhi],
    }
    # cost readout from recorded token usage
    pin = sum(int(r.get("prompt_tokens", 0)) for r in council)
    pout = sum(int(r.get("completion_tokens", 0)) for r in council)
    return {"n_judgements": n_judgements, "ranking_table": table, "head_to_head_dpo_vs_v4": h2h,
            "tokens": {"prompt": pin, "completion": pout}}


def _print_agg(agg: Dict[str, Any]) -> None:
    print("\n=== Stage-4 council: instructiveness mean rank (1=best) on GROUNDED, blinded ===")
    print(f"{'model':10} {'mean_rank':>10} {'95% CI':>16} {'n':>5}")
    for mk, s in sorted(agg["ranking_table"].items(), key=lambda kv: kv[1]["mean_rank"]):
        print(f"{mk:10} {s['mean_rank']:>10.3f} {str(s['ci95']):>16} {s['n_items']:>5}")
    h = agg["head_to_head_dpo_vs_v4"]
    print(f"\nv6-dpo vs v4 (per item, lower mean rank wins): "
          f"dpo_better={h['v6dpo_better']} v4_better={h['v4_better']} ties={h['ties']} n={h['n']}")
    print(f"  mean-rank advantage (v4 - v6dpo) = {h['mean_rank_advantage_dpo_minus_v4']:+.3f} "
          f"CI95={h['mean_rank_advantage_ci95']}  (positive => DPO ranked more instructive)")
    print(f"  tokens: prompt={agg['tokens']['prompt']:,} completion={agg['tokens']['completion']:,}")


def main(argv=None) -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items", type=int, default=96)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--judge-max-tokens", type=int, default=1500)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--aggregate-only", action="store_true")
    args = p.parse_args(argv)

    load_dotenv(settings.ROOT / ".env")

    bcfg.MODEL_ORDER = tuple(FIELD)
    bcfg.ANON_LABELS = bcfg.labels_for(len(FIELD))
    bcfg.JUDGE_MAX_TOKENS = args.judge_max_tokens

    if args.aggregate_only:
        agg = aggregate()
        (BENCH_DIR / "aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
        _print_agg(agg)
        return 0

    scenarios = _build_scenarios()
    by_id = {s["id"]: s for s in scenarios}
    n_gen = _write_generations(by_id)
    # write scenarios.jsonl (council reads scenarios from the subset we pass in)
    with (BENCH_DIR / "scenarios.jsonl").open("w", encoding="utf-8") as fh:
        for s in scenarios:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    subset = stratified_scenarios(scenarios, args.items, seed=args.seed)
    import collections
    dist = collections.Counter((s["tier"], s["phase"]) for s in subset)
    print(f"stage4 council: {len(subset)} items, field={FIELD}, judges={list(bcfg.JUDGE_KEYS)}, "
          f"gens={n_gen}", file=sys.stderr)
    print(f"  tier x phase: {dict(dist)}", file=sys.stderr)

    from src.eval.benchmark.council import run_council
    res = run_council(subset, ["grounded"], list(bcfg.JUDGE_KEYS),
                      concurrency=args.concurrency, timeout=args.timeout,
                      max_retries=args.max_retries)
    print(f"council run: {res}", file=sys.stderr)

    agg = aggregate()
    (BENCH_DIR / "aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    _print_agg(agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
