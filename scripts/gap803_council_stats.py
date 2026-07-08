#!/usr/bin/env python3
"""Instructiveness-council statistics: raw + self-preference-corrected ranks + CIs.

The blinded cross-family council (``data/benchmark_gap803/council.jsonl``) has each
of the 3 frontier judges (GPT-5.5 / Claude / Gemini) rank the unified 15-model
field per item. Because every judge also grades a model from its OWN lab, a raw
mean-rank leaderboard is contaminated by self-preference. This module produces:

* **raw** mean rank per model (all 3 judges) + 95% CI;
* **self-preference-corrected** mean rank per model — for the three frontier
  competitors, the vote from the SAME-family judge is dropped (leave-own-out), so
  no model is graded by its own lab; non-frontier models keep all 3 judges + CI;
* **per-judge self-preference deltas** — for each judge, how much better it ranks
  its own lab's model than the other two judges do (``own_mean − peers_mean``;
  negative ⇒ the judge favours its own family);
* explicit **council n** (items × judges) and top-1 rates.

Confidence intervals are a **cluster bootstrap by item** (the 3 judges rank the
same blinded responses for an item, so the item is the independent unit): items
are resampled with replacement, each model's mean rank recomputed, and the
2.5/97.5 percentiles reported. Pure functions over council rows; a ``main`` reads
the checkpoint and writes ``council_stats.json`` for the report + UI to consume.

Run::  BENCH_DIR=data/benchmark_gap803 ~/.venvs/mlx/bin/python -m scripts.gap803_council_stats
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: The unified 15-model council field (report row order). Kept in sync with
#: ``scripts.gap803_report.MODEL_ORDER`` / ``scripts.gap803_council.FIELD``.
FIELD: Tuple[str, ...] = (
    "ours_v3", "ours", "base",
    "gemma3_27b", "q3_32b", "q3_next80b", "llama33_70b",
    "dsv32", "glm5", "mistral3", "kimi25", "dsr1",
    "gpt", "claude", "gemini",
)
#: The three frontier judges == the three frontier competitors (same family key).
FRONTIER: Tuple[str, ...] = ("gpt", "claude", "gemini")


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# Per-model rank observations, grouped by item (for the cluster bootstrap)
# --------------------------------------------------------------------------- #

#: model -> item_id -> list of (judge, rank)
RankIndex = Dict[str, Dict[str, List[Tuple[str, int]]]]


def index_ranks(rows: Sequence[Dict[str, Any]]) -> Tuple[RankIndex, List[str]]:
    """Return (model -> item -> [(judge, rank)...], ordered list of item ids)."""
    idx: RankIndex = defaultdict(lambda: defaultdict(list))
    items: List[str] = []
    seen: set = set()
    for r in rows:
        mapping = r.get("label_to_model") or {}
        ranking = r.get("ranking") or []
        judge = r.get("judge", "?")
        item = f"{r.get('scenario_id')}|{r.get('condition')}"
        if item not in seen:
            seen.add(item)
            items.append(item)
        for pos, lab in enumerate(ranking, 1):
            mk = mapping.get(lab)
            if mk is None:
                continue
            idx[mk][item].append((judge, pos))
    return idx, items


def _flatten(item_map: Dict[str, List[Tuple[str, int]]],
             *, exclude_judge: Optional[str] = None) -> List[int]:
    out: List[int] = []
    for obs in item_map.values():
        for judge, rank in obs:
            if exclude_judge is not None and judge == exclude_judge:
                continue
            out.append(rank)
    return out


def _bootstrap_ci(
    item_map: Dict[str, List[Tuple[str, int]]],
    items: Sequence[str],
    *,
    exclude_judge: Optional[str] = None,
    n_boot: int = 2000,
    seed: int = 20260707,
) -> Optional[Tuple[float, float]]:
    """95% cluster-bootstrap CI for a model's mean rank (resample items)."""
    # Pre-flatten each item's contributing ranks once.
    per_item: Dict[str, List[int]] = {}
    for it in items:
        obs = item_map.get(it, [])
        vals = [rank for judge, rank in obs
                if not (exclude_judge is not None and judge == exclude_judge)]
        if vals:
            per_item[it] = vals
    pool = list(per_item.keys())
    if not pool:
        return None
    rng = random.Random(seed)
    means: List[float] = []
    n = len(pool)
    for _ in range(n_boot):
        acc: List[int] = []
        for _ in range(n):
            acc.extend(per_item[pool[rng.randrange(n)]])
        if acc:
            means.append(sum(acc) / len(acc))
    if not means:
        return None
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return (round(lo, 3), round(hi, 3))


# --------------------------------------------------------------------------- #
# Top-level stats
# --------------------------------------------------------------------------- #


def compute_council_stats(
    rows: Sequence[Dict[str, Any]],
    field: Sequence[str] = FIELD,
    frontier: Sequence[str] = FRONTIER,
    *,
    n_boot: int = 2000,
    seed: int = 20260707,
) -> Dict[str, Any]:
    """Raw + self-preference-corrected mean-rank leaderboard with 95% CIs."""
    if not rows:
        return {"n_items": 0, "n_judges": 0, "n_rankings": 0, "field": 0, "models": {}}

    idx, items = index_ranks(rows)
    n_items = len(items)
    judges = sorted({r.get("judge") for r in rows if r.get("judge")})
    n_judges = len(judges)
    field_size = max((len(r.get("label_to_model") or {}) for r in rows), default=len(field))
    frontier_set = set(frontier)

    models: Dict[str, Any] = {}
    for mk in field:
        item_map = idx.get(mk, {})
        raw_vals = _flatten(item_map)
        if not raw_vals:
            continue
        is_frontier = mk in frontier_set
        # corrected: drop the same-family judge's votes for a frontier competitor.
        excl = mk if is_frontier else None
        corr_vals = _flatten(item_map, exclude_judge=excl)
        top1 = sum(1 for v in raw_vals if v == 1)
        models[mk] = {
            "obs": len(raw_vals),
            "mean_rank": round(mean(raw_vals), 3),
            "ci95": _bootstrap_ci(item_map, items, n_boot=n_boot, seed=seed),
            "top1_pct": round(100.0 * top1 / len(raw_vals), 1),
            "corrected_obs": len(corr_vals),
            "corrected_mean_rank": round(mean(corr_vals), 3) if corr_vals else None,
            "corrected_ci95": _bootstrap_ci(item_map, items, exclude_judge=excl,
                                            n_boot=n_boot, seed=seed),
            "corrected_dropped_judge": (mk if is_frontier else None),
            "norm_rank": round((mean(raw_vals) - 1) / (field_size - 1), 4) if field_size > 1 else 0.0,
            "field": field_size,
            "frontier": is_frontier,
        }

    # Per-judge self-preference: own-family rank vs peers' rank of that same model.
    self_pref: Dict[str, Any] = {}
    deltas: List[float] = []
    for jk in frontier:
        item_map = idx.get(jk, {})
        own = [rank for obs in item_map.values() for judge, rank in obs if judge == jk]
        peers = [rank for obs in item_map.values() for judge, rank in obs if judge != jk]
        own_m = mean(own) if own else None
        peers_m = mean(peers) if peers else None
        delta = (own_m - peers_m) if (own_m is not None and peers_m is not None) else None
        if delta is not None:
            deltas.append(delta)
        self_pref[jk] = {
            "own_mean_rank": round(own_m, 3) if own_m is not None else None,
            "peers_mean_rank": round(peers_m, 3) if peers_m is not None else None,
            # own − peers; NEGATIVE ⇒ judge ranks its own family BETTER (lower) than peers.
            "delta_own_minus_peers": round(delta, 3) if delta is not None else None,
            "n_own": len(own),
            "n_peers": len(peers),
        }
    self_pref["_mean_signed_delta"] = round(mean(deltas), 3) if deltas else None
    self_pref["_mean_abs_delta"] = round(mean(abs(d) for d in deltas), 3) if deltas else None

    return {
        "n_items": n_items,
        "n_judges": n_judges,
        "n_rankings": len(rows),
        "field": field_size,
        "judges": judges,
        "models": models,
        "self_preference": self_pref,
    }


def raw_ranking(stats: Dict[str, Any]) -> List[str]:
    """Model keys sorted by RAW mean rank (best first)."""
    m = stats.get("models", {})
    return sorted(m.keys(), key=lambda k: m[k]["mean_rank"])


def corrected_ranking(stats: Dict[str, Any]) -> List[str]:
    """Model keys sorted by SELF-PREF-CORRECTED mean rank (best first)."""
    m = stats.get("models", {})
    def key(k: str) -> float:
        v = m[k].get("corrected_mean_rank")
        return v if v is not None else m[k]["mean_rank"]
    return sorted(m.keys(), key=key)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench-dir", default=os.environ.get("BENCH_DIR",
                   str(_ROOT / "data" / "benchmark_gap803")))
    p.add_argument("--council", default=None, help="Override council.jsonl path.")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default=None, help="Output JSON (default: <bench>/council_stats.json)")
    args = p.parse_args(argv)

    bench = Path(args.bench_dir)
    council_path = Path(args.council) if args.council else bench / "council.jsonl"
    rows = _read_jsonl(council_path)
    stats = compute_council_stats(rows, n_boot=args.n_boot)
    out_path = Path(args.out) if args.out else bench / "council_stats.json"
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Console summary.
    print(f"council: n_items={stats['n_items']} judges={stats['n_judges']} "
          f"rankings={stats['n_rankings']} field={stats['field']}")
    print(f"\n{'model':14} {'raw rank [95% CI]':>26} {'corrected [95% CI]':>26} {'top1':>6}")
    for mk in corrected_ranking(stats):
        d = stats["models"][mk]
        ci = d.get("ci95") or ("?", "?")
        cci = d.get("corrected_ci95") or ("?", "?")
        cmr = d.get("corrected_mean_rank")
        print(f"{mk:14} {d['mean_rank']:6.2f} [{ci[0]}, {ci[1]}]".ljust(42)
              + (f"{cmr:6.2f} [{cci[0]}, {cci[1]}]" if cmr is not None else "n/a").rjust(26)
              + f" {d['top1_pct']:5.1f}%")
    print("\nper-judge self-preference (own − peers; negative ⇒ favours own family):")
    sp = stats["self_preference"]
    for jk in FRONTIER:
        v = sp.get(jk, {})
        print(f"  {jk:8} own={v.get('own_mean_rank')} peers={v.get('peers_mean_rank')} "
              f"delta={v.get('delta_own_minus_peers')} (n_own={v.get('n_own')}, n_peers={v.get('n_peers')})")
    print(f"  mean signed delta={sp.get('_mean_signed_delta')} | mean |delta|={sp.get('_mean_abs_delta')}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
