#!/usr/bin/env python3
"""Aggregate the 803 gap eval into the DEFINITIVE balanced leaderboard.

Pipeline (each step reused / deterministic):

  merge     : concat per-model gen files -> generations.jsonl (dedup)
  objective : reuse src.eval.benchmark.objective -> objective.jsonl
              (fabrication / no_engine_speak / ply_cap / move_sound)
  report    : compute every metric + the weighted balanced ranking and write
              RESULTS_FULL_EVAL_803.md (repo root) + data/benchmark_gap803/*.json

Tier-appropriate move selection (the moat) is scored per position by RE-extracting
each coach's recommended move with the instrumented, pool-restricted extractor
(``extract_recommended_mode`` — genuine cue/prose pick vs a pool[0] fallback), and
comparing it to the canonical move from ``src.teacher.tier_select.select_tier_move``
(beginner=most-findable, intermediate=blend, advanced=sharpest). Fabrication /
no-engine-speak come from the reused objective scorer; move-safety (blunder-only)
from ``gap803_safety.py``; instructiveness from the reused council.

Run::

    ~/.venvs/mlx/bin/python -m scripts.gap803_report merge
    ~/.venvs/mlx/bin/python -m scripts.gap803_report objective
    ~/.venvs/mlx/bin/python -m scripts.gap803_report report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("BENCH_DIR", str(_ROOT / "data" / "benchmark_gap803"))

import chess  # noqa: E402

from src.eval.benchmark import config as bcfg  # noqa: E402
from scripts.divergence_analysis import extract_recommended_mode  # noqa: E402
from scripts.gap803_common import TIERS  # noqa: E402

BENCH = Path(os.environ["BENCH_DIR"])
SCN_PATH = BENCH / "scenarios.jsonl"
GEN_DIR = BENCH / "gen"
GEN_PATH = BENCH / "generations.jsonl"
OBJ_PATH = BENCH / "objective.jsonl"
COUNCIL_PATH = BENCH / "council.jsonl"
SAFETY_PATH = BENCH / "move_safety.json"
REPORT_MD = _ROOT / "RESULTS_FULL_EVAL_803.md"
METRICS_JSON = BENCH / "leaderboard.json"

#: Report row order + display + practical facts (params, local-runnability, cost).
#: local: 1.0 = 4-bit fits a 64GB Mac comfortably; 0.6 = tight-but-runnable;
#: 0.0 = far too large to run/fine-tune locally.
MODEL_ORDER: Tuple[str, ...] = (
    "ours", "base",
    "gemma3_27b", "q3_32b", "q3_next80b", "llama33_70b",
    "dsv32", "glm5", "mistral3", "kimi25", "dsr1",
    "gpt", "claude", "gemini",
)
DISPLAY: Dict[str, str] = {
    "ours": "OURS-v2 (Qwen3-1.7B tuned)", "base": "BASE (Qwen3-1.7B untuned)",
    "gpt": "GPT-5.5", "claude": "Claude Opus 4.8", "gemini": "Gemini 3.1 Pro",
    "q3_32b": "Qwen3-32B", "q3_next80b": "Qwen3-Next-80B-A3B", "gemma3_27b": "Gemma-3-27B-it",
    "llama33_70b": "Llama-3.3-70B", "dsv32": "DeepSeek-V3.2", "glm5": "GLM-5",
    "mistral3": "Mistral-Large-3 (675B)", "kimi25": "Kimi-K2.5", "dsr1": "DeepSeek-R1",
}
FAMILY: Dict[str, str] = {
    "ours": "ours", "base": "base", "gpt": "frontier", "claude": "frontier",
    "gemini": "frontier", **{k: "open" for k in (
        "q3_32b", "q3_next80b", "gemma3_27b", "llama33_70b", "dsv32", "glm5",
        "mistral3", "kimi25", "dsr1")},
}
#: (approx params, local_runnable score, note)
PRACTICAL: Dict[str, Tuple[str, float, str]] = {
    "ours": ("1.7B", 1.0, "already local"),
    "base": ("1.7B", 1.0, "already local"),
    "gemma3_27b": ("27B dense", 1.0, "4-bit ~15GB — comfortable"),
    "q3_32b": ("32B dense", 1.0, "4-bit ~18GB — comfortable"),
    "q3_next80b": ("80B/3B MoE", 0.6, "4-bit ~43GB — tight but fits; 3B active = fast"),
    "llama33_70b": ("70B dense", 0.6, "4-bit ~40GB — tight on 64GB"),
    "dsv32": ("~671B MoE", 0.0, "far exceeds 64GB"),
    "glm5": ("~355B+ MoE", 0.0, "far exceeds 64GB"),
    "mistral3": ("675B", 0.0, "far exceeds 64GB"),
    "kimi25": ("~1T MoE", 0.0, "far exceeds 64GB"),
    "dsr1": ("671B MoE", 0.0, "far exceeds 64GB"),
    "gpt": ("closed", 0.0, "API-only"),
    "claude": ("closed", 0.0, "API-only"),
    "gemini": ("closed", 0.0, "API-only"),
}


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# merge + objective
# --------------------------------------------------------------------------- #


def cmd_merge(_a: argparse.Namespace) -> int:
    seen: set = set()
    n = 0
    with GEN_PATH.open("w", encoding="utf-8") as out:
        for f in sorted(GEN_DIR.glob("*.jsonl")):
            for row in _read_jsonl(f):
                key = (row["scenario_id"], row["model"])
                if key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"merge: {n} generations -> {GEN_PATH}")
    by_model: Dict[str, int] = defaultdict(int)
    for k in seen:
        by_model[k[1]] += 1
    for m in MODEL_ORDER:
        if by_model.get(m):
            print(f"   {m:14} {by_model[m]}")
    return 0


def cmd_objective(_a: argparse.Namespace) -> int:
    from src.eval.benchmark.objective import run_objective
    scns = _read_jsonl(SCN_PATH)
    res = run_objective(scns)
    print(f"objective: {res}")
    return 0


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def _scenarios_by_id() -> Dict[str, Dict[str, Any]]:
    return {s["id"]: s for s in _read_jsonl(SCN_PATH)}


def _maia_rank(scn: Dict[str, Any], uci: Optional[str]) -> Optional[int]:
    if not uci:
        return None
    order = scn.get("pool_order") or []
    return order.index(uci) if uci in order else None


def compute_tier_metrics(scns_by_id: Dict[str, Dict[str, Any]]
                         ) -> Dict[str, Dict[str, Any]]:
    """Per model: tier-fit / differentiation / direction from re-extracted picks.

    Re-extracts each pick with the pool-restricted instrumented extractor so a
    genuine named sound move is separated from a pool[0] fallback, then compares
    to the ``select_tier_move`` canonical move (carried on the scenario as
    ``canonical_uci``).
    """
    # group generations by (model, pos_id) -> {tier: pick}
    gens = _read_jsonl(GEN_PATH)
    per: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for g in gens:
        scn = scns_by_id.get(g["scenario_id"])
        if scn is None:
            continue
        board = chess.Board(scn["fen"])
        pool = scn["sound_pool"]
        student_uci = scn["student_move"].get("uci") or ""
        rec_san, rec_uci, mode = extract_recommended_mode(g.get("output", ""), board, pool, student_uci)
        per[g["model"]][g["pos_id"]][scn["tier"]] = {
            "rec_uci": rec_uci, "mode": mode,
            "genuine": mode in ("cue", "prose"),
            "canonical_uci": scn.get("canonical_uci"),
            "engine_best_uci": scn.get("engine_best_uci"),
            "tier_move_uci": scn.get("tier_move_uci"),
            "discriminating": scn.get("discriminating"),
            "maia_rank": _maia_rank(scn, rec_uci),
        }

    out: Dict[str, Dict[str, Any]] = {}
    for model, positions in per.items():
        fit = {t: [0, 0] for t in TIERS}       # [hits, n]
        eng_mirror = {t: [0, 0] for t in TIERS}
        genuine = {t: [0, 0] for t in TIERS}
        findable_disc = {t: [0, 0] for t in TIERS}  # pick==most-findable on disc-for-tier
        n_pos = 0
        n_diff = 0
        n_full = 0            # positions with all 3 tiers present
        direction_ok = 0
        direction_n = 0
        mirror_all = 0
        for pos_id, tinfo in positions.items():
            if not all(t in tinfo for t in TIERS):
                # still use available tiers for per-tier fit, but skip cross-tier
                pass
            else:
                n_full += 1
                ucis = [tinfo[t]["rec_uci"] for t in TIERS]
                if len({u for u in ucis if u}) > 1:
                    n_diff += 1
                if all(tinfo[t]["rec_uci"] == tinfo[t]["engine_best_uci"] for t in TIERS):
                    mirror_all += 1
                # direction: beginner pick at least as findable as advanced
                br = tinfo["beginner"]["maia_rank"]
                ar = tinfo["advanced"]["maia_rank"]
                if br is not None and ar is not None:
                    direction_n += 1
                    if br < ar:
                        direction_ok += 1
                    elif br == ar:
                        direction_ok += 0.5
            n_pos += 1
            for t in TIERS:
                if t not in tinfo:
                    continue
                ti = tinfo[t]
                fit[t][1] += 1
                if ti["rec_uci"] and ti["rec_uci"] == ti["canonical_uci"]:
                    fit[t][0] += 1
                eng_mirror[t][1] += 1
                if ti["rec_uci"] == ti["engine_best_uci"]:
                    eng_mirror[t][0] += 1
                genuine[t][1] += 1
                if ti["genuine"]:
                    genuine[t][0] += 1
                if ti.get("discriminating"):
                    findable_disc[t][1] += 1
                    if ti["rec_uci"] and ti["rec_uci"] == ti["tier_move_uci"]:
                        findable_disc[t][0] += 1

        def _rate(hn):
            return hn[0] / hn[1] if hn[1] else None

        fit_by_tier = {t: _rate(fit[t]) for t in TIERS}
        fit_vals = [v for v in fit_by_tier.values() if v is not None]
        out[model] = {
            "n_pos": n_pos,
            "n_full": n_full,
            "tier_fit_by_tier": fit_by_tier,
            "tier_fit_mean": (sum(fit_vals) / len(fit_vals)) if fit_vals else None,
            "diff_rate": (n_diff / n_full) if n_full else None,
            "direction": (direction_ok / direction_n) if direction_n else None,
            "mirror_all": (mirror_all / n_full) if n_full else None,
            "eng_mirror_by_tier": {t: _rate(eng_mirror[t]) for t in TIERS},
            "genuine_rate": _rate(genuine["beginner"]),
            "findable_disc_by_tier": {t: _rate(findable_disc[t]) for t in TIERS},
        }
    return out


def compute_objective_metrics() -> Dict[str, Dict[str, Any]]:
    """Per model from objective.jsonl: fabrication, no-ES, ply, move_sound, parse."""
    rows = _read_jsonl(OBJ_PATH)
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)

    def rate(g, f):
        vals = [1 for r in g if r.get(f)]
        return len(vals) / len(g) if g else None

    out: Dict[str, Dict[str, Any]] = {}
    for m, g in by.items():
        out[m] = {
            "n": len(g),
            "fabrication": (sum(1 for r in g if r.get("fabricated")) / len(g)) if g else None,
            "no_engine_speak": rate(g, "no_engine_speak"),
            "ply_cap_ok": rate(g, "ply_cap_ok"),
            "move_sound": rate(g, "move_sound"),
            "move_parseable": rate(g, "move_parseable"),
            "avg_violations": (sum(r.get("n_violations", 0) for r in g) / len(g)) if g else None,
        }
    return out


def compute_council() -> Dict[str, Dict[str, Any]]:
    """Per model instructiveness from council.jsonl: mean rank, norm, top-1, rubric."""
    rows = _read_jsonl(COUNCIL_PATH)
    if not rows:
        return {}
    ranks: Dict[str, List[int]] = defaultdict(list)
    top1: Dict[str, int] = defaultdict(int)
    rubric: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    field_sizes = []
    for r in rows:
        mapping = r.get("label_to_model") or {}
        ranking = r.get("ranking") or []
        field_sizes.append(len(mapping))
        for pos, lab in enumerate(ranking, 1):
            mk = mapping.get(lab)
            if mk is None:
                continue
            ranks[mk].append(pos)
            if pos == 1:
                top1[mk] += 1
        for lab, sc in (r.get("scores") or {}).items():
            mk = mapping.get(lab)
            if mk is None:
                continue
            for dim, v in sc.items():
                try:
                    rubric[mk][dim].append(int(v))
                except (TypeError, ValueError):
                    pass
    field = max(field_sizes) if field_sizes else len(MODEL_ORDER)
    out: Dict[str, Dict[str, Any]] = {}
    for mk, rs in ranks.items():
        mean_rank = sum(rs) / len(rs)
        out[mk] = {
            "obs": len(rs),
            "mean_rank": mean_rank,
            "norm_rank": (mean_rank - 1) / (field - 1) if field > 1 else 0.0,
            "top1_pct": 100.0 * top1[mk] / len(rs) if rs else 0.0,
            "field": field,
            "rubric": {d: (sum(v) / len(v) if v else None) for d, v in rubric[mk].items()},
        }
    return out


def compute_safety() -> Dict[str, float]:
    """Per model blunder-free rate from move_safety.json (fallback: move_sound)."""
    if SAFETY_PATH.exists():
        data = json.loads(SAFETY_PATH.read_text(encoding="utf-8"))
        return {m: v for m, v in data.get("move_safe", {}).items()}
    return {}


# --------------------------------------------------------------------------- #
# balanced scoring
# --------------------------------------------------------------------------- #

GATE_SAFE = 0.98
GATE_NOES = 0.98

# balanced weights (tier + instructiveness highest; fabrication downweighted).
W_BALANCED = {"tier": 0.40, "instr": 0.40, "fab": 0.10, "practical": 0.10}
# best-base weights (tier-appropriateness is what we FINE-TUNE in -> low; the
# hard-to-add qualities -> high: instructiveness/capacity, faithfulness, local).
W_BASE = {"tier": 0.10, "instr": 0.35, "fab": 0.20, "practical": 0.35}


def _cost_blended(mk: str) -> float:
    m = bcfg.MODELS[mk]
    return float(m.price_in) + float(m.price_out)


def build_scores(tier: Dict[str, Any], obj: Dict[str, Any], council: Dict[str, Any],
                 safety: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    models = [m for m in MODEL_ORDER if m in obj]
    # cost normalization across present models (local = 0 = best)
    blended = {m: _cost_blended(m) for m in models}
    cmax = max(blended.values()) if blended else 1.0
    cost_score = {m: (1.0 - blended[m] / cmax) if cmax > 0 else 1.0 for m in models}
    # instructiveness normalization (norm_rank in [0,1]; lower better)
    scored: Dict[str, Dict[str, Any]] = {}
    for m in models:
        t = tier.get(m, {})
        tier_fit = t.get("tier_fit_mean")
        diff = t.get("diff_rate")
        direction = t.get("direction")
        tvals = [x for x in (tier_fit, diff, direction) if x is not None]
        tier_score = sum(tvals) / len(tvals) if tvals else None
        c = council.get(m, {})
        instr_score = (1.0 - c["norm_rank"]) if c else None
        fab = obj[m].get("fabrication")
        fab_score = (1.0 - fab) if fab is not None else None
        loc = PRACTICAL.get(m, ("", 0.0, ""))[1]
        practical = 0.6 * loc + 0.4 * cost_score.get(m, 0.0)
        safe = safety.get(m)
        noes = obj[m].get("no_engine_speak")
        gate_ok = (safe is None or safe >= GATE_SAFE) and (noes is None or noes >= GATE_NOES)

        def _wsum(weights: Dict[str, float]) -> Optional[float]:
            parts = {"tier": tier_score, "instr": instr_score, "fab": fab_score,
                     "practical": practical}
            num = den = 0.0
            for k, w in weights.items():
                v = parts[k]
                if v is None:
                    continue
                num += w * v
                den += w
            return (num / den) if den else None

        scored[m] = {
            "tier_score": tier_score, "instr_score": instr_score,
            "fab_score": fab_score, "practical": practical,
            "cost_score": cost_score.get(m), "local": loc,
            "gate_ok": gate_ok, "safe": safe, "no_engine_speak": noes,
            "balanced": _wsum(W_BALANCED), "base_fit": _wsum(W_BASE),
            "instr_imputed": bool(c and c.get("obs")) is False,
        }
    return scored


# --------------------------------------------------------------------------- #
# report writing (delegated to gap803_writer to keep this file focused)
# --------------------------------------------------------------------------- #


def compute_spend() -> Dict[str, Any]:
    """Estimate USD spend from generation + council token usage x gateway prices."""
    def _usd(mk: str, tin: int, tout: int) -> float:
        m = bcfg.MODELS[mk]
        return tin / 1e6 * float(m.price_in) + tout / 1e6 * float(m.price_out)

    groups = {g: {"calls": 0, "in": 0, "out": 0, "usd": 0.0}
              for g in ("open", "frontier_gen", "council", "local")}
    for f in GEN_DIR.glob("*.jsonl"):
        mk = f.stem
        if mk not in bcfg.MODELS:
            continue
        fam = FAMILY.get(mk)
        grp = "local" if fam in ("ours", "base") else ("frontier_gen" if fam == "frontier" else "open")
        for r in _read_jsonl(f):
            g = groups[grp]
            g["calls"] += 1
            g["in"] += int(r.get("prompt_tokens", 0))
            g["out"] += int(r.get("completion_tokens", 0))
            if grp != "local":
                g["usd"] += _usd(mk, int(r.get("prompt_tokens", 0)), int(r.get("completion_tokens", 0)))
    for r in _read_jsonl(COUNCIL_PATH):
        jk = r.get("judge")
        if jk not in bcfg.MODELS:
            continue
        g = groups["council"]
        g["calls"] += 1
        g["in"] += int(r.get("prompt_tokens", 0))
        g["out"] += int(r.get("completion_tokens", 0))
        g["usd"] += _usd(jk, int(r.get("prompt_tokens", 0)), int(r.get("completion_tokens", 0)))
    total = sum(g["usd"] for g in groups.values())
    for g in groups.values():
        g["usd"] = round(g["usd"], 2)
    return {**groups, "total_usd": round(total, 2)}


def cmd_report(_a: argparse.Namespace) -> int:
    scns_by_id = _scenarios_by_id()
    tier = compute_tier_metrics(scns_by_id)
    obj = compute_objective_metrics()
    council = compute_council()
    safety = compute_safety()
    scored = build_scores(tier, obj, council, safety)
    spend = compute_spend()

    council_rows = _read_jsonl(COUNCIL_PATH)
    n_council_items = len({r["scenario_id"] for r in council_rows})
    n_judges = len({r.get("judge") for r in council_rows}) or 3
    bundle = {"tier": tier, "objective": obj, "council": council,
              "safety": safety, "scored": scored, "spend": spend,
              "n_council_items": n_council_items, "n_judges": n_judges,
              "n_positions": len({s["pos_id"] for s in scns_by_id.values()})}
    METRICS_JSON.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {METRICS_JSON}")

    from scripts.gap803_writer import write_markdown
    write_markdown(REPORT_MD, MODEL_ORDER, DISPLAY, FAMILY, PRACTICAL,
                   tier, obj, council, safety, scored,
                   n_positions=bundle["n_positions"],
                   n_council_items=n_council_items, n_judges=n_judges,
                   w_balanced=W_BALANCED, w_base=W_BASE, spend=spend)
    print(f"wrote {REPORT_MD}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("merge").set_defaults(func=cmd_merge)
    sub.add_parser("objective").set_defaults(func=cmd_objective)
    sub.add_parser("report").set_defaults(func=cmd_report)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
