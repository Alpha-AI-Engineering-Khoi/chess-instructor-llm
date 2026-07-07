#!/usr/bin/env python3
"""Extend the v2 chess-coaching benchmark to BIGGER open-source models.

Reuses the shared harness (``src/eval/benchmark``) verbatim, on the SAME 100
held-out scenarios the v2 benchmark used, so the open models are directly
comparable to OURS-v2 / BASE / GPT-5.5 / Claude / Gemini.

Two phases (both resumable, checkpointed under ``data/benchmark_open/``):

  Phase 1 (cheap, high-signal): GROUNDED generation on each accessible open
  model (identical VERIFIED-FACTS input) + deterministic OBJECTIVE scoring
  (move-soundness, no-engine-speak, ply-cap, and the fabrication_rate the
  faithfulness verifier computes). Fabrication is THE metric.

  Phase 2 (cost-aware): one blinded, cross-family council that ranks a UNIFIED
  field (the 5 v2 competitors + the strongest open models) together per item on
  a reduced position subset, so every model gets a rank in the same field.

Sub-commands::

    python scripts/run_benchmark_open.py probe          # reachability
    python scripts/run_benchmark_open.py seed           # copy v2 scenarios + grounded 5-model gens/obj
    python scripts/run_benchmark_open.py generate       # Phase 1: grounded gen for open models
    python scripts/run_benchmark_open.py objective      # Phase 1: objective scoring
    python scripts/run_benchmark_open.py phase1         # print the fabrication leaderboard
    python scripts/run_benchmark_open.py council --field ours,base,gpt,claude,gemini,<open...> --n 50
    python scripts/run_benchmark_open.py report         # write RESULTS_OPEN_MODELS.md
    python scripts/run_benchmark_open.py status
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Point the harness at an isolated output dir BEFORE importing its config (paths
# are resolved from these env vars at import time). The v2 artifacts in
# data/benchmark/ are read-only inputs and never modified.
os.environ.setdefault("BENCH_DIR", str(ROOT / "data" / "benchmark_open"))

from config import settings  # noqa: E402
from src.eval.benchmark import config as bcfg  # noqa: E402

# OURS = the shipped v2 model (the thing the open models must beat to matter).
_OURS_PATH = os.environ.get("BENCH_OURS_MODEL", str(settings.MODELS / "mlx" / "chess-coach-v2"))
bcfg.MODELS["ours"] = replace(
    bcfg.MODELS["ours"],
    display=f"OURS ({Path(_OURS_PATH).name}, 1.7B tuned)",
    ident=_OURS_PATH,
)

from src.eval.benchmark import scenarios as scen_mod  # noqa: E402
from src.eval.benchmark import generate as gen_mod  # noqa: E402
from src.eval.benchmark import objective as obj_mod  # noqa: E402
from src.eval.benchmark import council as coun_mod  # noqa: E402
from src.eval.benchmark import aggregate as agg_mod  # noqa: E402
from src.eval.benchmark.io_utils import append_jsonl, done_keys, read_jsonl  # noqa: E402

log = logging.getLogger("benchmark_open")

SRC_DIR = ROOT / "data" / "benchmark"                 # v2 artifacts (inputs)
FRONTIER5: Tuple[str, ...] = ("ours", "base", "gpt", "claude", "gemini")
OPEN: Tuple[str, ...] = bcfg.OPEN_MODEL_ORDER
UNIFIED_ORDER: Tuple[str, ...] = FRONTIER5 + OPEN
REPORT_MD = ROOT / "RESULTS_OPEN_MODELS.md"


def _load_scn(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    scns = scen_mod.load_scenarios()
    if not scns:
        raise SystemExit("BLOCKED: no scenarios in data/benchmark_open/. Run `seed` first.")
    return scns[:limit] if limit else scns


# --------------------------------------------------------------------------- #
# seed: bring the v2 scenarios + grounded 5-model generations/objective across
# --------------------------------------------------------------------------- #


def _copy_jsonl_filtered(src: Path, dst: Path, key_fields: Sequence[str],
                         keep) -> int:
    """Append rows of ``src`` matching ``keep(row)`` to ``dst`` (idempotent)."""
    if not src.exists():
        raise SystemExit(f"BLOCKED: missing source {src}")
    existing = done_keys(dst, key_fields)
    n = 0
    for row in read_jsonl(src):
        if not keep(row):
            continue
        k = tuple(row[f] for f in key_fields)
        if k in existing:
            continue
        append_jsonl(dst, row)
        existing.add(k)
        n += 1
    return n


def cmd_seed(_a: argparse.Namespace) -> int:
    # Scenarios: byte-identical to the v2 held-out set.
    n_sc = _copy_jsonl_filtered(SRC_DIR / "scenarios.jsonl", bcfg.SCENARIOS_PATH,
                                ["id"], lambda r: True)
    # Grounded generations for the five v2 competitors (reused verbatim, free).
    # Objective is intentionally NOT copied: it is (re)computed by the `objective`
    # phase so every model — the five originals AND the open ones — is scored by
    # the *current* faithfulness verifier, which is what keeps them comparable.
    n_gen = _copy_jsonl_filtered(
        SRC_DIR / "generations.jsonl", bcfg.GENERATIONS_PATH,
        ["scenario_id", "model", "condition"],
        lambda r: r["condition"] == "grounded" and r["model"] in FRONTIER5,
    )
    print(f"seed: +{n_sc} scenarios, +{n_gen} grounded generations (5 v2 competitors)")
    print(f"  -> {bcfg.BENCH_DIR}")
    print("  (objective is recomputed by the `objective` phase for uniform scoring)")
    return 0


# --------------------------------------------------------------------------- #
# Phase 1: generation (grounded only) + objective
# --------------------------------------------------------------------------- #


def cmd_generate(a: argparse.Namespace) -> int:
    models = ([m.strip() for m in a.models.split(",") if m.strip()]
              if a.models else list(OPEN))
    gen_mod.run_generation(
        _load_scn(a.limit), models, ["grounded"],
        concurrency=a.concurrency, min_interval=a.min_interval,
        timeout=a.timeout, max_retries=a.max_retries,
    )
    return 0


def cmd_objective(a: argparse.Namespace) -> int:
    obj_mod.run_objective(_load_scn(a.limit))
    return 0


def _objective_grounded_by_model() -> Dict[str, Dict[str, float]]:
    rows = [r for r in read_jsonl(bcfg.OBJECTIVE_PATH) if r["condition"] == "grounded"]
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)

    def rate(g, field):
        return 100.0 * sum(1 for r in g if r[field]) / len(g)

    out: Dict[str, Dict[str, float]] = {}
    for mk, g in by.items():
        out[mk] = {
            "n": len(g),
            "move_parseable": rate(g, "move_parseable"),
            "move_sound": rate(g, "move_sound"),
            "no_engine_speak": rate(g, "no_engine_speak"),
            "ply_cap_ok": rate(g, "ply_cap_ok"),
            "fabrication": 100.0 * sum(1 for r in g if r["fabricated"]) / len(g),
            "avg_violations": sum(r["n_violations"] for r in g) / len(g),
        }
    return out


def cmd_phase1(_a: argparse.Namespace) -> int:
    stats = _objective_grounded_by_model()
    order = [m for m in UNIFIED_ORDER if m in stats]
    order.sort(key=lambda m: (stats[m]["fabrication"], -stats[m]["move_sound"]))
    print("\n=== PHASE 1 — GROUNDED OBJECTIVE LEADERBOARD (sorted by fabrication ↑) ===")
    hdr = f"{'model':<26} {'n':>3} {'fab%':>6} {'sound%':>7} {'noES%':>6} {'ply%':>6} {'parse%':>7} {'avgViol':>8}"
    print(hdr)
    print("-" * len(hdr))
    for mk in order:
        s = stats[mk]
        fam = bcfg.MODELS[mk].family
        tag = "  <- open" if fam == "open" else ""
        print(f"{bcfg.MODELS[mk].display:<26} {int(s['n']):>3} {s['fabrication']:>6.0f} "
              f"{s['move_sound']:>7.0f} {s['no_engine_speak']:>6.0f} {s['ply_cap_ok']:>6.0f} "
              f"{s['move_parseable']:>7.0f} {s['avg_violations']:>8.2f}{tag}")
    return 0


# --------------------------------------------------------------------------- #
# Phase 2: unified N-way council on a reduced subset
# --------------------------------------------------------------------------- #


def _set_field(field: Sequence[str]) -> None:
    """Point the harness at a specific unified field for a council run/report."""
    bcfg.MODEL_ORDER = tuple(field)
    bcfg.ANON_LABELS = bcfg.labels_for(len(field))


def cmd_council(a: argparse.Namespace) -> int:
    if a.field:
        field = [m.strip() for m in a.field.split(",") if m.strip()]
    else:
        stats = _objective_grounded_by_model()
        opens = [m for m in OPEN if m in stats]
        opens.sort(key=lambda m: (stats[m]["fabrication"], -stats[m]["move_sound"]))
        field = list(FRONTIER5) + opens[: a.top_open]
    bad = [m for m in field if m not in bcfg.MODELS]
    if bad:
        raise SystemExit(f"unknown model keys in field: {bad}")
    _set_field(field)
    bcfg.JUDGE_MAX_TOKENS = a.judge_max_tokens
    scns = _load_scn(None)[: a.n]
    print(f"council: field={field} (N={len(field)}), positions={len(scns)}, "
          f"judges={list(bcfg.JUDGE_KEYS)}, judge_max_tokens={bcfg.JUDGE_MAX_TOKENS}")
    coun_mod.run_council(
        scns, ["grounded"], list(bcfg.JUDGE_KEYS),
        concurrency=a.concurrency, min_interval=a.min_interval,
        timeout=a.timeout, max_retries=a.max_retries,
    )
    return 0


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _council_field_from_disk() -> List[str]:
    seen: Dict[str, int] = {}
    for r in read_jsonl(bcfg.COUNCIL_PATH):
        for mk in (r.get("label_to_model") or {}).values():
            seen[mk] = seen.get(mk, 0) + 1
    return [m for m in UNIFIED_ORDER if m in seen]


def cmd_report(_a: argparse.Namespace) -> int:
    from src.eval.benchmark import report_open  # local module (below)
    report_open.write_report(
        report_md=REPORT_MD, unified_order=list(UNIFIED_ORDER),
        frontier5=list(FRONTIER5),
        council_field=_council_field_from_disk(),
    )
    print(f"wrote {REPORT_MD}")
    return 0


def cmd_status(_a: argparse.Namespace) -> int:
    scns = read_jsonl(bcfg.SCENARIOS_PATH)
    gens = read_jsonl(bcfg.GENERATIONS_PATH)
    objs = read_jsonl(bcfg.OBJECTIVE_PATH)
    coun = read_jsonl(bcfg.COUNCIL_PATH)
    gm: Dict[str, int] = {}
    for g in gens:
        gm[g["model"]] = gm.get(g["model"], 0) + 1
    print("=== benchmark_open status ===")
    print(f"BENCH_DIR:   {bcfg.BENCH_DIR}")
    print(f"scenarios:   {len(scns)}")
    print(f"generations: {len(gens)} (grounded)  by model: "
          + ", ".join(f"{k}={v}" for k, v in sorted(gm.items())))
    print(f"objective:   {len(objs)}")
    print(f"council:     {len(coun)}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extend the chess-coach benchmark to open models.")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _gen_args(sp):
        sp.add_argument("--models", default=None, help="Comma keys (default: all open).")
        sp.add_argument("--limit", type=int, default=None)
        sp.add_argument("--concurrency", type=int, default=6)
        sp.add_argument("--min-interval", dest="min_interval", type=float, default=0.05)
        sp.add_argument("--timeout", type=float, default=300.0)
        sp.add_argument("--max-retries", dest="max_retries", type=int, default=5)

    sub.add_parser("probe").set_defaults(func=lambda a: _probe())
    sub.add_parser("seed").set_defaults(func=cmd_seed)

    pg = sub.add_parser("generate"); _gen_args(pg); pg.set_defaults(func=cmd_generate)
    po = sub.add_parser("objective"); po.add_argument("--limit", type=int, default=None)
    po.set_defaults(func=cmd_objective)
    sub.add_parser("phase1").set_defaults(func=cmd_phase1)

    pc = sub.add_parser("council")
    pc.add_argument("--field", default=None, help="Comma keys for the unified field.")
    pc.add_argument("--top-open", dest="top_open", type=int, default=4,
                    help="If --field omitted: 5 v2 models + this many top open models.")
    pc.add_argument("--n", type=int, default=50, help="Position subset size.")
    pc.add_argument("--judge-max-tokens", dest="judge_max_tokens", type=int, default=8000)
    pc.add_argument("--concurrency", type=int, default=6)
    pc.add_argument("--min-interval", dest="min_interval", type=float, default=0.05)
    pc.add_argument("--timeout", type=float, default=300.0)
    pc.add_argument("--max-retries", dest="max_retries", type=int, default=5)
    pc.set_defaults(func=cmd_council)

    sub.add_parser("report").set_defaults(func=cmd_report)
    sub.add_parser("status").set_defaults(func=cmd_status)
    return p


def _probe() -> int:
    import subprocess
    return subprocess.call([sys.executable, str(ROOT / "scripts" / "tfy_access_open.py")])


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
