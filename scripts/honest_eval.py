#!/usr/bin/env python3
"""Driver for the HONEST base-vs-tuned eval + the "train by prompting" hard test.

Everything runs through the SHIPPED gated pipeline (grounding + the shared
:func:`src.teacher.coach_gate.run_gate`), so the base and the tuned model differ
only in weights, and the prompt-engineered base differs only in its system
prompt. Phases (each resumable; keep cost modest — the full re-run happens after
v4 lands):

    seed      : build held-out DEV + VALIDATION slices (stratified, disjoint)
    optimize  : run the prompt-iteration loop -> best base system prompt per size
    gen       : gated-generate a contender on the validation slice
    reuse     : pull existing gap803 frontier / ours_v3 gens for the val positions
    judge     : blinded 6-dim cross-family council over the validation field
    report    : tier-fit + tier-coherence + council ranks + the four headline nums

Run (MLX venv)::

    P=~/.venvs/mlx/bin/python
    $P -m scripts.honest_eval seed --dev 8 --val 18
    $P -m scripts.honest_eval optimize --size 1p7 --rounds 3
    $P -m scripts.honest_eval optimize --size 32b --rounds 2
    $P -m scripts.honest_eval gen --model base_1p7
    $P -m scripts.honest_eval gen --model ours_1p7
    $P -m scripts.honest_eval gen --model pbase_1p7
    $P -m scripts.honest_eval gen --model base_32b
    $P -m scripts.honest_eval gen --model pbase_32b
    $P -m scripts.honest_eval reuse --models gpt,claude,gemini,ours_v3
    $P -m scripts.honest_eval judge
    $P -m scripts.honest_eval report
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# The honest eval keeps its own artifacts; grounding/prompt code reads gap803.
os.environ.setdefault("BENCH_DIR", str(_ROOT / "data" / "benchmark_gap803"))

from dotenv import load_dotenv  # noqa: E402

from config import settings  # noqa: E402
from scripts.gap803_common import stratified_positions  # noqa: E402
from src.eval.benchmark import config as bcfg  # noqa: E402  (canonical TFY model ids)

log = logging.getLogger("honest_eval")

GAP_BENCH = _ROOT / "data" / "benchmark_gap803"
GAP_SCN = GAP_BENCH / "scenarios.jsonl"
GAP_GEN_DIR = GAP_BENCH / "gen"
FRONTIER_IDS = GAP_BENCH / "frontier_ids.txt"

HB = _ROOT / "data" / "benchmark_honest"       # honest-eval artifacts
GEN_DIR = HB / "gen"
PROMPT_DIR = HB / "prompts"
DEV_IDS = HB / "dev_ids.txt"
VAL_IDS = HB / "val_ids.txt"
COUNCIL_PATH = HB / "council.jsonl"
REPORT_JSON = HB / "report.json"
REPORT_MD = Path(os.environ.get("HONEST_REPORT_MD", str(_ROOT / "RESULTS_HONEST_EVAL.md")))

BASE_1P7 = "mlx-community/Qwen3-1.7B-4bit"
OURS_1P7 = str(settings.MODELS / "mlx" / "chess-coach-v2")
Q3_32B = "aws-bedrock/qwen.qwen3-32b-v1-0"

MLX_MAX_TOKENS = 640


@dataclass(frozen=True)
class HModel:
    key: str
    display: str
    kind: str            # "mlx" | "tfy" | "reuse"
    ident: str           # mlx path / tfy id / gap803 gen model key (reuse)
    tuned: bool
    prompt: str          # "default" | "best_1p7" | "best_32b"
    reasoning_effort: Optional[str] = None
    gated: bool = True


HONEST_MODELS: Dict[str, HModel] = {
    "ours_1p7": HModel("ours_1p7", "OURS-v2 (1.7B tuned, gated)", "mlx", OURS_1P7, True, "default"),
    "base_1p7": HModel("base_1p7", "BASE (1.7B untuned, gated)", "mlx", BASE_1P7, False, "default"),
    "pbase_1p7": HModel("pbase_1p7", "PROMPT-BASE (1.7B engineered, gated)", "mlx", BASE_1P7, False, "best_1p7"),
    "base_32b": HModel("base_32b", "BASE-32B (Qwen3-32B untuned, gated)", "tfy", Q3_32B, False, "default"),
    "pbase_32b": HModel("pbase_32b", "PROMPT-BASE-32B (engineered, gated)", "tfy", Q3_32B, False, "best_32b"),
    "ours_v3": HModel("ours_v3", "OURS-v3 (32B tuned, reused ungated)", "reuse", "ours_v3", True, "default", gated=False),
    "gpt": HModel("gpt", "GPT-5.5 (frontier)", "reuse", "gpt", False, "default", "low", gated=False),
    "claude": HModel("claude", "Claude Opus 4.8 (frontier)", "reuse", "claude", False, "default", gated=False),
    "gemini": HModel("gemini", "Gemini 3.1 Pro (frontier)", "reuse", "gemini", False, "default", gated=False),
}

#: The unified council field (validation ranking) — ordered best→ for display.
FIELD: Tuple[str, ...] = (
    "ours_1p7", "base_1p7", "pbase_1p7",
    "pbase_32b", "base_32b", "ours_v3",
    "gpt", "claude", "gemini",
)
FRONTIER_KEYS: Tuple[str, ...] = ("gpt", "claude", "gemini")


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_all_scenarios() -> List[Dict[str, Any]]:
    if not GAP_SCN.exists():
        raise SystemExit(f"missing {GAP_SCN}; run gap803_gen seed first.")
    return _read_jsonl(GAP_SCN)


def _scenarios_by_pos(scns: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in scns:
        by[s["pos_id"]].append(s)
    return by


def _slice_scenarios(ids_path: Path, scns: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = set(ids_path.read_text(encoding="utf-8").split())
    return [s for s in scns if s["pos_id"] in keep]


def best_prompt_path(size: str) -> Path:
    return PROMPT_DIR / f"best_base_prompt_{size}.txt"


def _resolve_prompt(model: HModel) -> str:
    from src.eval.benchmark.prompts import load_system_prompt

    if model.prompt == "default":
        return load_system_prompt()
    size = "1p7" if model.prompt == "best_1p7" else "32b"
    p = best_prompt_path(size)
    if not p.exists():
        raise SystemExit(
            f"{model.key} needs the engineered prompt {p} — run "
            f"`optimize --size {size}` first."
        )
    return p.read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- #
# TFY backend helpers
# --------------------------------------------------------------------------- #


def _tfy_chat(model_id: str, *, max_tokens: int, reasoning_effort: Optional[str],
              timeout: float = 240.0, min_interval: float = 0.06, max_retries: int = 6):
    from src.eval.benchmark.backends import RateLimiter, TFYChat, make_tfy_client

    client = make_tfy_client(timeout)
    return TFYChat(client, model_id=model_id, max_tokens=max_tokens,
                   max_retries=max_retries, limiter=RateLimiter(min_interval),
                   reasoning_effort=reasoning_effort)


def _frontier_chat(role_key: str, *, max_tokens: int, **kw):
    """A TFY chat for a frontier judge/engineer, using the CANONICAL gateway id.

    The honest-eval ``HModel.ident`` for gpt/claude/gemini is the gap803 *reuse*
    key (e.g. ``"gpt"``), not a gateway model id — so TFY calls must resolve the
    real id + reasoning effort from :mod:`src.eval.benchmark.config`.
    """
    m = bcfg.MODELS[role_key]
    return _tfy_chat(m.ident, max_tokens=max_tokens, reasoning_effort=m.reasoning_effort, **kw)


# --------------------------------------------------------------------------- #
# seed
# --------------------------------------------------------------------------- #


def cmd_seed(a: argparse.Namespace) -> int:
    scns = _load_all_scenarios()
    by_pos = _scenarios_by_pos(scns)
    # Only positions with all three tiers present are usable (tier-fit + coherence).
    full_pos = {pid: rows for pid, rows in by_pos.items() if len(rows) == 3}

    frontier_ids = set(FRONTIER_IDS.read_text(encoding="utf-8").split()) if FRONTIER_IDS.exists() else set()
    # Validation draws from frontier positions so gpt/claude/gemini/ours_v3 gens
    # already exist there (reused, ungated reference) — no extra frontier spend.
    val_pool = [full_pos[p][0] for p in full_pos if p in frontier_ids] or [full_pos[p][0] for p in full_pos]
    val_reps = stratified_positions(val_pool, a.val, seed=a.seed)
    val_ids = {r["pos_id"] for r in val_reps}

    # DEV is disjoint from validation (held out); prefer frontier positions too.
    dev_pool = [full_pos[p][0] for p in full_pos if p not in val_ids and p in frontier_ids]
    if len(dev_pool) < a.dev:
        dev_pool = [full_pos[p][0] for p in full_pos if p not in val_ids]
    dev_reps = stratified_positions(dev_pool, a.dev, seed=a.seed + 1)
    dev_ids = {r["pos_id"] for r in dev_reps}

    HB.mkdir(parents=True, exist_ok=True)
    DEV_IDS.write_text("\n".join(sorted(dev_ids)) + "\n", encoding="utf-8")
    VAL_IDS.write_text("\n".join(sorted(val_ids)) + "\n", encoding="utf-8")

    import collections
    vd = collections.Counter((full_pos[p][0].get("source_tier"), full_pos[p][0]["phase"]) for p in val_ids)
    print(f"seed: DEV={len(dev_ids)} positions ({len(dev_ids)*3} scenarios), "
          f"VAL={len(val_ids)} positions ({len(val_ids)*3} scenarios)")
    print(f"  VAL from frontier_ids: {len(val_ids & frontier_ids)}/{len(val_ids)} "
          f"(reused frontier/ours_v3 gens available)")
    print(f"  VAL tier×phase: {dict(vd)}")
    print(f"  wrote {DEV_IDS} , {VAL_IDS}")
    return 0


# --------------------------------------------------------------------------- #
# optimize (prompt-iteration loop)
# --------------------------------------------------------------------------- #


def cmd_optimize(a: argparse.Namespace) -> int:
    load_dotenv(settings.ROOT / ".env")
    from src.eval.benchmark.prompts import load_system_prompt
    from src.eval.honest import promptopt as PO
    from src.eval.honest.gated import MLXSamplingCoach, TFYRunFn

    scns = _load_all_scenarios()
    dev = _slice_scenarios(DEV_IDS, scns)
    if not dev:
        raise SystemExit("no DEV scenarios; run `seed` first.")

    size = a.size
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    cache = HB / f"promptopt_cache_{size}.jsonl"

    seed_hook = None
    if size == "1p7":
        coach = MLXSamplingCoach(BASE_1P7, max_tokens=MLX_MAX_TOKENS)
        run_fn = coach.run
        import hashlib as _h

        def seed_hook(tag: str) -> None:  # noqa: ANN001
            coach.seed(int(_h.sha256(tag.encode()).hexdigest()[:8], 16))
        gate_on, max_attempts = True, a.max_attempts
        model_key = "pbase_1p7"
    elif size == "32b":
        chat = _tfy_chat(Q3_32B, max_tokens=4000, reasoning_effort=None)
        run_fn = TFYRunFn(chat).run
        gate_on, max_attempts = True, a.max_attempts
        model_key = "pbase_32b"
    else:
        raise SystemExit("--size must be 1p7 or 32b")

    judge = _frontier_chat(a.judge, max_tokens=2500)
    engineer = _frontier_chat(a.engineer, max_tokens=2500)

    result = PO.optimize(
        dev, run_fn, judge, engineer, load_system_prompt(),
        model_key=model_key, rounds=a.rounds, max_attempts=max_attempts,
        gate_on=gate_on, cache_path=cache, seed_hook=seed_hook,
    )
    best_prompt_path(size).write_text(result.best_prompt.strip() + "\n", encoding="utf-8")
    (HB / f"promptopt_history_{size}.json").write_text(
        json.dumps({"best_score": result.best_score, "history": result.history,
                    "best_prompt": result.best_prompt}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\noptimize[{size}]: best score={result.best_score:.2f} -> {best_prompt_path(size)}")
    for h in result.history:
        print(f"  round {h['round']:>1} {h['kind']:9} score={h['score']:6.2f} "
              f"tier_fit={h['tier_fit']:.3f} instr={h['instr_0_12']:.2f}/12 "
              + ("KEPT" if h.get("kept") else ("" if h["kind"] == "seed" else "reject")))
    return 0


# --------------------------------------------------------------------------- #
# gen (gated generation of one contender on the validation slice)
# --------------------------------------------------------------------------- #


def cmd_gen(a: argparse.Namespace) -> int:
    load_dotenv(settings.ROOT / ".env")
    from src.eval.honest.gated import MLXSamplingCoach, TFYRunFn, generate

    model = HONEST_MODELS[a.model]
    if model.kind == "reuse":
        raise SystemExit(f"{a.model} is a reuse model; use `reuse` not `gen`.")

    scns = _slice_scenarios(VAL_IDS, _load_all_scenarios())
    if not scns:
        raise SystemExit("no VALIDATION scenarios; run `seed` first.")
    system_prompt = _resolve_prompt(model)
    out = GEN_DIR / f"{a.model}.jsonl"
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    seedable = None
    if model.kind == "mlx":
        coach = MLXSamplingCoach(model.ident, max_tokens=MLX_MAX_TOKENS)
        run_fn = coach.run
        seedable = coach
    else:
        chat = _tfy_chat(model.ident, max_tokens=4000, reasoning_effort=model.reasoning_effort)
        run_fn = TFYRunFn(chat).run

    res = generate(scns, run_fn, a.model, out, system_prompt=system_prompt,
                   max_attempts=a.max_attempts, gate_on=model.gated, seedable=seedable)
    print(f"gen {a.model}: {res} -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# reuse (pull existing gap803 frontier / ours_v3 gens for validation positions)
# --------------------------------------------------------------------------- #


def cmd_reuse(a: argparse.Namespace) -> int:
    scns = _slice_scenarios(VAL_IDS, _load_all_scenarios())
    want_ids = {s["id"] for s in scns}
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    for mk in models:
        if mk not in HONEST_MODELS or HONEST_MODELS[mk].kind != "reuse":
            print(f"  skip {mk}: not a reuse model")
            continue
        src = GAP_GEN_DIR / f"{HONEST_MODELS[mk].ident}.jsonl"
        rows = [r for r in _read_jsonl(src) if r.get("scenario_id") in want_ids]
        # Re-key to the honest model key + gated schema shape (rec_uci recomputed at report).
        out = GEN_DIR / f"{mk}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "model": mk, "reused_ungated": True},
                                    ensure_ascii=False) + "\n")
        cov = len({r["scenario_id"] for r in rows})
        print(f"  reuse {mk}: {cov}/{len(want_ids)} val items from {src.name}")
    return 0


# --------------------------------------------------------------------------- #
# judge (6-dim blinded council over the validation field)
# --------------------------------------------------------------------------- #


def _outputs_by_model(field: Sequence[str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for mk in field:
        rows = _read_jsonl(GEN_DIR / f"{mk}.jsonl")
        out[mk] = {r["scenario_id"]: r.get("output", "") for r in rows}
    return out


def cmd_judge(a: argparse.Namespace) -> int:
    load_dotenv(settings.ROOT / ".env")
    from src.eval.honest import rubric as R

    scns = _slice_scenarios(VAL_IDS, _load_all_scenarios())
    field = [m.strip() for m in a.field.split(",")] if a.field else list(FIELD)
    field = [m for m in field if (GEN_DIR / f"{m}.jsonl").exists()]
    obm = _outputs_by_model(field)
    # Only rank items every field model produced.
    complete = [s for s in scns if all(s["id"] in obm.get(m, {}) for m in field)]
    log.info("judge: field=%s, %d/%d val items complete", field, len(complete), len(scns))

    judges = {jk: _frontier_chat(jk, max_tokens=a.judge_max_tokens)
              for jk in [j.strip() for j in a.judges.split(",")]}
    res = R.run_council(complete, obm, field, judges, COUNCIL_PATH,
                        condition="gated", concurrency=a.concurrency)
    print(f"judge: {res} -> {COUNCIL_PATH}")
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def _rec_by_model_pos_tier(field: Sequence[str], scns_by_id: Dict[str, Dict[str, Any]]
                           ) -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
    """model -> pos_id -> tier -> rec_uci (recomputed with the pool-restricted extractor)."""
    import chess

    from scripts.divergence_analysis import extract_recommended_mode

    out: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {
        mk: defaultdict(dict) for mk in field}
    for mk in field:
        for r in _read_jsonl(GEN_DIR / f"{mk}.jsonl"):
            scn = scns_by_id.get(r["scenario_id"])
            if scn is None:
                continue
            # Prefer the gate's own recommendation; fall back to re-extraction.
            rec = r.get("rec_uci")
            if not rec:
                board = chess.Board(scn["fen"])
                _san, rec, _mode = extract_recommended_mode(
                    r.get("output", ""), board, scn["sound_pool"],
                    scn["student_move"].get("uci") or "")
            out[mk][r["pos_id"]][scn["tier"]] = rec
    return out


def _tier_fit(field: Sequence[str], scns_by_id: Dict[str, Dict[str, Any]]
              ) -> Dict[str, Dict[str, Any]]:
    rec = _rec_by_model_pos_tier(field, scns_by_id)
    out: Dict[str, Dict[str, Any]] = {}
    for mk in field:
        by_tier = {t: [0, 0] for t in ("beginner", "intermediate", "advanced")}
        sound = [0, 0]
        for pos_id, picks in rec[mk].items():
            for tier, uci in picks.items():
                scn = scns_by_id.get(f"{pos_id}#{tier}")
                if scn is None:
                    continue
                by_tier[tier][1] += 1
                if uci and uci == scn.get("canonical_uci"):
                    by_tier[tier][0] += 1
                sound[1] += 1
                if uci and uci in set(scn.get("sound_uci", [])):
                    sound[0] += 1
        vals = [by_tier[t][0] / by_tier[t][1] for t in by_tier if by_tier[t][1]]
        out[mk] = {
            "tier_fit_by_tier": {t: (by_tier[t][0] / by_tier[t][1] if by_tier[t][1] else None)
                                 for t in by_tier},
            "tier_fit_mean": round(sum(vals) / len(vals), 4) if vals else None,
            "move_sound": round(sound[0] / sound[1], 4) if sound[1] else None,
        }
    return out


def _gate_stats(field: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for mk in field:
        rows = _read_jsonl(GEN_DIR / f"{mk}.jsonl")
        gated = [r for r in rows if not r.get("reused_ungated")]
        if not gated:
            out[mk] = {"gated": False, "n": len(rows)}
            continue
        att = [int(r.get("attempts", 1)) for r in gated]
        fb = sum(1 for r in gated if r.get("verified_fallback"))
        from src.eval.evaluate import find_engine_speak
        jarg = sum(1 for r in gated if not find_engine_speak(r.get("output", "")))
        out[mk] = {
            "gated": True, "n": len(gated),
            "mean_attempts": round(sum(att) / len(att), 3),
            "fallback_rate": round(fb / len(gated), 4),
            "no_jargon": round(jarg / len(gated), 4),
        }
    return out


def cmd_report(a: argparse.Namespace) -> int:
    from scripts import gap803_council_stats as CS
    from src.eval.honest import rubric as R

    scns = _slice_scenarios(VAL_IDS, _load_all_scenarios())
    scns_by_id = {s["id"]: s for s in scns}
    field = [m for m in FIELD if (GEN_DIR / f"{m}.jsonl").exists()]

    council_rows = _read_jsonl(COUNCIL_PATH)
    rank_stats = CS.compute_council_stats(council_rows, field=field, frontier=FRONTIER_KEYS)
    dims = R.dim_means(council_rows, field)
    tier = _tier_fit(field, scns_by_id)
    gates = _gate_stats(field)
    coherence = R.tier_coherence(_rec_by_model_pos_tier(field, scns_by_id), scns_by_id)

    def rank(mk: str) -> Optional[float]:
        m = rank_stats.get("models", {}).get(mk)
        return m["mean_rank"] if m else None

    def instr12(mk: str) -> Optional[float]:
        return dims.get(mk, {}).get("sum_0_12")

    headline = {
        "A_base_vs_tuned_1p7": {
            "tier_fit_delta": _delta(tier, "ours_1p7", "base_1p7", "tier_fit_mean"),
            "instr_rank": {"ours_1p7": rank("ours_1p7"), "base_1p7": rank("base_1p7"),
                           "delta_lower_is_better": _sub(rank("ours_1p7"), rank("base_1p7"))},
            "instr_0_12": {"ours_1p7": instr12("ours_1p7"), "base_1p7": instr12("base_1p7"),
                           "delta": _sub(instr12("ours_1p7"), instr12("base_1p7"))},
        },
        "B_litmus_prompt_vs_tune": {
            "1p7": _litmus(tier, rank, instr12, "pbase_1p7", "ours_1p7"),
            "32b": _litmus(tier, rank, instr12, "pbase_32b", "ours_v3",
                           caveat="ours_v3 reused UNGATED"),
        },
        "C_distance_to_frontier": {
            "best_frontier_rank": _best_frontier(rank),
            "ours_1p7_rank": rank("ours_1p7"),
            "ours_v3_rank": rank("ours_v3"),
            "gap_ours1p7_minus_bestfrontier": _sub(rank("ours_1p7"), _best_frontier(rank)[1]),
        },
        "D_tier_coherence_violation_rate": {
            mk: coherence.get(mk, {}).get("violation_rate") for mk in field},
    }

    report = {
        "field": field,
        "n_val_positions": len(scns) // 3,
        "council": {"n_items": rank_stats.get("n_items"), "n_judges": rank_stats.get("n_judges"),
                    "n_rankings": rank_stats.get("n_rankings")},
        "rank_stats": rank_stats,
        "dim_means": dims,
        "tier_fit": tier,
        "gate_stats": gates,
        "tier_coherence": coherence,
        "headline": headline,
    }
    HB.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report)
    print(f"report -> {REPORT_JSON}\nreport -> {REPORT_MD}")
    _print_headline(report)
    return 0


def _delta(d: Dict[str, Any], a_key: str, b_key: str, field: str) -> Dict[str, Any]:
    av = d.get(a_key, {}).get(field)
    bv = d.get(b_key, {}).get(field)
    return {a_key: av, b_key: bv, "delta": _sub(av, bv)}


def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(a - b, 4)


def _best_frontier(rank_fn) -> Tuple[Optional[str], Optional[float]]:
    best: Tuple[Optional[str], Optional[float]] = (None, None)
    for mk in FRONTIER_KEYS:
        r = rank_fn(mk)
        if r is not None and (best[1] is None or r < best[1]):
            best = (mk, r)
    return best


def _litmus(tier, rank_fn, instr_fn, prompt_key: str, tune_key: str,
            caveat: Optional[str] = None) -> Dict[str, Any]:
    pr, tr = rank_fn(prompt_key), rank_fn(tune_key)
    pf = tier.get(prompt_key, {}).get("tier_fit_mean")
    tf = tier.get(tune_key, {}).get("tier_fit_mean")
    pi, ti = instr_fn(prompt_key), instr_fn(tune_key)
    # "matches the tune" if the prompt-base is at least as good on instr rank AND tier-fit.
    matches = None
    if pr is not None and tr is not None and pf is not None and tf is not None:
        matches = bool(pr <= tr and pf >= tf)
    return {
        "prompt_base": prompt_key, "tune": tune_key,
        "instr_rank": {prompt_key: pr, tune_key: tr, "delta": _sub(pr, tr)},
        "tier_fit": {prompt_key: pf, tune_key: tf, "delta": _sub(pf, tf)},
        "instr_0_12": {prompt_key: pi, tune_key: ti, "delta": _sub(pi, ti)},
        "prompt_matches_tune": matches,
        "caveat": caveat,
    }


# --------------------------------------------------------------------------- #
# Markdown + console
# --------------------------------------------------------------------------- #


def _write_markdown(rep: Dict[str, Any]) -> None:
    field = rep["field"]
    rs = rep["rank_stats"]["models"]
    dims = rep["dim_means"]
    tier = rep["tier_fit"]
    coh = rep["tier_coherence"]
    gates = rep["gate_stats"]
    h = rep["headline"]

    def disp(mk: str) -> str:
        return HONEST_MODELS[mk].display

    lines: List[str] = []
    lines.append("# HONEST base-vs-tuned eval + the \"train by prompting\" hard test\n")
    lines.append(
        "First VALIDATION run on a held-out DEV/VAL slice. Every 1.7B/32B contender coaches the "
        "SAME positions through the **identical shipped pipeline** — grounding (Stockfish pool + "
        "Maia + verified facts) AND the shared faithfulness gate "
        "(`src.teacher.coach_gate.run_gate`, the exact code `src/api/server.py` runs) — so base vs "
        "tuned differ only in weights, and the prompt-base differs only in its system prompt. "
        "Frontier + OURS-v3 rows are REUSED ungated gap803 gens (low-fabrication reference); the "
        "core litmus (1.7B) is fully gated on both sides.\n")
    lines.append(f"- **Validation slice:** {rep['n_val_positions']} positions × 3 tiers; "
                 f"council n_items={rep['council']['n_items']}, judges={rep['council']['n_judges']}, "
                 f"rankings={rep['council']['n_rankings']}.\n")

    # Headline
    a = h["A_base_vs_tuned_1p7"]
    lines.append("## Headline\n")
    lines.append("**A. Training as the only variable (1.7B, identical gated pipeline):**")
    lines.append(f"- tier-appropriate move selection: OURS-v2 {a['tier_fit_delta']['ours_1p7']} vs "
                 f"BASE {a['tier_fit_delta']['base_1p7']} (**Δ {a['tier_fit_delta']['delta']}**).")
    lines.append(f"- instructiveness council mean rank (lower=better): OURS-v2 "
                 f"{a['instr_rank']['ours_1p7']} vs BASE {a['instr_rank']['base_1p7']} "
                 f"(**Δ {a['instr_rank']['delta_lower_is_better']}**).")
    lines.append(f"- instructiveness rubric sum (0-12): OURS-v2 {a['instr_0_12']['ours_1p7']} vs "
                 f"BASE {a['instr_0_12']['base_1p7']} (**Δ {a['instr_0_12']['delta']}**).\n")

    for size in ("1p7", "32b"):
        b = h["B_litmus_prompt_vs_tune"][size]
        verdict = ("PROMPT MATCHES/BEATS TUNE" if b["prompt_matches_tune"]
                   else "tune still wins" if b["prompt_matches_tune"] is False else "n/a")
        cav = f" _(caveat: {b['caveat']})_" if b.get("caveat") else ""
        lines.append(f"**B. Litmus [{size}] — can a well-prompted base match the tune?** "
                     f"**{verdict}**{cav}")
        lines.append(f"- instr rank: {b['prompt_base']} {b['instr_rank'][b['prompt_base']]} vs "
                     f"{b['tune']} {b['instr_rank'][b['tune']]} (Δ {b['instr_rank']['delta']}); "
                     f"tier-fit Δ {b['tier_fit']['delta']}; 6-dim Δ {b['instr_0_12']['delta']}.\n")

    c = h["C_distance_to_frontier"]
    lines.append(f"**C. Distance to frontier:** best frontier = {c['best_frontier_rank'][0]} "
                 f"(rank {c['best_frontier_rank'][1]}); OURS-v2 rank {c['ours_1p7_rank']}, "
                 f"OURS-v3 rank {c['ours_v3_rank']}; gap OURS-v2−bestfrontier = "
                 f"{c['gap_ours1p7_minus_bestfrontier']} rank positions.\n")

    lines.append("**D. Tier-coherence violation rate (deterministic):** "
                 + ", ".join(f"{disp(mk)} {h['D_tier_coherence_violation_rate'].get(mk)}"
                             for mk in field if h['D_tier_coherence_violation_rate'].get(mk) is not None)
                 + "\n")

    # Leaderboard
    lines.append("## Leaderboard (validation field)\n")
    lines.append("| Model | gated | tier-fit↑ | instr rank↓ | 6-dim/12↑ | move-sound↑ | tier-coh viol↓ |")
    lines.append("|---|:--:|---:|---:|---:|---:|---:|")
    order = sorted(field, key=lambda m: (rs.get(m, {}).get("mean_rank", 99)))
    for mk in order:
        r = rs.get(mk, {})
        lines.append(
            f"| {disp(mk)} | {'yes' if gates.get(mk,{}).get('gated') else 'reuse'} | "
            f"{_fmt(tier.get(mk,{}).get('tier_fit_mean'))} | {_fmt(r.get('mean_rank'))} | "
            f"{_fmt(dims.get(mk,{}).get('sum_0_12'))} | {_fmt(tier.get(mk,{}).get('move_sound'))} | "
            f"{_fmt(coh.get(mk,{}).get('violation_rate'))} |")
    lines.append("")

    # 6-dim breakdown
    lines.append("## Instructiveness rubric — six dimensions (mean 0/1/2)\n")
    lines.append("| Model | " + " | ".join(d.replace("_", " ") for d in R_SIX) + " |")
    lines.append("|---" + "|---:" * len(R_SIX) + "|")
    for mk in order:
        dd = dims.get(mk, {}).get("dims", {})
        lines.append(f"| {disp(mk)} | " + " | ".join(_fmt(dd.get(d)) for d in R_SIX) + " |")
    lines.append("")

    # Gate telemetry
    lines.append("## Gate telemetry (gated contenders)\n")
    lines.append("| Model | mean attempts | fallback rate | no-jargon |")
    lines.append("|---|---:|---:|---:|")
    for mk in order:
        g = gates.get(mk, {})
        if g.get("gated"):
            lines.append(f"| {disp(mk)} | {_fmt(g.get('mean_attempts'))} | "
                         f"{_fmt(g.get('fallback_rate'))} | {_fmt(g.get('no_jargon'))} |")
    lines.append("")

    lines.append("## Reproduce the FULL eval after v4 lands\n")
    lines.append(_full_eval_commands())
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


R_SIX = ("move_purpose", "transferable_principle", "board_specific_reason",
         "how_to_find", "level_calibration", "grounded_concise")


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{float(x):.3f}" if abs(x) < 10 else f"{float(x):.2f}"
    return str(x)


def _full_eval_commands() -> str:
    return (
        "```bash\n"
        "P=~/.venvs/mlx/bin/python\n"
        "# 1) (once) rebuild the held-out slices if the position set changed:\n"
        "$P -m scripts.honest_eval seed --dev 8 --val 18\n"
        "# 2) engineer the best base prompt per size (uses TrueFoundry judge+engineer):\n"
        "$P -m scripts.honest_eval optimize --size 1p7 --rounds 3\n"
        "$P -m scripts.honest_eval optimize --size 32b --rounds 2\n"
        "# 3) gated generation of every contender (1.7B local free; 32B via TFY):\n"
        "for m in base_1p7 ours_1p7 pbase_1p7 base_32b pbase_32b; do "
        "$P -m scripts.honest_eval gen --model $m; done\n"
        "# 4) reuse existing gap803 frontier + tuned-32B gens for the val positions:\n"
        "$P -m scripts.honest_eval reuse --models gpt,claude,gemini,ours_v3\n"
        "# 5) blinded 6-dim cross-family council + report:\n"
        "$P -m scripts.honest_eval judge --judges gpt,claude,gemini\n"
        "$P -m scripts.honest_eval report\n"
        "# For the DEFINITIVE full-scale run: raise --val (e.g. 150) and re-gen ours_1p7\n"
        "# against the v4 checkpoint (set OURS_1P7 / models/mlx/chess-coach-v4).\n"
        "```\n"
    )


def _print_headline(rep: Dict[str, Any]) -> None:
    h = rep["headline"]
    print("\n=== HEADLINE ===")
    a = h["A_base_vs_tuned_1p7"]
    print("A. base-vs-tuned (1.7B, identical gated pipeline):")
    print(f"   tier-fit Δ (ours-base) = {a['tier_fit_delta']['delta']}")
    print(f"   instr rank Δ (ours-base, <0 = ours better) = {a['instr_rank']['delta_lower_is_better']}")
    print(f"   6-dim/12 Δ (ours-base) = {a['instr_0_12']['delta']}")
    for size in ("1p7", "32b"):
        b = h["B_litmus_prompt_vs_tune"][size]
        print(f"B. litmus[{size}]: prompt_matches_tune={b['prompt_matches_tune']} "
              f"(instr rank Δ={b['instr_rank']['delta']}, tier-fit Δ={b['tier_fit']['delta']})"
              + (f"  [{b['caveat']}]" if b.get("caveat") else ""))
    c = h["C_distance_to_frontier"]
    print(f"C. distance to frontier: best={c['best_frontier_rank']}, ours_1p7 rank={c['ours_1p7_rank']}, "
          f"gap={c['gap_ours1p7_minus_bestfrontier']}")
    print(f"D. tier-coherence violation rate: {h['D_tier_coherence_violation_rate']}")


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("seed", help="Build held-out DEV + VALIDATION slices.")
    ps.add_argument("--dev", type=int, default=8)
    ps.add_argument("--val", type=int, default=18)
    ps.add_argument("--seed", type=int, default=3407)
    ps.set_defaults(func=cmd_seed)

    po = sub.add_parser("optimize", help="Prompt-iteration loop -> best base prompt.")
    po.add_argument("--size", choices=["1p7", "32b"], required=True)
    po.add_argument("--rounds", type=int, default=3)
    po.add_argument("--max-attempts", dest="max_attempts", type=int, default=4)
    po.add_argument("--judge", default="gpt", help="Loop instructiveness judge (default gpt: reliable at low effort).")
    po.add_argument("--engineer", default="claude", help="Prompt-engineer model (default claude).")
    po.set_defaults(func=cmd_optimize)

    pg = sub.add_parser("gen", help="Gated-generate one contender on the val slice.")
    pg.add_argument("--model", required=True, choices=[k for k, m in HONEST_MODELS.items() if m.kind != "reuse"])
    pg.add_argument("--max-attempts", dest="max_attempts", type=int, default=6)
    pg.set_defaults(func=cmd_gen)

    pr = sub.add_parser("reuse", help="Pull existing gap803 gens for reuse models.")
    pr.add_argument("--models", default="gpt,claude,gemini,ours_v3")
    pr.set_defaults(func=cmd_reuse)

    pj = sub.add_parser("judge", help="Blinded 6-dim council over the val field.")
    pj.add_argument("--field", default="")
    pj.add_argument("--judges", default="gpt,claude,gemini")
    pj.add_argument("--concurrency", type=int, default=5)
    pj.add_argument("--judge-max-tokens", dest="judge_max_tokens", type=int, default=4000)
    pj.set_defaults(func=cmd_judge)

    prep = sub.add_parser("report", help="Aggregate + write RESULTS_HONEST_EVAL.md.")
    prep.set_defaults(func=cmd_report)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    finally:
        try:
            from src.engine import maia_engine
            maia_engine.close_all()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
