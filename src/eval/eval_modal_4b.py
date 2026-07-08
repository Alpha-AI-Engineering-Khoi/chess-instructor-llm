#!/usr/bin/env python3
"""P1 — the HONEST 4B eval, ENTIRELY on Modal (no Mac).

Runs the exact honest base-vs-tuned eval + "train by prompting" litmus + frontier
council on a Modal GPU, so nothing depends on the laptop:

* generation for the three 4B contenders happens on the GPU (Unsloth base 4-bit,
  or base + our LoRA adapter from the ``chess-coach-lora`` Volume) via
  :func:`src.eval.gpu_gate.batched_gated_generate`, which reuses the shipped
  faithfulness gate helpers (``coach_gate`` + ``verify_text_ext``) verbatim;
* the prompt-engineered base is optimised with the REUSED
  :func:`src.eval.honest.promptopt.optimize` (GPU coach + TrueFoundry judge);
* the blinded cross-family council is the REUSED
  :func:`src.eval.honest.rubric.run_council` (TrueFoundry — org-funded, already
  Mac-independent);
* the report (headline A–E + deterministic gates + distinct-moves-per-level) is
  written to the ``chess-coach-eval`` Volume.

Cost-smart: the untuned ``base_4b`` + ``pbase_4b`` gens and the engineered prompt
are FIXED (the base model never changes), so they are generated ONCE into a shared
cache on the Volume and reused every iteration; only ``ours_4b`` (the new adapter)
is regenerated per iteration. The frontier / ours_v3 rows are the pre-computed
gap803 gens (baked into the image), sliced to the val positions.

Two functions so the loop only pays GPU rates for generation:
  * ``eval_generate`` (GPU) — optimize + gen ours/base/pbase + reuse -> Volume.
  * ``eval_judge_report`` (CPU) — council + report -> Volume, returns the headline.

Run manually (survives disconnect via .spawn on the server side)::

    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    MODAL_PROFILE=chess-instructor-2 modal run src/eval/eval_modal_4b.py \
        --iter-tag iter1 --adapter-dir /lora/chess-coach-4b-iter1/adapter

Deploy (so the orchestrator can call it detached)::

    MODAL_PROFILE=chess-instructor-2 modal deploy src/eval/eval_modal_4b.py
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import modal


def _setup_logging() -> None:
    """Surface the reused honest-eval modules' INFO logs in Modal container logs."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        force=True)

# --------------------------------------------------------------------------- #
# Names / infra
# --------------------------------------------------------------------------- #
APP_NAME = "chess-coach-eval-4b"
LORA_VOLUME = "chess-coach-lora"       # adapters + datasets (shared with trainer)
EVAL_VOLUME = "chess-coach-eval"       # eval inputs cache + run outputs + loop state
LORA_MOUNT = "/lora"
EVAL_MOUNT = "/eval"

BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
GPU = "A10G"
GEN_TIMEOUT_S = 5 * 3600
JUDGE_TIMEOUT_S = 4 * 3600
CUDA_TAG = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION = "3.11"

# The unified council field + roles (mirrors scripts.honest_eval).
FIELD = ("ours_4b", "base_4b", "pbase_4b", "ours_v3", "gpt", "claude", "gemini")
FRONTIER_KEYS = ("gpt", "claude", "gemini")
REUSE_MODELS = ("gpt", "claude", "gemini", "ours_v3")

# Remote (in-image) repo layout: settings.ROOT resolves to /root.
ROOT_REMOTE = "/root"
SCN_PATH = f"{ROOT_REMOTE}/data/benchmark_gap803/scenarios.jsonl"
VAL_IDS_PATH = f"{ROOT_REMOTE}/data/benchmark_honest/val_ids.txt"
DEV_IDS_PATH = f"{ROOT_REMOTE}/data/benchmark_honest/dev_ids.txt"
REUSE_GEN_DIR = f"{ROOT_REMOTE}/data/benchmark_gap803/gen"

if modal.is_local():
    REPO = Path(__file__).resolve().parents[2]
else:
    REPO = None

SECRET = modal.Secret.from_name("chess-eval-secrets")

_PY_IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"]

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install(
        "unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets",
        "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf",
        "python-chess", "openai", "python-dotenv",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": ROOT_REMOTE})
    .workdir(ROOT_REMOTE)
)

if modal.is_local():
    # Bake the code + the SMALL eval inputs (nothing gitignored/large) into the image
    # so a deployed run is fully self-contained (no Mac at runtime).
    image = (
        image
        .add_local_dir((REPO / "src").as_posix(), f"{ROOT_REMOTE}/src", copy=True, ignore=_PY_IGNORE)
        .add_local_dir((REPO / "config").as_posix(), f"{ROOT_REMOTE}/config", copy=True, ignore=_PY_IGNORE)
        .add_local_dir((REPO / "scripts").as_posix(), f"{ROOT_REMOTE}/scripts", copy=True, ignore=_PY_IGNORE)
        .add_local_dir((REPO / "prompts").as_posix(), f"{ROOT_REMOTE}/prompts", copy=True, ignore=_PY_IGNORE)
        .add_local_file((REPO / "data/benchmark_gap803/scenarios.jsonl").as_posix(), SCN_PATH, copy=True)
        .add_local_file((REPO / "data/benchmark_gap803/frontier_ids.txt").as_posix(),
                        f"{ROOT_REMOTE}/data/benchmark_gap803/frontier_ids.txt", copy=True)
        .add_local_file((REPO / "data/benchmark_honest/val_ids.txt").as_posix(), VAL_IDS_PATH, copy=True)
        .add_local_file((REPO / "data/benchmark_honest/dev_ids.txt").as_posix(), DEV_IDS_PATH, copy=True)
    )
    for _m in REUSE_MODELS:
        image = image.add_local_file(
            (REPO / f"data/benchmark_gap803/gen/{_m}.jsonl").as_posix(),
            f"{REUSE_GEN_DIR}/{_m}.jsonl", copy=True)

lora_vol = modal.Volume.from_name(LORA_VOLUME, create_if_missing=True)
eval_vol = modal.Volume.from_name(EVAL_VOLUME, create_if_missing=True)
app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# Shared helpers (run inside the container)
# --------------------------------------------------------------------------- #
def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _done_ids(p: Path) -> set:
    """scenario_ids already present in a gen jsonl (for coverage-aware caching)."""
    done: set = set()
    for r in _read_jsonl(p):
        sid = r.get("scenario_id")
        if sid:
            done.add(sid)
    return done


def _load_val_scenarios(max_positions: int = 0):
    scns = _read_jsonl(Path(SCN_PATH))
    keep = set(Path(VAL_IDS_PATH).read_text(encoding="utf-8").split())
    val = [s for s in scns if s["pos_id"] in keep]
    if max_positions:
        pos_order: List[str] = []
        for s in val:
            if s["pos_id"] not in pos_order:
                pos_order.append(s["pos_id"])
        chosen = set(pos_order[:max_positions])
        val = [s for s in val if s["pos_id"] in chosen]
    return scns, val


def _dev_scenarios():
    scns = _read_jsonl(Path(SCN_PATH))
    keep = set(Path(DEV_IDS_PATH).read_text(encoding="utf-8").split())
    return [s for s in scns if s["pos_id"] in keep]


def _frontier_chat(role_key: str, *, max_tokens: int):
    """A TrueFoundry chat for a frontier judge/engineer (canonical gateway id)."""
    from src.eval.benchmark import config as bcfg
    from src.eval.benchmark.backends import RateLimiter, TFYChat, make_tfy_client

    m = bcfg.MODELS[role_key]
    client = make_tfy_client(240.0)
    return TFYChat(client, model_id=m.ident, max_tokens=max_tokens, max_retries=6,
                   limiter=RateLimiter(0.06), reasoning_effort=m.reasoning_effort)


def _free_gpu(coach) -> None:
    import gc
    try:
        del coach.model
    except Exception:  # noqa: BLE001
        pass
    del coach
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# eval_generate (GPU): optimize + gen ours/base/pbase + reuse
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu=GPU, timeout=GEN_TIMEOUT_S,
              volumes={LORA_MOUNT: lora_vol, EVAL_MOUNT: eval_vol}, secrets=[SECRET])
def eval_generate(
    iter_tag: str,
    adapter_dir: str,
    *,
    max_val_positions: int = 0,
    optimize_rounds: int = 3,
    batch_size: int = 16,
    max_attempts: int = 6,
    force_base: bool = False,
) -> Dict[str, Any]:
    """Generate every contender's gated rows into ``/eval/runs/<iter_tag>/gen``.

    Only ``ours_4b`` is (re)generated per call; the fixed ``base_4b`` / ``pbase_4b``
    gens + engineered prompt are cached under ``/eval/gen_cache`` and reused.
    """
    sys.path.insert(0, ROOT_REMOTE)
    os.chdir(ROOT_REMOTE)
    _setup_logging()
    from src.eval.gpu_gate import GPUCoach, batched_gated_generate
    from src.eval.benchmark.prompts import load_system_prompt

    eval_vol.reload()
    lora_vol.reload()

    run_gen = Path(f"{EVAL_MOUNT}/runs/{iter_tag}/gen")
    cache = Path(f"{EVAL_MOUNT}/gen_cache")
    prompts_dir = Path(f"{EVAL_MOUNT}/prompts")
    for d in (run_gen, cache, prompts_dir):
        d.mkdir(parents=True, exist_ok=True)

    _scns, val = _load_val_scenarios(max_val_positions)
    n_pos = len({s["pos_id"] for s in val})
    print(f"[eval_generate] iter={iter_tag} val_positions={n_pos} "
          f"val_scenarios={len(val)} adapter={adapter_dir}")

    def _commit():
        eval_vol.commit()

    # 1) ours_4b — the per-iteration contender (base + this iter's LoRA adapter).
    coach = GPUCoach(BASE_MODEL, adapter_dir)
    batched_gated_generate(val, coach, "ours_4b", run_gen / "ours_4b.jsonl",
                           max_attempts=max_attempts, gate_on=True,
                           batch_size=batch_size, commit_cb=_commit)
    _free_gpu(coach)
    eval_vol.commit()

    # 2) base_4b / pbase_4b / engineered prompt — FIXED (base model never changes),
    #    so cache under /eval/gen_cache and reuse. Coverage-aware: the cache is only
    #    "done" when it holds a row for EVERY requested val scenario, and generation
    #    is resumable (skips done ids), so a small smoke run tops up to a later full
    #    run cleanly instead of leaving a stale partial cache.
    val_ids = {s["id"] for s in val}
    best_prompt_path = prompts_dir / "best_base_prompt_4b.txt"
    need_base = force_base or not val_ids.issubset(_done_ids(cache / "base_4b.jsonl"))
    need_prompt = force_base or not best_prompt_path.exists()
    need_pbase = force_base or not val_ids.issubset(_done_ids(cache / "pbase_4b.jsonl"))

    if need_base or need_prompt or need_pbase:
        base_coach = GPUCoach(BASE_MODEL, None)
        if need_prompt:
            _optimize_base_prompt(base_coach, best_prompt_path, optimize_rounds)
            eval_vol.commit()
        best_prompt = best_prompt_path.read_text(encoding="utf-8").strip()
        if need_base:
            batched_gated_generate(val, base_coach, "base_4b", cache / "base_4b.jsonl",
                                   max_attempts=max_attempts, gate_on=True,
                                   batch_size=batch_size, commit_cb=_commit)
            eval_vol.commit()
        if need_pbase:
            batched_gated_generate(val, base_coach, "pbase_4b", cache / "pbase_4b.jsonl",
                                   system_prompt=best_prompt, max_attempts=max_attempts,
                                   gate_on=True, batch_size=batch_size, commit_cb=_commit)
            eval_vol.commit()
        _free_gpu(base_coach)

    # Assemble the per-iter run dir: fresh ours_4b + cached base/pbase + reuse frontier.
    for mk in ("base_4b", "pbase_4b"):
        src = cache / f"{mk}.jsonl"
        if src.exists():
            shutil.copy(src, run_gen / f"{mk}.jsonl")

    reuse_stats = _do_reuse(val, run_gen)
    eval_vol.commit()

    manifest = {
        "iter_tag": iter_tag, "adapter_dir": adapter_dir,
        "val_positions": len({s["pos_id"] for s in val}), "val_scenarios": len(val),
        "gen_files": sorted(p.name for p in run_gen.glob("*.jsonl")),
        "reuse": reuse_stats,
    }
    (Path(f"{EVAL_MOUNT}/runs/{iter_tag}") / "generate_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    eval_vol.commit()
    print(f"[eval_generate] DONE {json.dumps(manifest)}")
    return manifest


def _optimize_base_prompt(base_coach, out_path: Path, rounds: int) -> None:
    """Reuse src.eval.honest.promptopt.optimize with the GPU base coach."""
    import hashlib

    from src.eval.benchmark.prompts import load_system_prompt
    from src.eval.honest import promptopt as PO

    dev = _dev_scenarios()
    if not dev:
        print("[optimize] no DEV scenarios; writing default prompt")
        out_path.write_text(load_system_prompt() + "\n", encoding="utf-8")
        return

    def seed_hook(tag: str) -> None:
        base_coach.seed(int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16))

    judge = _frontier_chat("gpt", max_tokens=2500)
    engineer = _frontier_chat("claude", max_tokens=2500)
    cache_path = Path(f"{EVAL_MOUNT}/prompts/promptopt_cache_4b.jsonl")
    result = PO.optimize(
        dev, base_coach.run, judge, engineer, load_system_prompt(),
        model_key="pbase_4b", rounds=rounds, max_attempts=4, gate_on=True,
        cache_path=cache_path, seed_hook=seed_hook,
    )
    out_path.write_text(result.best_prompt.strip() + "\n", encoding="utf-8")
    print(f"[optimize] best score={result.best_score:.2f} -> {out_path}")


def _do_reuse(val, run_gen: Path) -> Dict[str, int]:
    """Slice the baked gap803 frontier / ours_v3 gens to the val items."""
    want_ids = {s["id"] for s in val}
    stats: Dict[str, int] = {}
    for mk in REUSE_MODELS:
        src = Path(REUSE_GEN_DIR) / f"{mk}.jsonl"
        rows = [r for r in _read_jsonl(src) if r.get("scenario_id") in want_ids]
        out = run_gen / f"{mk}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "model": mk, "reused_ungated": True},
                                    ensure_ascii=False) + "\n")
        stats[mk] = len({r["scenario_id"] for r in rows})
    return stats


# --------------------------------------------------------------------------- #
# eval_judge_report (CPU): council + report
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=JUDGE_TIMEOUT_S,
              volumes={EVAL_MOUNT: eval_vol}, secrets=[SECRET])
def eval_judge_report(
    iter_tag: str,
    *,
    judges: str = "gpt,claude,gemini",
    concurrency: int = 6,
    judge_max_tokens: int = 4000,
    max_val_positions: int = 0,
) -> Dict[str, Any]:
    """Blinded council + headline report for ``iter_tag`` -> Volume; returns headline."""
    sys.path.insert(0, ROOT_REMOTE)
    os.chdir(ROOT_REMOTE)
    _setup_logging()
    from src.eval.honest import rubric as R

    eval_vol.reload()
    run_dir = Path(f"{EVAL_MOUNT}/runs/{iter_tag}")
    gen_dir = run_dir / "gen"
    council_path = run_dir / "council.jsonl"

    _scns, val = _load_val_scenarios(max_val_positions)
    field = [m for m in FIELD if (gen_dir / f"{m}.jsonl").exists()]
    obm = {mk: {r["scenario_id"]: r.get("output", "")
                for r in _read_jsonl(gen_dir / f"{mk}.jsonl")} for mk in field}
    complete = [s for s in val if all(s["id"] in obm.get(m, {}) for m in field)]
    print(f"[judge] iter={iter_tag} field={field} complete={len(complete)}/{len(val)}")

    judge_map = {jk: _frontier_chat(jk, max_tokens=judge_max_tokens)
                 for jk in [j.strip() for j in judges.split(",") if j.strip()]}
    res = R.run_council(complete, obm, field, judge_map, council_path,
                        condition="gated", concurrency=concurrency)
    eval_vol.commit()
    print(f"[judge] council: {res}")

    headline = _build_report(iter_tag, val, field)
    eval_vol.commit()
    return headline


def _rec_by_model_pos_tier(field: Sequence[str], gen_dir: Path,
                           scns_by_id: Dict[str, Dict[str, Any]]):
    """model -> pos_id -> tier -> rec_uci (gate's rec, else pool-restricted extract)."""
    import chess
    from collections import defaultdict

    from src.teacher.coach_gate import extract_recommended

    out: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {mk: defaultdict(dict) for mk in field}
    for mk in field:
        for r in _read_jsonl(gen_dir / f"{mk}.jsonl"):
            scn = scns_by_id.get(r["scenario_id"])
            if scn is None:
                continue
            rec = r.get("rec_uci")
            if not rec:
                board = chess.Board(scn["fen"])
                _san, rec = extract_recommended(
                    r.get("output", ""), board, scn["sound_pool"],
                    scn["student_move"].get("uci") or "")
            out[mk][r["pos_id"]][scn["tier"]] = rec
    return out


def _tier_fit(field, gen_dir, scns_by_id, rec):
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
        out[mk] = {"tier_fit_mean": round(sum(vals) / len(vals), 4) if vals else None,
                   "move_sound": round(sound[0] / sound[1], 4) if sound[1] else None}
    return out


def _gate_stats(field, gen_dir):
    from src.eval.evaluate import find_engine_speak

    out: Dict[str, Dict[str, Any]] = {}
    for mk in field:
        rows = _read_jsonl(gen_dir / f"{mk}.jsonl")
        gated = [r for r in rows if not r.get("reused_ungated")]
        if not gated:
            out[mk] = {"gated": False, "n": len(rows)}
            continue
        att = [int(r.get("attempts", 1)) for r in gated]
        fb = sum(1 for r in gated if r.get("verified_fallback"))
        jarg = sum(1 for r in gated if not find_engine_speak(r.get("output", "")))
        wf = sum(1 for r in gated if r.get("rec_uci"))
        out[mk] = {"gated": True, "n": len(gated),
                   "mean_attempts": round(sum(att) / len(att), 3),
                   "fallback_rate": round(fb / len(gated), 4),
                   "no_jargon": round(jarg / len(gated), 4),
                   "well_formed": round(wf / len(gated), 4)}
    return out


def _build_report(iter_tag: str, val, field) -> Dict[str, Any]:
    """Self-contained headline A–E + completion-criteria check -> Volume."""
    from scripts.gap803_council_stats import compute_council_stats
    from src.eval.honest import rubric as R

    run_dir = Path(f"{EVAL_MOUNT}/runs/{iter_tag}")
    gen_dir = run_dir / "gen"
    scns_by_id = {s["id"]: s for s in val}

    council = _read_jsonl(run_dir / "council.jsonl")
    rank_stats = compute_council_stats(council, field=field, frontier=FRONTIER_KEYS)
    rankmap = rank_stats.get("models", {})
    dims = R.dim_means(council, field)
    rec = _rec_by_model_pos_tier(field, gen_dir, scns_by_id)
    tier = _tier_fit(field, gen_dir, scns_by_id, rec)
    gates = _gate_stats(field, gen_dir)
    coh = R.tier_coherence(rec, scns_by_id)

    def rank(mk): return (rankmap.get(mk) or {}).get("mean_rank")
    def instr(mk): return (dims.get(mk) or {}).get("sum_0_12")
    def tf(mk): return (tier.get(mk) or {}).get("tier_fit_mean")
    def ms(mk): return (tier.get(mk) or {}).get("move_sound")

    # canonical beginner/advanced per position (for distinct-moves-per-level).
    canon: Dict[str, Dict[str, Optional[str]]] = {}
    for s in val:
        canon.setdefault(s["pos_id"], {})[s["tier"]] = s.get("canonical_uci")

    def distinct(mk):
        picks = rec.get(mk, {})
        n = d = 0
        for pid, tp in picks.items():
            cb, ca = canon.get(pid, {}).get("beginner"), canon.get(pid, {}).get("advanced")
            mb, ma = tp.get("beginner"), tp.get("advanced")
            if cb and ca and cb != ca and mb and ma:
                n += 1
                if mb != ma:
                    d += 1
        return {"differentiating_n": n, "distinct_rate": (round(d / n, 4) if n else None),
                "collapsed_BA": n - d}

    TUNED, BASE, PBASE = "ours_4b", "base_4b", "pbase_4b"

    def _sub(a, b): return None if a is None or b is None else round(a - b, 4)

    best_fr = None
    for mk in FRONTIER_KEYS:
        r = rank(mk)
        if r is not None and (best_fr is None or r < best_fr[1]):
            best_fr = (mk, r)

    litmus = None
    if None not in (rank(PBASE), rank(TUNED), tf(PBASE), tf(TUNED)):
        litmus = bool(rank(PBASE) <= rank(TUNED) and tf(PBASE) >= tf(TUNED))

    headline = {
        "A_base_vs_tuned": {
            "tier_fit": {BASE: tf(BASE), TUNED: tf(TUNED), "delta": _sub(tf(TUNED), tf(BASE))},
            "instr_rank_lower_better": {BASE: rank(BASE), TUNED: rank(TUNED),
                                        "delta_ours_minus_base": _sub(rank(TUNED), rank(BASE))},
            "instr_0_12": {BASE: instr(BASE), TUNED: instr(TUNED), "delta": _sub(instr(TUNED), instr(BASE))},
        },
        "B_litmus_prompt_vs_tune": {
            "instr_rank": {PBASE: rank(PBASE), TUNED: rank(TUNED), "delta": _sub(rank(PBASE), rank(TUNED))},
            "tier_fit": {PBASE: tf(PBASE), TUNED: tf(TUNED), "delta": _sub(tf(PBASE), tf(TUNED))},
            "prompt_matches_or_beats_tune": litmus,
        },
        "C_distance_to_frontier": {
            "best_frontier": best_fr, "ours_4b_rank": rank(TUNED),
            "gap_ours4b_minus_bestfrontier": _sub(rank(TUNED), best_fr[1] if best_fr else None),
            "ours_v3_rank_ref": rank("ours_v3"),
        },
        "D_gates_vs_100pct": {mk: {"move_sound": ms(mk),
                                   "no_engine_speak": (gates.get(mk) or {}).get("no_jargon"),
                                   "well_formed": (gates.get(mk) or {}).get("well_formed"),
                                   "gate_fallback_rate": (gates.get(mk) or {}).get("fallback_rate"),
                                   "mean_attempts": (gates.get(mk) or {}).get("mean_attempts")}
                              for mk in (TUNED, BASE, PBASE)},
        "E_distinct_moves_per_level": {mk: {**distinct(mk),
                                            "zigzag_rate": (coh.get(mk) or {}).get("zigzag_rate"),
                                            "flat_rate": (coh.get(mk) or {}).get("flat_rate"),
                                            "coherence_violation_rate": (coh.get(mk) or {}).get("violation_rate")}
                                       for mk in field},
    }

    # ---- completion-criteria check (RALPH_TASK) -------------------------- #
    dgate = headline["D_gates_vs_100pct"][TUNED]
    e = headline["E_distinct_moves_per_level"].get(TUNED, {})
    a = headline["A_base_vs_tuned"]
    criteria = {
        "gates_100pct": (dgate.get("move_sound") == 1.0 and dgate.get("no_engine_speak") == 1.0
                         and dgate.get("well_formed") == 1.0),
        "tier_fit_ge_060": (tf(TUNED) is not None and tf(TUNED) >= 0.60),
        "distinct_ge_095": (e.get("distinct_rate") is not None and e.get("distinct_rate") >= 0.95),
        "beats_base": (a["tier_fit"]["delta"] is not None and a["tier_fit"]["delta"] > 0
                       and a["instr_rank_lower_better"]["delta_ours_minus_base"] is not None
                       and a["instr_rank_lower_better"]["delta_ours_minus_base"] < 0),
        "beats_pbase_litmus": (litmus is False),  # tune still wins the litmus
    }
    criteria["all_met"] = all(v is True for v in criteria.values())

    report = {
        "iter_tag": iter_tag,
        "n_val_positions": len(val) // 3,
        "council": {"n_items": rank_stats.get("n_items"), "n_judges": rank_stats.get("n_judges"),
                    "n_rankings": rank_stats.get("n_rankings")},
        "field": field, "headline": headline, "criteria": criteria,
        "per_model": {mk: {"mean_rank": rank(mk), "tier_fit": tf(mk), "move_sound": ms(mk),
                           "instr_0_12": instr(mk), "coherence": coh.get(mk),
                           "gate": gates.get(mk), "distinct": distinct(mk) if mk in rec else None}
                      for mk in field},
    }
    (run_dir / "report_4b.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    _write_md(run_dir / "RESULTS_HONEST_EVAL_4B.md", report, rankmap, dims, tier, gates, coh, field)
    print(f"[report] iter={iter_tag} criteria={json.dumps(criteria)}")
    return report


def _fmt(x):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{float(x):.3f}" if abs(x) < 10 else f"{float(x):.2f}"
    return str(x)


_R_SIX = ("move_purpose", "transferable_principle", "board_specific_reason",
          "how_to_find", "level_calibration", "grounded_concise")


def _write_md(path: Path, rep, rankmap, dims, tier, gates, coh, field) -> None:
    h = rep["headline"]
    a = h["A_base_vs_tuned"]
    b = h["B_litmus_prompt_vs_tune"]
    c = h["C_distance_to_frontier"]
    L: List[str] = []
    L.append(f"# HONEST 4B base-vs-tuned eval — {rep['iter_tag']} (Modal, cloud)\n")
    L.append("Every gated 4B contender coaches the SAME held-out positions through the identical "
             "shipped pipeline (grounding + `coach_gate.run_gate` helpers) on a Modal GPU, so "
             "`base_4b` vs `ours_4b` differ ONLY in the LoRA weights and `pbase_4b` differs ONLY "
             "in its system prompt. Frontier + `ours_v3` rows are REUSED ungated references.\n")
    L.append(f"- Validation slice: {rep['n_val_positions']} positions x 3 tiers; council "
             f"n_items={rep['council']['n_items']}, judges={rep['council']['n_judges']}, "
             f"rankings={rep['council']['n_rankings']}.")
    L.append(f"- Completion criteria: {json.dumps(rep['criteria'])}\n")
    L.append("## Headline\n")
    L.append("**A. Training as the only variable:**")
    L.append(f"- tier-fit: ours_4b {_fmt(a['tier_fit']['ours_4b'])} vs base_4b "
             f"{_fmt(a['tier_fit']['base_4b'])} (Δ {_fmt(a['tier_fit']['delta'])}).")
    L.append(f"- instr rank (lower=better): ours_4b {_fmt(a['instr_rank_lower_better']['ours_4b'])} "
             f"vs base_4b {_fmt(a['instr_rank_lower_better']['base_4b'])} "
             f"(Δ {_fmt(a['instr_rank_lower_better']['delta_ours_minus_base'])}).")
    L.append(f"- 6-dim/12: ours_4b {_fmt(a['instr_0_12']['ours_4b'])} vs base_4b "
             f"{_fmt(a['instr_0_12']['base_4b'])} (Δ {_fmt(a['instr_0_12']['delta'])}).\n")
    verdict = ("PROMPT MATCHES/BEATS TUNE" if b["prompt_matches_or_beats_tune"]
               else "tune still wins" if b["prompt_matches_or_beats_tune"] is False else "n/a")
    L.append(f"**B. Litmus (best prompt-engineered base vs tune):** {verdict} "
             f"(instr rank Δ {_fmt(b['instr_rank']['delta'])}, tier-fit Δ {_fmt(b['tier_fit']['delta'])}).\n")
    bf = c["best_frontier"]
    L.append(f"**C. Distance to frontier:** best frontier {bf[0] if bf else '—'} "
             f"(rank {_fmt(bf[1] if bf else None)}); ours_4b rank {_fmt(c['ours_4b_rank'])}; "
             f"gap {_fmt(c['gap_ours4b_minus_bestfrontier'])}.\n")
    L.append("**D. Deterministic gates (target 100%):**")
    for mk in ("ours_4b", "base_4b", "pbase_4b"):
        g = h["D_gates_vs_100pct"][mk]
        L.append(f"- {mk}: move-sound {_fmt(g['move_sound'])}, no-engine-speak {_fmt(g['no_engine_speak'])}, "
                 f"well-formed {_fmt(g['well_formed'])} (fallback {_fmt(g['gate_fallback_rate'])}, "
                 f"attempts {_fmt(g['mean_attempts'])}).")
    L.append("")
    L.append("**E. Distinct-moves-per-level on DIFFERENTIATING positions (target ≥95%):**")
    for mk in ("ours_4b", "base_4b", "pbase_4b"):
        ee = h["E_distinct_moves_per_level"].get(mk, {})
        L.append(f"- {mk}: {_fmt(ee.get('distinct_rate'))} distinct over {ee.get('differentiating_n')} "
                 f"positions ({ee.get('collapsed_BA')} B==A collapses).")
    L.append("")
    order = sorted(field, key=lambda m: (rankmap.get(m, {}).get("mean_rank", 99)))
    L.append("## Leaderboard\n")
    L.append("| Model | gated | tier-fit↑ | instr rank↓ | 6-dim/12↑ | move-sound↑ | distinct↑ | coh-viol↓ |")
    L.append("|---|:--:|---:|---:|---:|---:|---:|---:|")
    hd = h["E_distinct_moves_per_level"]
    for mk in order:
        r = rankmap.get(mk, {})
        L.append(f"| {mk} | {'yes' if (gates.get(mk,{}) or {}).get('gated') else 'reuse'} | "
                 f"{_fmt((tier.get(mk,{}) or {}).get('tier_fit_mean'))} | {_fmt(r.get('mean_rank'))} | "
                 f"{_fmt((dims.get(mk,{}) or {}).get('sum_0_12'))} | "
                 f"{_fmt((tier.get(mk,{}) or {}).get('move_sound'))} | "
                 f"{_fmt((hd.get(mk) or {}).get('distinct_rate'))} | "
                 f"{_fmt((coh.get(mk,{}) or {}).get('violation_rate'))} |")
    L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Local entrypoint (manual P1 run)
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main(
    iter_tag: str = "iter1",
    adapter_dir: str = f"{LORA_MOUNT}/chess-coach-4b-iter1/adapter",
    max_val_positions: int = 0,
    optimize_rounds: int = 3,
    batch_size: int = 16,
    judges: str = "gpt,claude,gemini",
) -> None:
    print(f"=== {APP_NAME}: eval {iter_tag} (adapter={adapter_dir}) ===")
    gen = eval_generate.remote(
        iter_tag, adapter_dir, max_val_positions=max_val_positions,
        optimize_rounds=optimize_rounds, batch_size=batch_size)
    print("generate manifest:\n" + json.dumps(gen, indent=2, default=str))
    head = eval_judge_report.remote(iter_tag, judges=judges, max_val_positions=max_val_positions)
    print("headline:\n" + json.dumps(head.get("headline"), indent=2, default=str))
    print("criteria:\n" + json.dumps(head.get("criteria"), indent=2, default=str))
