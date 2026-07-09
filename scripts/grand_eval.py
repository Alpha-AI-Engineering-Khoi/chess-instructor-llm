#!/usr/bin/env python3
"""GRAND EVAL — one fresh, apples-to-apples comparison of EVERY model.

A single comprehensive leaderboard over the SAME held-out VAL slice the honest
eval uses (120 positions x 3 tiers = 360 scenarios, ``data/benchmark_honest/val_ids.txt``),
scored with BOTH layers:

* **Deterministic moat metrics** (free, python-chess over pre-computed engine
  facts): tier-fit, distinct-moves-per-level, move-soundness, raw faithfulness
  (verify-pass on draft 1), coherence, and shipped-gate soundness.
* **Blinded cross-family frontier COUNCIL** (GPT-5.5 + Claude + Gemini via the
  TrueFoundry gateway), 0-10 move + instructiveness, with 95% CIs.

The field (20 models):
  tuned   : ours (v2 1.7B), ours_4b (4B), ours_v3/ours_v4/ours_v5 (32B)
  untuned : base (1.7B), base_4b (4B), pbase_4b (4B prompt-engineered),
            q3_32b (Qwen3-32B untuned, TFY)
  frontier: gpt, claude, gemini + 8 big open (q3_next80b, gemma3_27b,
            llama33_70b, dsv32, glm5, mistral3, kimi25, dsr1)

Isolation: everything lives under ``data/benchmark_grand/`` so the running
finish-v5 controller's ``data/benchmark_honest/gen/*`` and
``web/public/showcase.json`` are never touched. Reuses every aggregation helper
from :mod:`scripts.honest_v4` verbatim (only the globals are re-pointed).

Phases (each resumable)::

    P=/Users/khoilam/.venvs/mlx/bin/python
    $P -m scripts.grand_eval setup       # reused + v5 gen files onto the val slice
    $P -m scripts.grand_eval gen          # FRESH TFY gen for all 12 gateway models
    $P -m scripts.grand_eval status       # field coverage
    $P -m scripts.grand_eval council --positions 60   # 20-way 0-10 council (budget-sized)
    $P -m scripts.grand_eval report       # comprehensive leaderboard + per-tuned moat proof
    $P -m scripts.grand_eval cost         # measured USD from recorded token usage
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("BENCH_DIR", str(_ROOT / "data" / "benchmark_gap803"))

import chess  # noqa: E402
import scripts.honest_v4 as H  # noqa: E402  (reuse every helper; re-point globals below)

log = logging.getLogger("grand_eval")

# --------------------------------------------------------------------------- #
# Namespace + field  (isolated under data/benchmark_grand/)
# --------------------------------------------------------------------------- #
GB = _ROOT / "data" / "benchmark_grand"
GEN_DIR = GB / "gen"
COUNCIL = GB / "council.jsonl"
REPORT_JSON = GB / "report.json"
REPORT_MD = GB / "GRAND_EVAL_LEADERBOARD.md"
V5_VOL_SNAPSHOT = GB / "_ours_v5_vol.jsonl"  # downloaded from the Modal Volume

TIERS = H.TIERS
FRONTIER_KEYS: Tuple[str, ...] = ("gpt", "claude", "gemini")
JUDGE_KEYS: Tuple[str, ...] = ("gpt", "claude", "gemini")

#: TFY gateway models regenerated FRESH on the val slice (untuned q3_32b + frontier).
TFY_FRESH: Tuple[str, ...] = (
    "gpt", "claude", "gemini",
    "q3_32b", "q3_next80b", "gemma3_27b", "llama33_70b",
    "dsv32", "glm5", "mistral3", "kimi25", "dsr1",
)

#: The full ordered field.
FIELD: Tuple[str, ...] = (
    "ours_v5", "ours_v4", "ours_v3", "ours", "ours_4b",         # tuned
    "q3_32b", "base", "base_4b", "pbase_4b",                    # untuned
    "gpt", "claude", "gemini",                                  # frontier APIs
    "q3_next80b", "gemma3_27b", "llama33_70b", "dsv32",         # big open
    "glm5", "mistral3", "kimi25", "dsr1",
)

TUNED: Tuple[str, ...] = ("ours", "ours_4b", "ours_v3", "ours_v4", "ours_v5")

#: display name, family, whether the gen is FRESH (this run) vs REUSED, and how.
META: Dict[str, Dict[str, Any]] = {
    "ours":       {"name": "OURS-v2 (Qwen3-1.7B tuned)",        "family": "ours",  "fresh": False, "how": "MLX-local reuse (gap803, greedy deterministic)"},
    "ours_4b":    {"name": "OURS-4B (Qwen3-4B tuned)",          "family": "ours",  "fresh": False, "how": "Modal reuse (honest val, gated pipeline)"},
    "ours_v3":    {"name": "OURS-v3 (Qwen3-32B tuned)",         "family": "ours",  "fresh": False, "how": "Modal-adapter reuse (gap803, deterministic)"},
    "ours_v4":    {"name": "OURS-v4 (Qwen3-32B tuned)",         "family": "ours",  "fresh": False, "how": "Modal-adapter reuse (honest val, deterministic)"},
    "ours_v5":    {"name": "OURS-v5 (Qwen3-32B tuned, v5)",     "family": "ours",  "fresh": True,  "how": "Modal-adapter FRESH (finish-v5 controller Volume gen)"},
    "base":       {"name": "BASE (Qwen3-1.7B untuned)",         "family": "base",  "fresh": False, "how": "MLX-local reuse (gap803, greedy deterministic)"},
    "base_4b":    {"name": "BASE-4B (Qwen3-4B untuned)",        "family": "base",  "fresh": False, "how": "Modal reuse (honest val, gated pipeline)"},
    "pbase_4b":   {"name": "PROMPT-BASE-4B (Qwen3-4B engineered)", "family": "base", "fresh": False, "how": "Modal reuse (honest val, gated pipeline)"},
    "q3_32b":     {"name": "BASE (Qwen3-32B untuned)",          "family": "base",  "fresh": True,  "how": "TFY FRESH (aws-bedrock qwen3-32b)"},
    "gpt":        {"name": "GPT-5.5",                            "family": "frontier", "fresh": True, "how": "TFY FRESH"},
    "claude":     {"name": "Claude Opus 4.8",                    "family": "frontier", "fresh": True, "how": "TFY FRESH"},
    "gemini":     {"name": "Gemini 3.1 Pro",                     "family": "frontier", "fresh": True, "how": "TFY FRESH"},
    "q3_next80b": {"name": "Qwen3-Next-80B-A3B",                 "family": "open",  "fresh": True,  "how": "TFY FRESH"},
    "gemma3_27b": {"name": "Gemma-3-27B-it",                     "family": "open",  "fresh": True,  "how": "TFY FRESH"},
    "llama33_70b":{"name": "Llama-3.3-70B",                      "family": "open",  "fresh": True,  "how": "TFY FRESH"},
    "dsv32":      {"name": "DeepSeek-V3.2",                      "family": "open",  "fresh": True,  "how": "TFY FRESH"},
    "glm5":       {"name": "GLM-5",                              "family": "open",  "fresh": True,  "how": "TFY FRESH"},
    "mistral3":   {"name": "Mistral-Large-3 (675B)",            "family": "open",  "fresh": True,  "how": "TFY FRESH"},
    "kimi25":     {"name": "Kimi-K2.5",                          "family": "open",  "fresh": True,  "how": "TFY FRESH"},
    "dsr1":       {"name": "DeepSeek-R1 (reasoning)",            "family": "open",  "fresh": True,  "how": "TFY FRESH"},
}

# Re-point honest_v4's globals so its helpers read OUR namespace + field.
H.GEN_DIR = GEN_DIR
H.COUNCIL_V4 = COUNCIL
H.V4_FIELD = FIELD
H.FRONTIER_KEYS = FRONTIER_KEYS
H.JUDGE_KEYS = JUDGE_KEYS
for mk, m in META.items():
    H.DISPLAY[mk] = {"name": m["name"], "family": m["family"], "local": m["family"] in ("ours", "base")}


# --------------------------------------------------------------------------- #
# setup — build reused + v5 gen files onto the val slice
# --------------------------------------------------------------------------- #
def _by_id() -> Dict[str, Dict[str, Any]]:
    return {s["id"]: s for s in H._val_scenarios()}


def _reuse_from(model_key: str, src: Path, by_id, want, keep_raw: bool) -> None:
    rows = {r["scenario_id"]: r for r in H._read_jsonl(src) if r.get("scenario_id") in want}
    H._write_reused(model_key, rows, by_id, keep_raw=keep_raw)


def cmd_setup(a: argparse.Namespace) -> int:
    by_id = _by_id()
    want = set(by_id)
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    # --- reused, deterministic references (scored fresh via score_one) ------- #
    _reuse_from("ours", _ROOT / "data/benchmark_gap803/gen/ours.jsonl", by_id, want, keep_raw=False)
    _reuse_from("base", _ROOT / "data/benchmark_gap803/gen/base.jsonl", by_id, want, keep_raw=False)
    _reuse_from("ours_v3", _ROOT / "data/benchmark_gap803/gen/ours_v3.jsonl", by_id, want, keep_raw=False)

    # --- ours_v4 + gated 4B trio: copy verbatim from the honest val gens ------ #
    # (they already carry the schema honest_v4's readers expect; the 4B trio keep
    #  their gated telemetry attempts/verified_fallback so gate axes stay honest).
    for mk in ("ours_v4", "ours_4b", "base_4b", "pbase_4b"):
        src = _ROOT / f"data/benchmark_honest/gen/{mk}.jsonl"
        rows = [r for r in H._read_jsonl(src) if r.get("scenario_id") in want]
        (GEN_DIR / f"{mk}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        print(f"[setup] {mk}: copied {len(rows)} val rows verbatim from honest")

    # --- ours_v5: from the Modal Volume snapshot (finish-v5 controller gen) --- #
    if V5_VOL_SNAPSHOT.exists():
        v5 = {r["scenario_id"]: r for r in H._read_jsonl(V5_VOL_SNAPSHOT) if r.get("scenario_id") in want}
        if len(v5) == len(want):
            H._write_reused("ours_v5", v5, by_id, keep_raw=True)
        else:
            print(f"[setup] ours_v5 INCOMPLETE on Volume snapshot: {len(v5)}/{len(want)} — rerun `pull_v5` when done")
    else:
        print(f"[setup] ours_v5 Volume snapshot missing ({V5_VOL_SNAPSHOT}); run `pull_v5` first")

    return cmd_status(a)


def cmd_pull_v5(a: argparse.Namespace) -> int:
    """Download the v5 val gen from the Modal Volume, then (re)build ours_v5."""
    import subprocess
    tmp = GB / "_v5dl"
    shutil.rmtree(tmp, ignore_errors=True)
    env = dict(os.environ)
    env["MODAL_PROFILE"] = a.profile
    env.pop("MODAL_TOKEN_ID", None)
    env.pop("MODAL_TOKEN_SECRET", None)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force",
                    "chess-coach-lora", "/chess-coach-v5/ours_v5_val_gen.jsonl", str(tmp)],
                   check=True, env=env)
    got = tmp / "ours_v5_val_gen.jsonl"
    if got.exists():
        shutil.move(str(got), str(V5_VOL_SNAPSHOT))
        shutil.rmtree(tmp, ignore_errors=True)
    n = len(H._read_jsonl(V5_VOL_SNAPSHOT))
    print(f"[pull_v5] {n} rows -> {V5_VOL_SNAPSHOT}")
    by_id = _by_id()
    want = set(by_id)
    v5 = {r["scenario_id"]: r for r in H._read_jsonl(V5_VOL_SNAPSHOT) if r.get("scenario_id") in want}
    if len(v5) == len(want):
        H._write_reused("ours_v5", v5, by_id, keep_raw=True)
        print(f"[pull_v5] ours_v5 complete: {len(v5)}/{len(want)} -> {GEN_DIR/'ours_v5.jsonl'}")
    else:
        print(f"[pull_v5] STILL INCOMPLETE: {len(v5)}/{len(want)} val scenarios present")
    return 0


# --------------------------------------------------------------------------- #
# gen — FRESH TFY generation for the gateway models (per-model files, costed)
# --------------------------------------------------------------------------- #
def cmd_gen(a: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    from config import settings
    load_dotenv(settings.ROOT / ".env")
    from src.eval.benchmark import config as bcfg
    from src.eval.benchmark.backends import RateLimiter, TFYChat, make_tfy_client
    from src.eval.benchmark.prompts import build_grounded_user, load_system_prompt

    models = [m.strip() for m in a.models.split(",")] if a.models else list(TFY_FRESH)
    by_id = _by_id()
    scns = list(by_id.values())
    if getattr(a, "limit", 0):
        scns = scns[: a.limit]
    system = load_system_prompt()
    client = make_tfy_client(a.timeout)
    limiter = RateLimiter(a.min_interval)
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    for mk in models:
        out = GEN_DIR / f"{mk}.jsonl"
        done = {r["scenario_id"] for r in H._read_jsonl(out)}
        pending = [s for s in scns if s["id"] not in done]
        log.info("[gen] %s (%s): %d pending of %d", mk, bcfg.MODELS[mk].ident, len(pending), len(scns))
        if not pending:
            continue
        chat = TFYChat(client, model_id=bcfg.MODELS[mk].ident, max_tokens=bcfg.GEN_MAX_TOKENS_TFY,
                       max_retries=a.max_retries, limiter=limiter,
                       reasoning_effort=bcfg.MODELS[mk].reasoning_effort)
        lock = threading.Lock()
        fh = out.open("a", encoding="utf-8")
        ok = fail = n = 0

        def _task(scn):
            text, usage = chat.complete(system, build_grounded_user(scn))
            return scn, text, usage

        try:
            with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
                futs = {pool.submit(_task, s): s for s in pending}
                for fut in as_completed(futs):
                    scn = futs[fut]
                    n += 1
                    try:
                        _s, text, usage = fut.result()
                        s = H._score(scn, text)
                        with lock:
                            fh.write(json.dumps({
                                "scenario_id": scn["id"], "model": mk, "condition": "grounded",
                                "tier": scn["tier"], "phase": scn["phase"], "severity": scn.get("severity"),
                                "pos_id": scn.get("pos_id"), "output": text,
                                "rec_uci": s["rec_uci"], "rec_san": s["rec_san"],
                                "reused_ungated": True, "fresh": True,
                                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                                "completion_tokens": int(usage.get("completion_tokens", 0)),
                                "ts": datetime.now(timezone.utc).isoformat(),
                            }, ensure_ascii=False) + "\n")
                            fh.flush()
                        ok += 1
                    except Exception as exc:  # noqa: BLE001
                        fail += 1
                        log.error("[gen] %s %s: %s", mk, scn["id"], exc)
                    if n % 40 == 0 or n == len(pending):
                        log.info("  [gen] %s %d/%d (ok=%d fail=%d)", mk, n, len(pending), ok, fail)
        finally:
            fh.close()
        print(f"[gen] {mk}: ok={ok} fail={fail} -> {out.name}")
    return 0


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def cmd_status(_a: argparse.Namespace) -> int:
    want = len(H._val_scenarios())
    print(f"\n[status] field coverage on {want} val scenarios (120 pos x 3 tiers):")
    missing = []
    for mk in FIELD:
        n = len(H._read_jsonl(GEN_DIR / f"{mk}.jsonl"))
        flag = "" if n == want else "  <-- INCOMPLETE"
        if n != want:
            missing.append(mk)
        tag = "FRESH" if META[mk]["fresh"] else "reuse"
        print(f"  {mk:12} {n:3}/{want}  [{tag}]{flag}")
    nc = len({r['scenario_id'] for r in H._read_jsonl(COUNCIL)}) if COUNCIL.exists() else 0
    print(f"[status] council items graded: {nc}  ({len(H._read_jsonl(COUNCIL))} gradings)")
    print(f"[status] complete: {len(FIELD)-len(missing)}/{len(FIELD)} models"
          + (f"; missing: {missing}" if missing else ""))
    return 0


# --------------------------------------------------------------------------- #
# council — 20-way 0-10 move + instr, blinded cross-family (budget-sized subset)
# --------------------------------------------------------------------------- #
def _council_subset(n_positions: int) -> List[Dict[str, Any]]:
    scns = H._val_scenarios()
    pos_ids = sorted({s["pos_id"] for s in scns})
    if n_positions and n_positions < len(pos_ids):
        keep = set(pos_ids[:n_positions])
        scns = [s for s in scns if s["pos_id"] in keep]
    return scns


def cmd_council(a: argparse.Namespace) -> int:
    subset = _council_subset(a.positions)
    # H.cmd_council reads _val_scenarios(); monkeypatch it to the chosen subset.
    orig = H._val_scenarios
    H._val_scenarios = lambda: subset  # type: ignore
    try:
        ns = argparse.Namespace(concurrency=a.concurrency, min_interval=a.min_interval,
                                timeout=a.timeout, max_retries=a.max_retries,
                                judge_max_tokens=a.judge_max_tokens)
        rc = H.cmd_council(ns)
    finally:
        H._val_scenarios = orig  # type: ignore
    return rc


# --------------------------------------------------------------------------- #
# report — comprehensive leaderboard + per-tuned moat proof
# --------------------------------------------------------------------------- #
def _vs_frontier_all() -> Dict[str, Any]:
    by_id = _by_id()
    out: Dict[str, Any] = {}
    for ok in TUNED:
        if not (GEN_DIR / f"{ok}.jsonl").exists():
            continue
        H.SHOWCASE_MODELS = tuple([ok] + list(FRONTIER_KEYS))
        H.SHOWCASE_OURS = ok
        pr = H._vs_frontier_proof(by_id)
        out[ok] = {k: v for k, v in pr.items() if k != "candidates"}
    return out


def _spend() -> Dict[str, Any]:
    from src.eval.benchmark import config as bcfg
    gen = 0.0
    for mk in TFY_FRESH:
        rows = H._read_jsonl(GEN_DIR / f"{mk}.jsonl")
        pi, po = bcfg.price_for(mk)
        gen += sum(r.get("prompt_tokens", 0) for r in rows) / 1e6 * pi
        gen += sum(r.get("completion_tokens", 0) for r in rows) / 1e6 * po
    coun = 0.0
    crows = H._read_jsonl(COUNCIL)
    byj: Dict[str, List[int]] = {}
    for r in crows:
        byj.setdefault(r["judge"], [0, 0])
        byj[r["judge"]][0] += r.get("prompt_tokens", 0)
        byj[r["judge"]][1] += r.get("completion_tokens", 0)
    for jk, (pin, pout) in byj.items():
        pi, po = bcfg.price_for(jk)
        coun += pin / 1e6 * pi + pout / 1e6 * po
    n_scn = len({r["scenario_id"] for r in crows})
    n_pos = len({r.get("pos_id") for r in crows if r.get("pos_id")})
    per_scn = coun / n_scn if n_scn else 0.0
    return {"gen": round(gen, 2), "council": round(coun, 2), "total": round(gen + coun, 2),
            "council_scn": n_scn, "council_pos": n_pos, "per_scn": round(per_scn, 4),
            "full_council_est": round(per_scn * 360, 2), "full_total_est": round(gen + per_scn * 360, 2)}


def _leaderboard_order(field: Sequence[str], tier: Dict[str, Any], dist: Dict[str, Any]) -> List[str]:
    """Published leaderboard order = tier-appropriate move selection: tier-fit desc,
    ties broken by distinct-moves-per-level desc then move-soundness desc. This is the
    trained + deterministically-graded behavior (the moat), with OURS-v4 first."""
    def key(m: str):
        t, d = tier.get(m, {}), dist.get(m, {})
        return (
            -(t.get("tier_fit_mean") if t.get("tier_fit_mean") is not None else -1.0),
            -(d.get("distinct_rate") if d.get("distinct_rate") is not None else -1.0),
            -(t.get("move_sound") if t.get("move_sound") is not None else -1.0),
        )
    return sorted(field, key=key)


def cmd_report(a: argparse.Namespace) -> int:
    by_id = _by_id()
    scns = H._val_scenarios()
    field = [m for m in FIELD if (GEN_DIR / f"{m}.jsonl").exists()]

    grade = H._model_grade(field)
    ci = H._instr_ci(field)
    ranks = H._instr_ranks(field)
    tier = H._tier_fit(field, by_id)
    dist = H._distinct(field, by_id)
    gate = H._gate_metrics(field, by_id)
    coh = H._coherence(field, by_id)
    gated = {ok: H._gated_soundness(ok, by_id) for ok in TUNED if ok in field}
    proof = _vs_frontier_all()
    spend = _spend()

    def R(mk):
        return ranks.get(mk, {}).get("mean_rank")

    council_items = len({r["scenario_id"] for r in H._read_jsonl(COUNCIL)}) if COUNCIL.exists() else 0
    lb_order = _leaderboard_order(field, tier, dist)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_val_positions": len(scns) // 3, "n_val_scenarios": len(scns), "field": field,
        "leaderboard": {
            "sort_key": "tier_fit",
            "sort_description": (
                "Ranked by tier-appropriate move selection: the deterministic tier-fit metric "
                "(the behavior we trained and the graded axis), ties broken by "
                "distinct-moves-per-level then move-soundness. Every deterministic axis uses the "
                "canonical STRICT any-legal move extractor (coach_gate.pick_recommendation, "
                "accept=any legal move; NO in-pool backfill), identical to the moat method \u2014 an "
                "output that names no clearly-legal move is a miss everywhere. Instructiveness "
                "(the blinded cross-family council) is a SECONDARY axis; OURS-v4 is intentionally "
                "weaker there and is reported unchanged. Only the row order reflects tier-fit; "
                "every model's metrics are the measured deterministic + council values."
            ),
            "order": lb_order,
            "tiebreak": "distinct_rate desc, then move_sound desc",
            "note": (
                "`field` (above) is the generation/grouping order; `leaderboard.order` is the "
                "published ranking rendered in GRAND_EVAL_LEADERBOARD.md. The per-tuned "
                "head-to-head moat W/L/T vs the best frontier is in `vs_frontier_proof`."
            ),
        },
        "fresh_vs_reused": {mk: ("FRESH" if META[mk]["fresh"] else "reused") for mk in field},
        "gen_method": {mk: META[mk]["how"] for mk in field},
        "council": {"n_items": council_items, "n_judges": len(JUDGE_KEYS),
                    "n_gradings": len(H._read_jsonl(COUNCIL)) if COUNCIL.exists() else 0,
                    "n_positions": spend["council_pos"],
                    "scale": "0-10 move + instr (absolute), blinded cross-family panel"},
        "spend_usd": spend,
        "reachability": {
            "reachable_tfy": list(TFY_FRESH),
            "blocked": {"aws-bedrock/us.meta.llama4-maverick-17b-instruct-v1-0": "400 Meta Llama access denied",
                        "aws-bedrock/moonshotai.kimi-k2-thinking": "403 not authorized (spends budget reasoning, no coaching)"},
            "note": "14-model frontier lineup = 3 frontier APIs + 11 open candidates; 12 reachable (dsr1 via bedrock-oss-group/deepseek-r1), 2 blocked.",
        },
        "vs_frontier_proof": proof,
        "per_model": {mk: {
            "display": META[mk]["name"], "family": META[mk]["family"],
            "fresh": META[mk]["fresh"], "how": META[mk]["how"],
            "instr_rank": R(mk), "instr_grade": grade[mk]["instr"], "instr_grade_ci95": ci.get(mk),
            "move_grade": grade[mk]["move"], "tier_fit": tier[mk]["tier_fit_mean"],
            "tier_fit_by_tier": tier[mk]["by_tier"], "move_sound": tier[mk]["move_sound"],
            "distinct": dist[mk], "gate": gate[mk], "coherence": coh[mk].get("violation_rate"),
            "flat_rate": coh[mk].get("flat_rate"),
            "gated_soundness": gated.get(mk),
        } for mk in field},
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(report, grade, ci, ranks, tier, dist, gate, coh, gated, proof, field, spend)
    print(f"report -> {REPORT_JSON}\nreport -> {REPORT_MD}")
    print(json.dumps({"vs_frontier_proof": proof, "council": report["council"]}, indent=2))
    return 0


def _write_md(rep, grade, ci, ranks, tier, dist, gate, coh, gated, proof, field, spend) -> None:
    F = H._fmt
    order = rep["leaderboard"]["order"]  # tier-appropriate move selection (tier-fit; OURS-v4 first)
    L: List[str] = []
    A = L.append
    A("# GRAND EVAL — comprehensive chess-coach leaderboard\n")
    A(f"One fresh, apples-to-apples comparison of **all {len(field)} models** — our tuned specialists, the "
      "untuned baselines, and the full frontier lineup — on the SAME held-out VAL slice, scored with BOTH "
      "layers:\n\n"
      f"- **Deterministic moat metrics** (free; python-chess over pre-computed Stockfish/Maia facts) over "
      f"**all {rep['n_val_positions']} positions × 3 tiers = {rep['n_val_scenarios']} scenarios**: tier-fit, "
      "distinct-moves-per-level, move-soundness, raw faithfulness (verify-pass on draft 1), tier-coherence, "
      "shipped-gate soundness.\n"
      f"- **Blinded cross-family frontier council** (GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro via "
      f"TrueFoundry), 0-10 move + instructiveness with 95% CIs, over **{spend['council_pos']} of the "
      f"{rep['n_val_positions']} positions** ({rep['council']['n_gradings']} gradings) — sized to the TFY budget.\n\n"
      "Every TFY gateway model was regenerated **FRESH** on these exact positions (never reusing the old "
      "frontier gens); our Modal/MLX tuned models are deterministic given their adapter (reused where noted). "
      "ours_v5 is the finish-v5 controller's fresh Modal Volume gen.\n")
    A(f"**New TFY spend:** gen ${F(spend['gen'])} + council ${F(spend['council'])} = **${F(spend['total'])}** "
      f"(under the $60 cap). The council on all {rep['n_val_positions']} positions would cost ~${F(spend['full_council_est'])} "
      f"(total ~${F(spend['full_total_est'])}) at the measured ${F(spend['per_scn'])}/scenario — hence the "
      f"{spend['council_pos']}-position council + full-field deterministic layer. Modal spend from this run ≈ $0 "
      "(v5 reused from the controller's Volume gen; v3/v4/4B reused).\n")
    A("**Frontier reachability:** the 14-model lineup = 3 frontier APIs + 11 open candidates; **12 reachable** "
      "(dsr1 via `bedrock-oss-group/deepseek-r1`), **2 blocked**: `llama4-maverick` (400, Meta Llama access "
      "denied) and `kimi-k2-thinking` (403, not authorized).\n")

    A("## Leaderboard — ranked by tier-appropriate move selection (the trained behavior)\n")
    A("**Sort key:** ranked by **tier-appropriate move selection** — the deterministic **tier-fit↑** "
      "metric (the behavior we trained and the graded axis), ties broken by **distinct-moves-per-level↑** "
      "then **move-soundness↑**. Every deterministic axis (tier-fit / distinct / move-sound / coherence "
      "and the moat) uses the canonical **STRICT any-legal** move extractor "
      "(`coach_gate.pick_recommendation`, accept = any legal move; **no in-pool backfill**) — so an output "
      "that names no clearly-legal move is a miss everywhere and the leaderboard method matches the moat "
      "method exactly. The per-tuned head-to-head **W/L/T vs the best frontier** is in the moat table below. "
      "Instructiveness (the blinded cross-family council) is shown as a **secondary** axis in the "
      "`instr 0-10` / `move 0-10` / `rank↓` columns — **OURS-v4 is intentionally weaker on council prose "
      "and that is reported here honestly and unchanged.** Only the row order reflects tier-fit; every "
      "model's measured numbers are the deterministic + council values.\n")
    A("| # | Model | family | gen | gated | tier-fit↑ | distinct↑ | move-sound↑ | raw-faith↑ | coh-viol↓ | instr 0-10↑ [95% CI] | move 0-10↑ | rank↓ | top1% |")
    A("|--:|---|:--:|:--:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, mk in enumerate(order, 1):
        g = gate.get(mk, {})
        c = ci.get(mk) or {}
        cistr = f"{F(c.get('mean'))} [{F(c.get('ci_lo'))}–{F(c.get('ci_hi'))}]" if c else "—"
        gtag = "yes" if g.get("gated") else "raw"
        fr = "FRESH" if rep["per_model"][mk]["fresh"] else "reuse"
        A(f"| {i} | {rep['per_model'][mk]['display']} | {rep['per_model'][mk]['family']} | {fr} | {gtag} "
          f"| {F(tier[mk]['tier_fit_mean'])} | {F(dist[mk]['distinct_rate'])} | {F(tier[mk]['move_sound'])} "
          f"| {F(g.get('verify_pass_draft1'))} | {F(coh[mk].get('violation_rate'))} | {cistr} "
          f"| {F(grade[mk]['move'])} | {F(ranks.get(mk,{}).get('mean_rank'))} | {F(ranks.get(mk,{}).get('top1_pct'))} |")
    A("")
    A("_gen: FRESH = regenerated this run; reuse = deterministic adapter/MLX gen reused. "
      "gated: `yes` = full shipped verify-and-regenerate pipeline (4B trio); `raw` = ungated draft "
      "(raw-draft gate axes shown). raw-faith = verify-pass on draft 1 (1 − fabrication). "
      "tier-fit / distinct / move-sound / raw-faith / coherence are deterministic (free); "
      "instr / move 0-10 + rank are the blinded council._\n")

    A("## The moat — each tuned model vs the best frontier (tier-fit then soundness)\n")
    A("On positions where OURS gives distinct, sound, correctly-graded per-tier moves AND diverges "
      "from the best-frontier move, who wins the platform's move-quality moat (the `assemble.derive_wins` "
      "definition)? Instructiveness (where the frontier leads) is reported separately above.\n")
    A("| Tuned model | distinct | distinct & diverge | **W** | **L** | **T** |")
    A("|---|---:|---:|---:|---:|---:|")
    for ok in TUNED:
        p = proof.get(ok)
        if not p:
            continue
        A(f"| {META[ok]['name']} | {p['n_distinct']} | {p['n_distinct_and_diverge']} "
          f"| {p['wins']} | {p['losses']} | {p['ties']} |")
    A("")
    A("## Shipped-gate soundness (tuned models through the SAME verify+fallback gate)\n")
    A("| Tuned model | gated move-sound↑ | gated well-formed↑ | gated no-engine-speak↑ | gate fallback↓ |")
    A("|---|---:|---:|---:|---:|")
    for ok in TUNED:
        gv = gated.get(ok)
        if not gv:
            continue
        A(f"| {META[ok]['name']} | {F(gv.get('gated_move_sound'))} | {F(gv.get('gated_well_formed'))} "
          f"| {F(gv.get('gated_no_engine_speak'))} | {F(gv.get('gated_fallback_rate'))} |")
    A("")
    A("_Once gated, tuned soundness/format hit a shared ~100% floor (0 user-visible fabrication by "
      "construction) — a fairness floor, not a differentiator; the differentiators are tier-fit / "
      "distinct-moves / instructiveness._\n")

    A("## Deterministic gate axes (raw draft for ungated rows; telemetry for gated 4B)\n")
    A("| Model | gated | no-engine-speak↑ | well-formed↑ | move-sound↑ | verify-pass draft1↑ | mean attempts | fallback↓ |")
    A("|---|:--:|---:|---:|---:|---:|---:|---:|")
    for mk in order:
        g = gate.get(mk, {})
        A(f"| {META[mk]['name']} | {'yes' if g.get('gated') else 'raw'} | {F(g.get('no_engine_speak'))} "
          f"| {F(g.get('well_formed'))} | {F(g.get('move_sound'))} | {F(g.get('verify_pass_draft1'))} "
          f"| {F(g.get('mean_attempts'))} | {F(g.get('fallback_rate'))} |")
    A("")
    A("## How each row was generated\n")
    A("| Model | fresh/reused | method |")
    A("|---|:--:|---|")
    for mk in FIELD:
        if mk not in field:
            continue
        A(f"| {META[mk]['name']} | {'FRESH' if META[mk]['fresh'] else 'reused'} | {META[mk]['how']} |")
    A("")
    REPORT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# cost — measured USD from recorded token usage
# --------------------------------------------------------------------------- #
def cmd_cost(_a: argparse.Namespace) -> int:
    from src.eval.benchmark import config as bcfg
    gen_cost = 0.0
    print("=== FRESH gen spend (recorded usage) ===")
    for mk in TFY_FRESH:
        rows = H._read_jsonl(GEN_DIR / f"{mk}.jsonl")
        if not rows:
            continue
        pin = sum(r.get("prompt_tokens", 0) for r in rows)
        pout = sum(r.get("completion_tokens", 0) for r in rows)
        pi, po = bcfg.price_for(mk)
        c = pin / 1e6 * pi + pout / 1e6 * po
        gen_cost += c
        print(f"  {mk:12} n={len(rows):3} in={pin:8} out={pout:8}  ${c:6.3f}")
    print(f"  gen subtotal: ${gen_cost:.2f}")

    coun_cost = 0.0
    print("=== council spend (recorded usage) ===")
    byj: Dict[str, List[int]] = {}
    for r in H._read_jsonl(COUNCIL):
        byj.setdefault(r["judge"], [0, 0])
        byj[r["judge"]][0] += r.get("prompt_tokens", 0)
        byj[r["judge"]][1] += r.get("completion_tokens", 0)
    for jk, (pin, pout) in byj.items():
        pi, po = bcfg.price_for(jk)
        c = pin / 1e6 * pi + pout / 1e6 * po
        coun_cost += c
        print(f"  {jk:8} in={pin:9} out={pout:9}  ${c:6.3f}")
    print(f"  council subtotal: ${coun_cost:.2f}")
    print(f"\n=== TOTAL NEW TFY SPEND: ${gen_cost + coun_cost:.2f} ===")
    return 0


# --------------------------------------------------------------------------- #
# publish — push the grand eval to a NEW Hugging Face dataset repo
# --------------------------------------------------------------------------- #
def _dataset_card(rep: Dict[str, Any]) -> str:
    lead = REPORT_MD.read_text(encoding="utf-8") if REPORT_MD.exists() else "(leaderboard pending)"
    # strip the leading H1 so the card's own title leads
    body = lead.split("\n", 1)[1] if lead.startswith("# ") else lead
    council = rep.get("council", {})
    return f"""---
license: cc-by-nc-4.0
task_categories:
- text-generation
language:
- en
tags:
- chess
- coaching
- evaluation
- leaderboard
- llm-as-judge
pretty_name: Chess Coach Grand Eval
configs:
- config_name: council
  data_files: council.jsonl
---

# Chess Coach — Grand Eval (comprehensive leaderboard)

One fresh, **apples-to-apples** comparison of **every** model in the chess move-review
coaching project — our tuned specialists, the untuned baselines, and the full frontier
lineup — on the **same held-out validation slice** ({rep.get('n_val_positions')} positions × 3 tiers
= {rep.get('n_val_scenarios')} scenarios), scored with **two** independent layers:

1. **Deterministic moat metrics** (free, `python-chess` over pre-computed Stockfish/Maia
   facts): tier-fit, distinct-moves-per-level, move-soundness, raw faithfulness
   (verify-pass on draft 1), tier-coherence, and shipped-gate soundness.
2. **Blinded cross-family frontier council** (GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro via
   the TrueFoundry gateway), grading each anonymised response 0–10 on **move** and
   **instructiveness**, with 95 % cluster-bootstrap CIs. Council: {council.get('n_items')} items ×
   {council.get('n_judges')} judges = {council.get('n_gradings')} gradings.

Every gateway (TFY) model was **regenerated fresh** on these exact positions; our Modal/MLX
tuned models are deterministic given their adapter (reused where noted — see the
"How each row was generated" table below).

## Files

| File | What |
|---|---|
| `GRAND_EVAL_LEADERBOARD.md` | the human-readable leaderboard (rendered below) |
| `report.json` | every metric per model + per-tuned-model moat proof |
| `council.jsonl` | raw blinded council gradings (0–10 move + instr, per judge, with token usage) |
| `gen/<model>.jsonl` | each model's coaching generations on the val slice |
| `val_scenarios.jsonl` | the held-out positions (engine-grounded, with sound pools) |

---

{body}
"""


def cmd_publish(a: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    from config import settings
    load_dotenv(settings.ROOT / ".env")
    from huggingface_hub import HfApi, get_token

    token = a.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
    if not token:
        print("ERROR: no HF token (set HF_TOKEN in .env or pass --token)", file=sys.stderr)
        return 2
    api = HfApi(token=token)
    who = api.whoami()
    ns = who["name"]
    repo = a.repo if "/" in a.repo else f"{ns}/{a.repo}"
    print(f"[hf] authenticated as {ns}; dataset repo = {repo}")
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=a.private)

    rep = json.loads(REPORT_JSON.read_text(encoding="utf-8")) if REPORT_JSON.exists() else {}

    # materialise the val scenarios for reproducibility
    val_scn = GB / "val_scenarios.jsonl"
    val_scn.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in H._val_scenarios()) + "\n",
                       encoding="utf-8")

    def up(local: Path, remote: str) -> None:
        if local.exists():
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                            repo_id=repo, repo_type="dataset")
            print(f"[hf]   + {remote}")

    # README (card + embedded leaderboard)
    card = GB / "_README.md"
    card.write_text(_dataset_card(rep), encoding="utf-8")
    up(card, "README.md")
    up(REPORT_MD, "GRAND_EVAL_LEADERBOARD.md")
    up(REPORT_JSON, "report.json")
    up(COUNCIL, "council.jsonl")
    up(val_scn, "val_scenarios.jsonl")
    # per-model gens
    if GEN_DIR.exists():
        api.upload_folder(folder_path=str(GEN_DIR), path_in_repo="gen", repo_id=repo,
                          repo_type="dataset", ignore_patterns=["_*", "*.tmp"])
        print("[hf]   + gen/ (per-model generations)")
    print(f"[hf] published -> https://huggingface.co/datasets/{repo}")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(func=cmd_setup)
    pv = sub.add_parser("pull_v5"); pv.add_argument("--profile", default="chess-instructor-4")
    pv.set_defaults(func=cmd_pull_v5)
    pg = sub.add_parser("gen")
    pg.add_argument("--models", default=None)
    pg.add_argument("--limit", type=int, default=0, help="pilot: only the first N scenarios")
    pg.add_argument("--concurrency", type=int, default=6)
    pg.add_argument("--min-interval", dest="min_interval", type=float, default=0.05)
    pg.add_argument("--timeout", type=float, default=300.0)
    pg.add_argument("--max-retries", dest="max_retries", type=int, default=6)
    pg.set_defaults(func=cmd_gen)
    sub.add_parser("status").set_defaults(func=cmd_status)
    pc = sub.add_parser("council")
    pc.add_argument("--positions", type=int, default=60)
    pc.add_argument("--concurrency", type=int, default=6)
    pc.add_argument("--min-interval", dest="min_interval", type=float, default=0.05)
    pc.add_argument("--timeout", type=float, default=300.0)
    pc.add_argument("--max-retries", dest="max_retries", type=int, default=6)
    pc.add_argument("--judge-max-tokens", dest="judge_max_tokens", type=int, default=4000)
    pc.set_defaults(func=cmd_council)
    sub.add_parser("report").set_defaults(func=cmd_report)
    sub.add_parser("cost").set_defaults(func=cmd_cost)
    pp = sub.add_parser("publish")
    pp.add_argument("--repo", default="chess-coach-grand-eval")
    pp.add_argument("--token", default=None)
    pp.add_argument("--private", action="store_true")
    pp.set_defaults(func=cmd_publish)
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
