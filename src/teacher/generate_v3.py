"""v3 chess-coaching dataset generator — larger, contrastive, method-teaching.

Same label recipe as v2 (deterministic tier-aware move selection +
grounded, forced-move, 3-part method-teaching GPT-5.5 labels + a faithfulness
retry), but:

1. Reads the **new, larger** curated position bank
   ``data/positions/v3_candidates.jsonl`` (2,423 motif-tagged contrastive
   multi-tier positions whose Stockfish sound pools are ALREADY computed) instead
   of ``candidates_v1.jsonl``. Stockfish is not re-run; only **Maia** per-tier
   policy is computed so :func:`src.teacher.tier_select.select_tier_move` can pick
   the human-findable move per tier.
2. Routes the teacher through the **TrueFoundry gateway** (``openai-group/gpt-5.5``)
   — ``base_url=TFY_BASE_URL``, ``api_key=TFY_API_KEY`` — per the v3 task, instead
   of OpenAI-direct. The :class:`~src.teacher.generate.TeacherClient` and its
   chat/json_object/reasoning path are unchanged and proven working on the gateway.
3. Emits **contrastive triples** (same position at all 3 tiers) for every position
   whose per-tier picks differ (the moat signal), plus a single-tier row at the
   position's source tier for the non-discriminating rest (coverage without 3x cost).
   ``--all-triples`` forces a triple for every position.

No held-out reserve is carved out: the definitive eval is the 803 gap set, verified
this session to have ZERO overlap with ``v3_candidates`` (and with v2 train/valid).

Everything is v3-suffixed; nothing v1/v2 is touched. Resume by job id + a costed
ledger with a ``--max-cost`` guard, identical to v2.

CLI
---
    python -m src.teacher.generate_v3 plan                 # build+cache plan (no spend)
    python -m src.teacher.generate_v3 smoke                # 3 real generations (measures cost)
    python -m src.teacher.generate_v3 generate             # run all pending jobs (resumable)
    python -m src.teacher.generate_v3 generate --max-cost 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from config import settings
from src.teacher.generate import RateLimiter, TeacherClient
from src.teacher.generate_v2 import (
    TIER_ORDER,
    _engine_block,
    _job,
    _pick_for,
    generate_one,
    plan_summary,
    write_cost,
)
from src.teacher.generate_v2 import board_key as _board_key
from src.teacher.generate_v2 import _phase as _phase_of

log = logging.getLogger("teacher.generate_v3")

# --- Paths (all v3-suffixed) ----------------------------------------------- #
V3_POSITIONS = settings.POSITIONS / "v3_candidates.jsonl"
V3_CANDIDATES = settings.GENERATED / "candidates_v3.jsonl"
PLAN_PATH = settings.GENERATED / "plan_v3.jsonl"
COST_PATH = settings.GENERATED / "cost_v3.json"

SEED = 3407


# --------------------------------------------------------------------------- #
# Load the v3 position bank (sound pools precomputed) into "coachable" records
# --------------------------------------------------------------------------- #


def load_coachable() -> List[Dict[str, Any]]:
    """Load v3_candidates.jsonl into records generate_v2's job builders consume.

    v3_candidates schema -> rec:
      id           <- id
      fen          <- fen
      tier         <- source_tier   (the position's native tier)
      student_move <- played_move    ({san,uci,cp_loss,severity})
      sound_pool   <- sound_pool     ([{san,uci,cp,pv}], best-first)
      phase        <- phase (fallback: recompute)
      severity     <- played_move.severity
    """
    out: List[Dict[str, Any]] = []
    seen: set = set()
    with V3_POSITIONS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            fen = d.get("fen")
            pool = d.get("sound_pool") or []
            if not fen or not pool or fen in seen:
                continue
            seen.add(fen)
            pm = d.get("played_move") or {}
            out.append(
                {
                    "id": str(d.get("id")),
                    "fen": fen,
                    "tier": d.get("source_tier") or "intermediate",
                    "student_move": {
                        "san": pm.get("san"),
                        "uci": pm.get("uci"),
                        "cp_loss": int(pm.get("cp_loss", 0) or 0),
                        "severity": pm.get("severity", "none"),
                    },
                    "sound_pool": pool,
                    "severity": pm.get("severity", "none"),
                    "phase": d.get("phase") or _phase_of(fen),
                    "primary_motif": d.get("primary_motif"),
                }
            )
    return out


# --------------------------------------------------------------------------- #
# Plan building (contrastive triples for discriminating positions + singles)
# --------------------------------------------------------------------------- #


def build_plan(*, all_triples: bool, single_limit: Optional[int]) -> List[Dict[str, Any]]:
    coachable = load_coachable()
    log.info("v3 coachable positions (from v3_candidates): %d", len(coachable))

    rng = random.Random(SEED)
    rng.shuffle(coachable)

    contrastive_jobs: List[Dict[str, Any]] = []
    single_jobs: List[Dict[str, Any]] = []
    diff_hist: Counter = Counter()
    motif_hist: Counter = Counter()
    scanned = 0

    for rec in coachable:
        scanned += 1
        picks: Dict[str, Dict[str, Any]] = {}
        maia_by_tier: Dict[str, List[Dict[str, Any]]] = {}
        for tier in TIER_ORDER:
            pick, maia6 = _pick_for(rec["fen"], tier, rec["sound_pool"])
            picks[tier] = pick
            maia_by_tier[tier] = maia6
        distinct = len({picks[t]["uci"] for t in TIER_ORDER})
        diff_hist[distinct] += 1
        motif_hist[rec.get("primary_motif") or "none"] += 1

        if all_triples or distinct >= 2:
            for tier in TIER_ORDER:
                contrastive_jobs.append(
                    _job("contrastive", rec, tier, picks[tier], maia_by_tier[tier])
                )
        else:
            t = rec["tier"] if rec["tier"] in TIER_ORDER else "intermediate"
            single_jobs.append(_job("single", rec, t, picks[t], maia_by_tier[t]))

        if scanned % 300 == 0:
            log.info("  scanned %d/%d (contrastive_jobs=%d single_jobs=%d)",
                     scanned, len(coachable), len(contrastive_jobs), len(single_jobs))

    if single_limit is not None:
        single_jobs = single_jobs[:single_limit]

    plan = contrastive_jobs + single_jobs
    _write_plan(plan)
    n_pos = scanned
    n_disc = diff_hist[2] + diff_hist[3]
    log.info("plan: %d jobs (%d contrastive rows + %d single rows) over %d positions",
             len(plan), len(contrastive_jobs), len(single_jobs), n_pos)
    log.info("distinct-pick histogram (1/2/3 distinct tier picks): %s", dict(sorted(diff_hist.items())))
    log.info("discriminating positions (>=2 distinct picks): %d/%d (%.0f%%)",
             n_disc, n_pos, 100 * n_disc / max(1, n_pos))
    log.info("top motifs: %s", dict(motif_hist.most_common(10)))
    return plan


def _write_plan(plan: List[Dict[str, Any]]) -> None:
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PLAN_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for job in plan:
            fh.write(json.dumps(job, ensure_ascii=False) + "\n")
    os.replace(tmp, PLAN_PATH)


def load_plan() -> List[Dict[str, Any]]:
    if not PLAN_PATH.exists():
        return []
    out: List[Dict[str, Any]] = []
    with PLAN_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _done_ids(path: Path) -> set:
    ids: set = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line).get("id")))
            except json.JSONDecodeError:
                continue
    return ids


# --------------------------------------------------------------------------- #
# TFY teacher client
# --------------------------------------------------------------------------- #


def _make_teacher(args: argparse.Namespace) -> TeacherClient:
    load_dotenv(settings.ROOT / ".env")
    provider = args.provider
    if provider == "tfy":
        key = os.environ.get("TFY_API_KEY")
        base = os.environ.get("TFY_BASE_URL")
        model = args.model or os.environ.get("TFY_TEACHER_MODEL") or "openai-group/gpt-5.5"
        if not key or not base:
            raise SystemExit("BLOCKED: TFY_API_KEY / TFY_BASE_URL missing from ROOT/.env")
        client = OpenAI(api_key=key, base_url=base, timeout=args.timeout, max_retries=0)
    else:  # openai-direct fallback
        key = os.environ.get("OPENAI_API_KEY")
        model = args.model or os.environ.get("TEACHER_MODEL") or settings.TEACHER_MODEL
        if not key:
            raise SystemExit("BLOCKED: OPENAI_API_KEY missing from ROOT/.env")
        client = OpenAI(api_key=key, timeout=args.timeout, max_retries=0)
    limiter = RateLimiter(args.min_interval)
    log.info("teacher: provider=%s model=%s effort=%s", provider, model, args.reasoning_effort)
    return TeacherClient(client, model=model, reasoning_effort=args.reasoning_effort,
                         max_retries=args.max_retries, limiter=limiter)


# --------------------------------------------------------------------------- #
# Generation driver (resumable, costed)
# --------------------------------------------------------------------------- #


def run_generation(args: argparse.Namespace) -> int:
    plan = load_plan()
    if not plan:
        log.info("no plan found; building it now ...")
        plan = build_plan(all_triples=args.all_triples, single_limit=args.single_limit)

    done = _done_ids(V3_CANDIDATES)
    pending = [j for j in plan if j["job_id"] not in done]
    if args.smoke:
        pending = pending[: args.smoke_n]
    log.info("generation: %d pending of %d planned jobs (%d already done)",
             len(pending), len(plan), len(done))
    if not pending:
        log.info("nothing to do — all planned jobs generated.")
        return 0

    teacher = _make_teacher(args)

    stop = threading.Event()
    write_lock = threading.Lock()
    V3_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    out_fh = V3_CANDIDATES.open("a", encoding="utf-8")
    written = failed = skipped = 0
    t0 = time.time()

    def worker(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if stop.is_set():
            return None
        return generate_one(job, teacher)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {pool.submit(worker, j): j for j in pending}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001 - one bad job must not abort
                    failed += 1
                    log.error("job %s failed: %s", job["job_id"], exc)
                    continue
                if row is None:
                    skipped += 1
                    continue
                with write_lock:
                    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    written += 1
                    if written % 25 == 0:
                        cost = write_cost(teacher, args.price_in, args.price_out,
                                          {"written": written, "failed": failed}, )
                        log.info("  wrote %d (fail=%d skip=%d) est_cost=$%.2f (%.1fs)",
                                 written, failed, skipped, cost["est_cost_usd"], time.time() - t0)
                        if args.max_cost and cost["est_cost_usd"] >= args.max_cost:
                            log.warning("MAX-COST $%.2f reached; stopping new work.", args.max_cost)
                            stop.set()
    finally:
        out_fh.close()

    cost = write_cost(teacher, args.price_in, args.price_out,
                      {"written": written, "failed": failed, "skipped": skipped})
    log.info("done: wrote=%d failed=%d skipped=%d -> %s", written, failed, skipped, V3_CANDIDATES)
    log.info("teacher usage: %s", teacher.usage_summary(args.price_in, args.price_out))
    log.info("cost ledger -> %s (est $%.2f)", COST_PATH, cost["est_cost_usd"])
    if args.smoke and written:
        _print_smoke(written)
    return 0


def _print_smoke(n: int) -> None:
    print("\n" + "=" * 78 + "\nSMOKE — first generated v3 rows\n" + "=" * 78)
    for line in V3_CANDIDATES.read_text(encoding="utf-8").splitlines()[-n:]:
        d = json.loads(line)
        to = d["teacher_output"]
        print(f"\n# {d['id']}  tier={d['tier']}  pick_rank={d['meta']['pick_pool_rank']} "
              f"engine_best={d['meta']['pick_is_engine_best']}  retries={d['meta']['faith_retries']} "
              f"fabricated={d['meta']['fabricated_final']}")
        print(f"  MOVE: {to['recommended_move_san']}")
        print(f"  COACH: {to['coaching'][:240]}")
        print(f"  METHOD: {to['method'][:240]}")
        print(f"  TAKEAWAY: {to['takeaway'][:160]}")


# override generate_v2.write_cost's COST_PATH by writing to v3 path -----------
# (generate_v2.write_cost writes to its own module COST_PATH; we re-point here)
def _write_cost_v3(teacher: TeacherClient, price_in: float, price_out: float,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with teacher._usage_lock:  # noqa: SLF001
        cin = teacher.prompt_tokens / 1_000_000 * price_in
        cout = teacher.completion_tokens / 1_000_000 * price_out
        payload = {
            "calls": teacher.calls,
            "prompt_tokens": teacher.prompt_tokens,
            "completion_tokens": teacher.completion_tokens,
            "reasoning_tokens": teacher.reasoning_tokens,
            "est_cost_usd": round(cin + cout, 4),
            "price_in_per_m": price_in,
            "price_out_per_m": price_out,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    if extra:
        payload.update(extra)
    COST_PATH.parent.mkdir(parents=True, exist_ok=True)
    COST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# use the v3 cost path
write_cost = _write_cost_v3  # noqa: F811


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _add_gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=("tfy", "openai"), default="tfy")
    p.add_argument("--model", default=None)
    p.add_argument("--reasoning-effort", dest="reasoning_effort",
                   default=settings.TEACHER_REASONING_EFFORT)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--min-interval", dest="min_interval", type=float, default=0.02)
    p.add_argument("--max-retries", dest="max_retries", type=int, default=6)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--all-triples", dest="all_triples", action="store_true",
                   help="Emit a contrastive triple for EVERY position (else only discriminating).")
    p.add_argument("--single-limit", dest="single_limit", type=int, default=None)
    p.add_argument("--max-cost", dest="max_cost", type=float, default=None)
    p.add_argument("--price-in", dest="price_in", type=float, default=1.25)
    p.add_argument("--price-out", dest="price_out", type=float, default=10.0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v3 larger, contrastive, method-teaching generator.")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("plan", help="Build + cache the plan (local Maia; no teacher calls).")
    pp.add_argument("--all-triples", dest="all_triples", action="store_true")
    pp.add_argument("--single-limit", dest="single_limit", type=int, default=None)
    pp.add_argument("--price-in", dest="price_in", type=float, default=1.25)
    pp.add_argument("--price-out", dest="price_out", type=float, default=10.0)
    pp.add_argument("--est-in", type=int, default=2200)
    pp.add_argument("--est-out", type=int, default=1400)
    pp.set_defaults(func=cmd_plan)

    pg = sub.add_parser("generate", help="Run all pending jobs (resumable, costed).")
    _add_gen_args(pg)
    pg.set_defaults(func=lambda a: run_generation(_with_smoke(a, False)))

    psk = sub.add_parser("smoke", help="Generate a few real rows to measure cost/quality.")
    _add_gen_args(psk)
    psk.add_argument("--n", dest="smoke_n", type=int, default=3)
    psk.set_defaults(func=lambda a: run_generation(_with_smoke(a, True)))

    return p


def _with_smoke(args: argparse.Namespace, smoke: bool) -> argparse.Namespace:
    args.smoke = smoke
    if not hasattr(args, "smoke_n"):
        args.smoke_n = 3
    return args


def cmd_plan(args: argparse.Namespace) -> int:
    plan = build_plan(all_triples=args.all_triples, single_limit=args.single_limit)
    print("\n=== PLAN (v3) ===")
    print(plan_summary(plan))
    est = len(plan) * (args.est_in / 1e6 * args.price_in + args.est_out / 1e6 * args.price_out)
    print(f"\nestimated teacher cost @ ~{args.est_in} in / {args.est_out} out tokens/call: "
          f"${est:.2f} (verify with `smoke`)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
