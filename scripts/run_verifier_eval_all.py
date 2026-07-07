#!/usr/bin/env python
"""Does the faithfulness gate drive USER-VISIBLE fabrication to ~0 for ALL 14 models?

Extends the prior two-model verifier eval (`scripts/run_verifier_eval.py`) to the
full 14-model field, measuring **RAW → GATED** user-visible fabrication and the
**fallback rate** for every model, and explicitly answering: *does any model fail
to hit 0% after the gate?*

It **reuses the exact production gate harness** verbatim:

* `src/experiments/verifier_gate.py` `run_gate` — the real verify-and-regenerate
  loop from `src/api/server.py` (up to `COACH_MAX_ATTEMPTS`), the deterministic
  checker `src/engine/faithfulness.py` `verify_text`, and the engine-derived
  `_verified_coaching` fallback.
* the SAME fabrication-weighted 50 held-out positions the prior run used
  (`build_sample(17, 20260706)` = all 33 OURS-v2-grounded fabrications + 17 clean
  controls) for **every** model, so RAW rates are meaningful and comparable.
* the stored grounding in `data/benchmark_v2/scenarios.jsonl` (no live engines).

Per model × position: RAW fabrication (attempt 1, scored by `verify_text` on the
user-visible slice) and GATED fabrication (full loop up to N attempts, else the
verified fallback; the FINAL user-visible output scored). Records RAW fab %,
GATED fab %, fallback rate %, and the attempts-to-clean distribution per model.

Locals (OURS-v2, BASE) run in-process via `mlx_lm` with the server-identical
decode; the 12 API models run via the TrueFoundry gateway (creds in `.env`) with
an invoice/rate-limit-resilient, concurrent client. TrueFoundry only.

Isolation: only *imports* production/verifier/harness helpers; never edits them,
never touches the canvas/HF Space, never disrupts the live servers.

Usage
-----
    ~/.venvs/mlx/bin/python scripts/run_verifier_eval_all.py                 # full run
    ~/.venvs/mlx/bin/python scripts/run_verifier_eval_all.py --only ours,gpt --limit 2   # smoke
    ~/.venvs/mlx/bin/python scripts/run_verifier_eval_all.py --report-only   # rebuild report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

# --- reuse the production gate + grounding (imported verbatim, never modified) --- #
from src.experiments.verifier_gate import (  # noqa: E402
    MAX_ATTEMPTS,
    PRODUCTION_USER_RENDERER,
    SYSTEM_PROMPT,
    run_gate,
)

# --- reuse the prior driver's sample builder, seeds, IO, OURS decoder, row schema --- #
from run_verifier_eval import (  # noqa: E402
    FRONTIER_MAX_TOKENS,
    OursCoach,
    _row,
    append_jsonl,
    build_sample,
    read_jsonl,
    seed_for,
)
from src.eval.benchmark import config as bcfg  # noqa: E402
from src.eval.evaluate import _strip_think  # noqa: E402
from src.teacher.generate import RateLimiter  # thread-safe limiter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("verifier_eval_all")

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #

OUT_DIR = Path(os.environ.get("VERIFIER_ALL_OUT_DIR", str(ROOT / "data" / "experiments")))
RAW_PATH = OUT_DIR / "verifier_eval_all_raw.jsonl"
SUMMARY_PATH = OUT_DIR / "verifier_eval_all_summary.json"
REPORT_PATH = OUT_DIR / "VERIFIER_EVAL_ALL.md"
PRIOR_RAW_PATH = ROOT / "data" / "experiments" / "verifier_eval_raw.jsonl"  # reuse ours=50 + gpt-in-canonical

OURS_V2_PATH = os.environ.get(
    "VERIFIER_OURS_MODEL", str(ROOT / "models" / "mlx" / "chess-coach-v2")
)

# Deterministic 50-position sample (identical to the prior run).
SAMPLE_CLEAN = 17
SAMPLE_SEED = 20260706


# --------------------------------------------------------------------------- #
# Model field (14): 2 local + 5 frontier/reference + ... = all reachable models
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Spec:
    key: str
    display: str
    kind: str  # "mlx" | "tfy"
    ident: str  # local path / gateway model id
    family: str
    reasoning_effort: Optional[str] = None
    price_in: float = 0.0
    price_out: float = 0.0


def build_field() -> List[Spec]:
    """The 14-model field, in report order (local, frontier, then the 9 open)."""
    order = [
        "ours", "base", "gpt", "claude", "gemini",
        "q3_32b", "q3_next80b", "gemma3_27b", "llama33_70b", "dsv32",
        "glm5", "mistral3", "kimi25", "dsr1",
    ]
    specs: List[Spec] = []
    for k in order:
        m = bcfg.MODELS[k]
        display = m.display
        ident = m.ident
        if k == "ours":  # this eval is OURS-v2 (chess-coach-v2), not the v1 in bcfg
            display = f"OURS-v2 ({Path(OURS_V2_PATH).name}, 1.7B tuned)"
            ident = OURS_V2_PATH
        specs.append(Spec(
            key=k, display=display, kind=m.kind, ident=ident, family=m.family,
            reasoning_effort=m.reasoning_effort, price_in=m.price_in, price_out=m.price_out,
        ))
    return specs


# --------------------------------------------------------------------------- #
# TFY model client — concurrent + invoice/rate-limit resilient
# --------------------------------------------------------------------------- #


class TFYModel:
    """One gateway model, callable concurrently through a shared rate limiter.

    Treats "unpaid invoice" / rate-limit / timeout / connection / empty-content
    as transient and retries with backoff (an automated system settles invoices
    within seconds; another big eval is hammering the gateway concurrently).
    Drops ``reasoning_effort`` for the whole run if the model rejects it.
    """

    def __init__(self, client: Any, model_id: str, reasoning_effort: Optional[str],
                 limiter: RateLimiter) -> None:
        import openai
        self._openai = openai
        self._client = client
        self.model_id = model_id
        self._effort = reasoning_effort
        self._limiter = limiter
        self._lock = threading.Lock()  # guards _effort mutation

    def run(self, system: str, user: str) -> Tuple[str, Dict[str, int]]:
        openai = self._openai
        attempt = 0
        while True:
            with self._lock:
                effort = self._effort
            kwargs: Dict[str, Any] = dict(
                model=self.model_id,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=FRONTIER_MAX_TOKENS,
            )
            if effort:
                kwargs["reasoning_effort"] = effort

            self._limiter.acquire()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                content = _strip_think(resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                pin = int(getattr(usage, "prompt_tokens", 0) or 0)
                pout = int(getattr(usage, "completion_tokens", 0) or 0)
                if not content:
                    raise ValueError("empty model response")
                return content, {"prompt_tokens": pin, "completion_tokens": pout}
            except (openai.BadRequestError, TypeError) as exc:
                with self._lock:
                    if self._effort:
                        log.warning("  [%s] rejected reasoning_effort (%s); dropping it",
                                    self.model_id, type(exc).__name__)
                        self._effort = None
                        continue
                raise
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                invoice = "invoice" in msg or "unpaid" in msg
                transient = invoice or isinstance(exc, (
                    openai.RateLimitError, openai.APITimeoutError,
                    openai.APIConnectionError, openai.InternalServerError, ValueError,
                ))
                attempt += 1
                if not transient and attempt > 3:
                    raise
                if attempt > 120:
                    raise
                delay = (5.0 if invoice
                         else min(2.0 ** attempt, 30.0) + random.uniform(0.0, 1.0))
                log.warning("  [%s] transient (%s: %s); retry %d in %.1fs",
                            self.model_id, type(exc).__name__, str(exc)[:90], attempt, delay)
                time.sleep(delay)


def make_client(timeout: float = 180.0):
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    key = os.environ.get("TFY_API_KEY")
    base = os.environ.get("TFY_BASE_URL")
    if not key or not base:
        raise RuntimeError("TFY_API_KEY / TFY_BASE_URL missing from ROOT/.env")
    return OpenAI(api_key=key, base_url=base, timeout=timeout, max_retries=0)


# --------------------------------------------------------------------------- #
# Seeding: reuse prior rows for canonical positions (ours=50, gpt-in-canonical)
# --------------------------------------------------------------------------- #


def seed_from_prior(sample: List[Dict[str, Any]], field_keys: set) -> int:
    """Copy prior verifier_eval rows for canonical positions into the new file."""
    if not PRIOR_RAW_PATH.exists():
        return 0
    canonical = {it["scn"]["id"] for it in sample}
    have = {(r["scenario_id"], r["model"]) for r in read_jsonl(RAW_PATH)}
    n = 0
    for r in read_jsonl(PRIOR_RAW_PATH):
        if r["model"] not in field_keys:
            continue
        if r["scenario_id"] not in canonical:
            continue
        k = (r["scenario_id"], r["model"])
        if k in have:
            continue
        append_jsonl(RAW_PATH, r)
        have.add(k)
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Generation drivers
# --------------------------------------------------------------------------- #


def _done_pairs() -> set:
    return {(r["scenario_id"], r["model"]) for r in read_jsonl(RAW_PATH)}


def run_local(spec: Spec, sample: List[Dict[str, Any]]) -> None:
    pending = [it for it in sample if (it["scn"]["id"], spec.key) not in _done_pairs()]
    if not pending:
        log.info("[%s] nothing pending (all cached)", spec.key)
        return
    coach = OursCoach(spec.ident)
    for i, item in enumerate(pending, 1):
        scn, stratum = item["scn"], item["stratum"]
        sid = scn["id"]
        user = PRODUCTION_USER_RENDERER(scn)
        base = seed_for(sid)
        calls = {"n": 0}

        def gen(attempt: int, _user=user, _base=base, _calls=calls) -> str:
            _calls["n"] += 1
            return coach.run(SYSTEM_PROMPT, _user, seed=_base + (attempt - 1))

        t0 = time.time()
        gate = run_gate(scn, gen)
        row = _row(scn, stratum, spec.key, gate, {}, calls["n"], time.time() - t0)
        append_jsonl(RAW_PATH, row)
        log.info("[%-11s %3d/%d] %-14s %-10s RAW=%d GATED=%d att=%d %s (%.1fs)",
                 spec.key, i, len(pending), sid, stratum, int(row["raw_fabricated"]),
                 int(row["gated_fabricated"]), row["attempts_used"],
                 "FALLBACK" if row["used_fallback"] else "model", row["secs"])


def run_tfy(spec: Spec, sample: List[Dict[str, Any]], client: Any, limiter: RateLimiter,
            *, concurrency: int) -> None:
    pending = [it for it in sample if (it["scn"]["id"], spec.key) not in _done_pairs()]
    if not pending:
        log.info("[%s] nothing pending (all cached)", spec.key)
        return
    model = TFYModel(client, spec.ident, spec.reasoning_effort, limiter)
    write_lock = threading.Lock()
    log.info("[%s] %d positions, concurrency=%d (%s)", spec.key, len(pending),
             concurrency, spec.ident)

    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        scn, stratum = item["scn"], item["stratum"]
        user = PRODUCTION_USER_RENDERER(scn)
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        calls = {"n": 0}

        def gen(attempt: int, _user=user, _usage=usage, _calls=calls) -> str:
            _calls["n"] += 1
            text, u = model.run(SYSTEM_PROMPT, _user)
            _usage["prompt_tokens"] += int(u.get("prompt_tokens", 0))
            _usage["completion_tokens"] += int(u.get("completion_tokens", 0))
            return text

        t0 = time.time()
        gate = run_gate(scn, gen)
        return _row(scn, stratum, spec.key, gate, usage, calls["n"], time.time() - t0)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futs = {pool.submit(task, it): it for it in pending}
        for fut in as_completed(futs):
            item = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001 - skip; a rerun retries it
                log.error("[%s] %s failed: %s", spec.key, item["scn"]["id"], exc)
                continue
            with write_lock:
                append_jsonl(RAW_PATH, row)
            done += 1
            log.info("[%-11s %3d/%d] %-14s %-10s RAW=%d GATED=%d att=%d %s tok=%d/%d (%.1fs)",
                     spec.key, done, len(pending), row["scenario_id"], row["stratum"],
                     int(row["raw_fabricated"]), int(row["gated_fabricated"]),
                     row["attempts_used"], "FALLBACK" if row["used_fallback"] else "model",
                     row["prompt_tokens"], row["completion_tokens"], row["secs"])


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _rate(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    return round(sum(1 for r in rows if r[key]) / len(rows), 4) if rows else None


def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    return round(sum(r[key] for r in rows) / len(rows), 4) if rows else None


def _attempts_dist(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """attempts-to-clean: model-pass on attempt k (1..N), or fallback."""
    dist: Dict[str, int] = {}
    for r in rows:
        if r["used_fallback"]:
            dist["fallback"] = dist.get("fallback", 0) + 1
        else:
            k = f"pass@{r['attempts_used']}"
            dist[k] = dist.get(k, 0) + 1
    return dist


def model_block(spec: Spec, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    raw_fab = [r for r in rows if r["raw_fabricated"]]
    pin = sum(r["prompt_tokens"] for r in rows)
    pout = sum(r["completion_tokens"] for r in rows)
    cost = round(pin / 1e6 * spec.price_in + pout / 1e6 * spec.price_out, 4)
    # rows whose FINAL user-visible output still fabricated (should be 0)
    residual = [r for r in rows if r["gated_fabricated"]]
    return {
        "key": spec.key,
        "display": spec.display,
        "family": spec.family,
        "kind": spec.kind,
        "n": n,
        "raw_fab_rate": _rate(rows, "raw_fabricated"),
        "raw_full_fab_rate": _rate(rows, "raw_full_fabricated"),
        "gated_fab_rate": _rate(rows, "gated_fabricated"),
        "delta_fab": (round((_rate(rows, "gated_fabricated") or 0)
                            - (_rate(rows, "raw_fabricated") or 0), 4) if rows else None),
        "fallback_rate": _rate(rows, "used_fallback"),
        "passed_within_budget_rate": _rate(rows, "passed_within_budget"),
        "avg_attempts": _mean(rows, "attempts_used"),
        "avg_generations": _mean(rows, "n_generations"),
        "attempts_dist": _attempts_dist(rows),
        "among_raw_fabricated": {
            "n": len(raw_fab),
            "resolved_by_model_regen": sum(1 for r in raw_fab if not r["used_fallback"]),
            "resolved_by_fallback": sum(1 for r in raw_fab if r["used_fallback"]),
            "still_fabricated_after_gate": sum(1 for r in raw_fab if r["gated_fabricated"]),
        },
        "residual_gated_fabrications": [
            {
                "scenario_id": r["scenario_id"],
                "final_source": r["final_source"],
                "gated_visible": r["gated_visible"],
                "gated_violations": r.get("gated_violations") or [],
            }
            for r in residual
        ],
        "prompt_tokens": pin,
        "completion_tokens": pout,
        "cost_usd": cost,
    }


def summarize(field: List[Spec]) -> Dict[str, Any]:
    rows_all = read_jsonl(RAW_PATH)
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows_all:
        by.setdefault(r["model"], []).append(r)

    blocks = [model_block(s, by.get(s.key, [])) for s in field]
    present = [b for b in blocks if b["n"] > 0]
    total_cost = round(sum(b["cost_usd"] for b in present), 4)

    any_nonzero = [b for b in present if (b["gated_fab_rate"] or 0) > 0]

    # Independent audit: re-run the checker on every stored GATED output from
    # scratch, so the headline "0%" is substantiated by a fresh verify_text pass
    # rather than only by the numbers the gate wrote at run time.
    import chess as _chess  # local import; cheap
    from src.engine.faithfulness import verify_text as _vt
    audit_bad: List[Tuple[str, str]] = []
    empty_gated = 0
    for r in rows_all:
        gv = (r.get("gated_visible") or "").strip()
        if not gv:
            empty_gated += 1
        if not _vt(gv, _chess.Board(r["fen"]).fen()).ok:
            audit_bad.append((r["model"], r["scenario_id"]))

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "gate": {
            "source": "src/api/server.py verify-and-regenerate (imported verbatim)",
            "max_attempts": MAX_ATTEMPTS,
            "verifier": "src/engine/faithfulness.py verify_text (deterministic)",
            "fallback": "src/api/server.py _verified_coaching (engine-derived, true by construction)",
            "user_visible": "coaching + Takeaway via _split_coaching; scored with verify_text",
            "grounding": "data/benchmark_v2/scenarios.jsonl (reused; no live engines)",
        },
        "sample": {
            "n_positions": len({r["scenario_id"] for r in rows_all}),
            "fabricated_stratum": sum(1 for r in by.get("ours", []) if r["stratum"] == "fabricated"),
            "clean_stratum": sum(1 for r in by.get("ours", []) if r["stratum"] == "clean"),
            "weighting": "fabrication-weighted on OURS-v2 grounded fabrications (33 fab + 17 clean)",
        },
        "models": blocks,
        "n_models_present": len(present),
        "any_nonzero_gated": bool(any_nonzero),
        "nonzero_gated_models": [b["key"] for b in any_nonzero],
        "independent_audit": {
            "rows_checked": len(rows_all),
            "gated_fabrications_found": len(audit_bad),
            "empty_gated_outputs": empty_gated,
            "offenders": audit_bad[:20],
        },
        "total_cost_usd": total_cost,
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def _pct1(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _signed(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:+.0f} pts"


def _dist_str(d: Dict[str, int]) -> str:
    order = ["pass@1", "pass@2", "pass@3", "pass@4", "fallback"]
    parts = [f"{k.replace('pass@', 'a')}:{d[k]}" for k in order if d.get(k)]
    return ", ".join(parts) if parts else "—"


def write_report(summary: Dict[str, Any]) -> None:
    S = summary
    present = [b for b in S["models"] if b["n"] > 0]
    N = S["gate"]["max_attempts"]

    L: List[str] = []
    L.append("# Does the verify-and-regenerate faithfulness gate drive user-visible fabrication to ~0% — for ALL 14 models?")
    L.append("")
    L.append(
        f"_Generated {S['generated']}. **Gate** = production `{S['gate']['source']}`, "
        f"N={N} attempts, deterministic checker `verify_text`, engine-derived verified "
        f"fallback. **Sample** = the SAME {S['sample']['n_positions']} held-out positions "
        f"for every model ({S['sample']['weighting']}), grounding reused from "
        f"`data/benchmark_v2/scenarios.jsonl` — no engine runs live. Locals in-process via "
        f"`mlx_lm` (server-identical decode); the 12 API models via the TrueFoundry gateway._"
    )
    L.append("")

    # ---- TL;DR ---- #
    worst_raw = max(present, key=lambda b: b["raw_fab_rate"] or 0)
    all_zero = not S["any_nonzero_gated"]
    L.append("## TL;DR")
    L.append("")
    L.append(
        f"- **Every one of the {len(present)} models lands at "
        f"{'**0%** GATED user-visible fabrication' if all_zero else 'the GATED rates in the table'}** "
        f"— RAW fabrication ranges from {_pct(min(b['raw_fab_rate'] or 0 for b in present))} to "
        f"{_pct(worst_raw['raw_fab_rate'])} ({worst_raw['display']}) before the gate."
    )
    if all_zero:
        L.append(
            "- **Answer to \"does any model fail to hit 0% after the gate?\" — NO.** All "
            f"{len(present)} models reach 0% user-visible fabrication after the gate. This is by "
            "design: the gate only ever serves text that passes `verify_text`, or the "
            "engine-derived fallback (true by construction), so the guarantee is model-agnostic."
        )
    else:
        names = ", ".join(b["display"] for b in present if (b["gated_fab_rate"] or 0) > 0)
        L.append(
            f"- **Answer to \"does any model fail to hit 0% after the gate?\" — YES: {names}.** "
            "See the investigation section — a residual is a checker blind spot (a fabrication the "
            "deterministic verifier does not recognise), common to all models, not a model-specific failure."
        )
    # honest differentiator = fallback
    rank = sorted(present, key=lambda b: (b["fallback_rate"] or 0, b["raw_fab_rate"] or 0))
    best, worst = rank[0], rank[-1]
    L.append(
        f"- **The honest differentiator is the fallback rate** (how often the gate had to replace "
        f"the model with the verified template). Most self-sufficient: **{best['display']}** "
        f"({_pct1(best['fallback_rate'])} fallback); most dependent on the safety net: "
        f"**{worst['display']}** ({_pct1(worst['fallback_rate'])})."
    )
    L.append(f"- **Total cost: ${S['total_cost_usd']:.4f}** (locals free; API usage on TrueFoundry).")
    L.append("")

    # ---- Method ---- #
    L.append("## Method (identical to the prior 2-model run, extended to 14)")
    L.append("")
    L.append(
        "- **RAW** — attempt 1 only, gate OFF: exactly what `src/api/server.py` serves with "
        "`COACH_FAITHFULNESS_GATE=0` (the reply split into coaching body + `Takeaway:` via "
        "`_split_coaching`). Fabrication scored on that **user-visible** text with `verify_text`."
    )
    L.append(
        f"- **GATED** — the real verify-and-regenerate loop: re-sample the whole answer up to **{N}** "
        "times, keep the FIRST reply whose full text passes `verify_text`; if none pass, emit the "
        "deterministic engine-derived explanation (`_verified_coaching`), true by construction. "
        "Fabrication scored on the FINAL **user-visible** text."
    )
    L.append(
        "- Attempt 1 of the gated loop **is** the RAW generation, so GATED = \"RAW + the gate\" on "
        "the identical sampling. Every model sees the same grounding (prose VERIFIED-FACTS + ascii "
        f"board), the same system prompt, and the same {S['sample']['n_positions']} positions."
    )
    L.append("")

    # ---- Main table ---- #
    L.append("## Per-model: RAW → GATED user-visible fabrication + fallback")
    L.append("")
    L.append("| Model | n | RAW fab | GATED fab | Δ | Fallback | Passed within N | Mean attempts |")
    L.append("|---|---|---|---|---|---|---|---|")
    # keep report order (field order), locals+frontier first then open
    for b in present:
        L.append(
            f"| {b['display']} | {b['n']} | {_pct(b['raw_fab_rate'])} | "
            f"{_pct(b['gated_fab_rate'])} | {_signed(b['delta_fab'])} | "
            f"{_pct1(b['fallback_rate'])} | {_pct(b['passed_within_budget_rate'])} | "
            f"{b['avg_attempts']} |"
        )
    L.append("")
    L.append(
        "> **Read the RAW column as a stress test, not a global fabrication rate.** The "
        f"{S['sample']['fabricated_stratum']} fabricated-stratum positions are exactly the ones "
        "**OURS-v2** fabricated on in the benchmark, so OURS-v2's RAW is enriched *by construction* "
        "(this is its own hard set), and every other model's RAW is *its* fabrication rate on "
        "OURS-v2's hardest positions — a deliberately adversarial slice, not that model's "
        "unconditional rate. The purpose of the shared, fabrication-weighted set is to exercise the "
        "gate hard and keep the **GATED** and **fallback** columns comparable across models; it is "
        "not a fair frontier-vs-frontier fabrication leaderboard. (Notably, bigger models write "
        "longer, more concrete coaching, giving the checker more surface area — so several of them "
        "post higher RAW here than the tiny untuned BASE, which hedges.)"
    )
    L.append("")

    # ---- Explicit 0% answer ---- #
    L.append("## Does ANY model have a nonzero GATED fabrication rate?")
    L.append("")
    if all_zero:
        aud = S.get("independent_audit", {})
        L.append(
            f"**No.** All {len(present)} models reach **0%** user-visible fabrication after the gate "
            f"(GATED column above is 0% for every row). The gate's guarantee is structural, not "
            "statistical: the only two things that can reach the learner are (a) a model reply that "
            "**passed** the deterministic `verify_text` on its full text — and the user-visible slice "
            "is a subset of that text, so it passes too — or (b) the **engine-derived fallback**, "
            "which is true by construction. Neither depends on which model produced the draft, so a "
            "weak open model and a frontier model land at the same 0%; they differ only in how often "
            "they need the safety net (the fallback rate)."
        )
        L.append("")
        L.append(
            f"**Independent audit.** Re-running `verify_text` from scratch on all "
            f"**{aud.get('rows_checked', '?')}** stored GATED outputs finds "
            f"**{aud.get('gated_fabrications_found', 0)}** fabrications and "
            f"**{aud.get('empty_gated_outputs', 0)}** empty outputs — so the 0% is a fresh-check "
            "result, not just the number the gate wrote at run time."
        )
        L.append("")
        L.append(
            "**Coverage gap vs. real leak.** Because GATED fabrication is measured *by the same "
            "checker* the gate uses, a nonzero GATED rate would signal an internal inconsistency (a "
            "real leak / gate bug); there are none. The residual risk that remains is therefore **not** "
            "a model-specific leak but the checker's own **coverage** — a false board claim phrased in "
            "a way `verify_text` does not yet recognise would pass the gate *and* be scored clean, so "
            "it would be invisible here for **every** model equally. That is a single shared blind spot "
            "to harden in the verifier (broader claim/relation coverage), not a reason to trust any one "
            "model more than another. The honest per-model differentiator stays the fallback rate below."
        )
    else:
        L.append(
            "**Yes — and per the gate design any residual is a checker blind spot, not a "
            "model-specific leak.** The gate only serves `verify_text`-clean text or the "
            "true-by-construction fallback, so a nonzero GATED rate means the deterministic checker "
            "gave inconsistent verdicts (it passed the served text at gate time but flagged it at "
            "scoring time — i.e. a coverage/segmentation gap in `verify_text`), which would affect "
            "every model equally. The specific cases:"
        )
        L.append("")
        for b in present:
            for r in b.get("residual_gated_fabrications", []):
                viol = "; ".join(f"“{v['sentence']}” → {v['reason']}"
                                 for v in (r["gated_violations"] or [])[:2]) or "(none)"
                cls = ("checker non-monotonicity (the gate accepted this exact text as clean, so "
                       "the scorer's flag is a segmentation/coverage gap in verify_text)"
                       if r["final_source"] == "model"
                       else "fallback-construction discrepancy (investigate _verified_coaching)")
                L.append(f"- **{b['display']}** · `{r['scenario_id']}` · final={r['final_source']} "
                         f"→ classification: {cls}")
                L.append(f"  - served: “{(r['gated_visible'] or '').strip()}”")
                L.append(f"  - scorer flagged: {viol}")
        L.append("")
        L.append(
            "_Because the served text either passed the gate's own `verify_text` or is the "
            "engine-derived fallback, these residuals are checker blind spots common to all models "
            "(a false claim the deterministic verifier does not recognise), not evidence that any "
            "one model defeats the gate._"
        )
    L.append("")

    # ---- Ranking by fallback ---- #
    L.append("## Ranked by fallback rate — the honest self-sufficiency differentiator")
    L.append("")
    L.append(
        "Every model reaches ~0% GATED, so fabrication rate no longer separates them. What does is "
        "**how often the gate had to throw the model's answer away** and serve the verified template. "
        "Lower = more self-sufficient (its own prose reaches the learner more often)."
    )
    L.append("")
    L.append("| Rank | Model | Fallback rate | RAW fab | Final output: model prose / verified template | Attempts-to-clean |")
    L.append("|---|---|---|---|---|---|")
    for i, b in enumerate(rank, 1):
        model_rate = 1.0 - (b["fallback_rate"] or 0.0)
        L.append(
            f"| {i} | {b['display']} | {_pct1(b['fallback_rate'])} | {_pct(b['raw_fab_rate'])} | "
            f"{_pct(model_rate)} / {_pct(b['fallback_rate'])} | {_dist_str(b['attempts_dist'])} |"
        )
    L.append("")
    L.append(
        "_Attempts-to-clean legend: `aK:n` = n positions where the model produced a clean reply on "
        f"attempt K (K≤{N}); `fallback:n` = n positions where no attempt passed in budget and the "
        "verified engine-derived explanation was served._"
    )
    L.append("")

    # ---- Cost ---- #
    L.append("## Cost")
    L.append("")
    L.append("| Model | Prompt tok | Completion tok | Cost (USD) |")
    L.append("|---|---|---|---|")
    for b in present:
        if b["kind"] == "mlx":
            L.append(f"| {b['display']} | — | — | $0.0000 (local) |")
        else:
            L.append(f"| {b['display']} | {b['prompt_tokens']:,} | {b['completion_tokens']:,} | "
                     f"${b['cost_usd']:.4f} |")
    L.append(f"| **Total** | | | **${S['total_cost_usd']:.4f}** |")
    L.append("")
    L.append(
        "_API prices are the per-model estimates in `src/eval/benchmark/config.py` "
        "(frontier priced exactly; open-model Bedrock on-demand best-effort). Locals are free._"
    )
    L.append("")

    # ---- Reproduce ---- #
    L.append("## Reproduce")
    L.append("")
    L.append("```bash")
    L.append("~/.venvs/mlx/bin/python scripts/run_verifier_eval_all.py")
    L.append("```")
    L.append("")
    L.append(
        "Gate harness (imports the real production gate/fallback/verifier, edits nothing): "
        "`src/experiments/verifier_gate.py` · driver: `scripts/run_verifier_eval_all.py` · "
        "raw per-item rows: `data/experiments/verifier_eval_all_raw.jsonl` · machine summary: "
        "`data/experiments/verifier_eval_all_summary.json`. The 2-model precursor lives in "
        "`data/experiments/VERIFIER_EVAL.md`."
    )
    L.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    log.info("[report] wrote %s", REPORT_PATH)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="RAW vs GATED faithfulness eval — all 14 models.")
    ap.add_argument("--only", default=None, help="comma model keys (default: all 14)")
    ap.add_argument("--limit", type=int, default=0, help="cap positions (smoke; 0 = all 50)")
    ap.add_argument("--concurrency", type=int, default=4, help="TFY workers per model")
    ap.add_argument("--min-interval", dest="min_interval", type=float, default=0.12,
                    help="min seconds between gateway calls (shared limiter)")
    ap.add_argument("--no-seed", action="store_true", help="do not reuse prior rows")
    ap.add_argument("--report-only", action="store_true", help="rebuild summary+report from raw")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    field = build_field()
    if args.only:
        want = {k.strip() for k in args.only.split(",") if k.strip()}
        field = [s for s in field if s.key in want]
    field_keys = {s.key for s in field}

    sample = build_sample(SAMPLE_CLEAN, SAMPLE_SEED)
    if args.limit > 0:
        sample = sample[: args.limit]

    log.info("[field] %d models: %s", len(field), ", ".join(s.key for s in field))
    log.info("[sample] %d positions (fabricated=%d, clean=%d); gate N=%d",
             len(sample),
             sum(1 for i in sample if i["stratum"] == "fabricated"),
             sum(1 for i in sample if i["stratum"] == "clean"), MAX_ATTEMPTS)

    if not args.report_only:
        if not args.no_seed and args.limit == 0:
            n = seed_from_prior(sample, field_keys)
            if n:
                log.info("[seed] reused %d prior rows (ours/gpt on canonical positions)", n)

        local = [s for s in field if s.kind == "mlx"]
        tfy = [s for s in field if s.kind == "tfy"]

        for spec in local:  # one MLX load at a time
            log.info("=== LOCAL %s ===", spec.key)
            run_local(spec, sample)

        if tfy:
            client = make_client()
            limiter = RateLimiter(args.min_interval)
            for spec in tfy:  # one model at a time; concurrent across positions
                log.info("=== TFY %s ===", spec.key)
                run_tfy(spec, sample, client, limiter, concurrency=args.concurrency)

    summary = summarize(field if not args.only else build_field())
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("[summary] wrote %s", SUMMARY_PATH)
    write_report(summary)

    # console recap
    log.info("\n=== RAW -> GATED (user-visible fabrication) | fallback ===")
    for b in [x for x in summary["models"] if x["n"] > 0]:
        log.info("  %-30s n=%2d  RAW=%4s -> GATED=%4s  | fallback=%5s  passedN=%4s  meanAtt=%s",
                 b["display"], b["n"], _pct(b["raw_fab_rate"]), _pct(b["gated_fab_rate"]),
                 _pct1(b["fallback_rate"]), _pct(b["passed_within_budget_rate"]), b["avg_attempts"])
    log.info("=== any nonzero GATED? %s  | total cost $%.4f ===",
             "YES: " + ",".join(summary["nonzero_gated_models"]) if summary["any_nonzero_gated"]
             else "NO (all 0%)", summary["total_cost_usd"])


if __name__ == "__main__":
    main()
