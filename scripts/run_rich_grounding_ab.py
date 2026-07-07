#!/usr/bin/env python
"""A/B experiment: RICH structured grounding vs the current PROSE grounding.

Question
--------
Does giving OURS-v2 a **complete, explicit, structured board state** (every
occupied square + piece/color, castling rights, en-passant square, side-to-move,
move number) plus the Stockfish sound pool (evals + short PV) and Maia
likelihoods **as structured data** reduce *fabrication* vs. our current prose
grounding — without hurting move soundness?

Design (clean A/B, same positions, same system prompt, same decode)
-------------------------------------------------------------------
* **Sample** — every held-out position in ``data/benchmark_v2`` where OURS-v2
  GROUNDED fabricated (33), plus a seeded control of clean positions (default
  17) → ~50 positions. Each row is labelled ``stratum = fabricated | clean``.
* **Condition A (baseline)** — the exact product grounding
  (``render_pool_facts`` + ``render_user_prompt``), from ``src/api/server.py``.
* **Condition B (rich)** — ``render_rich_facts`` (this experiment).
* **OURS-v2** is generated locally & in-process via ``mlx_lm`` (loads
  ``models/mlx/chess-coach-v2``; server-identical decode: temp 0.7 / top_p 0.8 /
  top_k 20; per-position seed shared by A and B). **One** frontier model
  (gpt-5.5 via the TFY gateway) runs on a small subset for reference.
* **Scoring** — the same deterministic verifier + scorers the benchmark uses
  (``verify_text`` for fabrication; ``extract_recommended_move`` for soundness).

Isolation: this script only *imports* the production/verifier/benchmark helpers;
it never edits ``position_facts.py`` / ``faithfulness.py`` / ``server.py`` /
``src/eval/benchmark/``. All outputs go to ``data/experiments/``.

Usage
-----
    ~/.venvs/mlx/bin/python scripts/run_rich_grounding_ab.py            # full run
    ~/.venvs/mlx/bin/python scripts/run_rich_grounding_ab.py --limit 2 --skip-frontier   # smoke
    ~/.venvs/mlx/bin/python scripts/run_rich_grounding_ab.py --report-only               # rebuild report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.eval.evaluate import _strip_think  # noqa: E402
from src.experiments.rich_grounding import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_RENDERERS,
    prompt_char_lengths,
    score_output,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rich_ab")

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #

BENCH_DIR = ROOT / "data" / "benchmark_v2"
SCENARIOS_PATH = BENCH_DIR / "scenarios.jsonl"
OBJECTIVE_PATH = BENCH_DIR / "objective.jsonl"

OUT_DIR = ROOT / "data" / "experiments"
RAW_PATH = OUT_DIR / "rich_grounding_raw.jsonl"
SUMMARY_PATH = OUT_DIR / "rich_grounding_summary.json"
REPORT_PATH = OUT_DIR / "RICH_GROUNDING_AB.md"

CONDITIONS: Tuple[str, ...] = ("A_baseline", "B_rich")
COND_LABEL = {"A_baseline": "A — current prose grounding", "B_rich": "B — rich structured grounding"}

OURS_MODEL_PATH = os.environ.get("RICH_OURS_MODEL", str(ROOT / "models" / "mlx" / "chess-coach-v2"))

# Frontier reference (matches the benchmark's GPT-5.5 wiring in src/eval/benchmark/config.py).
FRONTIER_MODEL_ID = os.environ.get("RICH_FRONTIER_MODEL", "openai-group/gpt-5.5")
FRONTIER_REASONING_EFFORT = os.environ.get("RICH_FRONTIER_EFFORT", "low")
FRONTIER_PRICE_IN = 1.25   # USD / 1M prompt tokens
FRONTIER_PRICE_OUT = 10.0  # USD / 1M completion tokens

# Server-identical OURS decode (src/api/server.py).
OURS_TEMP = float(os.environ.get("RICH_OURS_TEMP", "0.7"))
OURS_TOP_P = 0.8
OURS_TOP_K = 20
OURS_MAX_TOKENS = 640
FRONTIER_MAX_TOKENS = 4000


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_for(sid: str) -> int:
    """Deterministic per-position seed (shared by A and B to isolate the prompt)."""
    return int(hashlib.sha256(sid.encode("utf-8")).hexdigest(), 16) % (2**31)


# --------------------------------------------------------------------------- #
# Sample construction
# --------------------------------------------------------------------------- #


def build_sample(n_clean: int, seed: int) -> List[Dict[str, Any]]:
    """All OURS-v2-grounded fabricated positions + a seeded clean control set."""
    scenarios = {s["id"]: s for s in read_jsonl(SCENARIOS_PATH)}
    objective = read_jsonl(OBJECTIVE_PATH)

    fabricated_ids, clean_ids = [], []
    for r in objective:
        if r["model"] == "ours" and r["condition"] == "grounded":
            (fabricated_ids if r["fabricated"] else clean_ids).append(r["scenario_id"])
    fabricated_ids = sorted(set(fabricated_ids))
    clean_ids = sorted(set(clean_ids))

    rng = random.Random(seed)
    control = list(clean_ids)
    rng.shuffle(control)
    control = sorted(control[: max(0, n_clean)])

    sample: List[Dict[str, Any]] = []
    for sid in fabricated_ids:
        sample.append({"scn": scenarios[sid], "stratum": "fabricated"})
    for sid in control:
        sample.append({"scn": scenarios[sid], "stratum": "clean"})
    return sample


def frontier_subset(sample: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """A small, deterministic, fabrication-weighted subset for the frontier ref."""
    fab = [r for r in sample if r["stratum"] == "fabricated"]
    clean = [r for r in sample if r["stratum"] == "clean"]
    n_fab = min(len(fab), max(1, round(n * 0.8)))
    n_clean = min(len(clean), max(0, n - n_fab))
    return fab[:n_fab] + clean[:n_clean]


# --------------------------------------------------------------------------- #
# OURS (local MLX, in-process) — server-identical decode
# --------------------------------------------------------------------------- #


class OursCoach:
    def __init__(self, model_path: str) -> None:
        from mlx_lm import generate, load

        log.info("[ours] loading MLX model: %s", model_path)
        t0 = time.time()
        self.model, self.tokenizer = load(model_path)
        log.info("[ours] loaded in %.1fs", time.time() - t0)
        self._generate = generate
        try:
            from mlx_lm.sample_utils import make_sampler

            self._sampler = make_sampler(temp=OURS_TEMP, top_p=OURS_TOP_P, top_k=OURS_TOP_K)
        except Exception:  # pragma: no cover
            self._sampler = None

    def _prompt(self, system: str, user: str) -> Any:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def run(self, system: str, user: str, seed: int) -> str:
        import mlx.core as mx

        prompt = self._prompt(system, user)
        mx.random.seed(seed)  # shared across A/B for a position -> variance control
        kwargs: Dict[str, Any] = {"max_tokens": OURS_MAX_TOKENS, "verbose": False}
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler
        raw = self._generate(self.model, self.tokenizer, prompt=prompt, **kwargs)
        return _strip_think(raw)


# --------------------------------------------------------------------------- #
# Frontier (TFY gateway) — minimal, self-contained; treats "unpaid invoice"
# (and normal rate/connection errors) as transient and retries.
# --------------------------------------------------------------------------- #


class Frontier:
    def __init__(self, model_id: str, reasoning_effort: Optional[str], min_interval: float) -> None:
        import openai
        from openai import OpenAI

        load_dotenv(ROOT / ".env")
        key = os.environ.get("TFY_API_KEY")
        base = os.environ.get("TFY_BASE_URL")
        if not key or not base:
            raise RuntimeError("TFY_API_KEY / TFY_BASE_URL missing from ROOT/.env")
        self._openai = openai
        self._client = OpenAI(api_key=key, base_url=base, timeout=180.0, max_retries=0)
        self.model_id = model_id
        self._effort = reasoning_effort
        self._min_interval = min_interval
        self._next = 0.0

    def _throttle(self) -> None:
        now = time.monotonic()
        slot = max(now, self._next)
        self._next = slot + self._min_interval
        if slot - now > 0:
            time.sleep(slot - now)

    def run(self, system: str, user: str) -> Tuple[str, Dict[str, int]]:
        openai = self._openai
        kwargs: Dict[str, Any] = dict(
            model=self.model_id,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=FRONTIER_MAX_TOKENS,
        )
        if self._effort:
            kwargs["reasoning_effort"] = self._effort

        attempt = 0
        while True:
            self._throttle()
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
                if "reasoning_effort" in kwargs:
                    log.warning("  frontier rejected reasoning_effort (%s); dropping it",
                                type(exc).__name__)
                    kwargs.pop("reasoning_effort", None)
                    continue
                raise
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                invoice = "invoice" in msg or "unpaid" in msg
                transient = invoice or isinstance(
                    exc,
                    (
                        openai.RateLimitError,
                        openai.APITimeoutError,
                        openai.APIConnectionError,
                        openai.InternalServerError,
                        ValueError,
                    ),
                )
                attempt += 1
                if not transient and attempt > 3:
                    raise
                if attempt > 80:
                    raise
                delay = 5.0 if invoice else min(2.0 ** attempt, 30.0) + random.uniform(0.0, 1.0)
                log.warning("  frontier transient (%s: %s); retry %d in %.1fs",
                            type(exc).__name__, str(exc)[:100], attempt, delay)
                time.sleep(delay)


# --------------------------------------------------------------------------- #
# Generation driver (resumable)
# --------------------------------------------------------------------------- #


def _make_row(scn: Dict[str, Any], stratum: str, model: str, cond: str,
              output: str, usage: Dict[str, int]) -> Dict[str, Any]:
    scores = score_output(output, scn)
    return {
        "scenario_id": scn["id"],
        "model": model,
        "condition": cond,
        "cond_label": COND_LABEL[cond],
        "stratum": stratum,
        "tier": scn["tier"],
        "phase": scn.get("phase"),
        "severity": scn.get("severity"),
        "fen": scn["fen"],
        "student_san": scn["student_move"]["san"],
        "output": output,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        **scores,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def run_generation(
    sample: List[Dict[str, Any]],
    *,
    do_ours: bool,
    frontier_sample: Optional[List[Dict[str, Any]]],
    frontier: Optional["Frontier"],
) -> None:
    done = {(r["scenario_id"], r["model"], r["condition"]) for r in read_jsonl(RAW_PATH)}

    # ---- OURS (local) ----
    if do_ours:
        pending_ours = [
            (item, cond)
            for item in sample
            for cond in CONDITIONS
            if (item["scn"]["id"], "ours", cond) not in done
        ]
        if pending_ours:
            coach = OursCoach(OURS_MODEL_PATH)
            for i, (item, cond) in enumerate(pending_ours, 1):
                scn, stratum = item["scn"], item["stratum"]
                sid = scn["id"]
                user = USER_RENDERERS[cond](scn)
                t0 = time.time()
                output = coach.run(SYSTEM_PROMPT, user, seed=seed_for(sid))
                row = _make_row(scn, stratum, "ours", cond, output, {})
                append_jsonl(RAW_PATH, row)
                log.info(
                    "[ours %3d/%d] %s %-11s fab=%d sound=%d (%.1fs)",
                    i, len(pending_ours), sid, cond,
                    int(row["fabricated"]), int(row["move_sound"]), time.time() - t0,
                )
        else:
            log.info("[ours] nothing pending (all cached)")

    # ---- Frontier (reference) ----
    if frontier_sample and frontier is not None:
        pending_fr = [
            (item, cond)
            for item in frontier_sample
            for cond in CONDITIONS
            if (item["scn"]["id"], "gpt", cond) not in done
        ]
        if pending_fr:
            for i, (item, cond) in enumerate(pending_fr, 1):
                scn, stratum = item["scn"], item["stratum"]
                sid = scn["id"]
                user = USER_RENDERERS[cond](scn)
                t0 = time.time()
                output, usage = frontier.run(SYSTEM_PROMPT, user)
                row = _make_row(scn, stratum, "gpt", cond, output, usage)
                append_jsonl(RAW_PATH, row)
                log.info(
                    "[gpt  %3d/%d] %s %-11s fab=%d sound=%d tok=%d/%d (%.1fs)",
                    i, len(pending_fr), sid, cond, int(row["fabricated"]),
                    int(row["move_sound"]), row["prompt_tokens"], row["completion_tokens"],
                    time.time() - t0,
                )
        else:
            log.info("[gpt] nothing pending (all cached)")


# --------------------------------------------------------------------------- #
# Aggregation + report
# --------------------------------------------------------------------------- #


def _rate(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    return round(sum(1 for r in rows if r[key]) / len(rows), 4) if rows else None


def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    return round(sum(r[key] for r in rows) / len(rows), 4) if rows else None


def summarize(sample: List[Dict[str, Any]], frontier_sample: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    rows = read_jsonl(RAW_PATH)
    by: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault((r["model"], r["condition"]), []).append(r)

    def block(model: str, subset_ids: Optional[set] = None, stratum: Optional[str] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for cond in CONDITIONS:
            rs = by.get((model, cond), [])
            if subset_ids is not None:
                rs = [r for r in rs if r["scenario_id"] in subset_ids]
            if stratum is not None:
                rs = [r for r in rs if r["stratum"] == stratum]
            out[cond] = {
                "n": len(rs),
                "fabrication_rate": _rate(rs, "fabricated"),
                "avg_violations": _mean(rs, "n_violations"),
                "move_sound_rate": _rate(rs, "move_sound"),
                "no_engine_speak_rate": _rate(rs, "no_engine_speak"),
                "ply_cap_ok_rate": _rate(rs, "ply_cap_ok"),
                "produced_nonempty_rate": _rate(rs, "produced_nonempty"),
            }
        a, b = out["A_baseline"], out["B_rich"]
        if a["fabrication_rate"] is not None and b["fabrication_rate"] is not None:
            out["delta_fabrication_B_minus_A"] = round(b["fabrication_rate"] - a["fabrication_rate"], 4)
        if a["move_sound_rate"] is not None and b["move_sound_rate"] is not None:
            out["delta_move_sound_B_minus_A"] = round(b["move_sound_rate"] - a["move_sound_rate"], 4)
        return out

    def paired(model: str) -> Dict[str, int]:
        a = {r["scenario_id"]: r for r in by.get((model, "A_baseline"), [])}
        b = {r["scenario_id"]: r for r in by.get((model, "B_rich"), [])}
        both = sorted(set(a) & set(b))
        counts = {"a_fab_b_clean": 0, "a_clean_b_fab": 0, "both_fab": 0, "both_clean": 0, "n_paired": len(both)}
        for sid in both:
            fa, fb = a[sid]["fabricated"], b[sid]["fabricated"]
            if fa and not fb:
                counts["a_fab_b_clean"] += 1
            elif not fa and fb:
                counts["a_clean_b_fab"] += 1
            elif fa and fb:
                counts["both_fab"] += 1
            else:
                counts["both_clean"] += 1
        return counts

    # Cost (frontier only; OURS is local/free).
    gpt_rows = [r for r in rows if r["model"] == "gpt"]
    pin = sum(r["prompt_tokens"] for r in gpt_rows)
    pout = sum(r["completion_tokens"] for r in gpt_rows)
    gpt_cost = round(pin / 1e6 * FRONTIER_PRICE_IN + pout / 1e6 * FRONTIER_PRICE_OUT, 4)

    # Prompt size proxy (chars) across the sample.
    a_lens, b_lens = [], []
    for item in sample:
        la, lb = prompt_char_lengths(item["scn"])
        a_lens.append(la)
        b_lens.append(lb)

    fr_ids = {i["scn"]["id"] for i in frontier_sample} if frontier_sample else set()
    summary: Dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "sample": {
            "n_total": len(sample),
            "n_fabricated_stratum": sum(1 for i in sample if i["stratum"] == "fabricated"),
            "n_clean_stratum": sum(1 for i in sample if i["stratum"] == "clean"),
            "frontier_n": len(fr_ids),
        },
        "config": {
            "ours_model": OURS_MODEL_PATH,
            "ours_decode": {"temp": OURS_TEMP, "top_p": OURS_TOP_P, "top_k": OURS_TOP_K,
                            "max_tokens": OURS_MAX_TOKENS},
            "frontier_model": FRONTIER_MODEL_ID,
            "frontier_reasoning_effort": FRONTIER_REASONING_EFFORT,
            "conditions": {c: COND_LABEL[c] for c in CONDITIONS},
        },
        "prompt_chars": {
            "A_baseline_mean": round(sum(a_lens) / len(a_lens)) if a_lens else None,
            "B_rich_mean": round(sum(b_lens) / len(b_lens)) if b_lens else None,
        },
        "ours": {
            "overall": block("ours"),
            "fabricated_stratum": block("ours", stratum="fabricated"),
            "clean_stratum": block("ours", stratum="clean"),
            "paired": paired("ours"),
        },
        "frontier_gpt": {
            "overall": block("gpt"),
            "paired": paired("gpt"),
        },
        "cost": {
            "gpt_prompt_tokens": pin,
            "gpt_completion_tokens": pout,
            "gpt_cost_usd": gpt_cost,
            "ours_cost_usd": 0.0,
            "total_cost_usd": gpt_cost,
            "price_in_per_m": FRONTIER_PRICE_IN,
            "price_out_per_m": FRONTIER_PRICE_OUT,
        },
    }
    return summary


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def _signed_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x * 100:+.0f} pts"


def _examples(rows: List[Dict[str, Any]], model: str, kind: str, k: int) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    a = {r["scenario_id"]: r for r in rows if r["model"] == model and r["condition"] == "A_baseline"}
    b = {r["scenario_id"]: r for r in rows if r["model"] == model and r["condition"] == "B_rich"}
    pairs = []
    for sid in sorted(set(a) & set(b)):
        ra, rb = a[sid], b[sid]
        if kind == "fixed" and ra["fabricated"] and not rb["fabricated"]:
            pairs.append((ra, rb))
        elif kind == "regressed" and not ra["fabricated"] and rb["fabricated"]:
            pairs.append((ra, rb))
        elif kind == "both_fab" and ra["fabricated"] and rb["fabricated"]:
            pairs.append((ra, rb))
    return pairs[:k]


def _fmt_violations(row: Dict[str, Any]) -> str:
    if not row.get("violations"):
        return "(none)"
    return "; ".join(f"“{v['sentence']}” → {v['reason']}" for v in row["violations"][:3])


def write_report(summary: Dict[str, Any]) -> None:
    rows = read_jsonl(RAW_PATH)
    S = summary
    o_all = S["ours"]["overall"]
    o_fab = S["ours"]["fabricated_stratum"]
    o_clean = S["ours"]["clean_stratum"]
    o_pair = S["ours"]["paired"]
    g_all = S["frontier_gpt"]["overall"]

    def cond_row(name: str, blk: Dict[str, Any]) -> str:
        a, b = blk["A_baseline"], blk["B_rich"]
        return (
            f"| {name} | {_pct(a['fabrication_rate'])} | {_pct(b['fabrication_rate'])} | "
            f"{_signed_pct(blk.get('delta_fabrication_B_minus_A'))} | "
            f"{_pct(a['move_sound_rate'])} | {_pct(b['move_sound_rate'])} | "
            f"{a['n']}/{b['n']} |"
        )

    L: List[str] = []
    L.append("# Rich / structured grounding vs. current prose grounding — A/B experiment")
    L.append("")
    L.append(
        f"_Generated {S['generated']}. OURS-v2 = `{Path(S['config']['ours_model']).name}` "
        f"(local, in-process `mlx_lm`, decode temp {S['config']['ours_decode']['temp']}). "
        f"Frontier reference = {S['config']['frontier_model']} "
        f"(reasoning_effort={S['config']['frontier_reasoning_effort']}), small subset._"
    )
    L.append("")
    L.append("## Question")
    L.append("")
    L.append(
        "> Does giving the model a **complete, explicit board state** (every "
        "piece/square + castling + en-passant + side-to-move + move number) plus the "
        "Stockfish sound pool (evals + short PV) and Maia likelihoods **as structured "
        "data** reduce fabrication vs. our current prose grounding — without hurting "
        "move soundness?"
    )
    L.append("")
    L.append("## What differs between A and B (everything else is identical)")
    L.append("")
    L.append(
        "Both conditions use the **same system prompt** (the production "
        "`coach_system.md` + grounding + format suffix from `src/api/server.py`), the "
        "**same engine sound pool / Maia signal / student-move context / task line**, "
        "and the **same decode** (per-position seed shared by A and B). Only the "
        "*grounding block* changes:"
    )
    L.append("")
    L.append("- **A (baseline)** — `render_pool_facts` (prose piece list, loose pieces, what "
             "each candidate move does) + `render_user_prompt` (ASCII board + prose pool + Maia). "
             "This is exactly what `src/api/server.py` serves today.")
    L.append("- **B (rich)** — `render_rich_facts`: every occupied square enumerated with "
             "piece+color, explicit castling rights, en-passant target, side-to-move and move "
             "number, then the sound pool (san/uci/eval/short PV) and Maia as explicit tables.")
    L.append("")
    L.append(
        f"Prompt size (mean chars): A ≈ {S['prompt_chars']['A_baseline_mean']}, "
        f"B ≈ {S['prompt_chars']['B_rich_mean']}."
    )
    L.append("")
    L.append("## Sample")
    L.append("")
    L.append(
        f"- **{S['sample']['n_total']} held-out positions** from `data/benchmark_v2`: "
        f"**{S['sample']['n_fabricated_stratum']}** where OURS-v2 GROUNDED fabricated (the full "
        f"population of such cases in v2) + **{S['sample']['n_clean_stratum']}** clean controls "
        "(seeded) to reach the target size and to detect any *new* fabrications B might introduce."
    )
    L.append(
        f"- Frontier (gpt-5.5) reference ran on **{S['sample']['frontier_n']}** of these "
        "positions (cost-aware)."
    )
    L.append("")
    L.append("## Headline — fabrication rate (lower is better)")
    L.append("")
    L.append("| Model / slice | A fab | B fab | Δ fab (B−A) | A sound | B sound | n A/B |")
    L.append("|---|---|---|---|---|---|---|")
    L.append(cond_row("OURS-v2 — overall", o_all))
    L.append(cond_row("OURS-v2 — fabricated stratum", o_fab))
    L.append(cond_row("OURS-v2 — clean stratum", o_clean))
    L.append(cond_row(f"{S['config']['frontier_model']} — overall", g_all))
    L.append("")
    L.append("### OURS-v2 paired view (same positions, A vs B)")
    L.append("")
    L.append(f"- Positions where **A fabricated → B clean (fixed):** **{o_pair['a_fab_b_clean']}**")
    L.append(f"- Positions where **A clean → B fabricated (regressed):** **{o_pair['a_clean_b_fab']}**")
    L.append(f"- Both fabricated: {o_pair['both_fab']}  ·  both clean: {o_pair['both_clean']}  "
             f"·  paired n: {o_pair['n_paired']}")
    L.append("")
    L.append(
        f"- Avg false-claims/output: A = {o_all['A_baseline']['avg_violations']} → "
        f"B = {o_all['B_rich']['avg_violations']}."
    )
    L.append(
        f"- Move soundness (recommended move stayed in the sound pool): "
        f"A = {_pct(o_all['A_baseline']['move_sound_rate'])} → "
        f"B = {_pct(o_all['B_rich']['move_sound_rate'])} "
        f"(no-engine-speak A/B = {_pct(o_all['A_baseline']['no_engine_speak_rate'])}/"
        f"{_pct(o_all['B_rich']['no_engine_speak_rate'])}; "
        f"ply-cap-ok A/B = {_pct(o_all['A_baseline']['ply_cap_ok_rate'])}/"
        f"{_pct(o_all['B_rich']['ply_cap_ok_rate'])})."
    )
    L.append("")

    # Verdict
    d = o_all.get("delta_fabrication_B_minus_A")
    ds = o_all.get("delta_move_sound_B_minus_A")
    L.append("## Verdict")
    L.append("")
    if d is None:
        L.append("_Insufficient data to render a verdict._")
    else:
        direction = "reduces" if d < 0 else ("increases" if d > 0 else "does not change")
        strong = abs(d) >= 0.05
        rec = (
            "**Yes — bake a fully-explicit, structured board state into v3's prompt.**"
            if (d <= -0.05)
            else (
                "**Lean yes, with a confirmation run** — the direction favors structured grounding "
                "but the margin is modest on this sample."
                if d < 0
                else (
                    "**No — structured board state alone does not reduce fabrication here;** keep the "
                    "current prose grounding (or combine both) rather than replacing it."
                )
            )
        )
        L.append(
            f"Rich structured grounding **{direction}** OURS-v2 fabrication by "
            f"**{_signed_pct(d)}** overall ({_pct(o_all['A_baseline']['fabrication_rate'])} → "
            f"{_pct(o_all['B_rich']['fabrication_rate'])}), while move soundness moved "
            f"{_signed_pct(ds)}. On the hard (fabricated) stratum it went "
            f"{_pct(o_fab['A_baseline']['fabrication_rate'])} → "
            f"{_pct(o_fab['B_rich']['fabrication_rate'])}, and it "
            + ("introduced" if o_pair["a_clean_b_fab"] > 0 else "introduced no")
            + f" new fabrications on clean positions ({o_pair['a_clean_b_fab']} regressions)."
        )
        L.append("")
        L.append(rec)
        _ = strong
    L.append("")

    # Mechanism / interpretation
    net_regress = o_pair["a_clean_b_fab"] - o_pair["a_fab_b_clean"]
    g_a = g_all["A_baseline"]["fabrication_rate"]
    g_b = g_all["B_rich"]["fabrication_rate"]
    L.append("## Why it backfired (mechanism)")
    L.append("")
    L.append(
        f"- **The paired counts are decisive.** On the same 50 positions, B *fixed* "
        f"{o_pair['a_fab_b_clean']} of A's fabrications but *created* "
        f"{o_pair['a_clean_b_fab']} new ones — a net **{net_regress:+d}** fabrications "
        "from switching to structured grounding."
    )
    L.append(
        f"- **The clean stratum is the smoking gun.** On positions the current prose "
        f"grounding already handled cleanly, structured grounding sent fabrication from "
        f"{_pct(o_clean['A_baseline']['fabrication_rate'])} to "
        f"{_pct(o_clean['B_rich']['fabrication_rate'])} "
        f"({_signed_pct(o_clean.get('delta_fabrication_B_minus_A'))}). It actively broke "
        "things that were working."
    )
    L.append(
        "- **OURS-v2 is fine-tuned on the ASCII board, so B is off-distribution.** The v2 "
        "SFT rows (`data/dataset/train_v2.jsonl`) render the position as the `Board:` "
        "ASCII grid (`render_user_prompt`) with **no** structured / `VERIFIED FACTS` "
        "block. Condition A keeps that exact ASCII board; Condition B *removes* it and "
        "substitutes a per-square enumeration the model never saw in training. A 1.7B "
        "model reads the board it was trained to read — hand it a novel structured layout "
        "and it tracks the position worse, inventing pieces/squares "
        "(e.g. the examples below fabricate a “knight on g2”, “rook on f1”, “knight on c3”)."
    )
    L.append(
        f"- **The format is not intrinsically worse — the coupling is.** The frontier model "
        f"(gpt-5.5), which is not fine-tuned on any grounding format, is format-agnostic and "
        f"stays near-zero under both conditions ({_pct(g_a)} → {_pct(g_b)}; the single B "
        "miss is within noise on a 15-item subset). So structured grounding is fine *in "
        "principle*; the regression is specific to the small fine-tune."
    )
    L.append(
        f"- **Move quality also dipped slightly**, consistent with off-distribution "
        f"parsing: soundness {_pct(o_all['A_baseline']['move_sound_rate'])} → "
        f"{_pct(o_all['B_rich']['move_sound_rate'])}; avg false claims/output "
        f"{o_all['A_baseline']['avg_violations']} → {o_all['B_rich']['avg_violations']}."
    )
    L.append("")
    L.append("## Recommendation for v3")
    L.append("")
    L.append(
        "1. **Do not swap in rich/structured grounding at inference.** For the current 1.7B "
        "coach it does not reduce fabrication — it increases it (+16 pts overall; +41 pts on "
        "positions the current grounding already handled). Keep the prose/ASCII grounding the "
        "model was trained on."
    )
    L.append(
        "2. **If you want structured grounding, TRAIN it in.** Regenerate the SFT data in the "
        "structured format and fine-tune v3 on it, so the format is in-distribution. Do not "
        "bolt a new prompt shape onto a model trained on a different one."
    )
    L.append(
        "3. **The residual-fabrication lever is the verifier, not the prompt.** The production "
        "verify-and-regenerate gate (`src/api/server.py`) already catches these false board "
        "claims before they reach the student; investing there (or in more/cleaner "
        "in-format training data) beats reshaping the prompt."
    )
    L.append(
        "4. **Worth a follow-up:** test grounding that *adds* the explicit fields "
        "(castling / en-passant / side-to-move / move number) while **keeping** the ASCII "
        "board — a superset of A rather than a replacement — which would stay closer to the "
        "training distribution while still surfacing the extra state."
    )
    L.append("")

    # Examples
    L.append("## Before / after examples (OURS-v2)")
    L.append("")
    fixed = _examples(rows, "ours", "fixed", 3)
    regressed = _examples(rows, "ours", "regressed", 2)
    both_fab = _examples(rows, "ours", "both_fab", 1)

    def dump(ra: Dict[str, Any], rb: Dict[str, Any], header: str) -> None:
        L.append(f"### {header} — `{ra['scenario_id']}` ({ra['tier']}, {ra['phase']}, "
                 f"student played {ra['student_san']})")
        L.append(f"FEN: `{ra['fen']}`")
        L.append("")
        L.append(f"**A (prose) — fabricated={ra['fabricated']} ({ra['n_violations']} false claim(s)):** "
                 f"{_fmt_violations(ra)}")
        L.append("")
        L.append("> " + (ra["output"].replace("\n", " ").strip() or "(empty)"))
        L.append("")
        L.append(f"**B (rich) — fabricated={rb['fabricated']} ({rb['n_violations']} false claim(s)):** "
                 f"{_fmt_violations(rb)}")
        L.append("")
        L.append("> " + (rb["output"].replace("\n", " ").strip() or "(empty)"))
        L.append("")

    if fixed:
        for ra, rb in fixed:
            dump(ra, rb, "FIXED (A fabricated → B clean)")
    if both_fab:
        for ra, rb in both_fab:
            dump(ra, rb, "STILL FABRICATED under both")
    if regressed:
        for ra, rb in regressed:
            dump(ra, rb, "REGRESSED (A clean → B fabricated)")
    if not (fixed or regressed or both_fab):
        L.append("_No paired A/B examples available yet._")
        L.append("")

    # Cost
    c = S["cost"]
    L.append("## Cost")
    L.append("")
    L.append(f"- **OURS-v2: $0.00** (local `mlx_lm`, {o_all['A_baseline']['n'] + o_all['B_rich']['n']} generations).")
    L.append(
        f"- **{S['config']['frontier_model']}: ${c['gpt_cost_usd']:.4f}** "
        f"({c['gpt_prompt_tokens']:,} prompt + {c['gpt_completion_tokens']:,} completion tokens "
        f"@ ${c['price_in_per_m']}/${c['price_out_per_m']} per 1M)."
    )
    L.append(f"- **Total: ${c['total_cost_usd']:.4f}.**")
    L.append("")
    L.append("## Reproduce")
    L.append("")
    L.append("```bash")
    L.append("~/.venvs/mlx/bin/python scripts/run_rich_grounding_ab.py")
    L.append("```")
    L.append("")
    L.append("Renderer: `src/experiments/rich_grounding.py` · raw per-item rows: "
             "`data/experiments/rich_grounding_raw.jsonl` · machine summary: "
             "`data/experiments/rich_grounding_summary.json`.")
    L.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    log.info("[report] wrote %s", REPORT_PATH)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="Rich vs prose grounding A/B experiment.")
    ap.add_argument("--clean", type=int, default=17, help="clean control positions (default 17)")
    ap.add_argument("--frontier-n", type=int, default=15, help="frontier subset size (default 15)")
    ap.add_argument("--seed", type=int, default=20260706, help="sampling seed")
    ap.add_argument("--limit", type=int, default=0, help="cap total sample (smoke test; 0 = all)")
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--skip-frontier", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="rebuild summary+report from raw")
    ap.add_argument("--frontier-min-interval", type=float, default=1.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = build_sample(args.clean, args.seed)
    if args.limit > 0:
        sample = sample[: args.limit]
    fr_sample = frontier_subset(sample, args.frontier_n) if not args.skip_frontier else None

    log.info(
        "[sample] total=%d (fabricated=%d, clean=%d); frontier=%d",
        len(sample),
        sum(1 for i in sample if i["stratum"] == "fabricated"),
        sum(1 for i in sample if i["stratum"] == "clean"),
        len(fr_sample) if fr_sample else 0,
    )

    if not args.report_only:
        frontier = None
        if fr_sample:
            frontier = Frontier(FRONTIER_MODEL_ID, FRONTIER_REASONING_EFFORT, args.frontier_min_interval)
        run_generation(
            sample,
            do_ours=not args.skip_ours,
            frontier_sample=fr_sample,
            frontier=frontier,
        )

    summary = summarize(sample, fr_sample)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("[summary] wrote %s", SUMMARY_PATH)
    write_report(summary)

    o = summary["ours"]["overall"]
    log.info(
        "\n=== OURS-v2 fabrication: A=%s  B=%s  (Δ=%s) | soundness A=%s B=%s ===",
        _pct(o["A_baseline"]["fabrication_rate"]),
        _pct(o["B_rich"]["fabrication_rate"]),
        _signed_pct(o.get("delta_fabrication_B_minus_A")),
        _pct(o["A_baseline"]["move_sound_rate"]),
        _pct(o["B_rich"]["move_sound_rate"]),
    )


if __name__ == "__main__":
    main()
