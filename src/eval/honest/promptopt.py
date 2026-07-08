"""The hard test: automated "train by prompting" — iterate a base SYSTEM PROMPT.

The litmus for the whole project is: *a well-prompted base can't already do this
reliably*. To test it honestly we actually TRY to make the untuned base match the
tune by prompt engineering alone. This module runs a small automated loop:

    propose prompt -> score on a held-out DEV slice -> refine -> repeat -> keep best

* **Score** (objective, mirrors the eval's balanced weighting): tier-appropriate
  move selection (pick == :func:`select_tier_move`) 0.45 + instructiveness (a
  single frontier judge grades the six-dim rubric, absolute 0-12) 0.45 + a
  practical floor (no engine-speak, and the model produced a faithful answer
  without falling back to the deterministic template) 0.10, all in 0-100.
* **Refine**: a frontier "prompt engineer" model sees the current prompt, its DEV
  breakdown, and the worst failing examples, and proposes an improved SYSTEM
  PROMPT — constrained to keep the hard rules (no engine-speak, sound move only,
  ply cap, no fabrication, end with a Takeaway).

Everything is gated (the shipped :func:`src.teacher.coach_gate.run_gate`) so the
prompt-base is measured through the SAME pipeline as the base and the tune — the
only differences are the weights (base) and the system prompt (engineered). The
DEV slice is deliberately small and held out from the validation slice, and gens
are cached by (prompt, scenario), so the loop is cheap and resumable.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.eval.evaluate import _extract_json_object, find_engine_speak
from src.eval.honest.gated import RunFn, gated_row
from src.eval.honest.rubric import SIX_DIMS, _DIM_DESC

log = logging.getLogger("honest.promptopt")

# Objective weights (mirror the eval's balanced score).
W_TIER = 0.45
W_INSTR = 0.45
W_PRACTICAL = 0.10


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Single-response absolute instructiveness judge (cheap; used only in the loop)
# --------------------------------------------------------------------------- #

_ABS_SYSTEM = (
    "You are a strict, fair chess-coaching grader. You will see a position, the student's "
    "move, verified reference facts, and ONE coaching response for a student at a stated "
    "rating tier. Score the response on these six 0/1/2 dimensions (0 = absent, 1 = partial, "
    "2 = clearly present):\n"
    + "\n".join(f"- {d}: {_DIM_DESC[d]}" for d in SIX_DIMS)
    + "\n\nThe verified facts are for grading only. Return ONLY a JSON object: "
    '{"scores": {' + ", ".join(f'"{d}": 0' for d in SIX_DIMS) + '}, "note": "<short>"}'
)


def _abs_user(scn: Dict[str, Any], text: str) -> str:
    import chess

    from config import schema, settings
    from src.engine.position_facts import render_pool_facts
    from src.eval.benchmark.prompts import scenario_to_teacher_input

    ti = scenario_to_teacher_input(scn)
    facts = render_pool_facts(scn["fen"], ti["sound_pool"])
    sound = ", ".join(m["san"] for m in scn["sound_pool"])
    board = chess.Board(scn["fen"])
    t = settings.TIERS[scn["tier"]]
    return (
        f"STUDENT TIER: {scn['tier']} ({t['low']}-{t['high']}).\n"
        f"POSITION:\n{schema.ascii_board(scn['fen'])}\n"
        f"{'White' if board.turn else 'Black'} to move. The student played "
        f"{scn['student_move']['san']}.\n\n"
        f"VERIFIED REFERENCE (private):\n{facts}\n- Sound moves: {sound}.\n\n"
        f"COACHING RESPONSE:\n{text.strip() or '(no answer)'}\n\n"
        "Score the six dimensions. Reply with the single JSON object."
    )


def score_instructiveness(judge: Any, scn: Dict[str, Any], text: str) -> Optional[int]:
    """Absolute 0-12 instructiveness (sum of six 0/1/2 dims) from one judge, or None."""
    try:
        reply, _usage = judge.complete(_ABS_SYSTEM, _abs_user(scn, text))
    except Exception as exc:  # noqa: BLE001 - a judge hiccup must not abort the loop
        log.warning("abs judge failed on %s: %s", scn.get("id"), exc)
        return None
    obj = _extract_json_object(reply) or {}
    cell = obj.get("scores") or {}
    total = 0
    seen = False
    for d in SIX_DIMS:
        try:
            total += max(0, min(2, int(cell.get(d))))
            seen = True
        except (TypeError, ValueError):
            continue
    return total if seen else None


# --------------------------------------------------------------------------- #
# Evaluate one candidate prompt on the DEV slice
# --------------------------------------------------------------------------- #


@dataclass
class PromptScore:
    prompt: str
    score: float                 # combined 0-100
    tier_fit: float
    instr_0_12: float
    no_jargon: float
    clean_rate: float            # produced faithful text without deterministic fallback
    n: int
    worst: List[Dict[str, Any]] = field(default_factory=list)


def evaluate_prompt(
    dev: Sequence[Dict[str, Any]],
    run_fn: RunFn,
    judge: Any,
    system_prompt: str,
    *,
    model_key: str,
    max_attempts: int,
    gate_on: bool,
    cache_path: Optional[Path] = None,
    seed_hook: Optional[Callable[[str], None]] = None,
) -> PromptScore:
    """Gated-generate + score every DEV item for ``system_prompt``; combine to 0-100."""
    ph = prompt_hash(system_prompt)
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("prompt_hash") == ph:
                        cache[r["scenario_id"]] = r
                except Exception:  # noqa: BLE001
                    continue

    per_item: List[Dict[str, Any]] = []
    fh = cache_path.open("a", encoding="utf-8") if cache_path else None
    try:
        for scn in dev:
            row = cache.get(scn["id"])
            if row is None:
                if seed_hook is not None:
                    seed_hook(f"{model_key}|{ph}|{scn['id']}")
                gen = gated_row(scn, run_fn, model_key, system_prompt=system_prompt,
                                max_attempts=max_attempts, gate_on=gate_on)
                instr = score_instructiveness(judge, scn, gen["output"])
                row = {**gen, "prompt_hash": ph, "instr_0_12": instr}
                if fh is not None:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
            per_item.append({"scn": scn, "row": row})
    finally:
        if fh is not None:
            fh.close()

    n = len(per_item)
    tier_hits = jarg_ok = clean = 0
    instr_vals: List[int] = []
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for it in per_item:
        scn, row = it["scn"], it["row"]
        canonical = scn.get("canonical_uci")
        fit = 1.0 if (row.get("rec_uci") and row["rec_uci"] == canonical) else 0.0
        tier_hits += fit
        jarg = 1.0 if not find_engine_speak(row.get("output", "")) else 0.0
        jarg_ok += jarg
        cln = 0.0 if row.get("verified_fallback") else 1.0
        clean += cln
        instr = row.get("instr_0_12")
        if instr is not None:
            instr_vals.append(instr)
        instr_norm = (instr / 12.0) if instr is not None else 0.0
        item_score = W_TIER * fit + W_INSTR * instr_norm + W_PRACTICAL * (0.5 * jarg + 0.5 * cln)
        scored.append((item_score, {
            "id": scn["id"], "tier": scn["tier"], "pos_id": scn.get("pos_id"),
            "canonical_san": _san(scn, canonical), "picked_san": row.get("rec_san"),
            "tier_fit": bool(fit), "instr_0_12": instr, "no_jargon": bool(jarg),
            "verified_fallback": bool(row.get("verified_fallback")),
            "output": (row.get("output", "") or "")[:600],
            "student_move": scn["student_move"].get("san"),
        }))

    tier_fit = tier_hits / n if n else 0.0
    instr_mean = (sum(instr_vals) / len(instr_vals)) if instr_vals else 0.0
    no_jargon = jarg_ok / n if n else 0.0
    clean_rate = clean / n if n else 0.0
    combined = 100.0 * (
        W_TIER * tier_fit + W_INSTR * (instr_mean / 12.0)
        + W_PRACTICAL * (0.5 * no_jargon + 0.5 * clean_rate)
    )
    scored.sort(key=lambda x: x[0])
    worst = [d for _s, d in scored[:6]]
    return PromptScore(
        prompt=system_prompt, score=round(combined, 2), tier_fit=round(tier_fit, 4),
        instr_0_12=round(instr_mean, 3), no_jargon=round(no_jargon, 4),
        clean_rate=round(clean_rate, 4), n=n, worst=worst,
    )


def _san(scn: Dict[str, Any], uci: Optional[str]) -> Optional[str]:
    if not uci:
        return None
    for m in scn.get("sound_pool", []):
        if m.get("uci") == uci:
            return m.get("san")
    return uci


# --------------------------------------------------------------------------- #
# Prompt-engineer: propose an improved system prompt
# --------------------------------------------------------------------------- #

_ENGINEER_SYSTEM = (
    "You are an expert prompt engineer improving the SYSTEM PROMPT for a small chess-coaching "
    "language model. The model is given a position, the student's move, and verified engine "
    "analysis (a sound-move pool with human-likelihoods), and must recommend ONE move from that "
    "pool and coach the student. It is scored on: (A) tier-appropriate move selection — for "
    "BEGINNER pick the most human-FINDABLE sound move, for ADVANCED the sharpest sound move, for "
    "INTERMEDIATE a blend; (B) instructiveness on six dimensions (concrete move+purpose, a "
    "transferable principle, a board-specific reason, how to find it next time, level calibration, "
    "and being grounded+concise); (C) never using engine-speak/centipawns and never fabricating.\n\n"
    "You will get the CURRENT system prompt, its DEV-set scores, and failing examples. Propose a "
    "BETTER system prompt. HARD CONSTRAINTS you must keep: never mention centipawns/evaluations/"
    "\"engine\"/\"Stockfish\"; recommend only a move from the sound pool; respect the tier's ply "
    "cap; never invent a tactic; end every answer with one 'Takeaway:' line. Keep it concise "
    "(under ~350 words). Return ONLY the new system prompt text — no preamble, no code fences."
)


def _worst_block(worst: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for w in worst[:5]:
        lines.append(
            f"- [{w['tier']}] student played {w.get('student_move')}; "
            f"tier-appropriate move = {w.get('canonical_san')}, coach picked = {w.get('picked_san')} "
            f"(tier_fit={w['tier_fit']}, instr={w.get('instr_0_12')}/12, "
            f"jargon_ok={w['no_jargon']}, fell_back={w['verified_fallback']}).\n"
            f"    coaching: {(w.get('output') or '').strip()[:280]!r}"
        )
    return "\n".join(lines)


def propose_prompt(engineer: Any, current: str, sc: PromptScore) -> Optional[str]:
    user = (
        f"CURRENT SYSTEM PROMPT:\n\"\"\"\n{current}\n\"\"\"\n\n"
        f"DEV SCORES (n={sc.n}): combined={sc.score}/100 | "
        f"tier_fit={sc.tier_fit} | instructiveness={sc.instr_0_12}/12 | "
        f"no_jargon={sc.no_jargon} | faithful_without_fallback={sc.clean_rate}\n\n"
        f"WORST EXAMPLES:\n{_worst_block(sc.worst)}\n\n"
        "Diagnose why these failed (especially tier-appropriate selection and instructiveness) "
        "and return an improved SYSTEM PROMPT that would score higher. Return ONLY the prompt text."
    )
    try:
        reply, _usage = engineer.complete(_ENGINEER_SYSTEM, user)
    except Exception as exc:  # noqa: BLE001
        log.warning("prompt engineer failed: %s", exc)
        return None
    reply = (reply or "").strip()
    # Strip an accidental ```...``` code fence + a leading language tag line.
    if reply.startswith("```"):
        parts = reply.split("```")
        reply = parts[1] if len(parts) >= 2 else reply.strip("`")
        first, _, rest = reply.partition("\n")
        if first.strip().lower() in ("markdown", "md", "text", "txt", ""):
            reply = rest
        reply = reply.strip()
    return reply or None


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


@dataclass
class OptResult:
    best_prompt: str
    best_score: float
    history: List[Dict[str, Any]] = field(default_factory=list)


def optimize(
    dev: Sequence[Dict[str, Any]],
    run_fn: RunFn,
    judge: Any,
    engineer: Any,
    seed_prompt: str,
    *,
    model_key: str,
    rounds: int = 3,
    max_attempts: int = 4,
    gate_on: bool = True,
    cache_path: Optional[Path] = None,
    seed_hook: Optional[Callable[[str], None]] = None,
) -> OptResult:
    """Iterate the system prompt ``rounds`` times; keep the best-scoring one."""
    def _eval(p: str) -> PromptScore:
        return evaluate_prompt(dev, run_fn, judge, p, model_key=model_key,
                               max_attempts=max_attempts, gate_on=gate_on,
                               cache_path=cache_path, seed_hook=seed_hook)

    log.info("[%s] scoring seed prompt on %d DEV items ...", model_key, len(dev))
    cur = _eval(seed_prompt)
    best = cur
    history = [{"round": 0, "kind": "seed", "score": cur.score, "tier_fit": cur.tier_fit,
                "instr_0_12": cur.instr_0_12, "no_jargon": cur.no_jargon,
                "clean_rate": cur.clean_rate, "prompt_hash": prompt_hash(seed_prompt)}]
    log.info("[%s] seed score=%.2f (tier_fit=%.3f instr=%.2f/12)",
             model_key, cur.score, cur.tier_fit, cur.instr_0_12)

    for r in range(1, rounds + 1):
        cand_prompt = propose_prompt(engineer, best.prompt, best)
        if not cand_prompt:
            log.warning("[%s] round %d: no candidate proposed; stopping.", model_key, r)
            break
        cand = _eval(cand_prompt)
        improved = cand.score > best.score
        history.append({
            "round": r, "kind": "candidate", "score": cand.score, "tier_fit": cand.tier_fit,
            "instr_0_12": cand.instr_0_12, "no_jargon": cand.no_jargon,
            "clean_rate": cand.clean_rate, "prompt_hash": prompt_hash(cand_prompt),
            "improved": improved, "kept": improved,
        })
        log.info("[%s] round %d cand score=%.2f (tier_fit=%.3f instr=%.2f/12) %s",
                 model_key, r, cand.score, cand.tier_fit, cand.instr_0_12,
                 "KEEP" if improved else "reject")
        if improved:
            best = cand
    return OptResult(best_prompt=best.prompt, best_score=best.score, history=history)
