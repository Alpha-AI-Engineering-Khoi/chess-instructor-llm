"""Blinded, cross-family council: the 3 frontier models rank the 5 coaches.

For every (scenario x condition) item we collect all five coaching outputs,
anonymise them to labels A-E and shuffle the label->model mapping *per item*
(deterministically, so the same blind key is reproducible and shared with the
human-label export), then ask each of the three frontier judges to RANK them.

Primary criterion (the whole point): **how instructive and useful is this
coaching for a player at the stated tier** — not chess strength, not verbosity.
Judges also give a small per-response rubric (tier_calibration / clarity /
correctness). Because each judge grades its own lab's model too, the aggregation
can measure self-preference.

Resumable + costed: keyed by ``(scenario_id, condition, judge)``; token usage is
stored per row.
"""

from __future__ import annotations

import hashlib
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess

from config import schema, settings
from src.engine.position_facts import render_pool_facts
from src.eval.evaluate import _extract_json_object

from . import config as bcfg
from .backends import RateLimiter, TFYChat, make_tfy_client
from .io_utils import append_jsonl, done_keys, read_jsonl
from .prompts import scenario_to_teacher_input

log = logging.getLogger("benchmark.council")


# --------------------------------------------------------------------------- #
# Deterministic anonymisation (shared with the blind human-label export)
# --------------------------------------------------------------------------- #


def _seed_int(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:16], 16)


def anon_mapping(scenario_id: str, condition: str) -> Dict[str, str]:
    """Deterministic ``label -> model_key`` map for one item (A..E, shuffled)."""
    models = list(bcfg.MODEL_ORDER)
    rng = random.Random(_seed_int(bcfg.SEED, scenario_id, condition))
    rng.shuffle(models)
    return {bcfg.ANON_LABELS[i]: models[i] for i in range(len(models))}


# --------------------------------------------------------------------------- #
# Judge prompt
# --------------------------------------------------------------------------- #

_NUM_WORDS = {
    2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE", 6: "SIX", 7: "SEVEN", 8: "EIGHT",
    9: "NINE", 10: "TEN", 11: "ELEVEN", 12: "TWELVE", 13: "THIRTEEN", 14: "FOURTEEN",
}


def build_judge_system(labels: Optional[Sequence[str]] = None) -> str:
    """The judge system prompt, sized to ``labels`` (defaults to ``ANON_LABELS``).

    Identical wording to the original five-way prompt; only the count word and the
    label set/range are parameterised, so ranking N (>5) anonymized coaches uses
    the exact same rubric and instructions the frontier five-way run used.
    """
    labs = tuple(labels) if labels is not None else tuple(bcfg.ANON_LABELS)
    n = len(labs)
    word = _NUM_WORDS.get(n, str(n))
    label_range = f"{labs[0]}-{labs[-1]}" if n > 1 else labs[0]
    scores_example = ", ".join(
        (f'"{lab}": {{"tier_calibration": 0, "clarity": 0, "correctness": 0}}'
         if i == 0 else f'"{lab}": {{...}}')
        for i, lab in enumerate(labs)
    )
    return (
        "You are a strict, fair panel judge evaluating chess move-review COACHING for "
        "a student at a stated rating tier. You will see a position, the student's move, "
        f"verified reference facts, and {word} anonymized coaching responses labeled {label_range}.\n\n"
        f"RANK all {word.lower()} from best to worst. Your PRIMARY and decisive criterion is: "
        "**how INSTRUCTIVE and USEFUL is this coaching for a player at the stated tier** "
        "— will it actually help THIS student understand what went wrong and improve? "
        "This is NOT about raw chess strength, NOT about length or eloquence, and NOT "
        "about whether engine numbers are quoted (a good coach never quotes them).\n\n"
        "Also score each response on three 0/1/2 dimensions:\n"
        "- tier_calibration: ideas + depth fit the tier (simpler for beginners).\n"
        "- clarity: plain, well-organized, encouraging, easy to act on.\n"
        "- correctness: claims about the board/threats match the verified facts; "
        "the recommended move is sound. Contradicting the verified facts lowers this.\n\n"
        "The verified facts are for YOUR grading only; do not reward a response merely "
        "for restating them. Return ONLY a single JSON object, no prose, of the form:\n"
        '{"ranking": ["<best label>", "...", "<worst label>"], '
        '"scores": {' + scores_example + '}, '
        '"note": "<one short sentence>"}'
    )


#: Default five-way judge system prompt (backwards-compatible module constant).
_JUDGE_SYSTEM = build_judge_system()


def _reference_block(scn: Dict[str, Any]) -> str:
    """The verified, private reference the judge uses to grade correctness."""
    ti = scenario_to_teacher_input(scn)
    facts = render_pool_facts(scn["fen"], ti["sound_pool"])
    sound = ", ".join(m["san"] for m in scn["sound_pool"])
    return (
        f"{facts}\n"
        f"- Engine-sound moves (any of these is acceptable): {sound}.\n"
        f"- The student's move {scn['student_move']['san']} was a {scn['severity']}."
    )


def _build_judge_user(scn: Dict[str, Any], mapping: Dict[str, str],
                      outputs: Dict[str, str]) -> str:
    """Assemble the judge's user message (position + facts + 5 blinded answers)."""
    board = chess.Board(scn["fen"])
    t = settings.TIERS[scn["tier"]]
    lines = [
        f"STUDENT TIER: {scn['tier']} ({t['low']}-{t['high']}).",
        "POSITION:",
        schema.ascii_board(scn["fen"]),
        f"{'White' if board.turn else 'Black'} to move. "
        f"The student played {scn['student_move']['san']}.",
        "",
        "VERIFIED REFERENCE (private — for your grading only):",
        _reference_block(scn),
        "",
        "COACHING RESPONSES TO RANK:",
    ]
    for label in bcfg.ANON_LABELS:
        model_key = mapping[label]
        text = (outputs.get(model_key) or "").strip() or "(no answer)"
        lines.append(f"\n--- Response {label} ---\n{text}")
    lines.append(
        "\nRank all five by INSTRUCTIVENESS for this tier and score the rubric. "
        "Reply with the single JSON object."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Judge-output parsing
# --------------------------------------------------------------------------- #

_DIMS = ("tier_calibration", "clarity", "correctness")


def parse_judge(content: str) -> Tuple[List[str], Dict[str, Dict[str, int]], str]:
    """Parse a judge reply into (ranking, scores, note); defensive on bad JSON."""
    obj = _extract_json_object(content) or {}

    raw_rank = obj.get("ranking") or []
    ranking: List[str] = []
    for lab in raw_rank:
        lab = str(lab).strip().upper()[:1]
        if lab in bcfg.ANON_LABELS and lab not in ranking:
            ranking.append(lab)
    # Ensure a full permutation: append any missing labels (stable order).
    for lab in bcfg.ANON_LABELS:
        if lab not in ranking:
            ranking.append(lab)

    raw_scores = obj.get("scores") or {}
    scores: Dict[str, Dict[str, int]] = {}
    for lab in bcfg.ANON_LABELS:
        cell = raw_scores.get(lab) or {}
        scores[lab] = {}
        for dim in _DIMS:
            try:
                val = int(cell.get(dim))
            except (TypeError, ValueError):
                val = 0
            scores[lab][dim] = max(0, min(2, val))

    note = str(obj.get("note", ""))[:300]
    return ranking, scores, note


# --------------------------------------------------------------------------- #
# Council run
# --------------------------------------------------------------------------- #


def _outputs_by_model(condition: str, scenario_id: str,
                      gen_index: Dict[Tuple[str, str, str], str]) -> Optional[Dict[str, str]]:
    """Return ``{model_key: output}`` for one item, or None if any model missing."""
    out: Dict[str, str] = {}
    for mk in bcfg.MODEL_ORDER:
        key = (scenario_id, mk, condition)
        if key not in gen_index:
            return None
        out[mk] = gen_index[key]
    return out


def run_council(
    scenarios: Sequence[Dict[str, Any]],
    conditions: Sequence[str],
    judge_keys: Sequence[str],
    *,
    concurrency: int = 6,
    min_interval: float = 0.05,
    timeout: float = 300.0,
    max_retries: int = 4,
) -> Dict[str, int]:
    """Have each judge rank every complete item. Resumable + costed."""
    gen_index: Dict[Tuple[str, str, str], str] = {
        (g["scenario_id"], g["model"], g["condition"]): g.get("output", "")
        for g in read_jsonl(bcfg.GENERATIONS_PATH)
    }
    done = done_keys(bcfg.COUNCIL_PATH, ["scenario_id", "condition", "judge"])

    # Build the list of (scenario, condition, mapping, outputs, judge) tasks.
    tasks: List[Tuple[Dict[str, Any], str, Dict[str, str], Dict[str, str], str]] = []
    incomplete = 0
    for scn in scenarios:
        for cond in conditions:
            outputs = _outputs_by_model(cond, scn["id"], gen_index)
            if outputs is None:
                incomplete += 1
                continue
            mapping = anon_mapping(scn["id"], cond)
            for jk in judge_keys:
                if (scn["id"], cond, jk) in done:
                    continue
                tasks.append((scn, cond, mapping, outputs, jk))

    log.info("council: %d judge-tasks pending (%d incomplete items skipped)",
             len(tasks), incomplete)
    if not tasks:
        return {"ok": 0, "fail": 0, "pending": 0}

    client = make_tfy_client(timeout)
    limiter = RateLimiter(min_interval)
    judges = {
        jk: TFYChat(client, model_id=bcfg.MODELS[jk].ident,
                    max_tokens=bcfg.JUDGE_MAX_TOKENS, max_retries=max_retries,
                    limiter=limiter, reasoning_effort=bcfg.MODELS[jk].reasoning_effort)
        for jk in judge_keys
    }

    # Size the judge system prompt to the current field (5-way by default, more
    # when a driver sets a larger MODEL_ORDER/ANON_LABELS for a unified ranking).
    system = build_judge_system()

    ok = fail = done_n = 0

    def _task(item: Tuple[Dict[str, Any], str, Dict[str, str], Dict[str, str], str]):
        scn, cond, mapping, outputs, jk = item
        user = _build_judge_user(scn, mapping, outputs)
        text, usage = judges[jk].complete(system, user)
        ranking, scores, note = parse_judge(text)
        return item, ranking, scores, note, usage

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(_task, it): it for it in tasks}
        for fut in as_completed(futures):
            scn, cond, mapping, _outputs, jk = futures[fut]
            done_n += 1
            try:
                _item, ranking, scores, note, usage = fut.result()
                append_jsonl(
                    bcfg.COUNCIL_PATH,
                    {
                        "scenario_id": scn["id"],
                        "condition": cond,
                        "judge": jk,
                        "tier": scn["tier"],
                        "phase": scn["phase"],
                        "severity": scn["severity"],
                        "ranking": ranking,
                        "scores": scores,
                        "label_to_model": mapping,
                        "note": note,
                        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    },
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001 - skip; a rerun retries it
                fail += 1
                log.error("judge %s %s/%s failed: %s", jk, scn["id"], cond, exc)
            if done_n % 25 == 0 or done_n == len(tasks):
                log.info("  council: %d/%d (ok=%d fail=%d)", done_n, len(tasks), ok, fail)

    log.info("council done: ok=%d fail=%d", ok, fail)
    return {"ok": ok, "fail": fail, "pending": len(tasks)}
