"""The 6-dimension instructiveness rubric + the tier-coherence check (Part C).

**Instructiveness (blinded cross-family council).** The council still RANKS every
coach per item (that drives the instructiveness rank + distance-to-frontier), but
now also scores each response on SIX 0/1/2 dimensions — the spec's rubric:

1. ``move_purpose``          — names a concrete move AND its concrete purpose.
2. ``transferable_principle``— states a transferable idea, not just this move.
3. ``board_specific_reason`` — a reason grounded in THIS position (not generic).
4. ``how_to_find``           — tells the student how to find it next time.
5. ``level_calibration``     — pitched for the tier (simpler for beginners).
6. ``grounded_concise``      — grounded + concise, no engine-speak, no fabrication.

Judging is blinded (labels shuffled per item) and cross-family (the three
frontier judges), reusing the benchmark's retrying gateway client + the same
verified reference block the 3-dim council used for its correctness grade.

**Tier-coherence.** Deterministic, no LLM: for each position with all three tiers
present, the recommended moves should form a sensible gradient — beginner the
most human-findable sound move, advanced the sharpest. We flag two incoherences:

* ``zigzag``       — beginner == advanced but != intermediate (the spec's example
  of an incoherent tier pattern);
* ``inverted``     — the beginner pick is *strictly harder* for a human to find
  (worse Maia rank) than the advanced pick — the gradient runs backwards.

A position is a violation if it trips either. ``flat`` (all three identical) is
reported separately as no differentiation (not itself an incoherence).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess

from config import schema, settings
from src.engine.position_facts import render_pool_facts
from src.eval.benchmark.prompts import scenario_to_teacher_input
from src.eval.evaluate import _extract_json_object

log = logging.getLogger("honest.rubric")

SIX_DIMS: Tuple[str, ...] = (
    "move_purpose",
    "transferable_principle",
    "board_specific_reason",
    "how_to_find",
    "level_calibration",
    "grounded_concise",
)

_DIM_DESC: Dict[str, str] = {
    "move_purpose": "names ONE concrete move and its concrete purpose (a plan, not a vague nicety).",
    "transferable_principle": "states a transferable principle/idea the student can reuse elsewhere.",
    "board_specific_reason": "gives a reason grounded in THIS position (specific pieces/squares/threats), not a generic platitude.",
    "how_to_find": "tells the student how to FIND this kind of move next time (a method, cue, or question to ask).",
    "level_calibration": "is pitched for the stated tier — simpler ideas/shorter lines for beginners, sharper for advanced.",
    "grounded_concise": "is grounded and concise: no centipawns/engine-speak, no invented tactics, no rambling.",
}

_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def labels_for(n: int) -> Tuple[str, ...]:
    if not 1 <= n <= 26:
        raise ValueError(f"need 1..26 labels, got {n}")
    return tuple(_LABELS[:n])


# --------------------------------------------------------------------------- #
# Blinded anonymisation (deterministic, per item + field)
# --------------------------------------------------------------------------- #


def _seed_int(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:16], 16)


def anon_mapping(field: Sequence[str], scenario_id: str, condition: str,
                 seed: int = 20260707) -> Dict[str, str]:
    """Deterministic ``label -> model_key`` for one item (shuffled, field-sized)."""
    models = list(field)
    rng = random.Random(_seed_int(seed, scenario_id, condition, len(models)))
    rng.shuffle(models)
    labs = labels_for(len(models))
    return {labs[i]: models[i] for i in range(len(models))}


# --------------------------------------------------------------------------- #
# Judge prompt (6-dim)
# --------------------------------------------------------------------------- #


def build_judge_system(labels: Sequence[str]) -> str:
    n = len(labels)
    label_range = f"{labels[0]}-{labels[-1]}" if n > 1 else labels[0]
    dims_block = "\n".join(f"- {d}: {_DIM_DESC[d]}" for d in SIX_DIMS)
    example = ", ".join(
        (f'"{lab}": {{' + ", ".join(f'"{d}": 0' for d in SIX_DIMS) + "}}")
        if i == 0 else f'"{lab}": {{...}}'
        for i, lab in enumerate(labels)
    )
    return (
        "You are a strict, fair panel judge evaluating chess move-review COACHING for "
        "a student at a stated rating tier. You will see a position, the student's move, "
        f"verified reference facts, and {n} anonymized coaching responses labeled {label_range}.\n\n"
        f"RANK all {n} from best to worst. Your PRIMARY and decisive criterion is: **how "
        "INSTRUCTIVE and USEFUL is this coaching for a player at the stated tier** — will "
        "it actually help THIS student understand what went wrong and improve? This is NOT "
        "about raw chess strength, NOT about length or eloquence, and NOT about whether "
        "engine numbers are quoted (a good coach never quotes them).\n\n"
        "Also score EACH response on these six 0/1/2 dimensions (0 = absent, 1 = partial, "
        "2 = clearly present):\n"
        f"{dims_block}\n\n"
        "The verified facts are for YOUR grading only; do not reward a response merely for "
        "restating them, and lower grounded_concise / board_specific_reason when a response "
        "contradicts them. Return ONLY a single JSON object, no prose, of the form:\n"
        '{"ranking": ["<best label>", "...", "<worst label>"], '
        '"scores": {' + example + '}, "note": "<one short sentence>"}'
    )


def _reference_block(scn: Dict[str, Any]) -> str:
    ti = scenario_to_teacher_input(scn)
    facts = render_pool_facts(scn["fen"], ti["sound_pool"])
    sound = ", ".join(m["san"] for m in scn["sound_pool"])
    return (
        f"{facts}\n"
        f"- Engine-sound moves (any of these is acceptable): {sound}.\n"
        f"- The student's move {scn['student_move']['san']} was a {scn['severity']}."
    )


def build_judge_user(scn: Dict[str, Any], mapping: Dict[str, str],
                     outputs: Dict[str, str], labels: Sequence[str]) -> str:
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
    for label in labels:
        text = (outputs.get(mapping[label]) or "").strip() or "(no answer)"
        lines.append(f"\n--- Response {label} ---\n{text}")
    lines.append(
        f"\nRank all {len(labels)} by INSTRUCTIVENESS for this tier and score the six "
        "dimensions. Reply with the single JSON object."
    )
    return "\n".join(lines)


def parse_judge(content: str, labels: Sequence[str]) -> Tuple[List[str], Dict[str, Dict[str, int]], str]:
    """Parse a 6-dim judge reply into (ranking, scores, note); defensive on bad JSON."""
    obj = _extract_json_object(content) or {}
    labset = set(labels)

    ranking: List[str] = []
    for lab in obj.get("ranking") or []:
        lab = str(lab).strip().upper()[:1]
        if lab in labset and lab not in ranking:
            ranking.append(lab)
    for lab in labels:  # complete any missing labels (stable order)
        if lab not in ranking:
            ranking.append(lab)

    raw_scores = obj.get("scores") or {}
    scores: Dict[str, Dict[str, int]] = {}
    for lab in labels:
        cell = raw_scores.get(lab) or {}
        scores[lab] = {}
        for dim in SIX_DIMS:
            try:
                val = int(cell.get(dim))
            except (TypeError, ValueError):
                val = 0
            scores[lab][dim] = max(0, min(2, val))
    return ranking, scores, str(obj.get("note", ""))[:300]


# --------------------------------------------------------------------------- #
# Council runner (self-contained; reads a supplied per-model generations index)
# --------------------------------------------------------------------------- #


def _done_keys(path: Path) -> set:
    done: set = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                done.add((r["scenario_id"], r["condition"], r["judge"]))
            except Exception:  # noqa: BLE001
                continue
    return done


def run_council(
    scenarios: Sequence[Dict[str, Any]],
    outputs_by_model: Dict[str, Dict[str, str]],
    field: Sequence[str],
    judges: Dict[str, Any],  # judge_key -> TFYChat
    out_path: Path,
    *,
    condition: str = "gated",
    concurrency: int = 5,
    seed: int = 20260707,
    require_all: bool = True,
) -> Dict[str, int]:
    """Have each judge rank+score every complete item. Resumable + costed.

    ``outputs_by_model[model_key][scenario_id] = text``. ``field`` is the ordered
    competitor set (only these are ranked). An item is judged only when every
    field model has an output for it (``require_all``); otherwise it is skipped.
    """
    labels = labels_for(len(field))
    system = build_judge_system(labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_path)

    tasks: List[Tuple[Dict[str, Any], Dict[str, str], Dict[str, str], str]] = []
    incomplete = 0
    for scn in scenarios:
        outs = {mk: outputs_by_model.get(mk, {}).get(scn["id"]) for mk in field}
        if require_all and any(v is None for v in outs.values()):
            incomplete += 1
            continue
        outs = {mk: (v or "") for mk, v in outs.items()}
        mapping = anon_mapping(field, scn["id"], condition, seed=seed)
        for jk in judges:
            if (scn["id"], condition, jk) in done:
                continue
            tasks.append((scn, mapping, outs, jk))

    log.info("council(6dim): %d judge-tasks pending (%d incomplete items skipped)",
             len(tasks), incomplete)
    if not tasks:
        return {"ok": 0, "fail": 0, "pending": 0}

    ok = fail = done_n = 0
    lock = threading.Lock()
    fh = out_path.open("a", encoding="utf-8")

    def _task(item):
        scn, mapping, outs, jk = item
        user = build_judge_user(scn, mapping, outs, labels)
        text, usage = judges[jk].complete(system, user)
        ranking, scores, note = parse_judge(text, labels)
        return item, ranking, scores, note, usage

    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futs = {pool.submit(_task, it): it for it in tasks}
            for fut in as_completed(futs):
                scn, mapping, _outs, jk = futs[fut]
                done_n += 1
                try:
                    _item, ranking, scores, note, usage = fut.result()
                    with lock:
                        fh.write(json.dumps({
                            "scenario_id": scn["id"], "condition": condition, "judge": jk,
                            "tier": scn["tier"], "phase": scn["phase"], "severity": scn["severity"],
                            "pos_id": scn.get("pos_id"),
                            "ranking": ranking, "scores": scores, "label_to_model": mapping,
                            "note": note,
                            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                            "completion_tokens": int(usage.get("completion_tokens", 0)),
                            "ts": datetime.now(timezone.utc).isoformat(),
                        }, ensure_ascii=False) + "\n")
                        fh.flush()
                    ok += 1
                except Exception as exc:  # noqa: BLE001 - skip; a rerun retries it
                    fail += 1
                    log.error("judge %s %s failed: %s", jk, scn["id"], exc)
                if done_n % 20 == 0 or done_n == len(tasks):
                    log.info("  council(6dim): %d/%d (ok=%d fail=%d)", done_n, len(tasks), ok, fail)
    finally:
        fh.close()
    return {"ok": ok, "fail": fail, "pending": len(tasks)}


# --------------------------------------------------------------------------- #
# 6-dim aggregation from council rows
# --------------------------------------------------------------------------- #


def dim_means(rows: Sequence[Dict[str, Any]], field: Sequence[str]
              ) -> Dict[str, Dict[str, Any]]:
    """Per-model mean of each 0/1/2 dimension + a summed 0-12 instructiveness score."""
    acc: Dict[str, Dict[str, List[int]]] = {mk: defaultdict(list) for mk in field}
    for r in rows:
        mapping = r.get("label_to_model") or {}
        scores = r.get("scores") or {}
        for lab, mk in mapping.items():
            if mk not in acc:
                continue
            cell = scores.get(lab) or {}
            for dim in SIX_DIMS:
                if dim in cell:
                    acc[mk][dim].append(int(cell[dim]))
    out: Dict[str, Dict[str, Any]] = {}
    for mk in field:
        per_dim = {d: (round(mean(acc[mk][d]), 3) if acc[mk][d] else None) for d in SIX_DIMS}
        present = [per_dim[d] for d in SIX_DIMS if per_dim[d] is not None]
        out[mk] = {
            "dims": per_dim,
            "sum_0_12": round(sum(present), 3) if present else None,
            "n": max((len(acc[mk][d]) for d in SIX_DIMS), default=0),
        }
    return out


# --------------------------------------------------------------------------- #
# Tier-coherence (deterministic)
# --------------------------------------------------------------------------- #

TIERS: Tuple[str, ...] = ("beginner", "intermediate", "advanced")


def _maia_rank(scn: Dict[str, Any], uci: Optional[str]) -> Optional[int]:
    if not uci:
        return None
    order = scn.get("pool_order") or []
    return order.index(uci) if uci in order else None


def tier_coherence(
    rec_by_model_pos_tier: Dict[str, Dict[str, Dict[str, Optional[str]]]],
    scns_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Per-model tier-coherence violation rate over full-tier positions.

    ``rec_by_model_pos_tier[model][pos_id][tier] = rec_uci``. For each position
    with all three tier picks present we flag ``zigzag`` (b==a and b!=i) and
    ``inverted`` (beginner strictly harder to find than advanced by Maia rank);
    the position is a *violation* if it trips either.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for model, positions in rec_by_model_pos_tier.items():
        n_full = 0
        zig = inv = flat = viol = 0
        examples: List[Dict[str, Any]] = []
        for pos_id, picks in positions.items():
            if not all(t in picks and picks[t] for t in TIERS):
                continue
            n_full += 1
            b, i, a = picks["beginner"], picks["intermediate"], picks["advanced"]
            is_zig = (b == a and b != i)
            is_flat = (b == a == i)
            # findability gradient (needs each tier's own scenario for pool_order)
            br = _maia_rank(scns_by_id.get(f"{pos_id}#beginner", {}), b)
            ar = _maia_rank(scns_by_id.get(f"{pos_id}#advanced", {}), a)
            is_inv = (br is not None and ar is not None and br > ar)
            if is_zig:
                zig += 1
            if is_flat:
                flat += 1
            if is_inv:
                inv += 1
            if is_zig or is_inv:
                viol += 1
                if len(examples) < 8:
                    examples.append({
                        "pos_id": pos_id, "beginner": b, "intermediate": i, "advanced": a,
                        "zigzag": is_zig, "inverted": is_inv,
                        "beginner_maia_rank": br, "advanced_maia_rank": ar,
                    })
        out[model] = {
            "n_full": n_full,
            "violation_rate": round(viol / n_full, 4) if n_full else None,
            "zigzag_rate": round(zig / n_full, 4) if n_full else None,
            "inverted_rate": round(inv / n_full, 4) if n_full else None,
            "flat_rate": round(flat / n_full, 4) if n_full else None,
            "n_violations": viol,
            "examples": examples,
        }
    return out
