#!/usr/bin/env python3
"""Blinded, cross-family council: absolute move + instructiveness grades.

For every (position x tier) item we collect all 14 coaching responses, anonymise
them to labels A..N with a per-item deterministic shuffle, and ask each of the 3
frontier judges (GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro) to grade EACH blinded
response on two 0-10 axes:

* ``move``  — is the recommended move a good, tier-appropriate choice here?
* ``instr`` — how instructive/useful is the coaching for a student at this tier?

The judge never sees model identities and grades every lab's model (including its
own), so the grade is blinded and cross-family. Resumable + costed: keyed by
(scenario_id, judge); token usage stored per row. This is the ONE consistent
council scale used across train + test_new + test_reuse.

Run::  ~/.venvs/mlx/bin/python data/showcase/pipeline/council.py --split train --concurrency 6
"""
from __future__ import annotations

import argparse
import hashlib
import random
import string
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import (  # noqa: E402
    FIELD, ROOT, SPLIT_DIRS, append_jsonl, done_keys, read_jsonl,
)

sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

from config import schema, settings  # noqa: E402
from src.eval.benchmark import config as bcfg  # noqa: E402
from src.eval.evaluate import _extract_json_object  # noqa: E402
from src.engine.position_facts import render_pool_facts  # noqa: E402
from src.eval.benchmark.prompts import scenario_to_teacher_input  # noqa: E402

import chess  # noqa: E402

JUDGE_KEYS: Tuple[str, ...] = ("gpt", "claude", "gemini")
LABELS: Tuple[str, ...] = tuple(string.ascii_uppercase[: len(FIELD)])  # A..N
JUDGE_MAX_TOKENS = 1600
#: Grade an item as long as at least this many of the 14 models produced a
#: generation. This keeps grading resilient to a transient single-provider
#: outage (one model down != all items ungraded); the missing model's cell
#: simply stays null for that item.
MIN_PRESENT = 10


def _anon_mapping(scenario_id: str, present: Sequence[str]) -> Dict[str, str]:
    """Deterministic ``label -> model_key`` map (labels shuffled per item).

    Sized to the models actually PRESENT for this item, so a temporarily-missing
    model does not shift or block the blinded grading of the others.
    """
    digest = hashlib.sha256(f"{bcfg.SEED}|{scenario_id}|showcase".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    models = list(present)
    rng.shuffle(models)
    return {LABELS[i]: models[i] for i in range(len(models))}


def _judge_system(n: int) -> str:
    rng = f"{LABELS[0]}-{LABELS[n - 1]}"
    return (
        "You are a strict, fair panel judge evaluating chess move-review COACHING for a "
        "student at a stated rating tier. You will see a position, the student's move, "
        f"verified reference facts, and {n} anonymized coaching responses labeled {rng}.\n\n"
        "Grade EACH response on two integer axes from 0 to 10:\n"
        "- move: is the ONE recommended move a good, SOUND, and TIER-APPROPRIATE choice "
        "for this student (a move a player at this level could understand, find, and reuse)? "
        "10 = ideal tier-appropriate sound move; 0 = a blunder or clearly wrong-for-tier move. "
        "Judge the move against the verified sound moves, not your own engine preference.\n"
        "- instr: how INSTRUCTIVE and USEFUL is the coaching for a player at THIS tier — will "
        "it actually help them understand what went wrong and improve? This is NOT about raw "
        "chess strength, length, or eloquence, and NOT about quoting engine numbers (a good "
        "coach never does). 10 = genuinely illuminating and tier-calibrated; 0 = useless, "
        "confusing, or fabricated.\n\n"
        "The verified facts are for YOUR grading only; do not reward a response merely for "
        "restating them, and lower BOTH scores if a response contradicts the verified facts. "
        "Return ONLY a single JSON object, no prose, of the form:\n"
        '{"grades": {"' + LABELS[0] + '": {"move": 0, "instr": 0}, "..." : {"...": 0}}, '
        '"note": "<one short sentence>"}'
    )


def _reference_block(scn: Dict[str, Any]) -> str:
    ti = scenario_to_teacher_input(scn)
    facts = render_pool_facts(scn["fen"], ti["sound_pool"])
    sound = ", ".join(m["san"] for m in scn["sound_pool"])
    return (
        f"{facts}\n"
        f"- Engine-sound moves (any of these is acceptable): {sound}.\n"
        f"- The student's move {scn['student_move']['san']} was a {scn.get('severity','?')}."
    )


def _judge_user(scn: Dict[str, Any], mapping: Dict[str, str], outputs: Dict[str, str],
                labels: Sequence[str]) -> str:
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
        "COACHING RESPONSES TO GRADE:",
    ]
    for label in labels:
        text = (outputs.get(mapping[label]) or "").strip() or "(no answer)"
        lines.append(f"\n--- Response {label} ---\n{text}")
    lines.append("\nGrade every response on move (0-10) and instr (0-10). Reply with the single JSON object.")
    return "\n".join(lines)


def _parse(content: str, mapping: Dict[str, str],
           labels: Sequence[str]) -> Tuple[Dict[str, Dict[str, float]], str]:
    obj = _extract_json_object(content) or {}
    raw = obj.get("grades") or {}
    grades: Dict[str, Dict[str, float]] = {}

    def _clamp(v: Any) -> Optional[float]:
        try:
            return max(0.0, min(10.0, float(v)))
        except (TypeError, ValueError):
            return None

    for label in labels:
        cell = raw.get(label) or {}
        grades[mapping[label]] = {"move": _clamp(cell.get("move")), "instr": _clamp(cell.get("instr"))}
    note = str(obj.get("note", ""))[:300]
    return grades, note


def _gen_index(split_dir: Path) -> Dict[str, Dict[str, str]]:
    idx: Dict[str, Dict[str, str]] = {}
    for key in FIELD:
        for g in read_jsonl(split_dir / "gen" / f"{key}.jsonl"):
            idx.setdefault(g["scenario_id"], {})[key] = g.get("output", "")
    return idx


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True, choices=list(SPLIT_DIRS))
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--min-interval", type=float, default=0.05)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--require-full-field", action="store_true", default=True,
                   help="Only grade items where all 14 models have a generation.")
    args = p.parse_args(argv)

    load_dotenv(settings.ROOT / ".env")
    from src.eval.benchmark.backends import RateLimiter, TFYChat, make_tfy_client  # noqa: E402

    split_dir = SPLIT_DIRS[args.split]
    scns = read_jsonl(split_dir / "scenarios.jsonl")
    idx = _gen_index(split_dir)
    out = split_dir / "council.jsonl"
    done = done_keys(out, ["scenario_id", "judge"])

    tasks: List[Tuple[Dict[str, Any], Dict[str, str], Dict[str, str], List[str], str]] = []
    incomplete = 0
    for scn in scns:
        outputs = idx.get(scn["id"], {})
        present = [k for k in FIELD if outputs.get(k, "").strip()]
        if len(present) < MIN_PRESENT:
            incomplete += 1
            continue
        labels = list(LABELS[: len(present)])
        mapping = _anon_mapping(scn["id"], present)
        for jk in JUDGE_KEYS:
            if (scn["id"], jk) in done:
                continue
            tasks.append((scn, mapping, outputs, labels, jk))

    print(f"[council/{args.split}] {len(tasks)} judge-tasks pending "
          f"({incomplete} items skipped: <{MIN_PRESENT} models present)", file=sys.stderr)
    if not tasks:
        return 0

    client = make_tfy_client(args.timeout)
    limiter = RateLimiter(args.min_interval)
    judges = {
        jk: TFYChat(client, model_id=bcfg.MODELS[jk].ident, max_tokens=JUDGE_MAX_TOKENS,
                    max_retries=args.max_retries, limiter=limiter,
                    reasoning_effort=bcfg.MODELS[jk].reasoning_effort)
        for jk in JUDGE_KEYS
    }

    ok = fail = n = 0
    lock = threading.Lock()

    def _task(item):
        scn, mapping, outputs, labels, jk = item
        system = _judge_system(len(labels))
        text, usage = judges[jk].complete(system, _judge_user(scn, mapping, outputs, labels))
        grades, note = _parse(text, mapping, labels)
        return item, grades, note, usage

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(_task, it): it for it in tasks}
        for fut in as_completed(futs):
            scn, mapping, _o, labels, jk = futs[fut]
            n += 1
            try:
                _item, grades, note, usage = fut.result()
                with lock:
                    append_jsonl(out, {
                        "scenario_id": scn["id"], "tier": scn["tier"], "phase": scn["phase"],
                        "pos_id": scn.get("pos_id", scn["id"]), "judge": jk,
                        "n_present": len(labels),
                        "label_to_model": mapping, "grades": grades, "note": note,
                        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    })
                ok += 1
            except Exception as exc:  # noqa: BLE001 - a rerun retries it
                fail += 1
                print(f"  ! judge {jk} {scn['id']}: {exc}", file=sys.stderr)
            if n % 50 == 0 or n == len(tasks):
                print(f"  council {n}/{len(tasks)} (ok={ok} fail={fail})", file=sys.stderr)

    print(f"[council/{args.split}] done ok={ok} fail={fail} -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
