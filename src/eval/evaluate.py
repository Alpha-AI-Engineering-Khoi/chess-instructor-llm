#!/usr/bin/env python3
"""Base-vs-tuned evaluator for the chess move-review coach.

This is the *make-or-break* eval harness: it must exist (and be trusted) before
any fine-tuning, because the whole thesis of the project is "behavior from data"
demonstrated as a **base-vs-tuned delta on the same held-out scenarios**. If the
ruler is wrong, the experiment is meaningless.

What it does
------------
1. **Scenarios (held-out).** Take positions from ``data/positions/positions.jsonl``
   and, reusing the engine modules, compute for each: Stockfish ``sound_pool`` +
   ``classify_mistake`` and Maia ``human_moves``. These assemble a
   :class:`config.schema.TeacherInput` — the *exact* contract every stage shares.
   Positions with mistake severity ``"none"`` or an empty sound pool are skipped
   (nothing to teach). Scenarios are drawn round-robin across tiers so the
   per-tier table is populated.

2. **Model backends** turn ``coach_system.md`` + ``schema.render_user_prompt``
   into coaching text:
     - ``base``  — a local MLX model via ``mlx_lm`` (default
       ``mlx-community/Qwen3-1.7B-4bit``), chat-templated, ``<think>`` stripped.
     - ``tuned`` — the *same* interface pointed at a tuned MLX model path/repo.
       It is a stub for now (the tuned model does not exist yet): pass
       ``--tuned-path`` once trained; without it the tuned side is skipped.

3. **Scoring** has two independent layers:
     a. **Objective checks** (deterministic, cheap, run first) mirror the
        dataset filter's gates: produced non-empty coaching; recommended move is
        parseable AND in the sound pool; no engine-speak (regex); narrated line
        within the tier's ply cap.
     b. **LLM judge** — ``gpt-5.5-pro`` scores the five ``eval_rubric.md``
        dimensions (spec adherence, level calibration, no-engine-speak,
        truthfulness, task quality), each 0/1/2, returning JSON.

        .. note::
           We use an **OpenAI** judge here for convenience while bootstrapping.
           A judge from a **different model family than the teacher** (the teacher
           is GPT-5.5) is strongly preferable — you should not let a model grade
           its own family's homework. Swap in an Anthropic/Claude judge for the
           real base-vs-tuned report; the rubric and JSON contract are identical.

4. **Output.** A results table (printed) plus ``data/eval/results_<label>.json``:
   mean judge score per dimension and % passing each objective check, **per tier
   and overall**. The JSON is shaped so running ``base`` then ``tuned`` and
   passing ``--compare-to`` prints a side-by-side delta.

Security
--------
The OpenAI key is loaded from ``ROOT/.env`` and is **never** printed or written
to any artifact.

CLI
---
    # Evaluate the base model on 10 held-out scenarios (objective + judge):
    python src/eval/evaluate.py --model base --num-scenarios 10

    # Objective checks only (no API cost), useful while iterating:
    python src/eval/evaluate.py --model base --no-judge

    # Once a tuned model exists, then show the delta:
    python src/eval/evaluate.py --model tuned --tuned-path ./models/qwen3-coach-mlx
    python src/eval/evaluate.py --model tuned --tuned-path ./models/qwen3-coach-mlx \\
        --compare-to data/eval/results_base.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess

# --------------------------------------------------------------------------- #
# Make `config` / `src` importable whether run as a script or a module.
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import schema, settings  # noqa: E402
from src.engine import maia_engine, stockfish_engine  # noqa: E402
from src.teacher.coach_gate import pick_recommendation  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_MLX_MODEL: str = "mlx-community/Qwen3-1.7B-4bit"
DEFAULT_JUDGE_MODEL: str = "gpt-5.5-pro"
DEFAULT_NUM_SCENARIOS: int = 10
DEFAULT_MAX_TOKENS: int = 512

TIER_ORDER: Tuple[str, ...] = ("beginner", "intermediate", "advanced")

OBJECTIVE_CHECKS: Tuple[str, ...] = (
    "produced_nonempty",
    "move_parseable",
    "move_sound",
    "no_engine_speak",
    "ply_cap_ok",
)
JUDGE_DIMENSIONS: Tuple[str, ...] = (
    "spec_adherence",
    "level_calibration",
    "no_engine_speak",
    "truthfulness",
    "task_quality",
)

#: Patterns that constitute forbidden "engine-speak" (mirrors ``eval_rubric.md``).
#: Any match => the output leaks engine internals and fails the check.
_ENGINE_SPEAK_PATTERNS: Tuple[str, ...] = (
    r"\beval(?:uation|uations)?\b",         # "eval", "evaluation"
    r"\bengine\b",
    r"\bstockfish\b",
    r"\bcomputer\b",
    r"\bcentipawns?\b",
    r"\bcp\b",                               # the unit token itself
    r"\d+\s*cp\b",                           # "150cp", "150 cp"
    r"[+\-]\d+\.\d+",                        # signed decimal eval: "+1.3", "-2.5"
    r"[+\-]\d{2,}",                          # signed centipawn-ish int: "+150"
    r"(?<![a-hA-H])\b\d+\.\d+\b",            # bare decimal eval: "1.3" (not "e1.3")
    r"#\d+",                                 # engine mate notation "#3"
)
_ENGINE_SPEAK_RE = re.compile("|".join(_ENGINE_SPEAK_PATTERNS), re.IGNORECASE)

#: Shape of a SAN half-move (validated for legality separately via python-chess).
_SAN_SHAPE = re.compile(
    r"(?:O-O-O|O-O|0-0-0|0-0)"
    r"|(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)"
)
#: A leading move number like "12." or "3..." glued to (or standing before) a move.
_MOVE_NUMBER = re.compile(r"^\d+\.+(.*)$")
#: Game results that look move-ish but are not plies.
_RESULT_TOKENS = frozenset({"1-0", "0-1", "1/2-1/2", "1/2"})


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #


@dataclass
class Scenario:
    """One held-out coaching scenario derived from a position."""

    position_id: str
    tier: str
    teacher_input: schema.TeacherInput
    sound_uci: frozenset[str]
    student_uci: str
    user_prompt: str


@dataclass
class ScenarioResult:
    """Everything produced for one scenario by one model backend."""

    position_id: str
    tier: str
    recommended_san: Optional[str]
    recommended_uci: Optional[str]
    coaching: str
    objective: Dict[str, bool]
    engine_speak_hits: List[str]
    judge: Optional[Dict[str, Any]]

    def to_json(self, *, include_coaching: bool) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "position_id": self.position_id,
            "tier": self.tier,
            "recommended_san": self.recommended_san,
            "recommended_uci": self.recommended_uci,
            "objective": self.objective,
            "engine_speak_hits": self.engine_speak_hits,
            "judge": self.judge,
        }
        if include_coaching:
            row["coaching"] = self.coaching
        return row


# --------------------------------------------------------------------------- #
# Prompt loading
# --------------------------------------------------------------------------- #


def _read_prompt(name: str) -> str:
    """Read a prompt file from ``prompts/`` (e.g. ``coach_system.md``)."""
    return (settings.PROMPTS / name).read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- #
# Scenario building
# --------------------------------------------------------------------------- #


def _load_positions(path: Path) -> List[schema.Position]:
    """Load the newline-delimited positions catalog."""
    rows: List[schema.Position] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _interleave_by_tier(positions: Sequence[schema.Position]) -> List[schema.Position]:
    """Round-robin positions across tiers so a small draw still covers each tier."""
    by_tier: Dict[str, List[schema.Position]] = {t: [] for t in TIER_ORDER}
    for pos in positions:
        by_tier.setdefault(pos["tier"], []).append(pos)
    ordered: List[schema.Position] = []
    idx = 0
    while any(idx < len(v) for v in by_tier.values()):
        for tier in TIER_ORDER:
            bucket = by_tier.get(tier, [])
            if idx < len(bucket):
                ordered.append(bucket[idx])
        idx += 1
    return ordered


def build_scenario(
    pos: schema.Position,
    *,
    movetime_ms: int,
    tolerance_cp: int,
    multipv: int,
    maia_top_k: int,
) -> Optional[Scenario]:
    """Assemble a :class:`Scenario` from one position, or ``None`` if unteachable.

    Skips positions whose played move is not a real mistake (severity ``"none"``)
    or that have an empty sound pool — there is nothing to coach in either case.

    Parameters
    ----------
    pos:
        A row from ``positions.jsonl``.
    movetime_ms, tolerance_cp, multipv:
        Stockfish search / soundness parameters.
    maia_top_k:
        How many human-likely moves to attach.
    """
    fen = pos["fen"]
    tier = pos["tier"]
    played = pos["played_move_uci"]

    classified = stockfish_engine.classify_mistake(fen, played, movetime_ms=movetime_ms)
    if classified["severity"] == "none":
        return None

    pool_raw = stockfish_engine.sound_pool(
        fen, tolerance_cp=tolerance_cp, multipv=multipv, movetime_ms=movetime_ms
    )
    sound_pool: List[schema.SoundMove] = [
        {"san": m["san"], "uci": m["uci"], "cp": m["cp"], "pv": m["pv"]}
        for m in pool_raw
        if m.get("san") and m.get("uci")
    ]
    if not sound_pool:
        return None

    maia = maia_engine.human_moves(fen, tier, top_k=maia_top_k)["moves"]
    maia_moves: List[schema.MaiaMove] = [
        {"san": m["san"], "uci": m["uci"], "policy": m["policy"]} for m in maia
    ]

    student_move: schema.StudentMove = {
        "san": pos["played_move_san"],
        "uci": played,
        "cp_loss": int(classified["cp_loss"]),
        "severity": classified["severity"],
    }
    teacher_input: schema.TeacherInput = {
        "tier": tier,
        "fen": fen,
        "move_history_san": None,  # positions carry ply index, not SAN history
        "student_move": student_move,
        "sound_pool": sound_pool,
        "maia_human_moves": maia_moves,
    }
    return Scenario(
        position_id=pos["id"],
        tier=tier,
        teacher_input=teacher_input,
        sound_uci=frozenset(m["uci"] for m in sound_pool),
        student_uci=played,
        user_prompt=schema.render_user_prompt(teacher_input),
    )


def build_scenarios(
    positions: Sequence[schema.Position],
    *,
    num_scenarios: int,
    movetime_ms: int,
    tolerance_cp: int,
    multipv: int,
    maia_top_k: int,
) -> List[Scenario]:
    """Build up to ``num_scenarios`` teachable scenarios, balanced across tiers."""
    scenarios: List[Scenario] = []
    for pos in _interleave_by_tier(positions):
        if len(scenarios) >= num_scenarios:
            break
        try:
            scenario = build_scenario(
                pos,
                movetime_ms=movetime_ms,
                tolerance_cp=tolerance_cp,
                multipv=multipv,
                maia_top_k=maia_top_k,
            )
        except Exception as exc:  # noqa: BLE001 - one bad position shouldn't abort
            print(f"  ! skipped {pos.get('id')}: {exc}", file=sys.stderr)
            continue
        if scenario is not None:
            scenarios.append(scenario)
            print(
                f"  + scenario {len(scenarios):>2}/{num_scenarios} "
                f"[{scenario.tier}] {scenario.position_id} "
                f"(pool={len(scenario.teacher_input['sound_pool'])}, "
                f"sev={scenario.teacher_input['student_move']['severity']})",
                file=sys.stderr,
            )
    return scenarios


# --------------------------------------------------------------------------- #
# Model backends
# --------------------------------------------------------------------------- #


def _strip_think(text: str) -> str:
    """Remove Qwen ``<think>...</think>`` reasoning (closed or dangling)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


class MLXBackend:
    """A local MLX chat model that turns (system, user) into coaching text.

    Loads once via ``mlx_lm.load`` and generates greedily (``temp=0``) for
    reproducible evals. Works for BOTH the base and tuned models — the tuned run
    is just this backend pointed at a different model path/repo.
    """

    def __init__(self, model_path: str, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        from mlx_lm import generate, load  # imported lazily (heavy)

        self.model_path = model_path
        self.max_tokens = max_tokens
        self._generate = generate
        t0 = time.time()
        self.model, self.tokenizer = load(model_path)
        print(f"  loaded MLX model {model_path!r} in {time.time() - t0:.1f}s",
              file=sys.stderr)
        try:
            from mlx_lm.sample_utils import make_sampler

            self._sampler = make_sampler(temp=0.0)  # greedy => deterministic
        except Exception:  # noqa: BLE001 - older mlx_lm: fall back to defaults
            self._sampler = None

    def _apply_template(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:  # tokenizer without the enable_thinking kwarg
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

    def generate(self, system: str, user: str) -> str:
        """Produce coaching text for one prompt (``<think>`` stripped)."""
        prompt = self._apply_template(system, user)
        kwargs: Dict[str, Any] = {"max_tokens": self.max_tokens, "verbose": False}
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler
        raw = self._generate(self.model, self.tokenizer, prompt=prompt, **kwargs)
        return _strip_think(raw)


# --------------------------------------------------------------------------- #
# Objective checks (deterministic; run before the judge)
# --------------------------------------------------------------------------- #


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Return the first balanced ``{...}`` JSON object in ``text``, or ``None``."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    return None


def _clean_token(tok: str) -> str:
    """Strip surrounding punctuation and a leading move number from a token."""
    tok = tok.strip().strip(".,;:!?()[]{}\"'`*")
    m = _MOVE_NUMBER.match(tok)
    if m:
        tok = m.group(1)
    return tok


def _legal_uci(board: chess.Board, san_token: str) -> Optional[str]:
    """Return the UCI of ``san_token`` if it is a legal SAN in ``board``, else None."""
    try:
        move = board.parse_san(san_token)
    except (ValueError, AssertionError):
        return None
    return move.uci()


def extract_recommended_move(
    text: str, fen: str, student_uci: str
) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort extraction of the coach's recommended move as ``(san, uci)``.

    Strategy (deterministic; shares its avoid-framing-aware core with the shipped
    coach in :func:`src.teacher.coach_gate.pick_recommendation`):
      1. If the output embeds a JSON object with ``recommended_move_uci`` /
         ``recommended_move_san`` (a tuned model may emit structured output), use
         it when legal.
      2. Otherwise return the coach's ACTUAL recommended move via
         :func:`pick_recommendation` — the move after an explicit "I'd play X" /
         "the move: X" / "Play X again" cue (even if it equals the student's own
         move, e.g. the student already played the best move), else the first
         non-student move that is NOT framed as one to avoid.

    Crucially, a move the coach frames as one to AVOID ("rather than ...b2+",
    "a forcing move like ...b2+"), an opponent reply / continuation ("if White
    plays Ke4", "after d2+ ...f3"), or a bare square reference ("the rook goes to
    h3") is never mistaken for the recommendation. Any legal move is accepted (the
    move need not be sound — soundness is scored separately), so an unsound pick
    is reported honestly rather than hidden.

    Returns ``(None, None)`` when no recommended move can be recovered.
    """
    board = chess.Board(fen)

    obj = _extract_json_object(text)
    if obj:
        uci = obj.get("recommended_move_uci")
        san = obj.get("recommended_move_san")
        if isinstance(uci, str):
            try:
                mv = chess.Move.from_uci(uci.strip())
                if mv in board.legal_moves:
                    return board.san(mv), mv.uci()
            except ValueError:
                pass
        if isinstance(san, str):
            got = _legal_uci(board, san.strip())
            if got is not None:
                return board.san(chess.Move.from_uci(got)), got

    picked = pick_recommendation(text, board, student_uci, accept=lambda _u: True)
    return picked if picked is not None else (None, None)


def find_engine_speak(text: str) -> List[str]:
    """Return the distinct forbidden engine-speak snippets found in ``text``."""
    hits = [m.group(0).strip() for m in _ENGINE_SPEAK_RE.finditer(text)]
    seen: Dict[str, None] = {}
    for h in hits:
        seen.setdefault(h, None)
    return list(seen)


def longest_narrated_line(text: str) -> int:
    """Length (in plies) of the longest run of consecutive SAN moves in prose.

    A deterministic proxy for "how deep a line did the coach narrate": move
    numbers act as connectors, any non-move word breaks the run. Used for the
    ply-cap check.
    """
    longest = 0
    run = 0
    for raw in text.split():
        tok = _clean_token(raw)
        if tok == "":  # a bare move number like "12." -> connector
            continue
        if tok in _RESULT_TOKENS:
            run = 0
            continue
        if _SAN_SHAPE.fullmatch(tok):
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def objective_checks(scenario: Scenario, coaching: str) -> Tuple[
    Dict[str, bool], Optional[str], Optional[str], List[str]
]:
    """Run the deterministic gates; return (checks, rec_san, rec_uci, speak_hits)."""
    ply_cap = settings.TIERS[scenario.tier]["ply_cap"]
    rec_san, rec_uci = extract_recommended_move(
        coaching, scenario.teacher_input["fen"], scenario.student_uci
    )
    speak_hits = find_engine_speak(coaching)

    checks: Dict[str, bool] = {
        "produced_nonempty": bool(coaching.strip()),
        "move_parseable": rec_uci is not None,
        "move_sound": rec_uci is not None and rec_uci in scenario.sound_uci,
        "no_engine_speak": len(speak_hits) == 0,
        "ply_cap_ok": longest_narrated_line(coaching) <= ply_cap,
    }
    return checks, rec_san, rec_uci, speak_hits


# --------------------------------------------------------------------------- #
# LLM judge (gpt-5.5-pro via the Responses API)
# --------------------------------------------------------------------------- #


class OpenAIJudge:
    """Rubric judge backed by ``gpt-5.5-pro`` through the OpenAI Responses API.

    ``gpt-5.5-pro`` is a reasoning model served on ``v1/responses`` (not
    ``chat/completions``); we request ``reasoning.effort="high"`` and force JSON
    output. The key is read from the environment (loaded from ``ROOT/.env``) and
    never logged.

    Reminder: an OpenAI judge grading GPT-5.5-teacher data is same-family; prefer
    a different-family judge (e.g. Claude) for the headline base-vs-tuned report.
    """

    def __init__(
        self, model: str = DEFAULT_JUDGE_MODEL, *, timeout: float = 180.0
    ) -> None:
        from openai import OpenAI  # lazy import

        tfy_key = os.environ.get("TFY_API_KEY")
        tfy_base = os.environ.get("TFY_BASE_URL")
        if tfy_key and tfy_base:
            # Cross-family judge via the TrueFoundry gateway (OpenAI-compatible
            # chat.completions). Preferred for the headline: avoids grading
            # GPT-5.5 teacher data with a GPT-5.5 judge (preference leakage).
            self.model = (
                model if (model and model != DEFAULT_JUDGE_MODEL)
                else (os.environ.get("TFY_JUDGE_MODEL") or "claude-group/claude-opus-4-8")
            )
            self._client = OpenAI(api_key=tfy_key, base_url=tfy_base, timeout=timeout, max_retries=1)
            self._use_chat = True
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "No TFY_API_KEY/TFY_BASE_URL and no OPENAI_API_KEY set in ROOT/.env."
                )
            self.model = model
            self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=1)
            self._use_chat = False
        self._instructions = self._build_instructions()

    @staticmethod
    def _build_instructions() -> str:
        rubric = _read_prompt("eval_rubric.md")
        tier_guides = _read_prompt("tier_guides.md")
        return (
            "You are a strict, fair evaluator of chess move-review COACHING for "
            "students at a stated rating tier. You are grading one model's output "
            "against the project's single behavior spec. Grade ONLY what is "
            "written; never reward fluency that violates the spec.\n\n"
            "=== BEHAVIOR SPEC ===\n"
            f"{settings.BEHAVIOR_SPEC}\n\n"
            "=== RUBRIC (score each dimension 0, 1, or 2) ===\n"
            f"{rubric}\n\n"
            "=== PER-TIER LEVELING GUIDES ===\n"
            f"{tier_guides}\n\n"
            "You are given, as private reference, the same prompt the coach saw "
            "(including internal engine numbers). Those numbers are for YOUR "
            "grading only; the coach must NOT repeat them — leaking any of them "
            "is an automatic 0 on 'no_engine_speak'.\n\n"
            "Return a single JSON object, nothing else, with integer fields "
            "spec_adherence, level_calibration, no_engine_speak, truthfulness, "
            "task_quality (each 0, 1, or 2) and a short string field 'rationale' "
            "(one sentence, no engine numbers)."
        )

    def _user_input(self, scenario: Scenario, coaching: str) -> str:
        return (
            f"STUDENT TIER: {scenario.tier}\n\n"
            "COACH PROMPT (private reference — includes internal numbers the coach "
            "must not echo):\n"
            f"{scenario.user_prompt}\n\n"
            "COACH OUTPUT TO GRADE (verbatim):\n"
            "<<<\n"
            f"{coaching}\n"
            ">>>\n\n"
            "Score the five rubric dimensions and reply with a single JSON object."
        )

    def score(self, scenario: Scenario, coaching: str) -> Dict[str, Any]:
        """Score one output; returns the parsed judge dict (raises on hard failure)."""
        if self._use_chat:
            # TrueFoundry gateway: OpenAI-compatible chat.completions. Do NOT pass
            # `temperature` (newest Claude models on the gateway reject it, 400).
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._instructions},
                    {"role": "user", "content": self._user_input(scenario, coaching)},
                ],
                max_tokens=800,
            )
            content = resp.choices[0].message.content or ""
        else:
            resp = self._client.responses.create(
                model=self.model,
                reasoning={"effort": "high"},
                instructions=self._instructions,
                input=self._user_input(scenario, coaching),
                text={"format": {"type": "json_object"}},
            )
            content = resp.output_text or ""
        obj = _extract_json_object(content) or {}
        result: Dict[str, Any] = {}
        for dim in JUDGE_DIMENSIONS:
            val = obj.get(dim)
            try:
                ival = int(val)
            except (TypeError, ValueError):
                ival = 0
            result[dim] = max(0, min(2, ival))
        result["rationale"] = str(obj.get("rationale", ""))[:400]
        return result


# --------------------------------------------------------------------------- #
# Evaluation loop
# --------------------------------------------------------------------------- #


def evaluate_scenarios(
    scenarios: Sequence[Scenario],
    backend: MLXBackend,
    system_prompt: str,
    judge: Optional[OpenAIJudge],
    *,
    judge_workers: int = 4,
) -> List[ScenarioResult]:
    """Generate coaching, run objective checks (sequential), then judge (parallel).

    Generation is local and fast, so it runs sequentially. Judge calls are slow
    network round-trips (a reasoning model at high effort), so they run in a
    small thread pool — cutting wall-clock time roughly ``judge_workers``x while
    each call keeps its own timeout.
    """
    results: List[ScenarioResult] = []
    for i, scenario in enumerate(scenarios, start=1):
        t0 = time.time()
        coaching = backend.generate(system_prompt, scenario.user_prompt)
        checks, rec_san, rec_uci, hits = objective_checks(scenario, coaching)
        results.append(
            ScenarioResult(
                position_id=scenario.position_id,
                tier=scenario.tier,
                recommended_san=rec_san,
                recommended_uci=rec_uci,
                coaching=coaching,
                objective=checks,
                engine_speak_hits=hits,
                judge=None,
            )
        )
        print(
            f"  gen [{i:>2}/{len(scenarios)}] [{scenario.tier}] "
            f"{scenario.position_id}: rec={rec_san or '-'} "
            f"obj={sum(checks.values())}/{len(OBJECTIVE_CHECKS)} "
            f"({time.time() - t0:.1f}s)",
            file=sys.stderr,
        )

    if judge is not None:
        _judge_all(scenarios, results, judge, workers=judge_workers)
    return results


def _judge_all(
    scenarios: Sequence[Scenario],
    results: List[ScenarioResult],
    judge: OpenAIJudge,
    *,
    workers: int,
) -> None:
    """Judge every scenario in parallel, writing scores back into ``results``."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n = len(scenarios)
    print(f"  judging {n} outputs with {min(workers, n)} workers "
          f"(model={judge.model}) ...", file=sys.stderr)
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(workers, n))) as pool:
        futures = {
            pool.submit(judge.score, scenarios[i], results[i].coaching): i
            for i in range(n)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i].judge = fut.result()
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - one bad call must not abort
                status = f"FAILED ({type(exc).__name__})"
                print(f"  ! judge failed on {scenarios[i].position_id}: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
            done += 1
            print(f"  judged [{done:>2}/{n}] {scenarios[i].position_id}: {status}",
                  file=sys.stderr)
    print(f"  judging done in {time.time() - t0:.1f}s", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _aggregate_group(results: Sequence[ScenarioResult]) -> Dict[str, Any]:
    """Aggregate objective pass-rates and judge means for a set of results."""
    n = len(results)
    obj_rate: Dict[str, Optional[float]] = {}
    for check in OBJECTIVE_CHECKS:
        vals = [1.0 if r.objective.get(check) else 0.0 for r in results]
        obj_rate[check] = _mean(vals)

    judged = [r.judge for r in results if r.judge is not None]
    judge_mean: Dict[str, Optional[float]] = {}
    for dim in JUDGE_DIMENSIONS:
        vals = [float(j[dim]) for j in judged if dim in j]
        judge_mean[dim] = _mean(vals)

    return {
        "n": n,
        "n_judged": len(judged),
        "objective_pass_rate": obj_rate,
        "judge_mean": judge_mean,
    }


def aggregate(results: Sequence[ScenarioResult]) -> Dict[str, Any]:
    """Overall + per-tier aggregation."""
    by_tier: Dict[str, Any] = {}
    for tier in TIER_ORDER:
        group = [r for r in results if r.tier == tier]
        if group:
            by_tier[tier] = _aggregate_group(group)
    return {"overall": _aggregate_group(results), "by_tier": by_tier}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt_pct(x: Optional[float]) -> str:
    return "  -  " if x is None else f"{x * 100:4.0f}%"


def _fmt_score(x: Optional[float]) -> str:
    return "  -  " if x is None else f"{x:4.2f}"


def _columns(agg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    cols: List[Tuple[str, Dict[str, Any]]] = []
    for tier in TIER_ORDER:
        if tier in agg["by_tier"]:
            cols.append((tier[:5], agg["by_tier"][tier]))
    cols.append(("ALL", agg["overall"]))
    return cols


def render_table(label: str, meta: Dict[str, Any], agg: Dict[str, Any]) -> str:
    """Render the human-readable results table for one model."""
    cols = _columns(agg)
    lines: List[str] = []
    lines.append("=" * 74)
    lines.append(f"CHESS-COACH EVAL — model: {label}   ({meta['model_path']})")
    lines.append(
        f"scenarios: {agg['overall']['n']}   judged: {agg['overall']['n_judged']}"
        f"   judge: {meta['judge_model']}"
    )
    lines.append("=" * 74)

    header = f"{'':<22}" + "".join(f"{name:>10}" for name, _ in cols)
    lines.append("OBJECTIVE CHECKS (% passing)")
    lines.append(header)
    lines.append("-" * 74)
    for check in OBJECTIVE_CHECKS:
        row = f"{check:<22}"
        for _, data in cols:
            row += f"{_fmt_pct(data['objective_pass_rate'][check]):>10}"
        lines.append(row)

    lines.append("")
    lines.append("LLM-JUDGE (mean 0-2)")
    lines.append(header)
    lines.append("-" * 74)
    for dim in JUDGE_DIMENSIONS:
        row = f"{dim:<22}"
        for _, data in cols:
            row += f"{_fmt_score(data['judge_mean'][dim]):>10}"
        lines.append(row)
    lines.append("=" * 74)
    return "\n".join(lines)


def render_delta(
    base: Dict[str, Any], tuned: Dict[str, Any]
) -> str:
    """Render a base-vs-tuned side-by-side delta table (overall only)."""
    b, t = base["aggregate"]["overall"], tuned["aggregate"]["overall"]
    lines: List[str] = []
    lines.append("=" * 74)
    lines.append(
        f"DELTA — base ({base['model_label']}) vs tuned ({tuned['model_label']})"
    )
    lines.append("=" * 74)
    lines.append(f"{'':<22}{'base':>12}{'tuned':>12}{'delta':>12}")
    lines.append("OBJECTIVE (% passing)")
    lines.append("-" * 74)
    for check in OBJECTIVE_CHECKS:
        bv = b["objective_pass_rate"].get(check)
        tv = t["objective_pass_rate"].get(check)
        dv = None if bv is None or tv is None else (tv - bv)
        lines.append(
            f"{check:<22}{_fmt_pct(bv):>12}{_fmt_pct(tv):>12}"
            f"{('  -  ' if dv is None else f'{dv * 100:+4.0f}%'):>12}"
        )
    lines.append("")
    lines.append("LLM-JUDGE (mean 0-2)")
    lines.append("-" * 74)
    for dim in JUDGE_DIMENSIONS:
        bv = b["judge_mean"].get(dim)
        tv = t["judge_mean"].get(dim)
        dv = None if bv is None or tv is None else (tv - bv)
        lines.append(
            f"{dim:<22}{_fmt_score(bv):>12}{_fmt_score(tv):>12}"
            f"{('  -  ' if dv is None else f'{dv:+4.2f}'):>12}"
        )
    lines.append("=" * 74)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Base-vs-tuned evaluator for the chess move-review coach.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=("base", "tuned"), default="base",
                   help="Which backend to evaluate.")
    p.add_argument("--mlx-model", default=DEFAULT_MLX_MODEL,
                   help="MLX repo/path for the base model.")
    p.add_argument("--tuned-path", default=None,
                   help="MLX repo/path for the tuned model (required for --model tuned).")
    p.add_argument("--label", default=None,
                   help="Output label (defaults to the --model value).")
    p.add_argument("--num-scenarios", type=int, default=DEFAULT_NUM_SCENARIOS,
                   help="How many held-out scenarios to build.")
    p.add_argument("--positions", default=str(settings.POSITIONS / "positions.jsonl"),
                   help="Path to positions.jsonl.")
    p.add_argument("--movetime", type=int, default=settings.DEFAULT_MOVETIME_MS,
                   help="Stockfish ms per position.")
    p.add_argument("--tolerance", type=int, default=settings.SOUND_TOLERANCE_CP,
                   help="Sound-pool cp tolerance.")
    p.add_argument("--multipv", type=int, default=settings.MULTIPV,
                   help="Stockfish MultiPV for the sound pool.")
    p.add_argument("--maia-top-k", type=int, default=6,
                   help="Human-likely moves to attach per scenario.")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="Max new tokens per generation.")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help="OpenAI judge model (Responses API).")
    p.add_argument("--judge-workers", type=int, default=4,
                   help="Parallel judge calls (network-bound).")
    p.add_argument("--judge-timeout", type=float, default=180.0,
                   help="Per-judge-call timeout in seconds.")
    p.add_argument("--no-judge", action="store_true",
                   help="Skip the LLM judge (objective checks only; no API cost).")
    p.add_argument("--compare-to", default=None,
                   help="Path to another results_*.json to print a delta against.")
    p.add_argument("--out", default=None,
                   help="Override the results JSON output path.")
    return p


def _load_env() -> None:
    """Load ROOT/.env so OPENAI_API_KEY is available (value never printed)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(settings.ROOT / ".env")
    except Exception:  # noqa: BLE001 - dotenv optional; env may already be set
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Builds scenarios, evaluates one backend, writes results."""
    args = _build_parser().parse_args(argv)
    _load_env()

    label = args.label or args.model

    # Resolve the model backend path (tuned is a stub until a path is provided).
    if args.model == "tuned" and not args.tuned_path:
        print(
            "tuned model not available yet (stub). Train it, then re-run with "
            "--tuned-path <mlx_model_dir>. The base run is what proves the harness.",
            file=sys.stderr,
        )
        return 0
    model_path = args.tuned_path if args.model == "tuned" else args.mlx_model

    # 1) Scenarios --------------------------------------------------------- #
    print(f"[1/4] Building up to {args.num_scenarios} held-out scenarios "
          f"(movetime={args.movetime}ms, tol={args.tolerance}cp, "
          f"multipv={args.multipv}) ...", file=sys.stderr)
    positions = _load_positions(Path(args.positions))
    scenarios = build_scenarios(
        positions,
        num_scenarios=args.num_scenarios,
        movetime_ms=args.movetime,
        tolerance_cp=args.tolerance,
        multipv=args.multipv,
        maia_top_k=args.maia_top_k,
    )
    if not scenarios:
        print("No teachable scenarios could be built.", file=sys.stderr)
        return 1

    # 2) Backend + judge --------------------------------------------------- #
    print(f"[2/4] Loading backend {model_path!r} ...", file=sys.stderr)
    backend = MLXBackend(model_path, max_tokens=args.max_tokens)
    system_prompt = _read_prompt("coach_system.md")
    judge: Optional[OpenAIJudge] = None
    if not args.no_judge:
        print(f"[2/4] Initializing judge {args.judge_model!r} ...", file=sys.stderr)
        judge = OpenAIJudge(args.judge_model, timeout=args.judge_timeout)

    # 3) Evaluate ---------------------------------------------------------- #
    print(f"[3/4] Evaluating {len(scenarios)} scenarios with {label!r} ...",
          file=sys.stderr)
    results = evaluate_scenarios(
        scenarios, backend, system_prompt, judge, judge_workers=args.judge_workers
    )
    agg = aggregate(results)

    # 4) Persist + render -------------------------------------------------- #
    meta = {
        "model_label": label,
        "backend": "mlx",
        "model_path": model_path,
        "judge_model": judge.model if judge is not None else "none",
        "params": {
            "num_scenarios": len(scenarios),
            "movetime_ms": args.movetime,
            "tolerance_cp": args.tolerance,
            "multipv": args.multipv,
            "max_tokens": args.max_tokens,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    payload = {
        **meta,
        "aggregate": agg,
        "scenarios": [r.to_json(include_coaching=True) for r in results],
    }

    out_dir = settings.DATA / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"results_{label}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[4/4] Wrote {out_path}", file=sys.stderr)

    print()
    print(render_table(label, meta, agg))
    print(f"\nresults JSON: {out_path}")

    if args.compare_to:
        other_path = Path(args.compare_to)
        if other_path.is_file():
            other = json.loads(other_path.read_text(encoding="utf-8"))
            base, tuned = (other, payload) if label != "base" else (payload, other)
            print()
            print(render_delta(base, tuned))
        else:
            print(f"\n(compare-to not found: {other_path})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
