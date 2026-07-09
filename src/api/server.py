"""FastAPI backend for the chess-coaching studio (THE ANALYSIS ROOM).

This is a *thin* HTTP layer over the pieces the repo already builds. It does not
re-implement any chess logic; it wires them together for the Next.js front end:

* **Stockfish** (:mod:`src.engine.stockfish_engine`) supplies engine truth — the
  sound-move pool and, when the student played a move, how bad it was.
* **Maia** (:mod:`src.engine.maia_engine`) supplies human-likelihood — which sound
  moves a player at the chosen tier would actually consider (best effort; the API
  degrades gracefully if lc0 / the weights are unavailable).
* **schema** (:mod:`config.schema`) assembles those facts into the exact
  ``TeacherInput`` prompt text the model was trained on (``render_user_prompt``).
* **The MLX model** (via ``mlx_lm``) reads ``prompts/coach_system.md`` + that
  prompt and produces plain-language coaching that recommends one sound move.

Model backend
-------------
The MLX model is loaded **once** at startup. The default is the *base*
``mlx-community/Qwen3-1.7B-4bit`` so the server runs immediately with no tuned
weights. Point at the fine-tuned coach with a single env var::

    COACH_MODEL_PATH=/path/to/tuned-mlx-model   # a fused MLX model dir/repo
    # ...or keep the base model and apply an MLX LoRA adapter:
    COACH_ADAPTER_PATH=/path/to/mlx-adapter

Run
---
    # from the repo root, with the MLX venv python
    ~/.venvs/mlx/bin/python -m uvicorn src.api.server:app --port 8000
    # point at the tuned model
    COACH_MODEL_PATH=./models/qwen3-coach-mlx \
        ~/.venvs/mlx/bin/python -m uvicorn src.api.server:app --port 8000

No secrets are needed: the model is local and engine analysis is computed live.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Path bootstrap: allow `uvicorn src.api.server:app` (and direct execution) to
# import the project packages (`config` / `src`) that assume the repo root is on
# the path. The repo uses namespace packages (no __init__.py), so this is what
# makes the imports below resolve regardless of the launch cwd.
# --------------------------------------------------------------------------- #
_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chess  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from config import settings  # noqa: E402
from config.schema import (  # noqa: E402
    MaiaMove,
    SoundMove,
    StudentMove,
    TeacherInput,
    render_user_prompt,
)
from src.engine.maia_engine import human_moves  # noqa: E402
from src.engine.position_facts import render_pool_facts  # noqa: E402
from src.engine.stockfish_engine import classify_mistake, sound_pool  # noqa: E402

# The gate + verified fallback live in a single shared module so the live coach
# and the HONEST base-vs-tuned eval run byte-identical code (only weights differ).
from src.teacher.coach_gate import run_gate  # noqa: E402

# Re-export the historical ``_``-prefixed helper names other modules import from
# here (e.g. ``src.demo.app`` uses ``_pick_fallback_move`` / ``_verified_coaching``);
# these are intentional re-exports, not dead code.
from src.teacher.coach_gate import (  # noqa: E402,F401
    extract_recommended as _extract_recommended,
    finalize_verified as _finalize_verified,
    pick_fallback_move as _pick_fallback_move,
    split_coaching as _split_coaching,
    verified_coaching as _verified_coaching,
)

try:  # optional: load COACH_MODEL_PATH / COACH_ADAPTER_PATH etc. from ROOT/.env
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    pass

log = logging.getLogger("api.server")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Default MLX model — the *base* Qwen3, so the server runs with no tuned weights.
DEFAULT_MODEL: str = "mlx-community/Qwen3-1.7B-4bit"

#: Selected model + optional adapter (the one-env-var tuned-model swap).
COACH_MODEL_PATH: str = os.environ.get("COACH_MODEL_PATH", DEFAULT_MODEL)
COACH_ADAPTER_PATH: Optional[str] = os.environ.get("COACH_ADAPTER_PATH") or None

#: Generation settings (Qwen3 "non-thinking" recommended sampling). Lowering the
#: temperature was tried for faithfulness but measured WORSE (the small model locks
#: onto a fabricated pattern more deterministically), so we keep the default.
GEN_MAX_TOKENS: int = 640
GEN_TEMP: float = 0.7
GEN_TOP_P: float = 0.8
GEN_TOP_K: int = 20

#: Faithfulness gate (VERIFY-AND-REGENERATE). After the model writes a reply we
#: check every board claim against the real position (:mod:`src.engine.faithfulness`).
#: If ANY claim is false, we RE-SAMPLE the whole answer and re-check — never
#: stripping sentences — keeping the first reply that verifies clean, and
#: short-circuiting as soon as one passes. If no attempt verifies within the
#: budget, we emit a deterministic, engine-derived explanation that is true by
#: construction. Set ``COACH_FAITHFULNESS_GATE=0`` to disable (used only to
#: re-measure the ungated fabrication rate); ``COACH_MAX_ATTEMPTS`` caps re-tries.
MAX_COACH_ATTEMPTS: int = max(1, int(os.environ.get("COACH_MAX_ATTEMPTS", "6")))
FAITHFULNESS_GATE: bool = (
    os.environ.get("COACH_FAITHFULNESS_GATE", "1").strip().lower()
    not in ("0", "false", "no", "off")
)

#: The coach system prompt shipped in the repo (reused verbatim) plus a small,
#: clearly-scoped output-format hint so the free-text reply reliably parses into
#: a coaching body + a single "Takeaway:" line for the UI's pull-quote.
_COACH_SYSTEM: str = (settings.PROMPTS / "coach_system.md").read_text(encoding="utf-8").strip()
_GROUNDING: str = (
    "\n\nYou will be given a VERIFIED FACTS block listing the exact pieces on the "
    "board, which pieces are loose, and what each candidate move concretely does. "
    "Ground EVERY concrete claim — pieces, squares, captures, threats — in that "
    "block. Never mention a piece, square, or capture that is not in the facts. If "
    "you are unsure a detail is true, leave it out and speak about the plan instead."
)
_FORMAT_SUFFIX: str = (
    "\n\nWrite your reply as plain prose for the student: two to four short "
    "sentences of coaching, then a final separate line that begins exactly with "
    '"Takeaway:" stating one transferable idea in a single sentence. Do not use '
    "markdown, headings, or bullet points."
)
SYSTEM_PROMPT: str = _COACH_SYSTEM + _GROUNDING + _FORMAT_SUFFIX

#: Browser origins allowed to call this API cross-origin (the CORS allowlist).
#:
#: The LIVE demo is a Hugging Face **Static Space** whose exported Next.js client
#: fetches this API DIRECTLY from the browser (see ``web/next.config.ts`` ->
#: ``NEXT_PUBLIC_API_BASE`` points at the Modal endpoint). Because that fetch is
#: cross-origin and sends ``Content-Type: application/json``, the browser issues a
#: CORS *preflight* (OPTIONS) first; if the Space origin is not on this list the
#: preflight is rejected ("Disallowed CORS origin", no ``Access-Control-Allow-Origin``)
#: and the live coach never responds. So the deployed Space origin MUST be allowed
#: here (localhost stays for ``next dev`` / preview). This same module is imported
#: verbatim by the Modal serve scripts (``src/serve/serve_v4_4bit_modal.py`` does
#: ``self._app = server.app``), so listing the origin here fixes the deployed app.
#: Extra origins can be added at deploy time via ``COACH_CORS_ORIGINS``
#: (comma-separated) without a code change — handy if the Space is ever renamed.
CORS_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Live Hugging Face Static Space — the browser origin of the shipped demo.
    "https://khoilamalphaai-chess-coach-studio.static.hf.space",
]
_EXTRA_CORS: str = os.environ.get("COACH_CORS_ORIGINS", "")
if _EXTRA_CORS.strip():
    for _o in _EXTRA_CORS.split(","):
        _o = _o.strip()
        if _o and _o not in CORS_ORIGINS:
            CORS_ORIGINS.append(_o)


# --------------------------------------------------------------------------- #
# MLX coach model (loaded once, generation serialized)
# --------------------------------------------------------------------------- #


def _strip_think(text: str) -> str:
    """Remove Qwen3 ``<think>...</think>`` reasoning blocks from ``text``."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text and "<think>" not in text:  # dangling close tag
        text = text.split("</think>", 1)[1]
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


class Coach:
    """A local MLX chat model that turns (system, user) into coaching text.

    Loads once via ``mlx_lm.load`` and serializes generation behind a lock (MLX
    generation is not safe to run concurrently). Works for BOTH the base and the
    tuned models — the tuned run is just this pointed at a different model
    path/repo (``COACH_MODEL_PATH``) or an MLX LoRA adapter (``COACH_ADAPTER_PATH``).
    """

    def __init__(self, model_path: str, adapter_path: Optional[str] = None) -> None:
        from mlx_lm import generate, load  # imported lazily (heavy)

        self.model_path = model_path
        self.adapter_path = adapter_path
        self._generate = generate
        t0 = time.time()
        if adapter_path:
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        else:
            self.model, self.tokenizer = load(model_path)
        log.info("loaded MLX model %r in %.1fs", model_path, time.time() - t0)

        try:
            from mlx_lm.sample_utils import make_sampler

            self._sampler = make_sampler(temp=GEN_TEMP, top_p=GEN_TOP_P, top_k=GEN_TOP_K)
        except Exception:  # pragma: no cover - older mlx_lm: fall back to defaults
            self._sampler = None
        self._lock = threading.Lock()

    def _apply_template(self, system: str, user: str) -> Any:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:  # tokenizer without the enable_thinking kwarg
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def run(self, system: str, user: str, max_tokens: int = GEN_MAX_TOKENS) -> str:
        """Produce coaching text for one prompt (``<think>`` stripped)."""
        prompt = self._apply_template(system, user)
        kwargs: Dict[str, Any] = {"max_tokens": max_tokens, "verbose": False}
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler
        with self._lock:
            raw = self._generate(self.model, self.tokenizer, prompt=prompt, **kwargs)
        return _strip_think(raw)


#: The process-wide coach (populated in the lifespan startup).
_COACH: Optional[Coach] = None


def get_coach() -> Coach:
    if _COACH is None:  # pragma: no cover - startup guarantees this
        raise HTTPException(status_code=503, detail="Coach model is not loaded yet.")
    return _COACH


def _is_tuned() -> bool:
    """True when pointed at anything other than the plain base model."""
    return COACH_MODEL_PATH != DEFAULT_MODEL or bool(COACH_ADAPTER_PATH)


# --------------------------------------------------------------------------- #
# Chess + parsing helpers
# --------------------------------------------------------------------------- #


def _parse_move(board: chess.Board, text: str) -> chess.Move:
    """Parse a UCI or SAN move legal in ``board`` (raises ``ValueError`` if not)."""
    text = text.strip()
    if not text:
        raise ValueError("empty move string")
    try:
        cand = chess.Move.from_uci(text)
        if cand in board.legal_moves:
            return cand
    except ValueError:
        pass
    try:
        return board.parse_san(text)
    except ValueError as exc:
        raise ValueError(f"illegal or unparseable move {text!r}") from exc


# --------------------------------------------------------------------------- #
# Move parsing + the recommendation extractor, faithfulness gate, and verified
# fallback all live in :mod:`src.teacher.coach_gate` (imported above as the
# historical ``_extract_recommended`` / ``_split_coaching`` / ``_pick_fallback_move``
# / ``_finalize_verified`` / ``_verified_coaching`` names) so the shipped pipeline
# and the honest eval share one implementation.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class CoachRequest(BaseModel):
    fen: str = Field(..., description="Position in Forsyth-Edwards Notation.")
    tier: str = Field("beginner", description="beginner | intermediate | advanced")
    student_move: Optional[str] = Field(
        None, description="Optional SAN or UCI move the student played."
    )


class EngineMove(BaseModel):
    san: str
    uci: str
    cp: int
    pv: List[str] = Field(default_factory=list)


class StudentInfo(BaseModel):
    san: str
    uci: str
    cp_loss: int
    severity: str


class EngineBlock(BaseModel):
    best_san: str
    best_cp: int
    sound_pool: List[EngineMove]
    student_move: Optional[StudentInfo]


class MaiaInfo(BaseModel):
    san: str
    uci: str
    policy: float


class CoachMeta(BaseModel):
    model: str
    tuned: bool
    notes: List[str] = Field(default_factory=list)
    #: How many generations the faithfulness gate needed (1 = clean first try).
    attempts: int = 1
    #: True when every attempt failed and a verified, engine-derived reply was used.
    verified_fallback: bool = False


class CoachResponse(BaseModel):
    recommended_move_san: str
    recommended_move_uci: str
    coaching: str
    takeaway: str
    concepts_used: List[str]
    side_to_move: str
    engine: EngineBlock
    maia: List[MaiaInfo]
    meta: CoachMeta


class Example(BaseModel):
    label: str
    fen: str
    tier: str
    student_move: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    adapter: Optional[str]
    tuned: bool
    ready: bool


# --------------------------------------------------------------------------- #
# Curated examples (the first is the classic early-queen 2.Qh5 sortie)
# --------------------------------------------------------------------------- #

EXAMPLES: List[Example] = [
    Example(
        label="Early queen: 1.e4 e5 2.Qh5?",
        fen="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        tier="beginner",
        student_move="Qh5",
    ),
    Example(
        label="Fried Liver: is 4.Ng5 sound?",
        fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        tier="intermediate",
        student_move="Ng5",
    ),
    Example(
        label="Italian: White to plan (2.Nf3 Nc6)",
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        tier="intermediate",
        student_move=None,
    ),
    Example(
        label="Queen's Gambit: should Black take on c4?",
        fen="rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
        tier="advanced",
        student_move="dxc4",
    ),
    Example(
        label="Scholar's mate try: 3.Qh5 hitting e5",
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3",
        tier="beginner",
        student_move="Qh5",
    ),
]


# --------------------------------------------------------------------------- #
# App + lifespan (load the model once on startup)
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _COACH
    banner = f"[api] loading MLX coach: {COACH_MODEL_PATH}"
    if COACH_ADAPTER_PATH:
        banner += f"  (+ adapter: {COACH_ADAPTER_PATH})"
    log.info(banner)
    _COACH = Coach(COACH_MODEL_PATH, COACH_ADAPTER_PATH)
    log.info("[api] coach ready (%s)", "tuned" if _is_tuned() else "base")
    yield
    # Tear down any cached lc0 (Maia) processes on shutdown.
    try:
        from src.engine import maia_engine

        maia_engine.close_all()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


app = FastAPI(title="Chess Coach API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=COACH_MODEL_PATH,
        adapter=COACH_ADAPTER_PATH,
        tuned=_is_tuned(),
        ready=_COACH is not None,
    )


@app.get("/api/examples", response_model=List[Example])
def examples() -> List[Example]:
    return EXAMPLES


def _sound_pool_for(fen: str) -> List[SoundMove]:
    """Stockfish sound-move pool (kept small for the UI), best-first."""
    raw = sound_pool(
        fen,
        tolerance_cp=settings.SOUND_TOLERANCE_CP,
        multipv=settings.MULTIPV,
        movetime_ms=settings.DEFAULT_MOVETIME_MS,
    )
    return [
        SoundMove(san=m["san"], uci=m["uci"], cp=int(m["cp"]), pv=m["pv"])
        for m in raw
        if m.get("san") and m.get("uci")
    ]


def _maia_best_effort(fen: str, tier: str, notes: List[str]) -> List[MaiaMove]:
    """Maia human-move likelihoods; empty (with a UI note) if unavailable."""
    try:
        result = human_moves(fen, tier, top_k=6)
        return [
            MaiaMove(san=m["san"], uci=m["uci"], policy=float(m["policy"]))
            for m in result["moves"]
        ]
    except Exception as exc:  # noqa: BLE001 - Maia is a helpful signal, not required
        log.warning("Maia unavailable (%s); continuing without human-move signal", exc)
        notes.append("Human-likelihood (Maia) analysis was unavailable for this position.")
        return []


# --------------------------------------------------------------------------- #
# Shared coaching pipeline (ONE unit reused by /api/coach and /api/coach_all)
#
# The per-position engine truth (Stockfish sound pool + student-move severity) is
# tier-independent, so it is split out from the per-tier work (tier-specific Maia
# net + the trained prompt + the gated generation). /api/coach computes it for one
# tier; /api/coach_all computes it ONCE and reuses it across all three, so the
# Studio's "fetch every band up front" costs a single Stockfish pass, not three.
# --------------------------------------------------------------------------- #


def _validate_position(fen_in: str) -> tuple[str, chess.Board]:
    """Validate + parse a FEN into ``(stripped_fen, board)`` or raise HTTPException."""
    fen = (fen_in or "").strip()
    if not fen:
        raise HTTPException(status_code=400, detail="A FEN is required.")
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {exc}") from exc
    if not board.is_valid():
        raise HTTPException(status_code=400, detail="The FEN is not a legal position.")
    if board.is_game_over():
        raise HTTPException(status_code=422, detail="The game is already over in this position.")
    return fen, board


def _validate_tier(tier_in: str) -> str:
    """Normalize + validate a tier label, or raise HTTPException."""
    tier = (tier_in or "beginner").strip().lower()
    if tier not in settings.TIERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tier {tier!r}. Expected one of: {', '.join(settings.TIERS)}.",
        )
    return tier


def _sound_pool_or_raise(fen: str) -> List[SoundMove]:
    """Stockfish sound-move pool (best-first) or the matching HTTPException.

    Empty only for a terminal position (already rejected upstream) -> 422.
    """
    try:
        pool = _sound_pool_for(fen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - engine/binary problems -> 500
        log.exception("sound_pool failed")
        raise HTTPException(status_code=500, detail=f"Engine error: {exc}") from exc
    if not pool:
        raise HTTPException(status_code=422, detail="No legal moves to analyze.")
    return pool


def _classify_student(
    board: chess.Board, fen: str, student_move: Optional[str]
) -> tuple[Optional[StudentInfo], StudentMove]:
    """Classify the (optional) student move -> ``(student_info, student_schema)``.

    Tier-independent, so the all-tier endpoint runs this (and its Stockfish
    ``classify_mistake`` pass) exactly once for the whole batch.
    """
    if student_move and student_move.strip():
        try:
            move = _parse_move(board, student_move)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            cls = classify_mistake(fen, move.uci(), movetime_ms=settings.DEFAULT_MOVETIME_MS)
        except Exception as exc:  # noqa: BLE001
            log.exception("classify_mistake failed")
            raise HTTPException(status_code=500, detail=f"Engine error: {exc}") from exc
        student_schema = StudentMove(
            san=board.san(move),
            uci=move.uci(),
            cp_loss=int(cls["cp_loss"]),
            severity=str(cls["severity"]),
        )
        return StudentInfo(**student_schema), student_schema
    return None, StudentMove(san="(none provided)", uci="", cp_loss=0, severity="none")


def _coach_for_tier(
    board: chess.Board,
    tier: str,
    pool: List[SoundMove],
    best: SoundMove,
    student_info: Optional[StudentInfo],
    student_schema: StudentMove,
) -> CoachResponse:
    """Run the SHIPPED coach-gate pipeline for ONE tier over an already-computed
    engine pool + student classification, returning the standard CoachResponse.

    Only the tier-specific work happens here — the tier's Maia human-likelihoods,
    the trained prompt, and the gated model generation. The engine truth (sound
    pool + student severity) is passed in, so this is byte-for-byte the answer a
    single ``/api/coach`` call would produce for the tier; ``/api/coach_all`` just
    reuses the same pool across the three tiers.
    """
    notes: List[str] = []

    # Maia human-likelihoods are tier-specific (maia-1100 / 1500 / 1900), so they
    # are read per tier with the CORRECT net for the rating band (best effort).
    maia_moves = _maia_best_effort(board.fen(), tier, notes)

    # Assemble the exact TeacherInput contract + render the trained prompt.
    teacher_input: TeacherInput = TeacherInput(
        tier=tier,
        fen=board.fen(),
        move_history_san=None,
        student_move=student_schema,
        sound_pool=pool,
        maia_human_moves=maia_moves,
    )
    # Prepend the VERIFIED FACTS block so the model explains from truth (piece list
    # + per-move facts in words) instead of guessing off the ASCII board.
    facts = render_pool_facts(board.fen(), list(pool))
    user_prompt = f"{facts}\n\n{render_user_prompt(teacher_input)}"

    # Run the coach model behind the VERIFY-AND-REGENERATE faithfulness gate — the
    # shared :func:`run_gate` unit (:mod:`src.teacher.coach_gate`), the SAME code
    # the honest base-vs-tuned eval runs. If nothing verifies within the attempt
    # budget it returns a deterministic, engine-derived explanation, true by
    # construction.
    student_uci = student_schema.get("uci") or ""
    try:
        result = run_gate(
            lambda system, user: get_coach().run(system, user),
            SYSTEM_PROMPT, user_prompt, board.fen(), list(pool), student_uci,
            max_attempts=MAX_COACH_ATTEMPTS, gate_on=FAITHFULNESS_GATE,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("model generation failed")
        raise HTTPException(status_code=500, detail=f"Model error: {exc}") from exc

    attempts = result.attempts
    verified_fallback = result.verified_fallback
    rec_san = result.rec_san or best["san"]
    rec_uci = result.rec_uci or best["uci"]
    coaching_body, takeaway = result.body, result.takeaway
    if verified_fallback:
        notes.append(
            f"The model kept stating false board facts across {MAX_COACH_ATTEMPTS} "
            "attempts, so a verified, engine-derived explanation was used instead."
        )
    elif FAITHFULNESS_GATE and attempts > 1:
        notes.append(
            f"The coach's first {attempts - 1} draft(s) stated a false board "
            f"fact and were re-generated; attempt {attempts} verified clean."
        )

    return CoachResponse(
        recommended_move_san=rec_san,
        recommended_move_uci=rec_uci,
        coaching=coaching_body or (result.raw or "").strip(),
        takeaway=takeaway,
        concepts_used=[],
        side_to_move="white" if board.turn == chess.WHITE else "black",
        engine=EngineBlock(
            best_san=best["san"],
            best_cp=int(best["cp"]),
            sound_pool=[
                EngineMove(san=m["san"], uci=m["uci"], cp=int(m["cp"]), pv=list(m.get("pv") or []))
                for m in pool
            ],
            student_move=student_info,
        ),
        maia=[MaiaInfo(san=m["san"], uci=m["uci"], policy=float(m["policy"])) for m in maia_moves],
        meta=CoachMeta(
            model=COACH_MODEL_PATH,
            tuned=_is_tuned(),
            notes=notes,
            attempts=attempts,
            verified_fallback=verified_fallback,
        ),
    )


@app.post("/api/coach", response_model=CoachResponse)
def coach(req: CoachRequest) -> CoachResponse:
    """Ground a position in engine + Maia analysis, run the coach, return JSON."""
    fen, board = _validate_position(req.fen)
    tier = _validate_tier(req.tier)
    pool = _sound_pool_or_raise(fen)
    student_info, student_schema = _classify_student(board, fen, req.student_move)
    return _coach_for_tier(board, tier, pool, pool[0], student_info, student_schema)


class CoachAllRequest(BaseModel):
    fen: str = Field(..., description="Position in Forsyth-Edwards Notation.")
    student_move: Optional[str] = Field(
        None, description="Optional SAN or UCI move the student played."
    )


class CoachAllResponse(BaseModel):
    """All three rating bands for one position, each a standard CoachResponse."""

    beginner: CoachResponse
    intermediate: CoachResponse
    advanced: CoachResponse


@app.post("/api/coach_all", response_model=CoachAllResponse)
def coach_all(req: CoachAllRequest) -> CoachAllResponse:
    """Coach a position at ALL THREE tiers from ONE engine pass.

    The Studio fetches every band up front so the Beginner/Intermediate/Advanced
    buttons switch with no new model call. Doing that as three separate
    ``/api/coach`` calls recomputes the (expensive) Stockfish sound pool + the
    student-move severity three times over the identical position. This endpoint
    computes that engine truth exactly ONCE and then runs the SAME shipped
    coach-gate pipeline per tier (:func:`_coach_for_tier`), so each returned value
    is exactly what a single ``/api/coach`` call would produce for that tier — just
    cheaper to obtain together.
    """
    fen, board = _validate_position(req.fen)
    pool = _sound_pool_or_raise(fen)  # computed ONCE, shared across every tier
    best = pool[0]
    student_info, student_schema = _classify_student(board, fen, req.student_move)  # ONCE
    return CoachAllResponse(
        beginner=_coach_for_tier(board, "beginner", pool, best, student_info, student_schema),
        intermediate=_coach_for_tier(
            board, "intermediate", pool, best, student_info, student_schema
        ),
        advanced=_coach_for_tier(board, "advanced", pool, best, student_info, student_schema),
    )


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Chess Coach API",
        "model": COACH_MODEL_PATH,
        "tuned": _is_tuned(),
        "endpoints": [
            "/api/health",
            "/api/examples",
            "POST /api/coach",
            "POST /api/coach_all",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
