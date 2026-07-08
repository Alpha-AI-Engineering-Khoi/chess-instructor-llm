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
from typing import Any, Dict, List, Optional, Tuple

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
from src.engine.faithfulness import verify_text  # noqa: E402
from src.engine.faithfulness_ext import verify_text_ext  # noqa: E402
from src.engine.maia_engine import human_moves  # noqa: E402
from src.engine.position_facts import move_facts, render_pool_facts  # noqa: E402
from src.engine.stockfish_engine import classify_mistake, sound_pool  # noqa: E402

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

#: Origins the Next.js dev/preview server runs on.
CORS_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

#: SAN token (incl. castling / promotion / check markers).
_SAN_RE = re.compile(r"(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)")

#: Phrases that typically precede the coach's recommended move.
_CUE_RE = re.compile(
    r"(?:i['\u2019]?d\s+play|i\s+would\s+play|i['\u2019]?ll\s+play|i\s+play|"
    r"recommend(?:ed)?(?:\s+move)?(?:\s+is)?|best\s+move\s+is|go\s+with|"
    r"choose|consider|play)\s*[:\-]?\s*",
    re.IGNORECASE,
)

#: Splits the coaching body from the trailing "Takeaway:" line.
_TAKEAWAY_RE = re.compile(r"\b(?:key\s+)?take[-\s]?away\s*:\s*", re.IGNORECASE)


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


def _extract_recommended(
    text: str, board: chess.Board, pool: List[SoundMove], student_uci: str
) -> Tuple[Optional[str], Optional[str]]:
    """Extract the recommended move (SAN, UCI) from the coach's free text.

    The recommendation is always a *sound* move that is NOT the student's own
    move. Strategy: a sound move right after a cue phrase ("I'd play ..."), then
    the first sound move named anywhere in the prose, and finally the engine's
    best sound move. We never return the student's move (a coach that phrases the
    pick as "develop the knight to f3" rather than "Nf3" must not be mis-read as
    recommending the mistake the student just played).
    """
    pool_ucis = {m["uci"] for m in pool}

    def _try(token: str) -> Optional[Tuple[str, str]]:
        try:
            move = board.parse_san(token)
        except ValueError:
            return None
        return board.san(move), move.uci()

    # 1) Cue phrase -> a move that is not the student's.
    for cue in _CUE_RE.finditer(text):
        window = text[cue.end() : cue.end() + 16]
        match = _SAN_RE.search(window)
        if match:
            parsed = _try(match.group(1))
            if parsed and parsed[1] != student_uci and parsed[1] in pool_ucis:
                return parsed

    # 2) First sound move named anywhere in the prose (never the student's).
    for match in _SAN_RE.finditer(text):
        parsed = _try(match.group(1))
        if parsed and parsed[1] != student_uci and parsed[1] in pool_ucis:
            return parsed

    # 3) Fallback: the engine's best sound move (guaranteed != the mistake).
    if pool:
        return pool[0]["san"], pool[0]["uci"]
    return None, None


#: A markdown horizontal rule on its own line (base models sometimes emit these).
_HR_LINE_RE = re.compile(r"(?m)^[ \t]*[-*_]{3,}[ \t]*$")


def _split_coaching(text: str) -> Tuple[str, str]:
    """Split the reply into (coaching_body, takeaway).

    Splits at the FIRST "Takeaway:" marker: the body is everything before it and
    the takeaway is the single line after it. Anything past that (small models
    sometimes repeat the whole answer) is dropped, and stray markdown rules are
    removed, so the UI never shows duplicated text or a "Takeaway:" inside the
    body.
    """
    text = (text or "").strip()
    match = _TAKEAWAY_RE.search(text)
    if not match:
        body, takeaway = text, ""
    else:
        body = text[: match.start()].strip()
        rest = text[match.end() :].strip()
        takeaway = rest.split("\n", 1)[0].strip()
        if not body:
            body = text
    body = _HR_LINE_RE.sub("", body).strip()
    return body, takeaway


# --------------------------------------------------------------------------- #
# Verified fallback (guaranteed-truthful coaching, no LLM)
# --------------------------------------------------------------------------- #


def _pick_fallback_move(
    board: chess.Board, pool: List[SoundMove], student_uci: str
) -> Optional[chess.Move]:
    """A sound move for the verified fallback — prefer one that isn't the student's."""
    ordered = [m for m in pool if m.get("uci") and m["uci"] != student_uci]
    ordered += [m for m in pool if m.get("uci") and m["uci"] == student_uci]
    for m in ordered:
        try:
            mv = chess.Move.from_uci(m["uci"])
        except ValueError:
            continue
        if mv in board.legal_moves:
            return mv
    return None


def _finalize_verified(
    board: chess.Board, san: str, body: str, takeaway: str
) -> Tuple[str, str]:
    """Assert the deterministic text is faithful; if an edge case slipped a false
    claim through, swap in a claim-free template wholesale (never strips a line)."""
    if verify_text_ext(f"{body} {takeaway}", board.fen()).ok:
        return body, takeaway
    body = (
        f"I'd play {san}. It's a sound, engine-approved move that keeps your "
        "position solid and your king safe."
    )
    takeaway = "When unsure, choose a safe developing move and don't leave a piece undefended."
    return body, takeaway


def _verified_coaching(board: chess.Board, move: chess.Move) -> Tuple[str, str]:
    """Deterministic ``(coaching, takeaway)`` built ONLY from verified move facts.

    Truthful by construction: every concrete claim is derived from
    :func:`move_facts` (computed from the board with python-chess) and phrased so
    it also holds on the CURRENT position, so it passes :func:`verify_text`
    untouched. Used only when the model cannot produce a faithful explanation
    within the attempt budget — the student still gets a guaranteed-true
    explanation of a sound move instead of a fabricated one.
    """
    f = move_facts(board, move)
    san = f["san"]

    if f["castle"]:
        body = (
            f"I'd play {san}. Castling gets your king to safety and brings a rook "
            "toward the center where it can help."
        )
        takeaway = "Castle early — get your king safe, then start making plans."
        return _finalize_verified(board, san, body, takeaway)

    # What the piece itself does (each phrase is true on the current board).
    if f["is_capture"]:
        if board.is_en_passant(move):
            lead = "captures a pawn en passant"
        elif f["captured"]:
            lead = f"captures the {f['captured']} on {f['to']}"
        else:
            lead = f"makes a capture on {f['to']}"
    elif f["develops"]:
        lead = f"develops the {f['piece']}"
    else:
        lead = f"brings the {f['piece']} to {f['to']}"

    tail: List[str] = []
    # The king is covered by "gives check"; don't also list it under "pressures".
    attacks = [(s, n) for s, n in f["attacks"] if n != "king"]
    if attacks:
        tgts = ", ".join(f"the {n} on {s}" for s, n in attacks[:2])
        tail.append(f"and pressures {tgts}")
    if f["defends"]:
        tgts = ", ".join(f"the {n} on {s}" for s, n in f["defends"][:1])
        tail.append(f"while covering {tgts}")
    if f["is_check"]:
        tail.append("and gives check")

    sentence = f"It {lead}"
    if tail:
        sentence += " " + " ".join(tail)
    body = f"I'd play {san}. {sentence}."

    if f["is_check"]:
        takeaway = "A check with a point forces your opponent to react on your terms."
    elif f["is_capture"]:
        takeaway = "Look for safe captures that win material or trade in your favor."
    elif f["develops"]:
        takeaway = "Develop your pieces toward the center before you attack."
    elif f["attacks"]:
        takeaway = "Put your pieces on squares where they do the most work."
    else:
        takeaway = "Prefer purposeful moves that improve a piece and keep your king safe."
    return _finalize_verified(board, san, body, takeaway)


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


@app.post("/api/coach", response_model=CoachResponse)
def coach(req: CoachRequest) -> CoachResponse:
    """Ground a position in engine + Maia analysis, run the coach, return JSON."""
    fen = (req.fen or "").strip()
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

    tier = (req.tier or "beginner").strip().lower()
    if tier not in settings.TIERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tier {tier!r}. Expected one of: {', '.join(settings.TIERS)}.",
        )

    notes: List[str] = []

    # 1) Engine soundness (always). Empty only for a terminal position (handled).
    try:
        pool = _sound_pool_for(fen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - engine/binary problems -> 500
        log.exception("sound_pool failed")
        raise HTTPException(status_code=500, detail=f"Engine error: {exc}") from exc
    if not pool:
        raise HTTPException(status_code=422, detail="No legal moves to analyze.")

    best = pool[0]  # best-first; rank-1 line is the engine's best move

    # 2) Student move (optional) -> mistake severity + red/rust arrow later.
    student_info: Optional[StudentInfo] = None
    student_schema: StudentMove
    if req.student_move and req.student_move.strip():
        try:
            move = _parse_move(board, req.student_move)
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
        student_info = StudentInfo(**student_schema)
    else:
        student_schema = StudentMove(san="(none provided)", uci="", cp_loss=0, severity="none")

    # 3) Maia human-likelihoods (best effort).
    maia_moves = _maia_best_effort(fen, tier, notes)

    # 4) Assemble the exact TeacherInput contract + render the trained prompt.
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

    # 5) Run the coach model behind a VERIFY-AND-REGENERATE faithfulness gate.
    #    The tuned 1.7B model nails style but sometimes fabricates board facts, so
    #    we check every board claim against the real position and RE-SAMPLE the
    #    whole answer whenever any is false — keeping the FIRST reply that verifies
    #    clean and short-circuiting there. We never strip sentences. The engine
    #    analysis above is computed once; only generation repeats.
    student_uci = student_schema.get("uci") or ""
    fen_norm = board.fen()
    attempts = 0
    verified_reply: Optional[str] = None
    try:
        if FAITHFULNESS_GATE:
            for attempts in range(1, MAX_COACH_ATTEMPTS + 1):
                candidate = get_coach().run(SYSTEM_PROMPT, user_prompt)
                if verify_text_ext(candidate, fen_norm).ok:
                    verified_reply = candidate
                    break
        else:
            attempts = 1
            verified_reply = get_coach().run(SYSTEM_PROMPT, user_prompt)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("model generation failed")
        raise HTTPException(status_code=500, detail=f"Model error: {exc}") from exc

    # 6) Turn the (verified) reply into a recommendation + coaching / takeaway. If
    #    NOTHING verified within the budget, emit a deterministic, engine-derived
    #    explanation of a sound move that is truthful by construction.
    verified_fallback = False
    if verified_reply is not None:
        rec_san, rec_uci = _extract_recommended(verified_reply, board, pool, student_uci)
        coaching_body, takeaway = _split_coaching(verified_reply)
        if rec_san is None or rec_uci is None:
            # Should be impossible (pool is non-empty), but never 500 on a good request.
            rec_san, rec_uci = best["san"], best["uci"]
        if FAITHFULNESS_GATE and attempts > 1:
            notes.append(
                f"The coach's first {attempts - 1} draft(s) stated a false board "
                f"fact and were re-generated; attempt {attempts} verified clean."
            )
    else:
        verified_fallback = True
        fb_move = _pick_fallback_move(board, pool, student_uci) or chess.Move.from_uci(best["uci"])
        coaching_body, takeaway = _verified_coaching(board, fb_move)
        rec_san, rec_uci = board.san(fb_move), fb_move.uci()
        notes.append(
            f"The model kept stating false board facts across {MAX_COACH_ATTEMPTS} "
            "attempts, so a verified, engine-derived explanation was used instead."
        )

    return CoachResponse(
        recommended_move_san=rec_san,
        recommended_move_uci=rec_uci,
        coaching=coaching_body or (verified_reply or "").strip(),
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


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Chess Coach API",
        "model": COACH_MODEL_PATH,
        "tuned": _is_tuned(),
        "endpoints": ["/api/health", "/api/examples", "POST /api/coach"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
