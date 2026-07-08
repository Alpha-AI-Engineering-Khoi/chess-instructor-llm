"""Local Gradio demo for the engine-grounded chess-coaching model (Apple Silicon / MLX).

This app ties together the pieces the rest of the repo already builds:

* **Stockfish** (``src.engine.stockfish_engine``) supplies engine *truth* — a pool
  of *sound* candidate moves and, if the student played a move, how bad it was.
* **Maia** (``src.engine.maia_engine``) supplies *human*-likelihood — which sound
  moves a player at the chosen rating tier would actually consider.
* **schema** (``config.schema``) assembles those facts into the exact
  ``TeacherInput`` prompt text the model was trained on (``render_user_prompt``).
* **The model** (MLX via ``mlx_lm``) reads the system prompt + that prompt and
  produces plain-language coaching that recommends exactly one sound move.

The demo renders the position with a GREEN arrow for the recommended move (and a
RED arrow for the student's move, if given), shows the recommended move in SAN,
and shows the coaching text.

Model backend
-------------
The MLX model is selected by ``--model-path`` (or ``MODEL_PATH``); the default is
the **base** ``mlx-community/Qwen3-1.7B-4bit`` so the demo runs immediately. To
show the fine-tuned behaviour later, point ``--model-path`` at a local MLX model
directory (e.g. a fused checkpoint), or pass an MLX-format LoRA adapter directory
via ``--adapter-path`` / ``ADAPTER_PATH``. Nothing else changes.

Run
---
    # from the repo root, with the MLX venv python
    ~/.venvs/mlx/bin/python src/demo/app.py

    # point at a tuned MLX model later
    ~/.venvs/mlx/bin/python src/demo/app.py --model-path /path/to/tuned-mlx-model
    # ...or an MLX LoRA adapter directory
    ~/.venvs/mlx/bin/python src/demo/app.py --adapter-path /path/to/mlx-adapter

No secrets are used. Engine analysis is computed live for every submission.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Path bootstrap: allow ``python src/demo/app.py`` from the repo root to import
# the project packages (``config`` / ``src``) that assume the root is on path.
# --------------------------------------------------------------------------- #
_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chess  # noqa: E402
import chess.svg  # noqa: E402
import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from config import settings  # noqa: E402
from config.schema import (  # noqa: E402
    MaiaMove,
    SoundMove,
    StudentMove,
    TeacherInput,
    render_user_prompt,
)
from src.engine.faithfulness_ext import verify_text_ext  # noqa: E402
from src.engine.maia_engine import human_moves  # noqa: E402
from src.engine.stockfish_engine import classify_mistake, sound_pool  # noqa: E402

# Reuse the API's faithfulness-gate helpers so the demo applies the SAME
# verify-and-regenerate + deterministic-fallback path as ``src/api/server.py``.
# Importing the server module is side-effect-free here: it builds a FastAPI app
# object and reads the coach prompt, but loads no model and starts no server.
from src.api.server import (  # noqa: E402
    MAX_COACH_ATTEMPTS as COACH_MAX_ATTEMPTS,
    _pick_fallback_move,
    _verified_coaching,
)

try:  # optional: load MODEL_PATH / ADAPTER_PATH etc. from the repo .env
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    pass

from mlx_lm import generate, load  # noqa: E402  (import after path bootstrap)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Default MLX model — the *base* Qwen3, so the demo runs with no tuned weights.
DEFAULT_MODEL: str = "mlx-community/Qwen3-1.7B-4bit"

#: Arrow colours for the rendered board.
GREEN: str = "#15803d"  # recommended move
RED: str = "#dc2626"  # student's move

#: Generation settings (Qwen3 "non-thinking" recommended sampling).
GEN_MAX_TOKENS: int = 512
GEN_TEMP: float = 0.7
GEN_TOP_P: float = 0.8
GEN_TOP_K: int = 20

#: The coach system prompt shipped in the repo.
SYSTEM_PROMPT: str = (settings.PROMPTS / "coach_system.md").read_text(encoding="utf-8")

#: Example positions for the dropdown -> (fen, tier, student move). The first is
#: the classic early-queen sortie (2.Qh5) called out in the task.
EXAMPLES: Dict[str, Dict[str, str]] = {
    "Early queen: 1.e4 e5 — is 2.Qh5 good? (beginner)": {
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "tier": "beginner",
        "move": "Qh5",
    },
    "Opening: what should White play from the start? (beginner)": {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "tier": "beginner",
        "move": "",
    },
    "Italian: 1.e4 e5 2.Nf3 Nc6 — best plan for White? (intermediate)": {
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "tier": "intermediate",
        "move": "",
    },
    "Two Knights: is 4.Ng5 (Fried Liver) sound here? (intermediate)": {
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "tier": "intermediate",
        "move": "Ng5",
    },
    "Queen's Gambit: 1.d4 d5 2.c4 — should Black take on c4? (advanced)": {
        "fen": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
        "tier": "advanced",
        "move": "dxc4",
    },
}

#: SAN token (incl. castling / promotion / check markers).
_SAN_RE = re.compile(
    r"(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)"
)

#: Phrases that typically precede the coach's recommended move.
_CUE_RE = re.compile(
    r"(?:i['\u2019]?d\s+play|i\s+would\s+play|i['\u2019]?ll\s+play|i\s+play|"
    r"recommend(?:ed)?(?:\s+move)?(?:\s+is)?|best\s+move\s+is|go\s+with|"
    r"choose|consider|play)\s*[:\-]?\s*",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Model backend (loaded once, reused)
# --------------------------------------------------------------------------- #

_MODEL: Optional[Any] = None
_TOKENIZER: Optional[Any] = None
_MODEL_PATH: str = os.environ.get("MODEL_PATH", DEFAULT_MODEL)
_ADAPTER_PATH: Optional[str] = os.environ.get("ADAPTER_PATH") or None


def get_model() -> Tuple[Any, Any]:
    """Load (and cache) the MLX model + tokenizer selected by the CLI/env.

    Returns
    -------
    tuple
        ``(model, tokenizer)`` — cached after the first call so a running demo
        never reloads weights.
    """
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        if _ADAPTER_PATH:
            _MODEL, _TOKENIZER = load(_MODEL_PATH, adapter_path=_ADAPTER_PATH)
        else:
            _MODEL, _TOKENIZER = load(_MODEL_PATH)
    return _MODEL, _TOKENIZER


def _strip_think(text: str) -> str:
    """Remove Qwen3 ``<think>...</think>`` reasoning blocks from ``text``."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text and "<think>" not in text:  # dangling close tag
        text = text.split("</think>", 1)[1]
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def run_model(system_prompt: str, user_prompt: str, max_tokens: int = GEN_MAX_TOKENS) -> str:
    """Run the MLX model on a chat-templated (system, user) prompt.

    Parameters
    ----------
    system_prompt:
        The coach system prompt.
    user_prompt:
        The rendered ``TeacherInput`` text.
    max_tokens:
        Generation cap.

    Returns
    -------
    str
        The model's reply with any ``<think>`` block stripped.
    """
    model, tokenizer = get_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # ``enable_thinking=False`` is a no-op for templates that ignore it.
    try:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:  # pragma: no cover - older templates
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    kwargs: Dict[str, Any] = {"max_tokens": max_tokens, "verbose": False}
    try:
        from mlx_lm.sample_utils import make_sampler

        kwargs["sampler"] = make_sampler(
            temp=GEN_TEMP, top_p=GEN_TOP_P, top_k=GEN_TOP_K
        )
    except Exception:  # pragma: no cover - fall back to greedy defaults
        pass

    text = generate(model, tokenizer, prompt=prompt, **kwargs)
    return _strip_think(text)


# --------------------------------------------------------------------------- #
# Chess helpers
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


def build_teacher_input(
    fen: str, tier: str, student_move: str
) -> Tuple[TeacherInput, List[str]]:
    """Assemble a :class:`~config.schema.TeacherInput` from live engine analysis.

    Runs Stockfish ``sound_pool`` (always), Maia ``human_moves`` (best effort),
    and Stockfish ``classify_mistake`` (only when a student move is supplied).

    Parameters
    ----------
    fen:
        Position in Forsyth-Edwards Notation (assumed already validated).
    tier:
        ``"beginner"`` / ``"intermediate"`` / ``"advanced"``.
    student_move:
        Optional SAN/UCI move; empty string means "no move played".

    Returns
    -------
    tuple
        ``(teacher_input, notes)`` where ``notes`` are human-readable warnings
        (e.g. Maia unavailable) to surface in the UI.
    """
    board = chess.Board(fen)
    notes: List[str] = []

    pool: List[SoundMove] = [
        SoundMove(san=m["san"], uci=m["uci"], cp=m["cp"], pv=m["pv"])
        for m in sound_pool(fen)
    ]

    maia: List[MaiaMove] = []
    try:
        result = human_moves(fen, tier, top_k=6)
        maia = [
            MaiaMove(san=m["san"], uci=m["uci"], policy=m["policy"])
            for m in result["moves"]
        ]
    except Exception as exc:  # lc0 / weights missing, etc. -> degrade gracefully
        notes.append(
            f"Maia human-move analysis unavailable ({type(exc).__name__}): "
            f"{exc}. Proceeding with engine soundness only."
        )

    if student_move and student_move.strip():
        move = _parse_move(board, student_move)
        cls = classify_mistake(fen, move.uci())
        student = StudentMove(
            san=board.san(move),
            uci=move.uci(),
            cp_loss=int(cls["cp_loss"]),
            severity=str(cls["severity"]),
        )
    else:
        student = StudentMove(san="(none provided)", uci="", cp_loss=0, severity="none")

    teacher_input = TeacherInput(
        tier=tier,
        fen=board.fen(),
        move_history_san=None,
        student_move=student,
        sound_pool=pool,
        maia_human_moves=maia,
    )
    return teacher_input, notes


def extract_recommended(
    text: str, board: chess.Board, pool: List[SoundMove]
) -> Tuple[Optional[str], Optional[str], str]:
    """Extract the recommended move (SAN + UCI) from the model's coaching text.

    Strategy: prefer a legal move right after a cue phrase ("I'd play ..."), then
    any legal move that is in the sound pool, then any legal move, and finally
    fall back to the top sound move so an arrow can always be drawn.

    Parameters
    ----------
    text:
        The model's (think-stripped) reply.
    board:
        The position the move must be legal in.
    pool:
        The engine sound pool (used to prefer sound recommendations / fallback).

    Returns
    -------
    tuple
        ``(san, uci, source)`` where ``source`` is ``"model"``,
        ``"sound-pool fallback"`` or ``"none"``. ``san``/``uci`` may be ``None``
        only when there are no legal moves at all.
    """
    pool_ucis = {m["uci"] for m in pool}

    def _try(token: str) -> Optional[Tuple[str, str]]:
        try:
            move = board.parse_san(token)
        except ValueError:
            return None
        return board.san(move), move.uci()

    # 1) cue-phrase first (most reliable for a well-formed reply).
    for cue in _CUE_RE.finditer(text):
        window = text[cue.end() : cue.end() + 16]
        match = _SAN_RE.search(window)
        if match:
            parsed = _try(match.group(1))
            if parsed:
                return parsed[0], parsed[1], "model"

    # 2) any legal SAN, preferring one from the sound pool.
    first_legal: Optional[Tuple[str, str]] = None
    for match in _SAN_RE.finditer(text):
        parsed = _try(match.group(1))
        if not parsed:
            continue
        if parsed[1] in pool_ucis:
            return parsed[0], parsed[1], "model"
        if first_legal is None:
            first_legal = parsed
    if first_legal is not None:
        return first_legal[0], first_legal[1], "model"

    # 3) fallback to the engine's top sound move.
    if pool:
        return pool[0]["san"], pool[0]["uci"], "sound-pool fallback"
    return None, None, "none"


# --------------------------------------------------------------------------- #
# Board rendering (cairosvg -> PNG image, else inline SVG via gr.HTML)
# --------------------------------------------------------------------------- #


def _svg_to_png(svg: str) -> Optional[bytes]:
    """Rasterise ``svg`` to PNG bytes via cairosvg, or ``None`` on any failure."""
    try:
        import cairosvg

        return cairosvg.svg2png(bytestring=svg.encode("utf-8"))
    except Exception:  # pragma: no cover - environment specific (libcairo, etc.)
        return None


def _detect_image_mode() -> bool:
    """Return ``True`` if cairosvg can rasterise SVG (so we can use ``gr.Image``)."""
    return _svg_to_png("<svg xmlns='http://www.w3.org/2000/svg' width='4' height='4'/>") is not None


#: Whether the board is shown as a rasterised PNG (True) or inline SVG (False).
USE_IMAGE: bool = _detect_image_mode()


def _board_svg(fen: str, rec_uci: Optional[str], student_uci: Optional[str]) -> str:
    """Build the board SVG with a green (recommended) and/or red (student) arrow."""
    board = chess.Board(fen)
    arrows: List[chess.svg.Arrow] = []
    if student_uci:
        mv = chess.Move.from_uci(student_uci)
        arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color=RED))
    if rec_uci:
        mv = chess.Move.from_uci(rec_uci)
        arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color=GREEN))
    return chess.svg.board(board, orientation=board.turn, arrows=arrows, size=420)


def render_board_payload(
    fen: str, rec_uci: Optional[str], student_uci: Optional[str]
) -> Any:
    """Return the board as a PIL image (image mode) or an HTML SVG string.

    The return type matches whichever output component the demo created — a
    :class:`PIL.Image.Image` for ``gr.Image`` or an HTML ``str`` for ``gr.HTML``.
    """
    svg = _board_svg(fen, rec_uci, student_uci)
    if USE_IMAGE:
        png = _svg_to_png(svg)
        if png is not None:
            return Image.open(BytesIO(png))
        return None
    return f"<div style='max-width:440px'>{svg}</div>"


def _empty_board_payload() -> Any:
    """Return an empty payload appropriate for the active board component."""
    return None if USE_IMAGE else ""


def _details_md(
    ti: TeacherInput,
    notes: List[str],
    rec_info: Optional[Tuple[Optional[str], str]],
    student_err: Optional[str] = None,
) -> str:
    """Render a Markdown summary of the live engine analysis for the accordion."""
    lines: List[str] = [
        "*Computed live for this position: Stockfish sound-move pool + "
        "Maia human-move likelihoods.*",
    ]
    sm = ti["student_move"]
    if sm["san"] != "(none provided)":
        lines.append(
            f"**Student move:** {sm['san']} — severity **{sm['severity']}** "
            f"(loses ~{sm['cp_loss']} cp)."
        )
    if student_err:
        lines.append(f"**Note:** could not parse student move — {student_err}")
    sound = ", ".join(m["san"] for m in ti["sound_pool"][:6]) or "—"
    lines.append(f"**Engine-sound moves:** {sound}")
    maia = (
        ", ".join(
            f"{m['san']} {round(m['policy'] * 100)}%" for m in ti["maia_human_moves"][:6]
        )
        or "—"
    )
    lines.append(f"**Human-likely (Maia):** {maia}")
    if rec_info is not None:
        san, source = rec_info
        lines.append(f"**Parsed recommendation:** {san or '—'} (from {source}).")
    for note in notes:
        lines.append(f"_{note}_")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# Main inference callback
# --------------------------------------------------------------------------- #


def coach(fen: str, tier: str, student_move: str) -> Tuple[Any, str, str, str]:
    """Full pipeline for one submission: engine grounding -> model -> render.

    Parameters
    ----------
    fen:
        Position FEN (from the textbox / example dropdown).
    tier:
        Rating tier radio value.
    student_move:
        Optional SAN/UCI move.

    Returns
    -------
    tuple
        ``(board_payload, recommended_san, coaching_markdown, details_markdown)``
        matching the four demo output components.
    """
    fen = (fen or "").strip()
    if not fen:
        return _empty_board_payload(), "", "Enter a FEN or pick an example position.", ""

    try:
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("the position is not legal")
    except Exception as exc:
        return _empty_board_payload(), "", f"**Invalid FEN.** {exc}", ""

    tier = tier if tier in settings.TIERS else "beginner"

    # Validate the student's move up front so we can draw its red arrow even if
    # something else fails, and so we can warn on an unparseable move.
    student_uci: Optional[str] = None
    student_err: Optional[str] = None
    if student_move and student_move.strip():
        try:
            student_uci = _parse_move(board, student_move).uci()
        except ValueError as exc:
            student_err = str(exc)

    try:
        ti, notes = build_teacher_input(
            fen, tier, "" if student_err else student_move
        )
    except Exception as exc:
        return (
            render_board_payload(fen, None, student_uci),
            "",
            f"**Engine error.** {exc}",
            "",
        )

    user_prompt = render_user_prompt(ti)
    fen_norm = board.fen()
    pool = ti["sound_pool"]
    student_uci_norm = student_uci or ""

    # Faithfulness gate (VERIFY-AND-REGENERATE), mirroring src/api/server.py so the
    # demo NEVER shows an ungated fabrication. Re-sample the whole reply up to
    # COACH_MAX_ATTEMPTS and keep the first one whose every board claim verifies
    # against the real position; if none verify, fall back to a deterministic,
    # engine-derived explanation of a sound move that is truthful by construction.
    try:
        verified_reply: Optional[str] = None
        attempts = 0
        for attempts in range(1, COACH_MAX_ATTEMPTS + 1):
            candidate = run_model(SYSTEM_PROMPT, user_prompt)
            if verify_text_ext(candidate, fen_norm).ok:
                verified_reply = candidate
                break
    except Exception as exc:
        return (
            render_board_payload(fen, None, student_uci),
            "",
            f"**Model error.** {exc}",
            _details_md(ti, notes, None, student_err),
        )

    if verified_reply is not None:
        rec_san, rec_uci, source = extract_recommended(verified_reply, board, pool)
        coaching_md = verified_reply.strip() or "*(the model returned an empty response)*"
        if attempts > 1:
            notes.append(
                f"The coach's first {attempts - 1} draft(s) stated a false board "
                f"fact and were re-generated; attempt {attempts} verified clean."
            )
    else:
        fb_move = _pick_fallback_move(board, pool, student_uci_norm)
        if fb_move is None:  # no sound move to explain — never fabricate
            rec_san, rec_uci, source = extract_recommended("", board, pool)
            coaching_md = "*(no verifiable coaching could be produced for this position)*"
        else:
            body, takeaway = _verified_coaching(board, fb_move)
            rec_san, rec_uci, source = board.san(fb_move), fb_move.uci(), "verified fallback"
            coaching_md = f"{body}\n\n**Takeaway:** {takeaway}"
            notes.append(
                f"The model kept stating false board facts across "
                f"{COACH_MAX_ATTEMPTS} attempts, so a verified, engine-derived "
                "explanation was used instead."
            )

    board_payload = render_board_payload(fen, rec_uci, student_uci)
    details_md = _details_md(ti, notes, (rec_san, source), student_err)
    return board_payload, rec_san or "(could not parse)", coaching_md, details_md


def on_example(label: Optional[str]) -> Tuple[Any, Any, Any]:
    """Populate the FEN / tier / move inputs from a selected example label."""
    example = EXAMPLES.get(label or "")
    if not example:
        return gr.update(), gr.update(), gr.update()
    return example["fen"], example["tier"], example["move"]


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #


def build_demo() -> gr.Blocks:
    """Construct and return the Gradio :class:`gr.Blocks` demo."""
    first = next(iter(EXAMPLES.values()))
    legend = "🟢 Green arrow = recommended move  ·  🔴 Red arrow = your move"

    with gr.Blocks(theme=gr.themes.Soft(), title="Chess Coach") as demo:
        gr.Markdown("# \u265f\ufe0f Chess Coach (Qwen3-1.7B, engine-grounded)")
        gr.Markdown(
            "Engine analysis — a **Stockfish** sound-move pool plus **Maia** "
            "human-move likelihoods — is computed **live** for every position. "
            "The model must recommend one sound move and coach it in plain "
            "language for the chosen rating tier."
        )

        with gr.Row():
            with gr.Column(scale=1):
                example_dd = gr.Dropdown(
                    choices=list(EXAMPLES.keys()),
                    label="Example positions",
                    value=None,
                    info="Pick one to fill in the FEN, tier, and move below.",
                )
                fen_in = gr.Textbox(
                    label="FEN",
                    value=first["fen"],
                    lines=2,
                    info="Paste any position in Forsyth-Edwards Notation.",
                )
                tier_in = gr.Radio(
                    choices=list(settings.TIERS.keys()),
                    value=first["tier"],
                    label="Rating tier",
                )
                move_in = gr.Textbox(
                    label="Student's move (optional)",
                    value=first["move"],
                    placeholder="SAN or UCI, e.g. Qh5 or d1h5",
                )
                submit = gr.Button("Coach me", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown(f"**Board** · {legend}")
                if USE_IMAGE:
                    board_out: Any = gr.Image(
                        type="pil", label="Position", height=440
                    )
                else:
                    board_out = gr.HTML(label="Position")
                rec_out = gr.Textbox(
                    label="Recommended move (SAN)", interactive=False
                )
                gr.Markdown("### Coaching")
                coaching_out = gr.Markdown()

        with gr.Accordion("Engine analysis (live)", open=False):
            details_out = gr.Markdown()

        example_dd.change(
            on_example, inputs=[example_dd], outputs=[fen_in, tier_in, move_in]
        )
        submit.click(
            coach,
            inputs=[fen_in, tier_in, move_in],
            outputs=[board_out, rec_out, coaching_out, details_out],
        )

    return demo


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: parse args, warm the model, and launch the demo."""
    global _MODEL_PATH, _ADAPTER_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", DEFAULT_MODEL),
        help="MLX model repo id or local path (default: %(default)s).",
    )
    parser.add_argument(
        "--adapter-path",
        default=os.environ.get("ADAPTER_PATH") or None,
        help="Optional MLX-format LoRA adapter directory to apply to the model.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link."
    )
    args = parser.parse_args(argv)

    _MODEL_PATH = args.model_path
    _ADAPTER_PATH = args.adapter_path

    banner = f"[chess-coach] loading MLX model: {_MODEL_PATH}"
    if _ADAPTER_PATH:
        banner += f"  (+ adapter: {_ADAPTER_PATH})"
    print(banner, flush=True)
    get_model()  # warm now so a load failure surfaces before the server starts
    print(
        f"[chess-coach] model ready. board mode = "
        f"{'PNG image' if USE_IMAGE else 'inline SVG'}",
        flush=True,
    )

    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
