"""Central configuration for the chess-instructor pipeline.

Single source of truth for tiers, engine tolerances, Maia mapping, paths, and
model ids. Imported by ingest / teacher / filter / eval so every stage agrees.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
POSITIONS = DATA / "positions"
TRANSCRIPTS = DATA / "transcripts"
GENERATED = DATA / "generated"
DATASET = DATA / "dataset"
MODELS = ROOT / "models"
PROMPTS = ROOT / "prompts"

# --- Tiers (rating bands) -------------------------------------------------
# Coaching granularity is coarse on purpose: fine (100-elo) buckets are false
# precision no judge can grade. Three tiers = gradeable + data-efficient.
TIERS = {
    "beginner":     {"low": 1000, "high": 1200, "maia": "maia-1100", "ply_cap": 2},
    "intermediate": {"low": 1300, "high": 1600, "maia": "maia-1500", "ply_cap": 4},
    "advanced":     {"low": 1700, "high": 2000, "maia": "maia-1900", "ply_cap": 6},
}

def tier_for_rating(rating: int) -> str | None:
    for name, t in TIERS.items():
        if t["low"] <= rating <= t["high"]:
            return name
    return None

# --- Engine (Stockfish) tolerances ---------------------------------------
STOCKFISH_BIN = "/opt/homebrew/bin/stockfish"
SOUND_TOLERANCE_CP = 150      # a move within this of best is "sound" (teachable)
BLUNDER_CP = 250              # cp_loss >= this is a blunder (never recommend)
MISTAKE_CP = 100
INACCURACY_CP = 50
DEFAULT_MOVETIME_MS = 300
MULTIPV = 8

# --- Maia -----------------------------------------------------------------
MAIA_DIR = MODELS / "maia"

# --- Teacher / judge models ----------------------------------------------
TEACHER_MODEL = "gpt-5.5"            # override via .env TEACHER_MODEL
TEACHER_REASONING_EFFORT = "high"    # "maximum reasoning"
TEACHER_MODEL_HARD = "gpt-5.5-pro"   # optional, for hard positions
# Judge must be a DIFFERENT family than the teacher (no grading own homework).
JUDGE_MODEL = "claude"               # resolve to a concrete Anthropic id when wired

# --- Behavior Spec (the graded contract) ---------------------------------
# The ONE trained behavior is tier-appropriate MOVE SELECTION; the explanation is a
# secondary, verifier-gated DISPLAY layer, not a separately optimized objective. This
# string is read by the eval judge (src.eval.evaluate) as the spec it grades against,
# so it must stay a single valid string covering both the graded move and the gated
# prose. See BRAINLIFT.md for the full one-behavior treatment.
BEHAVIOR_SPEC = """\
The model's ONE trained behavior is tier-appropriate MOVE SELECTION. Given a position, \
the student's rating tier, the move the student played, and full-strength engine \
analysis (a sound-move pool with evals + short lines) plus the tier's human-move \
likelihoods, the coach recommends exactly ONE move drawn from the sound pool — the \
most human-findable sound move for a Beginner, the engine's sharpest sound line for \
an Advanced player — rendered as that move plus a short principle tag (e.g. \
"Nf3 - develop toward the center"). That single choice is what is trained and graded, \
on three deterministic clauses: sound (not a blunder), tier-appropriate (equals the \
canonical tier move), and distinct across levels (a Beginner and an Advanced player \
are not handed the same move on a differentiating position). The fuller explanation is \
a SECONDARY, verifier-gated display layer, NOT a separately optimized objective: when \
shown it stays tied to a concrete plan and the student's actual mistake, and NEVER \
states raw engine numbers/centipawns, cites lines deeper than the tier's ply cap, \
recommends a blunder, or fabricates a tactic absent from the analysis; a \
verify-and-regenerate gate replaces any prose whose board facts do not check out with \
a truthful, engine-derived explanation of the SAME move, and ends with one \
transferable takeaway. Prose richness is explicitly not the trained goal."""
