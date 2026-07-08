"""Shared config + helpers for the honest SHOWCASE eval pipeline.

Everything the showcase produces lives under ``data/showcase/`` (raw artifacts)
and the single web asset ``web/public/showcase.json``. This module never writes
into ``data/benchmark_gap803``, ``data/dataset`` or ``models`` — those are the
protected live-platform / v3-training artifacts and are only ever *read*.

The showcase reuses the exact, already-trusted benchmark machinery:

* grounding      -> ``scripts.build_gap_positions.analyze_one`` (Stockfish sound
                    pool + Maia per-tier likelihoods; the same function that built
                    ``gap_positions.jsonl`` and ``v3_candidates.jsonl``).
* scenario shape -> ``scripts.gap803_common.to_scenario`` / ``flatten``.
* model registry -> ``src.eval.benchmark.config`` (the 14 FQNs, prices, effort).
* backends       -> ``src.eval.benchmark.backends`` (local MLX + TFY gateway).
* objective      -> ``src.eval.benchmark.objective.score_one``.

So every number in the showcase is byte-identical in provenance to the
definitive 803 benchmark; only the *positions* and the *split labels* differ.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --- repo root on path (this file lives at data/showcase/pipeline/common.py) - #
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.eval.benchmark import config as bcfg  # noqa: E402

TIERS: Tuple[str, ...] = ("beginner", "intermediate", "advanced")

# --------------------------------------------------------------------------- #
# The 14 models (same field as the definitive benchmark; reuse the FQNs).
# The task's "untuned Qwen3-32B" and "Qwen3-32B" are the same registry entry
# (q3_32b, an untuned open base). ours -> the LIVE local v2 coach; base -> the
# untuned 1.7B. Order is fixed so anonymisation + tables are reproducible.
# --------------------------------------------------------------------------- #
FIELD: Tuple[str, ...] = (
    "ours", "base",
    "gpt", "claude", "gemini",
    "q3_32b", "q3_next80b", "gemma3_27b", "llama33_70b",
    "dsv32", "glm5", "mistral3", "kimi25", "dsr1",
)

#: The frontier references (used for the honest ours-wins / ours-loses contrast).
FRONTIER_KEYS: Tuple[str, ...] = ("gpt", "claude", "gemini")
OURS_KEY = "ours"

#: The local, free models (run on-device via MLX).
LOCAL_KEYS: Tuple[str, ...] = ("ours", "base")

#: Where the shipped OURS-v2 coach actually lives (config default ident is v1).
OURS_V2_PATH = str(settings.MODELS / "mlx" / "chess-coach-v2")

# --------------------------------------------------------------------------- #
# Paths — everything under data/showcase/ (+ the one web asset).
# --------------------------------------------------------------------------- #
SHOWCASE_DIR = settings.DATA / "showcase"
TRAIN_DIR = SHOWCASE_DIR / "train"
TEST_NEW_DIR = SHOWCASE_DIR / "test_new"
TEST_REUSE_DIR = SHOWCASE_DIR / "test_reuse"
REPORT_PATH = SHOWCASE_DIR / "SHOWCASE_REPORT.md"
COST_PATH = SHOWCASE_DIR / "cost.json"
WEB_SHOWCASE = ROOT / "web" / "public" / "showcase.json"

SPLIT_DIRS: Dict[str, Path] = {
    "train": TRAIN_DIR,
    "test_new": TEST_NEW_DIR,
    "test_reuse": TEST_REUSE_DIR,
}


def ensure_dirs() -> None:
    for d in (SHOWCASE_DIR, TRAIN_DIR, TEST_NEW_DIR, TEST_REUSE_DIR):
        (d / "gen").mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model metadata for the output schema.
# --------------------------------------------------------------------------- #
def model_meta(key: str) -> Dict[str, Any]:
    m = bcfg.MODELS[key]
    display = m.display
    if key == "ours":
        display = "OURS-v2 (1.7B tuned)"
    return {
        "key": key,
        "name": display,
        "family": m.family,
        "local": key in LOCAL_KEYS,
    }


def resolved_ident(key: str) -> str:
    """Gateway id or local MLX path; ours -> the shipped v2 coach."""
    if key == "ours":
        return os.environ.get("BENCH_OURS_MODEL", OURS_V2_PATH)
    return bcfg.MODELS[key].ident


# --------------------------------------------------------------------------- #
# JSONL io (atomic-ish append, resumable done-sets).
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


_APPEND_LOCK = threading.Lock()


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()


def done_keys(path: Path, fields: Sequence[str]) -> set:
    done: set = set()
    for row in read_jsonl(path):
        try:
            done.add(tuple(row[f] for f in fields))
        except KeyError:
            continue
    return done


def load_scenarios(split_dir: Path) -> List[Dict[str, Any]]:
    return read_jsonl(split_dir / "scenarios.jsonl")


# --------------------------------------------------------------------------- #
# Cost readout (uses the benchmark's estimated per-1M prices).
# --------------------------------------------------------------------------- #
def price_for(key: str) -> Tuple[float, float]:
    m = bcfg.MODELS[key]
    return float(m.price_in), float(m.price_out)


def usd_for(key: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = price_for(key)
    return (prompt_tokens * pin + completion_tokens * pout) / 1_000_000.0
