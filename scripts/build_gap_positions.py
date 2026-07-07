#!/usr/bin/env python3
"""Build a curated, gap-EXPOSING chess-position set (EVAL + v3 TRAINING pool).

The one behavior this whole project is about is *tier-appropriate move selection*:
for a stated ELO tier, recommend the move that is BOTH sound (inside Stockfish's
tolerance pool) AND human-findable at that tier (the move a real player at that
level would actually play), which is frequently **not** the engine's sharpest #1.
``data/analysis/GAP_REPORT.md`` proves the frontier models mostly fail this — they
default to the engine best regardless of level.

This script mines the positions where that behavior can actually be *exercised* —
"discriminating" positions where, for at least one tier, the most-findable sound
move differs from the engine's #1 — and packages them two ways:

* ``data/eval/gap_positions.jsonl``      — a curated, held-out EVAL set (~500-1000).
* ``data/positions/v3_candidates.jsonl`` — the larger TRAINING candidate pool
  (same schema, ready for the teacher-distillation step to coach at all 3 tiers).

It is **pure engine analysis** (Stockfish sound pool + Maia per-tier likelihoods);
it never loads an LLM and never touches the running app/model server. It only
adds Stockfish + lc0 (Maia) load, so it is deliberately resource-bounded
(``--workers``, Stockfish ``Threads=1``) to coexist with the live platform.

Definition (matches ``scripts/frontier_gap.py`` / ``GAP_REPORT.md``)
-------------------------------------------------------------------
For a position and a tier, rank the SOUND POOL by that tier's Maia policy; the
top one is the tier's **most-findable sound move** = the tier-appropriate move.
The position **discriminates** for that tier iff the tier-appropriate move != the
engine's #1 (equivalently, the engine best's Maia-rank inside the pool is > 0).

Gates applied before a position is eligible (EVAL or TRAINING):
  * legal/valid FEN, not terminal, engine analysis succeeded;
  * not trivially decided (no forced mate in the best line and |best eval| <
    ``--trivial-cp``, default 800cp);
  * at least two sound moves (a real choice exists);
  * discriminates for >= 1 tier;
  * board (placement + side-to-move) is absent from ``train_v2`` / ``valid_v2`` /
    ``benchmark_v2`` — zero leakage — and unique within this build.

Two stages
----------
``analyze`` : stream the candidate pool through the engines, writing one rich row
              per analyzed position to an intermediate JSONL (resumable).
``curate``  : dedup, gate, partition into EVAL (diverse + sharp) + TRAINING, and
              write ``GAP_POSITIONS_REPORT.md``.
``all``     : analyze then curate.

Run (repo root, pinned interpreter)::

    ~/.venvs/mlx/bin/python -m scripts.build_gap_positions analyze --limit 40 --workers 4   # smoke
    ~/.venvs/mlx/bin/python -m scripts.build_gap_positions analyze --workers 6 --resume
    ~/.venvs/mlx/bin/python -m scripts.build_gap_positions curate  --eval-target 800
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
import chess.engine

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import settings  # noqa: E402
from src.engine import maia_engine  # noqa: E402
from src.engine import position_facts as pf  # noqa: E402
from scripts.divergence_analysis import build_heldin_keys, pos_key, _phase  # noqa: E402

TIER_ORDER: Tuple[str, ...] = ("beginner", "intermediate", "advanced")
MATE_SCORE: int = 100_000
PV_MAX_PLIES: int = 6

#: Severity thresholds (clamped cp_loss), identical to stockfish_engine._severity.
_SEVERITY: Sequence[Tuple[int, str]] = ((50, "none"), (100, "inaccuracy"), (250, "mistake"))

#: Default intermediate + output paths.
DEFAULT_ANALYSIS = "data/positions/v3_analysis.jsonl"
DEFAULT_EVAL_OUT = "data/eval/gap_positions.jsonl"
DEFAULT_TRAIN_OUT = "data/positions/v3_candidates.jsonl"
DEFAULT_REPORT = "data/eval/GAP_POSITIONS_REPORT.md"


# --------------------------------------------------------------------------- #
# Thread-local Stockfish engines (one per worker; Threads=1 to stay light)
# --------------------------------------------------------------------------- #

_TL = threading.local()
_ENGINES_LOCK = threading.Lock()
_ALL_ENGINES: List[chess.engine.SimpleEngine] = []


def _sf_engine(hash_mb: int) -> chess.engine.SimpleEngine:
    """Return this worker-thread's own Stockfish engine (opened once, reused).

    Each worker owns a single engine configured with ``Threads=1`` so the whole
    run's Stockfish CPU footprint is bounded by ``--workers`` — it will not starve
    the live platform's own engine usage.
    """
    eng = getattr(_TL, "engine", None)
    if eng is None:
        eng = chess.engine.SimpleEngine.popen_uci(settings.STOCKFISH_BIN)
        try:
            eng.configure({"Threads": 1, "Hash": hash_mb})
        except Exception:  # noqa: BLE001 - keep going with defaults
            pass
        _TL.engine = eng
        with _ENGINES_LOCK:
            _ALL_ENGINES.append(eng)
    return eng


def _close_engines() -> None:
    with _ENGINES_LOCK:
        for eng in _ALL_ENGINES:
            try:
                eng.quit()
            except Exception:  # noqa: BLE001
                pass
        _ALL_ENGINES.clear()


def _severity(cp_loss: int) -> str:
    for upper, label in _SEVERITY:
        if cp_loss < upper:
            return label
    return "blunder"


def _pv_to_san(board: chess.Board, pv: Sequence[chess.Move], cap: int = PV_MAX_PLIES) -> List[str]:
    preview = board.copy(stack=False)
    out: List[str] = []
    for mv in list(pv)[:cap]:
        try:
            out.append(preview.san(mv))
        except (ValueError, AssertionError):
            break
        preview.push(mv)
    return out


def _decode_line(board: chess.Board, info: chess.engine.InfoDict) -> Optional[Dict[str, Any]]:
    pov = info["score"].pov(board.turn)
    pv = list(info.get("pv") or [])
    if not pv:
        return None
    try:
        san = board.san(pv[0])
    except (ValueError, AssertionError):
        return None
    return {
        "uci": pv[0].uci(),
        "san": san,
        "cp": int(pov.score(mate_score=MATE_SCORE)),
        "mate": pov.mate(),
        "pv": _pv_to_san(board, pv),
    }


# --------------------------------------------------------------------------- #
# Deterministic, board-derived motif tags (truth by construction, no LLM)
# --------------------------------------------------------------------------- #


def _has_passed_pawn(board: chess.Board, color: chess.Color) -> bool:
    """Cheap passed-pawn test for ``color`` (no enemy pawn ahead on same/adj file)."""
    pawns = board.pieces(chess.PAWN, color)
    enemy = board.pieces(chess.PAWN, not color)
    for sq in pawns:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        blocked = False
        for ef in (f - 1, f, f + 1):
            if ef < 0 or ef > 7:
                continue
            for esq in enemy:
                if chess.square_file(esq) != ef:
                    continue
                er = chess.square_rank(esq)
                ahead = er > r if color == chess.WHITE else er < r
                if ahead:
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            return True
    return False


def _any_pin(board: chess.Board) -> bool:
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        if board.is_pinned(piece.color, sq):
            return True
    return False


#: Priority order for choosing a single "primary" motif per position (for
#: diversity bucketing); the full set is stored alongside.
_MOTIF_PRIORITY: Tuple[str, ...] = (
    "promotion", "enemy_hanging", "fork_shot", "pin", "own_hanging",
    "check_available", "passed_pawn", "capture_available", "quiet_positional",
)


def motif_tags(board: chess.Board, pool: List[Dict[str, Any]]) -> Tuple[List[str], str]:
    """Return ``(all_tags, primary_tag)`` — deterministic tactical/positional motifs.

    Computed from the board + the sound pool with ``python-chess`` only, so the
    labels are verifiable (they feed the report's motif diversity + curation).
    """
    stm = board.turn
    tags: set[str] = set()

    if pf.hanging_pieces(board, not stm):
        tags.add("enemy_hanging")
    if pf.hanging_pieces(board, stm):
        tags.add("own_hanging")

    # 7th-rank pawn about to promote (either side).
    for color in (chess.WHITE, chess.BLACK):
        seventh = 6 if color == chess.WHITE else 1
        if any(chess.square_rank(sq) == seventh for sq in board.pieces(chess.PAWN, color)):
            tags.add("promotion")

    has_check = has_capture = has_fork = False
    for mv in board.legal_moves:
        if not has_capture and board.is_capture(mv):
            has_capture = True
        if not has_check and board.gives_check(mv):
            has_check = True
    # Fork-ish: a SOUND move that lands attacking >= 2 enemy pieces.
    for m in pool:
        try:
            mv = chess.Move.from_uci(m["uci"])
        except ValueError:
            continue
        if mv not in board.legal_moves:
            continue
        if len(pf.move_facts(board, mv)["attacks"]) >= 2:
            has_fork = True
            break
    if has_check:
        tags.add("check_available")
    if has_capture:
        tags.add("capture_available")
    if has_fork:
        tags.add("fork_shot")

    if _any_pin(board):
        tags.add("pin")
    if _has_passed_pawn(board, stm) or _has_passed_pawn(board, not stm):
        tags.add("passed_pawn")

    tactical = {"enemy_hanging", "own_hanging", "promotion", "check_available",
                "fork_shot", "pin"}
    if not (tags & tactical):
        tags.add("quiet_positional")

    primary = next((t for t in _MOTIF_PRIORITY if t in tags), "quiet_positional")
    return sorted(tags), primary


# --------------------------------------------------------------------------- #
# Per-position engine analysis
# --------------------------------------------------------------------------- #


def _student_move(
    board: chess.Board, played_uci: Optional[str], best_cp: int,
    line_cp: Dict[str, int], eng: chess.engine.SimpleEngine, movetime_ms: int,
) -> Dict[str, Any]:
    """Classify the human move actually played in the source game (context)."""
    if not played_uci:
        return {"san": "(none)", "uci": "", "cp_loss": 0, "severity": "none"}
    try:
        mv = chess.Move.from_uci(played_uci)
    except ValueError:
        return {"san": "(none)", "uci": "", "cp_loss": 0, "severity": "none"}
    if mv not in board.legal_moves:
        return {"san": "(none)", "uci": "", "cp_loss": 0, "severity": "none"}
    san = board.san(mv)
    if played_uci in line_cp:
        cp = line_cp[played_uci]
    else:
        info = eng.analyse(board, chess.engine.Limit(time=movetime_ms / 1000.0), root_moves=[mv])
        info = info[0] if isinstance(info, list) else info
        cp = int(info["score"].pov(board.turn).score(mate_score=MATE_SCORE))
    cp_loss = max(0, best_cp - cp)
    return {"san": san, "uci": mv.uci(), "cp_loss": int(cp_loss), "severity": _severity(cp_loss)}


def analyze_one(
    rec: Dict[str, Any], *, movetime_ms: int, tolerance_cp: int, multipv: int,
    trivial_cp: int, hash_mb: int,
) -> Optional[Dict[str, Any]]:
    """Full Stockfish + Maia(x3) analysis of one candidate position.

    Returns a rich row (including non-discriminating ones, so rates are honest) or
    ``None`` if the FEN is invalid/terminal or the engine produced no line.
    """
    fen = rec["fen"]
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    if not board.is_valid() or board.is_game_over():
        return None

    eng = _sf_engine(hash_mb)
    infos = eng.analyse(board, chess.engine.Limit(time=movetime_ms / 1000.0), multipv=multipv)
    if isinstance(infos, dict):
        infos = [infos]
    lines: List[Dict[str, Any]] = []
    for info in infos:
        dec = _decode_line(board, info)
        if dec:
            lines.append(dec)
    if not lines:
        return None

    best = lines[0]
    best_cp, best_mate = int(best["cp"]), best["mate"]
    line_cp = {ln["uci"]: int(ln["cp"]) for ln in lines}

    # Sound pool: within tolerance of best AND never a blunder (mirrors sound_pool).
    max_loss = min(tolerance_cp, settings.BLUNDER_CP - 1)
    pool = [
        {"san": ln["san"], "uci": ln["uci"], "cp": int(ln["cp"]), "pv": ln["pv"]}
        for ln in lines
        if best_cp - int(ln["cp"]) <= max_loss
    ]
    n_sound = len(pool)
    n_legal = board.legal_moves.count()
    trivial = (best_mate is not None) or (abs(best_cp) >= trivial_cp)

    student = _student_move(board, rec.get("played_move_uci"), best_cp, line_cp, eng, movetime_ms)

    # Per-tier Maia: policy over ALL sound-pool moves -> most-findable sound move.
    pool_ucis = [m["uci"] for m in pool]
    san_by_uci = {m["uci"]: m["san"] for m in pool}
    maia_by_tier: Dict[str, Any] = {}
    tier_moves: Dict[str, Optional[str]] = {}
    discriminating_tiers: List[str] = []
    strong_tiers: List[str] = []
    for tier in TIER_ORDER:
        try:
            res = maia_engine.human_moves(fen, tier, top_k=max(n_legal, 8))["moves"]
        except Exception as exc:  # noqa: BLE001
            print(f"    ! maia failed ({tier}) {rec.get('id')}: {exc}", file=sys.stderr)
            res = []
        policy = {m["uci"]: float(m["policy"]) for m in res}
        top6 = [{"san": m["san"], "uci": m["uci"], "policy": float(m["policy"])} for m in res[:6]]
        order = sorted(pool_ucis, key=lambda u: (-policy.get(u, 0.0), u))
        rank = {u: i for i, u in enumerate(order)}
        tier_uci = order[0] if order else None
        eb_rank = rank.get(best["uci"])
        disc = bool(tier_uci is not None and tier_uci != best["uci"])
        tmove_pol = policy.get(tier_uci, 0.0) if tier_uci else 0.0
        best_pol = policy.get(best["uci"], 0.0)
        pol_gap = tmove_pol - best_pol
        # "strong" = the findable move is genuinely human-likely and clearly more
        # so than the engine best (filters out ties/near-ties = real teaching forks).
        strong = bool(disc and tmove_pol >= 0.10 and pol_gap >= 0.05)
        maia_by_tier[tier] = {
            "net": maia_engine.net_for_tier(tier),
            "top": top6,
            "pool_policy": {u: round(policy.get(u, 0.0), 4) for u in pool_ucis},
            "pool_order": order,
            "tier_move": (
                {"san": san_by_uci.get(tier_uci, ""), "uci": tier_uci,
                 "policy": round(tmove_pol, 4)} if tier_uci else None
            ),
            "engine_best_maia_rank": eb_rank,
            "policy_gap": round(pol_gap, 4),
            "discriminating": disc,
            "strong": strong,
        }
        tier_moves[tier] = tier_uci
        if disc:
            discriminating_tiers.append(tier)
        if strong:
            strong_tiers.append(tier)

    distinct_tier_moves = len({v for v in tier_moves.values() if v})
    tags, primary_motif = motif_tags(board, pool)
    max_gap = max((maia_by_tier[t]["policy_gap"] for t in discriminating_tiers), default=0.0)

    why = ""
    if discriminating_tiers:
        # Explain the sharpest discriminating tier (largest findability gap).
        st = max(discriminating_tiers, key=lambda t: maia_by_tier[t]["policy_gap"])
        tm = maia_by_tier[st]["tier_move"]
        why = (
            f"For {st}, the most human-findable sound move is {tm['san']} "
            f"(Maia {round(tm['policy'] * 100)}%), not the engine's #1 {best['san']} "
            f"(Maia {round(maia_by_tier[st]['pool_policy'].get(best['uci'], 0.0) * 100)}%); "
            f"recommending the engine best would over-level a {st}-tier player. "
            f"Discriminates for: {', '.join(discriminating_tiers)}."
        )

    return {
        "id": rec.get("id"),
        "fen": fen,
        "board_key": pos_key(fen),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "phase": _phase(fen),
        "motifs": tags,
        "primary_motif": primary_motif,
        "source_tier": rec.get("tier"),
        "mover_rating": rec.get("mover_rating"),
        "time_control": rec.get("time_control"),
        "played_move": student,
        "engine_best": {"san": best["san"], "uci": best["uci"], "cp": best_cp, "mate": best_mate},
        "best_cp": best_cp,
        "n_sound": n_sound,
        "n_legal": n_legal,
        "trivial": trivial,
        "sound_pool": pool,
        "maia_by_tier": maia_by_tier,
        "tier_moves": tier_moves,
        "discriminating_tiers": discriminating_tiers,
        "n_discriminating_tiers": len(discriminating_tiers),
        "strong_tiers": strong_tiers,
        "n_strong_tiers": len(strong_tiers),
        "discriminating": bool(discriminating_tiers),
        "strong_discriminating": bool(strong_tiers),
        "n_distinct_tier_moves": distinct_tier_moves,
        "max_policy_gap": round(max_gap, 4),
        "why_discriminating": why,
        # eligibility for the curated gap set (EVAL or TRAINING candidate)
        "eligible": bool(
            (not trivial) and n_sound >= 2 and len(discriminating_tiers) >= 1
        ),
    }


# --------------------------------------------------------------------------- #
# Candidate loading + dedup
# --------------------------------------------------------------------------- #


def _held_in_keys(train_v2: Path, valid_v2: Path, benchmarks: List[Path]) -> set:
    """Board (placement + side-to-move) keys already used by v2 corpora."""
    keys = build_heldin_keys(train_v2, valid_v2)
    for bp in benchmarks:
        if not bp.exists():
            continue
        for raw in bp.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                fen = json.loads(raw).get("fen")
            except json.JSONDecodeError:
                continue
            if fen:
                keys.add(pos_key(fen))
    return keys


def load_candidates(paths: List[Path], held_in: set) -> List[Dict[str, Any]]:
    """Load candidate positions, dropping v2-leaked + duplicate boards."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for path in paths:
        if not path.exists():
            print(f"  [warn] missing candidate file {path}", file=sys.stderr)
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue
            fen = d.get("fen")
            if not fen:
                continue
            key = pos_key(fen)
            if key in held_in or key in seen:
                continue
            seen.add(key)
            out.append(d)
    return out


def _load_done_ids(path: Path) -> set:
    done: set = set()
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                done.add(json.loads(raw)["id"])
            except Exception:  # noqa: BLE001
                continue
    return done


# --------------------------------------------------------------------------- #
# Stage: analyze
# --------------------------------------------------------------------------- #


def stage_analyze(args: argparse.Namespace) -> int:
    def _abs(x: str) -> Path:
        p = Path(x)
        return p if p.is_absolute() else _ROOT / p

    out_path = _abs(args.analysis_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    held_in = _held_in_keys(
        _abs(args.train_v2), _abs(args.valid_v2), [_abs(b) for b in args.benchmarks]
    )
    print(f"[1/3] held-in board keys (train_v2+valid_v2+benchmarks): {len(held_in)}", file=sys.stderr)

    cand_paths = [_abs(p) for p in args.positions]
    cands = load_candidates(cand_paths, held_in)
    print(f"[2/3] candidate positions (held-out, unique board): {len(cands)}", file=sys.stderr)

    rng = random.Random(args.seed)
    rng.shuffle(cands)  # de-bias any file ordering before an optional --limit
    if args.limit:
        cands = cands[: args.limit]
        print(f"      smoke --limit -> {len(cands)} positions", file=sys.stderr)

    done = _load_done_ids(out_path) if args.resume else set()
    if args.resume and done:
        cands = [c for c in cands if c.get("id") not in done]
        print(f"      resuming: {len(done)} already analyzed, {len(cands)} remaining", file=sys.stderr)

    print(f"[3/3] analyzing with {args.workers} Stockfish workers (Threads=1) + "
          f"Maia x3 (movetime={args.movetime}ms, multipv={args.multipv}) ...", file=sys.stderr)

    stats = Counter()
    t0 = time.time()
    n_written = 0
    write_lock = threading.Lock()
    mode = "a" if (args.resume and done) else "w"

    def _work(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return analyze_one(
            rec, movetime_ms=args.movetime, tolerance_cp=args.tolerance,
            multipv=args.multipv, trivial_cp=args.trivial_cp, hash_mb=args.hash,
        )

    try:
        with out_path.open(mode, encoding="utf-8") as fh, \
                ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_work, rec): rec for rec in cands}
            for i, fut in enumerate(as_completed(futs), 1):
                rec = futs[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    stats["failed"] += 1
                    print(f"  ! {rec.get('id')} FAILED: {exc}", file=sys.stderr)
                    continue
                if row is None:
                    stats["skipped"] += 1
                    continue
                with write_lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                n_written += 1
                stats["analyzed"] += 1
                if row["eligible"]:
                    stats["eligible"] += 1
                if row["strong_discriminating"]:
                    stats["strong"] += 1
                if i % 100 == 0 or i == len(futs):
                    dt = time.time() - t0
                    rate = stats["eligible"] / max(1, stats["analyzed"])
                    print(
                        f"  [{i}/{len(futs)}] analyzed={stats['analyzed']} "
                        f"eligible={stats['eligible']} ({rate:.0%}) strong={stats['strong']} "
                        f"skip={stats['skipped']} fail={stats['failed']} "
                        f"| {dt:.0f}s ({dt / max(1, stats['analyzed']):.2f}s/pos)",
                        file=sys.stderr,
                    )
    finally:
        _close_engines()
        try:
            maia_engine.close_all()
        except Exception:  # noqa: BLE001
            pass

    dt = time.time() - t0
    print(f"DONE analyze — wrote {n_written} rows to {out_path} in {dt:.0f}s. "
          f"eligible={stats['eligible']}/{stats['analyzed']} strong={stats['strong']}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# Stage: curate (dedup, gate, partition, report) — no engines
# --------------------------------------------------------------------------- #


def _eval_slug(row: Dict[str, Any]) -> Dict[str, Any]:
    """Project an analysis row into the compact, self-describing EVAL/TRAIN row."""
    return {
        "id": row["id"],
        "fen": row["fen"],
        "board_key": row["board_key"],
        "side_to_move": row["side_to_move"],
        "phase": row["phase"],
        "motifs": row["motifs"],
        "primary_motif": row["primary_motif"],
        "source_tier": row["source_tier"],
        "mover_rating": row["mover_rating"],
        "played_move": row["played_move"],
        "engine_best": row["engine_best"],
        "best_cp": row["best_cp"],
        "n_sound": row["n_sound"],
        "n_legal": row["n_legal"],
        "sound_pool": row["sound_pool"],
        "maia_by_tier": row["maia_by_tier"],
        "tier_moves": row["tier_moves"],
        "discriminating_tiers": row["discriminating_tiers"],
        "n_discriminating_tiers": row["n_discriminating_tiers"],
        "strong_tiers": row["strong_tiers"],
        "n_distinct_tier_moves": row["n_distinct_tier_moves"],
        "max_policy_gap": row["max_policy_gap"],
        "why_discriminating": row["why_discriminating"],
    }


def curate_eval(
    rows: List[Dict[str, Any]], target: int, seed: int = 3407,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified EVAL/TRAINING split; return ``(eval_rows, train_rows)``.

    Splits every ``(phase x source_tier x primary_motif)`` cell between the two
    outputs so BOTH the held-out EVAL set and the larger TRAINING pool carry the
    full diversity across all three axes — no phase/tier/motif is starved from
    either side (the earlier greedy round-robin funnelled all ~162 endgames into
    EVAL). Abundant phases are split proportionally to hit ``target``; the scarce
    ENDGAME phase is split ~evenly so EVAL keeps a usable endgame sample while
    TRAINING retains endgames too. Each cell with >= 2 rows contributes >= 1 to
    each side, guaranteeing maximal representation everywhere. Within a cell the
    order is a seeded shuffle (a representative, not cherry-picked-sharp, sample).
    """
    rng = random.Random(seed)
    total = len(rows)
    n_endgame = sum(1 for r in rows if r["phase"] == "endgame")
    endgame_frac = 0.5  # even split for the scarce phase (keeps both sides supplied)
    denom = max(1, total - n_endgame)
    base_frac = min(0.9, max(0.0, (target - endgame_frac * n_endgame) / denom))

    cells: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cells[(r["phase"], r["source_tier"] or "unknown", r["primary_motif"])].append(r)

    eval_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    for key in sorted(cells):
        lst = cells[key]
        rng.shuffle(lst)
        n = len(lst)
        frac = endgame_frac if key[0] == "endgame" else base_frac
        if n == 1:
            (eval_rows if rng.random() < frac else train_rows).append(lst[0])
            continue
        take = int(round(frac * n))
        take = max(1, min(take, n - 1))  # >= 1 to EACH side for full diversity
        eval_rows.extend(lst[:take])
        train_rows.extend(lst[take:])
    return eval_rows, train_rows


def _dist(rows: List[Dict[str, Any]], key: str) -> Counter:
    c: Counter = Counter()
    for r in rows:
        c[r.get(key)] += 1
    return c


def _motif_dist(rows: List[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        for m in r["motifs"]:
            c[m] += 1
    return c


def _tier_disc_rates(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Per-tier discriminating + strong rates over ``rows`` (eligible analyzed)."""
    out: Dict[str, Dict[str, float]] = {}
    n = max(1, len(rows))
    for tier in TIER_ORDER:
        disc = sum(1 for r in rows if r["maia_by_tier"][tier]["discriminating"])
        strong = sum(1 for r in rows if r["maia_by_tier"][tier].get("strong"))
        out[tier] = {"disc": disc / n, "strong": strong / n, "disc_n": disc, "strong_n": strong}
    return out


def stage_curate(args: argparse.Namespace) -> int:
    def _abs(x: str) -> Path:
        p = Path(x)
        return p if p.is_absolute() else _ROOT / p

    analysis_path = _abs(args.analysis_out)
    if not analysis_path.exists():
        print(f"missing {analysis_path}; run `analyze` first.", file=sys.stderr)
        return 1

    all_rows = [json.loads(ln) for ln in analysis_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"[1/4] loaded {len(all_rows)} analyzed rows", file=sys.stderr)

    # Re-verify held-out property at curation time (defensive; zero leakage).
    held_in = _held_in_keys(
        _abs(args.train_v2), _abs(args.valid_v2), [_abs(b) for b in args.benchmarks]
    )
    leaked = sum(1 for r in all_rows if r["board_key"] in held_in)

    # Dedup by board + keep eligible discriminating rows only.
    seen: set = set()
    eligible: List[Dict[str, Any]] = []
    dup = 0
    for r in all_rows:
        if r["board_key"] in held_in:
            continue
        if r["board_key"] in seen:
            dup += 1
            continue
        seen.add(r["board_key"])
        if r.get("eligible"):
            eligible.append(r)
    print(f"[2/4] leaked(dropped)={leaked} dup(dropped)={dup} "
          f"eligible discriminating held-out={len(eligible)}", file=sys.stderr)

    if not eligible:
        print("no eligible discriminating positions — nothing to write.", file=sys.stderr)
        return 1

    # EVAL target: within [--eval-min, --eval-target] but never > 45% of the pool,
    # so the TRAINING candidate pool stays clearly larger.
    cap = int(0.45 * len(eligible))
    target = min(args.eval_target, cap) if cap >= args.eval_min else min(args.eval_target, len(eligible))
    target = max(min(target, len(eligible)), min(args.eval_min, len(eligible)))
    eval_rows_full, train_rows_full = curate_eval(eligible, target, seed=getattr(args, "seed", 3407))
    print(f"[3/4] EVAL={len(eval_rows_full)}  TRAINING={len(train_rows_full)} "
          f"(eval target={target}, cap45%={cap})", file=sys.stderr)

    eval_out = _abs(args.eval_out)
    train_out = _abs(args.train_out)
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    train_out.parent.mkdir(parents=True, exist_ok=True)

    with eval_out.open("w", encoding="utf-8") as fh:
        for r in eval_rows_full:
            slug = _eval_slug(r)
            slug["split"] = "eval"
            fh.write(json.dumps(slug, ensure_ascii=False) + "\n")
    with train_out.open("w", encoding="utf-8") as fh:
        for r in train_rows_full:
            slug = _eval_slug(r)
            slug["split"] = "train_candidate"
            fh.write(json.dumps(slug, ensure_ascii=False) + "\n")

    print(f"      wrote EVAL   -> {eval_out}", file=sys.stderr)
    print(f"      wrote TRAIN  -> {train_out}", file=sys.stderr)

    # --- report ---------------------------------------------------------- #
    write_report(_abs(args.report), all_rows, eligible, eval_rows_full, train_rows_full,
                 held_in_n=len(held_in), leaked=leaked, dup=dup, target=target, args=args)
    print(f"[4/4] wrote report -> {_abs(args.report)}", file=sys.stderr)
    return 0


def write_report(
    path: Path, all_rows: List[Dict[str, Any]], eligible: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]], train_rows: List[Dict[str, Any]],
    *, held_in_n: int, leaked: int, dup: int, target: int, args: argparse.Namespace,
) -> None:
    analyzed = [r for r in all_rows]
    # Analyzed-level rates (over non-trivial, real-choice positions for an honest
    # "% where tier-move != engine #1").
    decidable = [r for r in analyzed if (not r["trivial"]) and r["n_sound"] >= 2]
    disc = [r for r in decidable if r["discriminating"]]
    strong = [r for r in decidable if r["strong_discriminating"]]
    rate = len(disc) / max(1, len(decidable))
    strong_rate = len(strong) / max(1, len(decidable))
    tier_rates = _tier_disc_rates(decidable)

    def _tbl(counter: Counter, order: Optional[Sequence[str]] = None) -> str:
        items = [(k, counter.get(k, 0)) for k in order] if order else counter.most_common()
        return "\n".join(f"| {k} | {v} |" for k, v in items)

    both = set(r["board_key"] for r in eval_rows) & set(r["board_key"] for r in train_rows)

    lines: List[str] = []
    lines.append("# Gap-Position Set — curated positions that expose the tier-move selection gap\n")
    lines.append(
        "The target behavior (see `data/analysis/GAP_REPORT.md`): for a stated ELO tier, "
        "recommend the move that is **sound** (in Stockfish's tolerance pool) **and "
        "human-findable at that tier** (top Maia move *inside* the sound pool) — which is "
        "often NOT the engine's #1. A position is **discriminating** for a tier when that "
        "tier-appropriate move differs from the engine's #1; those are the only positions "
        "where 'just give the best move' (what the frontier does) is provably wrong for a "
        "lower tier.\n"
    )
    lines.append("## Method\n")
    lines.append(
        f"- **Candidate pool:** `{', '.join(args.positions)}` — real rated Lichess decision "
        f"positions (reused from the existing polite sample; no new API load).\n"
        f"- **Engines:** Stockfish 18 (`sound_pool`, tolerance {args.tolerance}cp, multipv "
        f"{args.multipv}, movetime {args.movetime}ms, Threads=1) + Maia (`maia-1100/1500/1900`) "
        f"per tier via lc0 `nodes=1`.\n"
        f"- **Tier-appropriate move:** the sound-pool move with the highest Maia policy at "
        f"that tier (same definition as `frontier_gap.py`).\n"
        f"- **Gates:** legal/valid FEN, not terminal, engine OK, not trivially decided "
        f"(no forced mate & |best eval| < {args.trivial_cp}cp), ≥2 sound moves, discriminates "
        f"for ≥1 tier.\n"
        f"- **Dedup:** board (placement + side-to-move) absent from `train_v2` + `valid_v2` + "
        f"`benchmark_v2` and unique within this build.\n"
    )

    lines.append("## 1. Headline counts\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Positions analyzed | {len(analyzed)} |")
    lines.append(f"| Decidable (non-trivial, ≥2 sound) | {len(decidable)} |")
    lines.append(f"| **Discriminating (tier-move ≠ engine #1, ≥1 tier)** | **{len(disc)} ({rate:.1%} of decidable)** |")
    lines.append(f"| Strong discriminating (findable ≥10%, gap ≥5%) | {len(strong)} ({strong_rate:.1%}) |")
    lines.append(f"| Eligible held-out discriminating (deduped) | {len(eligible)} |")
    lines.append(f"| **EVAL set → `{Path(args.eval_out).as_posix()}`** | **{len(eval_rows)}** |")
    lines.append(f"| **TRAINING pool → `{Path(args.train_out).as_posix()}`** | **{len(train_rows)}** |")
    lines.append("")

    lines.append("## 2. How discriminating — % where the tier-move ≠ engine #1\n")
    lines.append("Per tier, over the decidable analyzed positions (this is the crux rate):\n")
    lines.append("| Tier | discriminating | strong |")
    lines.append("|---|---|---|")
    for t in TIER_ORDER:
        tr = tier_rates[t]
        lines.append(f"| {t} | {tr['disc_n']} ({tr['disc']:.1%}) | {tr['strong_n']} ({tr['strong']:.1%}) |")
    lines.append("")
    dtm = _dist(disc, "n_discriminating_tiers")
    lines.append("Number of tiers a position discriminates for (of 3):\n")
    lines.append("| tiers | positions |")
    lines.append("|---|---|")
    lines.append(_tbl(dtm, order=[1, 2, 3]))
    lines.append("")
    ddm = _dist(disc, "n_distinct_tier_moves")
    lines.append("Distinct tier-appropriate moves across the 3 tiers (2–3 ⇒ the move itself "
                 "should change with level — the highest-value contrastive positions):\n")
    lines.append("| distinct moves | positions |")
    lines.append("|---|---|")
    lines.append(_tbl(ddm, order=[1, 2, 3]))
    lines.append("")

    for sec, (name, rows) in enumerate((("EVAL", eval_rows), ("TRAINING", train_rows)), start=3):
        lines.append(f"## {sec}. {name} composition\n")
        lines.append(f"**Phase** ({name}):\n")
        lines.append("| phase | positions |")
        lines.append("|---|---|")
        lines.append(_tbl(_dist(rows, "phase"), order=["opening", "middlegame", "endgame"]))
        lines.append("")
        lines.append(f"**Source rating tier** ({name}):\n")
        lines.append("| tier | positions |")
        lines.append("|---|---|")
        lines.append(_tbl(_dist(rows, "source_tier"), order=list(TIER_ORDER)))
        lines.append("")
        lines.append(f"**Primary motif** ({name}):\n")
        lines.append("| motif | positions |")
        lines.append("|---|---|")
        lines.append(_tbl(_dist(rows, "primary_motif")))
        lines.append("")
        lines.append(f"**All motif tags** ({name}, positions may carry several):\n")
        lines.append("| motif | positions |")
        lines.append("|---|---|")
        lines.append(_tbl(_motif_dist(rows)))
        lines.append("")

    lines.append("## 5. Dedup / leakage confirmation\n")
    lines.append(f"- Held-in board keys (train_v2 + valid_v2 + benchmark_v2): **{held_in_n}**.")
    lines.append(f"- Analyzed rows whose board leaked into v2 corpora: **{leaked}** (dropped).")
    lines.append(f"- Duplicate boards within the pool: **{dup}** (dropped).")
    lines.append(f"- EVAL ∩ TRAINING (board overlap): **{len(both)}** (must be 0).")
    lines.append("- Every emitted position is held-out and unique by board — **zero leakage**.")
    lines.append("")

    lines.append("## 6. Example discriminating positions (EVAL)\n")
    lines.append("Positions where the tier-appropriate move genuinely CHANGES across tiers "
                 "(distinct moves ≥ 2) are the sharpest illustration of the gap — the model "
                 "must give a *different* move by level, not the engine's #1:\n")
    contrastive = sorted(
        (r for r in eval_rows if r["n_distinct_tier_moves"] >= 2),
        key=lambda r: (-r["n_distinct_tier_moves"], -r["max_policy_gap"]),
    )
    # spread examples across phases for variety
    picks: List[Dict[str, Any]] = []
    by_phase: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in contrastive:
        by_phase[r["phase"]].append(r)
    idx = 0
    phases = ["opening", "middlegame", "endgame"]
    while len(picks) < 14 and any(by_phase[p] for p in phases):
        p = phases[idx % len(phases)]
        if by_phase[p]:
            picks.append(by_phase[p].pop(0))
        idx += 1
    for r in picks:
        b = r["engine_best"]["san"]
        tm = "  ".join(
            f"{t[0].upper()}:{(r['maia_by_tier'][t]['tier_move'] or {}).get('san', '?')}"
            for t in TIER_ORDER
        )
        lines.append(
            f"- `{r['id']}` [{r['phase']}/{r['primary_motif']}] engine#1 **{b}** — {tm}"
        )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--analysis-out", default=DEFAULT_ANALYSIS,
                   help="Intermediate per-position analysis JSONL (resumable).")
    p.add_argument("--train-v2", default="data/dataset/train_v2.jsonl")
    p.add_argument("--valid-v2", default="data/dataset/valid_v2.jsonl")
    p.add_argument("--benchmarks", nargs="+", default=["data/benchmark_v2/scenarios.jsonl"],
                   help="Benchmark scenario files (explicit FEN) to dedup against.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    pa = sub.add_parser("analyze", help="Engine-analyze the candidate pool (resumable).")
    _add_common(pa)
    pa.add_argument("--positions", nargs="+", default=["data/positions/positions_v1.jsonl"],
                    help="Candidate position JSONL file(s) (Lichess sampler schema).")
    pa.add_argument("--workers", type=int, default=6, help="Stockfish worker threads (each Threads=1).")
    pa.add_argument("--movetime", type=int, default=settings.DEFAULT_MOVETIME_MS)
    pa.add_argument("--multipv", type=int, default=settings.MULTIPV)
    pa.add_argument("--tolerance", type=int, default=settings.SOUND_TOLERANCE_CP)
    pa.add_argument("--trivial-cp", type=int, default=800,
                    help="|best eval| >= this (or a forced mate) ⇒ trivially decided, excluded.")
    pa.add_argument("--hash", type=int, default=64, help="Per-engine hash MB (kept small).")
    pa.add_argument("--limit", type=int, default=0, help="Smoke cap on positions (0 = all).")
    pa.add_argument("--seed", type=int, default=3407)
    pa.add_argument("--resume", action="store_true", help="Skip ids already in --analysis-out.")

    pc = sub.add_parser("curate", help="Dedup + partition EVAL/TRAINING + report (no engines).")
    _add_common(pc)
    pc.add_argument("--eval-out", default=DEFAULT_EVAL_OUT)
    pc.add_argument("--train-out", default=DEFAULT_TRAIN_OUT)
    pc.add_argument("--report", default=DEFAULT_REPORT)
    pc.add_argument("--eval-target", type=int, default=800, help="Desired EVAL size (500-1000).")
    pc.add_argument("--eval-min", type=int, default=500)
    pc.add_argument("--seed", type=int, default=3407)
    # curate re-reads analysis rows; needs --positions only for the report text.
    pc.add_argument("--positions", nargs="+", default=["data/positions/positions_v1.jsonl"])
    pc.add_argument("--tolerance", type=int, default=settings.SOUND_TOLERANCE_CP)
    pc.add_argument("--multipv", type=int, default=settings.MULTIPV)
    pc.add_argument("--movetime", type=int, default=settings.DEFAULT_MOVETIME_MS)
    pc.add_argument("--trivial-cp", type=int, default=800)

    pall = sub.add_parser("all", help="analyze then curate with defaults.")
    _add_common(pall)
    pall.add_argument("--positions", nargs="+", default=["data/positions/positions_v1.jsonl"])
    pall.add_argument("--workers", type=int, default=6)
    pall.add_argument("--movetime", type=int, default=settings.DEFAULT_MOVETIME_MS)
    pall.add_argument("--multipv", type=int, default=settings.MULTIPV)
    pall.add_argument("--tolerance", type=int, default=settings.SOUND_TOLERANCE_CP)
    pall.add_argument("--trivial-cp", type=int, default=800)
    pall.add_argument("--hash", type=int, default=64)
    pall.add_argument("--limit", type=int, default=0)
    pall.add_argument("--seed", type=int, default=3407)
    pall.add_argument("--resume", action="store_true")
    pall.add_argument("--eval-out", default=DEFAULT_EVAL_OUT)
    pall.add_argument("--train-out", default=DEFAULT_TRAIN_OUT)
    pall.add_argument("--report", default=DEFAULT_REPORT)
    pall.add_argument("--eval-target", type=int, default=800)
    pall.add_argument("--eval-min", type=int, default=500)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "analyze":
        return stage_analyze(args)
    if args.stage == "curate":
        return stage_curate(args)
    if args.stage == "all":
        rc = stage_analyze(args)
        if rc != 0:
            return rc
        # curate defaults present on the 'all' namespace.
        return stage_curate(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
