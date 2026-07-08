"""Ground raw (fen, played_move, tier) records into the full gap-schema.

Thin, resumable driver around ``scripts.build_gap_positions.analyze_one`` — the
same Stockfish(sound pool) + Maia(per-tier likelihood) analysis that produced the
definitive benchmark positions. Used for BOTH the training sample and the new
Lichess test positions so their grounding is byte-identical to the 803 set.

Also exposes ``dedup_keys(...)`` — the board (placement + side-to-move) keys to
avoid, so the new test positions are provably disjoint from train_v2/v3, the
candidate pools, and every existing benchmark scenario set.
"""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from common import ROOT, read_jsonl  # noqa: E402  (pipeline dir on sys.path)

import json  # noqa: E402

sys.path.insert(0, str(ROOT))
from config import settings  # noqa: E402
from scripts.build_gap_positions import analyze_one, _close_engines  # noqa: E402
from scripts.divergence_analysis import build_heldin_keys, pos_key  # noqa: E402
from src.engine import maia_engine  # noqa: E402


def _fens_from_jsonl(path: Path) -> List[str]:
    out: List[str] = []
    for row in read_jsonl(path):
        fen = row.get("fen")
        if not fen and isinstance(row.get("teacher_input"), dict):
            fen = row["teacher_input"].get("fen")
        if fen:
            out.append(fen)
    return out


def dedup_keys(
    *,
    include_train_v2: bool = True,
    include_v3: bool = True,
    include_candidates: bool = True,
    include_benchmarks: bool = True,
    extra_fen_files: Sequence[Path] = (),
) -> set:
    """Board keys the new test positions must NOT collide with.

    Union of: SFT training corpora (train_v2/valid_v2, train_v3/valid_v3, parsed
    back from their ASCII boards), the candidate pools (candidates_v2,
    v3_candidates, gap_positions), and every explicit-FEN benchmark scenario set.
    """
    keys: set = set()
    ds = settings.DATASET
    if include_train_v2:
        keys |= build_heldin_keys(ds / "train_v2.jsonl", ds / "valid_v2.jsonl")
    if include_v3:
        keys |= build_heldin_keys(ds / "train_v3.jsonl", ds / "valid_v3.jsonl")

    fen_files: List[Path] = list(extra_fen_files)
    if include_candidates:
        fen_files += [
            settings.GENERATED / "candidates_v2.jsonl",
            settings.POSITIONS / "v3_candidates.jsonl",
            settings.DATA / "eval" / "gap_positions.jsonl",
        ]
    if include_benchmarks:
        fen_files += [
            settings.DATA / "benchmark_gap803" / "scenarios.jsonl",
            settings.DATA / "benchmark_v2" / "scenarios.jsonl",
            settings.DATA / "benchmark" / "scenarios.jsonl",
        ]
    for fp in fen_files:
        if not fp.exists():
            continue
        for fen in _fens_from_jsonl(fp):
            keys.add(pos_key(fen))
    return keys


def _done_ids(out_path: Path) -> set:
    done: set = set()
    for row in read_jsonl(out_path):
        rid = row.get("id")
        if rid is not None:
            done.add(rid)
    return done


def ground_records(
    records: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    workers: int = 6,
    movetime_ms: int = settings.DEFAULT_MOVETIME_MS,
    tolerance_cp: int = settings.SOUND_TOLERANCE_CP,
    multipv: int = settings.MULTIPV,
    trivial_cp: int = 800,
    hash_mb: int = 64,
    resume: bool = True,
    progress_every: int = 200,
) -> int:
    """Ground ``records`` (Lichess-sampler schema) -> gap-schema rows in ``out_path``.

    Resumable: rows whose ``id`` is already present are skipped. Returns the number
    newly written. Non-analysable (illegal/terminal/no-line) positions are dropped.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path) if resume else set()
    todo = [r for r in records if r.get("id") not in done]
    print(f"[ground] {len(todo)} to analyse ({len(done)} already done) -> {out_path}",
          file=sys.stderr)
    if not todo:
        return 0

    write_lock = threading.Lock()
    n_written = 0
    n_skip = 0
    n_fail = 0
    t0 = time.time()

    def _work(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return analyze_one(
            rec, movetime_ms=movetime_ms, tolerance_cp=tolerance_cp,
            multipv=multipv, trivial_cp=trivial_cp, hash_mb=hash_mb,
        )

    mode = "a" if (resume and done) else "w"
    try:
        with out_path.open(mode, encoding="utf-8") as fh, \
                ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_work, rec): rec for rec in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                rec = futs[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    n_fail += 1
                    print(f"  ! {rec.get('id')} FAILED: {exc}", file=sys.stderr)
                    continue
                if row is None:
                    n_skip += 1
                    continue
                with write_lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                n_written += 1
                if i % progress_every == 0 or i == len(todo):
                    dt = time.time() - t0
                    print(f"  [{i}/{len(todo)}] written={n_written} skip={n_skip} "
                          f"fail={n_fail} | {dt:.0f}s ({dt/max(1,i):.2f}s/pos)",
                          file=sys.stderr)
    finally:
        _close_engines()
        try:
            maia_engine.close_all()
        except Exception:  # noqa: BLE001
            pass

    print(f"[ground] DONE wrote {n_written} rows (skip={n_skip} fail={n_fail}) in "
          f"{time.time()-t0:.0f}s", file=sys.stderr)
    return n_written
