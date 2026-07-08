"""Gated coaching generation — the shipped pipeline, for any model + prompt.

Produces one coaching answer per scenario the *same way the live coach does*:
grounded prompt (Stockfish pool + Maia + verified facts) -> the model ->
:func:`src.teacher.coach_gate.run_gate` (verify-and-regenerate + verified
fallback). Because the gate is the shared shipped unit, running the base and the
tuned model through :func:`generate` differs only in the weights (and, for the
"train by prompting" contender, the *system prompt*) — never the tools or gate.

Two backends, both exposed as a ``run_fn(system, user) -> text``:

* :class:`MLXSamplingCoach` — local MLX (free), sampling with the SAME knobs the
  server ships (``temp=0.7, top_p=0.8, top_k=20``) so the gate's re-samples truly
  explore (a greedy backend would return the identical draft every attempt, which
  would make the gate a no-op). Seeded per scenario for reproducibility.
* :class:`TFYRunFn` — the TrueFoundry gateway (frontier + the 32B open base),
  reusing the benchmark's retrying ``TFYChat``.

Rows are written in the benchmark generation schema (so the objective scorer and
council read them unchanged) plus gate telemetry (``attempts``,
``verified_fallback``, ``rec_uci``). Resumable: keyed by ``scenario_id``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from src.eval.benchmark.prompts import build_user_prompt, load_system_prompt
from src.eval.evaluate import _strip_think
from src.teacher.coach_gate import run_gate

log = logging.getLogger("honest.gated")

RunFn = Callable[[str, str], str]


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class MLXSamplingCoach:
    """Local MLX chat model matching the server's sampling (seedable, thinking off).

    ``model_path`` is a fused MLX dir/repo; ``adapter_path`` optionally applies an
    MLX LoRA on top of a base. :meth:`seed` reseeds MLX's RNG so a scenario's
    generation (and thus each gate re-sample) is reproducible run-to-run while
    still differing attempt-to-attempt (the RNG advances between calls).
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        *,
        max_tokens: int = 640,
        temp: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
    ) -> None:
        from mlx_lm import generate, load  # heavy; import lazily

        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self._generate = generate
        t0 = time.time()
        if adapter_path:
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        else:
            self.model, self.tokenizer = load(model_path)
        print(f"  loaded MLX {model_path!r}"
              + (f" +adapter {adapter_path!r}" if adapter_path else "")
              + f" in {time.time() - t0:.1f}s", file=sys.stderr)
        try:
            from mlx_lm.sample_utils import make_sampler

            self._sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
        except Exception:  # noqa: BLE001 - older mlx_lm: default sampler
            self._sampler = None
        self._lock = threading.Lock()

    def seed(self, seed: int) -> None:
        try:
            import mlx.core as mx

            mx.random.seed(int(seed) & 0x7FFFFFFF)
        except Exception:  # noqa: BLE001 - seeding is best-effort
            pass

    def _apply_template(self, system: str, user: str) -> Any:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def run(self, system: str, user: str) -> str:
        prompt = self._apply_template(system, user)
        kwargs: Dict[str, Any] = {"max_tokens": self.max_tokens, "verbose": False}
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler
        with self._lock:
            raw = self._generate(self.model, self.tokenizer, prompt=prompt, **kwargs)
        return _strip_think(raw)


class TFYRunFn:
    """Adapt a benchmark ``TFYChat`` (frontier / 32B open base) to a ``run_fn``."""

    def __init__(self, chat: Any) -> None:
        self._chat = chat

    def run(self, system: str, user: str) -> str:
        text, _usage = self._chat.complete(system, user)
        return text


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def _scn_seed(model_key: str, scenario_id: str) -> int:
    h = hashlib.sha256(f"{model_key}|{scenario_id}".encode()).hexdigest()
    return int(h[:8], 16)


def _done_ids(path: Path) -> set:
    done: set = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["scenario_id"])
                except Exception:  # noqa: BLE001
                    continue
    return done


def gated_row(
    scn: Dict[str, Any],
    run_fn: RunFn,
    model_key: str,
    *,
    system_prompt: str,
    max_attempts: int,
    gate_on: bool,
) -> Dict[str, Any]:
    """Generate one gated coaching row for ``scn`` (benchmark schema + telemetry)."""
    user = build_user_prompt(scn, "grounded")
    result = run_gate(
        run_fn, system_prompt, user, scn["fen"], scn["sound_pool"],
        scn["student_move"].get("uci") or "",
        max_attempts=max_attempts, gate_on=gate_on,
    )
    return {
        "scenario_id": scn["id"],
        "model": model_key,
        "condition": "gated",
        "tier": scn["tier"],
        "phase": scn["phase"],
        "severity": scn["severity"],
        "pos_id": scn.get("pos_id"),
        "output": result.text,
        "rec_uci": result.rec_uci,
        "rec_san": result.rec_san,
        "attempts": result.attempts,
        "verified_fallback": result.verified_fallback,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def generate(
    scenarios: Sequence[Dict[str, Any]],
    run_fn: RunFn,
    model_key: str,
    out_path: Path,
    *,
    system_prompt: Optional[str] = None,
    max_attempts: int = 6,
    gate_on: bool = True,
    seedable: Optional[MLXSamplingCoach] = None,
    progress_every: int = 20,
) -> Dict[str, int]:
    """Gated-generate every scenario to ``out_path`` (resumable by scenario_id).

    ``seedable`` (when ``run_fn`` is an :class:`MLXSamplingCoach`) is reseeded per
    scenario for reproducibility. Returns ``{ok, fail, skipped}``.
    """
    system = system_prompt if system_prompt is not None else load_system_prompt()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)
    todo = [s for s in scenarios if s["id"] not in done]
    log.info("%s: %d pending of %d (%d done)", model_key, len(todo), len(scenarios), len(done))
    ok = fail = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for i, scn in enumerate(todo, 1):
            if seedable is not None:
                seedable.seed(_scn_seed(model_key, scn["id"]))
            try:
                row = gated_row(
                    scn, run_fn, model_key, system_prompt=system,
                    max_attempts=max_attempts, gate_on=gate_on,
                )
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                ok += 1
            except Exception as exc:  # noqa: BLE001 - one item must not abort the run
                fail += 1
                log.error("%s %s failed: %s", model_key, scn["id"], exc)
            if i % progress_every == 0 or i == len(todo):
                dt = time.time() - t0
                log.info("  %s %d/%d (%.2fs/it, eta %.0fm)",
                         model_key, i, len(todo), dt / i, dt / i * (len(todo) - i) / 60)
    return {"ok": ok, "fail": fail, "skipped": len(done)}
