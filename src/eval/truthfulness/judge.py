"""Cross-family LLM-judge truthfulness pass — the non-circular residual metric.

The deterministic checkers (:mod:`src.engine.faithfulness` +
:mod:`src.engine.faithfulness_ext`) decide the claims a board computation *can*
decide: piece locations, single-move consequences, simple relations, counts. They
deliberately abstain on everything else. This module covers that residual:

* multi-move tactical claims — "this wins the queen", "leads to mate in three";
* evaluations / assessments — "you're winning", "the position is equal";
* any concrete claim that is false or **unsupported** by the verified facts but
  is beyond a one-ply board check.

It asks an **independent panel of >=2 cross-family judges** (GPT / Claude / Gemini
via the TrueFoundry gateway) to flag such claims. It is designed to run on the
*gated* coach output (after the deterministic gate has already scrubbed the
mechanical falsehoods), so what it measures is the truthfulness that survives the
gate — hence "non-circular residual". Judges never grade their own family's
homework by construction (three different labs), and no judge is the coach.

Testability: this module has **no import-time dependency** on the model backends
or MLX. Judge *clients* are injected (any object exposing
``complete(system, user) -> (text, usage)``, exactly like
:class:`src.eval.benchmark.backends.TFYChat`). :func:`default_panel` builds the
real cross-family panel lazily, so an offline unit test can pass mock clients and
never import the gateway stack.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import chess

log = logging.getLogger("truthfulness.judge")

__all__ = [
    "ChatClient",
    "JudgeClient",
    "TruthfulnessResult",
    "TruthfulnessJudge",
    "assess_truthfulness",
    "default_panel",
    "build_system_prompt",
    "build_user_prompt",
    "parse_judge_reply",
    "aggregate",
]

Usage = Dict[str, int]


class ChatClient(Protocol):
    """Anything shaped like :class:`backends.TFYChat` — one ``complete`` call."""

    def complete(self, system: str, user: str) -> Tuple[str, Usage]:  # pragma: no cover - protocol
        ...


@dataclass
class JudgeClient:
    """A named judge: a cross-family label plus its chat client."""

    name: str
    client: ChatClient


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def build_system_prompt() -> str:
    """Judge instructions: flag only concrete FALSE / UNSUPPORTED claims."""
    return (
        "You are a strict chess fact-checker auditing a coach's written feedback to a "
        "student. You are NOT grading style, tone, usefulness, or pedagogy. Your ONLY "
        "job is to catch CONCRETE claims that are FALSE or UNSUPPORTED.\n\n"
        "You are given: the position (FEN + board), whose move it is, the move the coach "
        "recommends, and a block of VERIFIED engine facts (these are ground truth — "
        "computed from the board, never guessed). Then you see the coaching text.\n\n"
        "FLAG a claim when it is one of:\n"
        "  - a tactical claim that spans multiple moves and is not true "
        "(e.g. 'this wins the queen', 'forces mate', 'traps the rook') and is not "
        "supported by the verified facts;\n"
        "  - a position assessment that is clearly wrong given the facts "
        "(e.g. 'you're winning' when material/threats say otherwise);\n"
        "  - any specific board/consequence claim that contradicts the verified facts.\n\n"
        "Do NOT flag:\n"
        "  - subjective coaching advice, plans, or general principles "
        "('control the center', 'develop your pieces');\n"
        "  - hedged or approximate language ('this looks a bit better', 'you might consider');\n"
        "  - anything you cannot show to be false. When unsure, DO NOT flag it. "
        "Precision matters far more than recall.\n\n"
        "Quote the offending claim verbatim (a short phrase) and give a one-sentence reason "
        "grounded in the facts. Return ONLY a single JSON object, no prose:\n"
        '{"truthful": true|false, "flagged": [{"claim": "<verbatim phrase>", '
        '"reason": "<why it is false/unsupported>"}]}\n'
        'If nothing is wrong, return {"truthful": true, "flagged": []}.'
    )


def build_user_prompt(
    coaching_text: str,
    fen: str,
    recommended_move: Optional[str],
    engine_facts_text: str,
) -> str:
    """Assemble the judge's user message (board + facts + coaching to audit)."""
    board_ascii = fen
    side = ""
    try:
        board = chess.Board(fen)
        board_ascii = str(board)
        side = "White" if board.turn == chess.WHITE else "Black"
    except ValueError:
        pass

    parts = [
        f"FEN: {fen}",
        f"Side to move: {side}" if side else "",
        "BOARD:",
        board_ascii,
        f"Coach's recommended move: {recommended_move}" if recommended_move else "",
        "",
        "VERIFIED ENGINE FACTS (ground truth — trust these over the coaching text):",
        engine_facts_text.strip() or "(none provided)",
        "",
        "COACHING TEXT TO FACT-CHECK (verbatim):",
        "<<<",
        coaching_text.strip(),
        ">>>",
        "",
        "Return the single JSON object flagging any false or unsupported concrete claim.",
    ]
    return "\n".join(p for p in parts if p != "")


# --------------------------------------------------------------------------- #
# Parsing one judge reply
# --------------------------------------------------------------------------- #

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """First balanced ``{...}`` JSON object in ``text`` (self-contained; no deps)."""
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
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    return None


def parse_judge_reply(text: str) -> Dict[str, Any]:
    """Parse one judge's reply into ``{"truthful": bool, "flagged": [...]}``.

    Defensive: the ``flagged`` list is the source of truth (a non-empty list means
    not truthful, whatever the model wrote in ``truthful``). Malformed items are
    coerced; a bare ``truthful:false`` with no items becomes one unspecified flag.
    """
    obj = _extract_json_object(text) or {}
    raw_flags = obj.get("flagged") or obj.get("flags") or []
    flagged: List[Dict[str, str]] = []
    if isinstance(raw_flags, list):
        for item in raw_flags:
            if isinstance(item, dict):
                claim = str(item.get("claim") or item.get("quote") or item.get("text") or "").strip()
                reason = str(item.get("reason") or item.get("why") or item.get("explanation") or "").strip()
            else:
                claim, reason = str(item).strip(), ""
            if claim or reason:
                flagged.append({"claim": claim or "(unspecified)", "reason": reason})

    explicit = obj.get("truthful")
    if not flagged and explicit is False:
        note = str(obj.get("reason") or obj.get("note") or "").strip()
        flagged.append({"claim": "(unspecified)", "reason": note or "judge marked untruthful"})

    return {"truthful": len(flagged) == 0, "flagged": flagged}


# --------------------------------------------------------------------------- #
# Aggregation across the panel
# --------------------------------------------------------------------------- #

def aggregate(
    per_judge: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    mode: str = "any",
) -> Dict[str, Any]:
    """Combine per-judge verdicts into the panel result.

    ``per_judge`` is a sequence of ``(judge_name, parsed_reply)`` for judges that
    returned successfully. ``mode``:

    * ``"any"``      — the panel is truthful iff *no* judge flagged anything
      (strict; a single reviewer's objection sinks it).
    * ``"majority"`` — the panel is truthful iff a strict majority of judges found
      it truthful (ties resolve to *not* truthful, i.e. err toward flagging).

    ``agreement`` is how much the panel agreed on the truthful/flagged split,
    ``max(n_truthful, n_flagged) / n_judges`` (1.0 = unanimous), independent of mode.
    """
    n = len(per_judge)
    flagged: List[Dict[str, str]] = []
    n_truthful = 0
    for name, r in per_judge:
        if r["truthful"]:
            n_truthful += 1
        for f in r["flagged"]:
            flagged.append({"claim": f["claim"], "reason": f["reason"], "judge": name})
    n_flagged = n - n_truthful

    if n == 0:
        return {"truthful": True, "flagged": [], "n_judges": 0, "agreement": 0.0}

    if mode == "majority":
        truthful = n_truthful > (n / 2.0)
    else:  # "any"
        truthful = n_flagged == 0

    agreement = max(n_truthful, n_flagged) / n
    return {
        "truthful": truthful,
        "flagged": flagged,
        "n_judges": n,
        "agreement": round(agreement, 4),
    }


# --------------------------------------------------------------------------- #
# The panel
# --------------------------------------------------------------------------- #

@dataclass
class TruthfulnessResult:
    """Structured panel verdict (also exposed as a plain dict via ``to_dict``)."""

    truthful: bool
    flagged: List[Dict[str, str]]
    n_judges: int
    agreement: float
    judge_verdicts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    usage: Usage = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truthful": self.truthful,
            "flagged": self.flagged,
            "n_judges": self.n_judges,
            "agreement": self.agreement,
            "judge_verdicts": self.judge_verdicts,
            "errors": self.errors,
            "usage": self.usage,
        }


class TruthfulnessJudge:
    """Runs a cross-family panel over one coaching output and aggregates verdicts."""

    def __init__(
        self,
        judges: Sequence[JudgeClient],
        *,
        aggregation: str = "any",
        concurrency: int = 4,
    ) -> None:
        if len(judges) < 2:
            raise ValueError("need >=2 cross-family judges for an independent panel")
        if aggregation not in ("any", "majority"):
            raise ValueError("aggregation must be 'any' or 'majority'")
        self.judges = list(judges)
        self.aggregation = aggregation
        self.concurrency = max(1, concurrency)

    def assess(
        self,
        coaching_text: str,
        fen: str,
        recommended_move: Optional[str],
        engine_facts_text: str,
    ) -> TruthfulnessResult:
        """Fact-check one coaching output; return the aggregated panel verdict."""
        system = build_system_prompt()
        user = build_user_prompt(coaching_text, fen, recommended_move, engine_facts_text)

        per_judge: List[Tuple[str, Dict[str, Any]]] = []
        verdicts: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}
        usage: Usage = {"prompt_tokens": 0, "completion_tokens": 0}

        def _one(j: JudgeClient) -> Tuple[str, Optional[Dict[str, Any]], Optional[Usage], Optional[str]]:
            try:
                text, u = j.client.complete(system, user)
                return j.name, parse_judge_reply(text), u, None
            except Exception as exc:  # noqa: BLE001 - a failed judge is skipped, not fatal
                return j.name, None, None, f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(self.judges))) as pool:
            futures = [pool.submit(_one, j) for j in self.judges]
            for fut in as_completed(futures):
                name, parsed, u, err = fut.result()
                if err is not None or parsed is None:
                    errors[name] = err or "no result"
                    log.warning("truthfulness judge %s failed: %s", name, errors[name])
                    continue
                per_judge.append((name, parsed))
                verdicts[name] = parsed
                if u:
                    usage["prompt_tokens"] += int(u.get("prompt_tokens", 0))
                    usage["completion_tokens"] += int(u.get("completion_tokens", 0))

        agg = aggregate(per_judge, mode=self.aggregation)
        return TruthfulnessResult(
            truthful=agg["truthful"],
            flagged=agg["flagged"],
            n_judges=agg["n_judges"],
            agreement=agg["agreement"],
            judge_verdicts=verdicts,
            errors=errors,
            usage=usage,
        )


# --------------------------------------------------------------------------- #
# Live panel factory (lazy heavy imports; keys from ROOT/.env TFY_API_KEY)
# --------------------------------------------------------------------------- #

def default_panel(
    judge_keys: Optional[Sequence[str]] = None,
    *,
    timeout: float = 180.0,
    max_retries: int = 4,
    min_interval: float = 0.05,
    max_tokens: int = 1500,
) -> List[JudgeClient]:
    """Build the real cross-family panel via the TrueFoundry gateway.

    Reuses :func:`src.eval.benchmark.backends.make_tfy_client` /
    :class:`~src.eval.benchmark.backends.TFYChat` and the model registry in
    :mod:`src.eval.benchmark.config`. Imports are deferred to here so importing
    this module (and unit-testing the panel with mocks) never pulls in the gateway
    or MLX stack. ``TFYChat`` already omits ``temperature`` and retries transient
    errors. Requires ``TFY_API_KEY`` / ``TFY_BASE_URL`` in ``ROOT/.env``.
    """
    from src.eval.benchmark import config as bcfg
    from src.eval.benchmark.backends import RateLimiter, TFYChat, make_tfy_client

    keys = tuple(judge_keys) if judge_keys is not None else tuple(bcfg.JUDGE_KEYS)
    if len(keys) < 2:
        raise ValueError("need >=2 cross-family judges for an independent panel")

    client = make_tfy_client(timeout)
    limiter = RateLimiter(min_interval)
    panel: List[JudgeClient] = []
    for key in keys:
        model = bcfg.MODELS[key]
        chat = TFYChat(
            client,
            model_id=model.ident,
            max_tokens=max_tokens,
            max_retries=max_retries,
            limiter=limiter,
            reasoning_effort=model.reasoning_effort,
        )
        panel.append(JudgeClient(name=key, client=chat))
    return panel


def assess_truthfulness(
    coaching_text: str,
    fen: str,
    recommended_move: Optional[str],
    engine_facts_text: str,
    *,
    judge_keys: Optional[Sequence[str]] = None,
    aggregation: str = "any",
    **panel_kwargs: Any,
) -> Dict[str, Any]:
    """Convenience one-shot: build the default panel and fact-check one output.

    Returns the plain-dict verdict
    ``{"truthful", "flagged", "n_judges", "agreement", ...}``. For tests or custom
    wiring, construct :class:`TruthfulnessJudge` with your own
    :class:`JudgeClient` list instead.
    """
    panel = default_panel(judge_keys, **panel_kwargs)
    judge = TruthfulnessJudge(panel, aggregation=aggregation)
    return judge.assess(coaching_text, fen, recommended_move, engine_facts_text).to_dict()
