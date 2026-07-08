#!/usr/bin/env python3
"""SMALL v5-style demo generator (sanctioned, sparing TFY teacher use).

Reuses EXISTING grounded prompts from train_v4 (identical VERIFIED FACTS + pool +
Maia grounding) and only upgrades the TEACHER INSTRUCTION to the proposed v5 spec:
  * takeaway MUST name a transferable principle from a controlled vocabulary
  * "How to find it" MUST be a reusable thinking routine (<=2 sentences), not
    board narration
  * clean lead: address the student's move; NEVER restate the move as a command
    ("Play X." / "The move is X.") -> kills the dangling/restate artifacts
  * beginner vocabulary cap (no tempo/prophylaxis/outpost/etc.)
Then renders with render_assistant_target_v2 and prints OLD vs NEW + element
coverage, as an A/B proof that the v5 prompt moves the weak axes.

Caps total teacher calls (default 4). Uses TFY openai-group/gpt-5.5.
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv
from openai import OpenAI

from config import schema  # noqa: E402
from src.teacher.generate import RateLimiter, TeacherClient  # noqa: E402
from audit_v5_instructiveness import (  # noqa: E402
    parse_user, parse_assistant, PRINCIPLE_FAMILIES, TACTIC_FAMILIES, principle_hits,
)

try:
    from src.engine.faithfulness import verify_text
except Exception:  # noqa: BLE001
    verify_text = None

CONTROLLED_PRINCIPLES = [
    "King safety first (get the king out of the center)",
    "Develop before you attack",
    "Make threats while you develop (gain time)",
    "Don't leave pieces hanging (count attackers vs defenders)",
    "Put rooks on open files",
    "A rook on the 7th rank is powerful",
    "Create and push a passed pawn",
    "Improve your worst-placed piece",
    "Trade pieces when you are AHEAD in material; keep pieces when BEHIND",
    "Prophylaxis: stop the opponent's plan before your own",
    "Control the center",
    "Look for a move that does two jobs at once",
]

V5_INSTRUCTION = (
    "\n\nWrite the coaching to this v5 spec:\n"
    "1) Start coaching by ADDRESSING THE STUDENT'S ACTUAL MOVE. Do NOT restate the "
    "recommended move as a command (never write 'Play X.', 'The move is X.', 'THE "
    "MOVE:'), and never begin with a dangling dash. The row already opens with "
    "\"I'd play <MOVE>.\" so your 'coaching' must NOT repeat that.\n"
    "2) 'method' = ONE reusable thinking routine a player at this tier can run on "
    "OTHER positions to FIND this kind of move (<=2 sentences). It is a checklist/"
    "question, not a narration of this game.\n"
    "3) 'takeaway' = ONE short sentence that NAMES a transferable principle. Prefer "
    "one of these named principles (adapt wording to the tier):\n   - "
    + "\n   - ".join(CONTROLLED_PRINCIPLES) + "\n"
    "4) Keep it grounded in ONLY the VERIFIED FACTS and concise (<=140 words total).\n"
)

BEGINNER_VOCAB_BAN = (
    " For this BEGINNER student, do NOT use the words: tempo, prophylaxis, outpost, "
    "initiative, imbalance, zugzwang, bishop pair. Use plain language a 1000-1200 "
    "player knows.\n"
)

SYSTEM = (ROOT / "prompts" / "coach_system.md").read_text(encoding="utf-8").strip()


def forced_directive(tier: str, san: str) -> str:
    return (
        f"\n\nPRE-SELECTED TEACHING MOVE for this {tier} player: {san}. You MUST teach "
        f"exactly this move (set recommended_move_san={san}). Return STRICT JSON with "
        f"keys: coaching, method, takeaway."
    )


def build_v5_user(existing_user: str, tier: str, san: str) -> str:
    instr = V5_INSTRUCTION + (BEGINNER_VOCAB_BAN if tier == "beginner" else "")
    return existing_user + forced_directive(tier, san) + instr


def elements(text: str, tier: str) -> str:
    pa = parse_assistant(text)
    strat = principle_hits(text, PRINCIPLE_FAMILIES)
    tac = principle_hits(text, TACTIC_FAMILIES)
    tk_strat = principle_hits(pa["takeaway"] or "", PRINCIPLE_FAMILIES)
    tk_any = principle_hits(pa["takeaway"] or "", {**PRINCIPLE_FAMILIES, **TACTIC_FAMILIES})
    after = re.sub(r"^I'?d play [^.]+\.\s*", "", text.strip())
    dangling = bool(re.match(r"^[—\-–]\s*(and|but|so|then|in fact)\b", after, re.I))
    restate = bool(re.match(r"^(THE MOVE|The move is|This is the move|Play\s+\S+\.|Consider)", after, re.I))
    words = len(re.findall(r"\S+", text))
    return (f"principle_in_takeaway={'YES' if (tk_strat or tk_any) else 'no'} "
            f"named_families={strat+tac} artifact={'DANGLING' if dangling else ('RESTATE' if restate else 'clean')} "
            f"words={words}")


def load_unique(path: Path):
    seen, out = set(), []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            msgs = d.get("messages") or []
            u = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            a = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            if u and a and (u, a) not in seen:
                seen.add((u, a)); out.append((u, a))
    return out


def main():
    load_dotenv(ROOT / ".env")
    rows = load_unique(ROOT / "data" / "dataset" / "train_v4.jsonl")

    # index board -> tier -> (user, assistant, rec)
    from collections import defaultdict
    board = defaultdict(dict)
    beg_no_principle = None
    for u, a in rows:
        pu, pa = parse_user(u), parse_assistant(a)
        if pu["board"] and pu["tier"] and pa["rec"] and pu["tier"] not in board[pu["board"]]:
            board[pu["board"]][pu["tier"]] = (u, a, pa["rec"])
        if beg_no_principle is None and pu["tier"] == "beginner":
            if not principle_hits(pa["takeaway"] or "", {**PRINCIPLE_FAMILIES, **TACTIC_FAMILIES}):
                beg_no_principle = (u, a, pa["rec"])

    # pick first full-gradient board (3 distinct recs), small pool preferred
    gradient = None
    for b, tm in board.items():
        if len(tm) == 3:
            recs = {tm[t][2] for t in tm}
            if len(recs) == 3:
                gradient = (b, tm); break

    key = os.environ["TFY_API_KEY"]; base = os.environ["TFY_BASE_URL"]
    model = os.environ.get("TFY_TEACHER_MODEL") or "openai-group/gpt-5.5"
    client = OpenAI(api_key=key, base_url=base, timeout=300, max_retries=0)
    teacher = TeacherClient(client, model=model, reasoning_effort="medium",
                            max_retries=3, limiter=RateLimiter(0.05))
    print(f"teacher: {model} (effort=medium)\n")

    calls = 0
    MAX_CALLS = 4

    def gen(u, tier, san):
        raw = teacher.complete(SYSTEM, build_v5_user(u, tier, san))
        to = {"recommended_move_san": san, "coaching": raw.get("coaching", ""),
              "method": raw.get("method", ""), "takeaway": raw.get("takeaway", "")}
        new = schema.render_assistant_target_v2(to)  # type: ignore[arg-type]
        faith = "n/a"
        if verify_text is not None:
            # need fen -> not available from prompt; skip strict, note only
            faith = "not-checked(demo)"
        return new, faith

    if gradient:
        b, tm = gradient
        print("=" * 80)
        print("DEMO 1 — SAME POSITION, CONTRASTIVE TIERS (full gradient)\n")
        print(b, "\n")
        for tier in ("beginner", "intermediate", "advanced"):
            if calls >= MAX_CALLS:
                break
            u, a, rec = tm[tier]
            new, faith = gen(u, tier, rec); calls += 1
            print(f"\n--- {tier.upper()}  (canonical move: {rec}) ---")
            old = parse_assistant(a)["full"]
            print(f"  OLD [{elements(old, tier)}]")
            print(f"      {old[:300]}")
            print(f"  NEW [{elements(new, tier)}]")
            print(f"      {new[:400]}")

    if beg_no_principle and calls < MAX_CALLS:
        u, a, rec = beg_no_principle
        print("\n" + "=" * 80)
        print("DEMO 2 — BEGINNER takeaway that named NO principle -> v5 fix\n")
        new, faith = gen(u, "beginner", rec); calls += 1
        old = parse_assistant(a)["full"]
        print(f"  OLD [{elements(old, 'beginner')}]")
        print(f"      {old[:300]}")
        print(f"  NEW [{elements(new, 'beginner')}]")
        print(f"      {new[:400]}")

    print(f"\n\ntotal teacher calls: {calls}")
    print("usage:", teacher.usage_summary(1.25, 10.0))


if __name__ == "__main__":
    main()
