"""Probe TrueFoundry access to the BIGGER open-source chat/reasoning models.

Lists the gateway model ids, then for each candidate FQN (from the extend-benchmark
brief) checks (a) whether it is listed and (b) whether this key can actually CALL it
with a tiny 1-token chat completion. Prints OK vs DENY(code)/MISSING. Never prints
the key. Also dumps every ``bedrock``/``oss`` id it can see so name variants surface.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

# Chat/reasoning open models to test (coder-only + vision-only excluded per brief).
CANDIDATES = [
    "aws-bedrock/qwen.qwen3-32b-v1-0",
    "aws-bedrock/qwen.qwen3-next-80b-a3b",
    "aws-bedrock/google.gemma-3-27b-it",
    "aws-bedrock/us.meta.llama3-3-70b-instruct-v1-0",
    "aws-bedrock/us.meta.llama4-maverick-17b-instruct-v1-0",
    "aws-bedrock/deepseek.v3.2",
    "aws-bedrock/zai.glm-5",
    "aws-bedrock/mistral.mistral-large-3-675b-instruct",
    "aws-bedrock/moonshotai.kimi-k2.5",
    # Optional reasoning models (include only if their output fits the coach format).
    "aws-bedrock/deepseek.r1",
    "aws-bedrock/moonshotai.kimi-k2-thinking",
]


def main() -> int:
    load_dotenv()
    key = os.environ.get("TFY_API_KEY")
    base = os.environ.get("TFY_BASE_URL")
    if not key or not base:
        print("BLOCKED: TFY_API_KEY / TFY_BASE_URL missing from .env")
        return 1
    c = OpenAI(api_key=key, base_url=base, timeout=40, max_retries=0)

    try:
        ids = [m.id for m in c.models.list().data]
    except Exception as e:  # noqa: BLE001
        print(f"BLOCKED: models.list() failed: {type(e).__name__}: {str(e)[:200]}")
        return 1
    idset = set(ids)
    print(f"gateway lists {len(ids)} models\n")

    bedrock_like = sorted(i for i in ids if "bedrock" in i.lower() or "oss" in i.lower())
    print("=== bedrock/oss ids visible ({}): ===".format(len(bedrock_like)))
    for i in bedrock_like:
        print("   ", i)
    print()

    print("=== probing candidates (1-token chat call) ===")
    reachable, denied, missing = [], [], []
    for m in CANDIDATES:
        listed = m in idset
        try:
            c.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=1,
            )
            print(f"OK        listed={listed!s:<5} {m}")
            reachable.append(m)
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "status_code", "?")
            msg = str(e)[:110].replace("\n", " ")
            tag = "MISSING" if (not listed and code in (404, "?")) else f"DENY {code}"
            print(f"{tag:<9} listed={listed!s:<5} {m}  :: {msg}")
            (missing if tag == "MISSING" else denied).append(m)

    print("\n=== SUMMARY ===")
    print(f"reachable ({len(reachable)}):")
    for m in reachable:
        print("   ", m)
    print(f"denied ({len(denied)}):")
    for m in denied:
        print("   ", m)
    print(f"missing ({len(missing)}):")
    for m in missing:
        print("   ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
