"""Update the public dataset card (README.md) for chess-coach-move-review.

This is a docs-only helper. It rewrites the Hugging Face dataset card for
CLARITY and does not touch the data files, the model, or any other repo.

To avoid ever breaking the dataset viewer, it does NOT hardcode the YAML
frontmatter. It downloads the current card, keeps the live frontmatter
(the ``configs:`` block that drives the viewer) byte-for-byte, and swaps in
only the human-readable body below.

Usage::

    ~/.venvs/mlx/bin/python -m scripts.update_dataset_card --dry-run   # print, no push
    ~/.venvs/mlx/bin/python -m scripts.update_dataset_card             # push to HF

The HF write token is read from the repo ``.env`` (HF_TOKEN /
HUGGING_FACE_HUB_TOKEN) or the ambient environment / cached CLI login.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "khoilamalphaai/chess-coach-move-review"

# The card body (everything after the YAML frontmatter). Numbers are pulled
# from the source repo (BRAINLIFT.md, README.md, RESULTS_FULL_EVAL_803.md,
# data/analysis/V6_REBUILD_REPORT.md) and the live data files.
BODY = r"""# Chess coach move-review SFT dataset

Supervised fine-tuning data for one specific, trained behavior: given a chess
position and the student's rating tier (Beginner, Intermediate, or Advanced),
select the tier-appropriate instructive move and tag it with a short principle,
for example "Nf3, develop toward the center."

That single move choice is the trained objective, and it is deterministically
checkable. The plain-English explanation rendered beside the move is a secondary
display layer. It is still present in the SFT loss, but it is not separately
optimized and it is not the evaluation claim. The truthfulness of that prose is
enforced at serving time by a separate non-LLM verifier, not by the fine-tune.

This framing matches the project thesis documents (`BRAINLIFT.md` and `README.md`
in the source repo): the fine-tune reproduces a tier-selection policy that a
prompt on the same base does not, while the canonical move itself is defined by
engine and human-move grounding, not by the model weights.

## Shipped version: v4 (default)

`v4/train.jsonl` is the training set behind the shipped coach, the
[chess-coach-32b-v4-qlora](https://huggingface.co/khoilamalphaai/chess-coach-32b-v4-qlora)
adapter and its sibling small-model coaches. The `v4` config is the default:

```python
from datasets import load_dataset

ds = load_dataset("khoilamalphaai/chess-coach-move-review")  # resolves to v4
print(ds)                            # DatasetDict with train / validation
print(ds["train"][0]["messages"])    # a system / user / assistant chat triple
```

Earlier iterations remain available and are selected by name:

```python
ds_v3 = load_dataset("khoilamalphaai/chess-coach-move-review", "v3")
```

## Splits and sizes

| Config | Train rows | Validation rows | Notes |
|---|---:|---:|---|
| v4 (default) | 8,745 | 353 | Shipped SFT set, tuned to maximize tier-appropriate move selection. |
| v3 | 6,772 | 356 | Prior 32B all-rounder, balancing move choice and prose. |
| v2 | 2,457 | 129 | The genuinely small (1.7B) learnability result, contrastive multi-tier pairs. |
| v1 | 1,376 | 72 | Earliest iteration. |

Validation splits are game-disjoint holdouts, and the shipped model's evaluation
slice is checked to have zero board-key overlap with its training data.

## Row schema

Every row is a JSON object with a single key, `messages`: an OpenAI-style chat
list of exactly three turns (`system`, `user`, `assistant`). There is no separate
metadata column; all grounding lives inside the text of the turns.

| Turn | What it contains |
|---|---|
| `system` | The coach specification. It states the role, describes the inputs, gives the job (recommend exactly one move from the sound list, the single most instructive move for this tier, which is often not the top engine move), and lists the hard rules: never mention centipawns or the engine, respect the position's ply cap, cite only tactics that truly exist, simplify without falsifying, and never recommend an unsound move. |
| `user` | The engine-grounded prompt for one position. It carries a VERIFIED FACTS block (side to move, the exact pieces on each square, which pieces are loose or attacked, and what each candidate move concretely does), the student rating tier, an ASCII board, the move the student played with its mistake severity, the Stockfish sound-candidate pool (with internal evaluations and short lines, explicitly marked never to be quoted), the Maia human-move likelihoods for that tier, the ply cap, and the exact output format to follow. |
| `assistant` | The coaching label. It opens with "I'd play <MOVE>." in standard algebraic notation, gives two to four sentences that tie the move to the student's mistake and a concrete plan plus how to find it, and closes with one "Takeaway:" line. The MOVE is the trained, checkable target; the surrounding sentences are the display layer. |

## Tiers and the leveling signal

Three tiers are used throughout: Beginner (1000-1200), Intermediate (1300-1600),
and Advanced (1700-2000). The dataset is built as contrastive multi-tier sets:
the same position is taught at more than one tier so the correct label varies by
level. A beginner is guided to a sound, human-findable move; an advanced player is
guided toward the sharper engine-best line. This is what directly supervises the
model to change its move by tier, the one behavior a prompt on the base does not
reliably provide.

## How it was built

The dataset is fully synthetic, distilled from a frontier teacher on top of real
engine grounding.

| Step | Process |
|---|---|
| Positions | Sampled from real Lichess games (the CC0 Open Database) across the three tiers, each paired with the move a human actually played. |
| Grounding | Every position is analyzed with Stockfish (a tolerance-gated sound-move pool plus mistake severity) and Maia (human-move likelihood at the tier). A short deterministic rule, `select_tier_move`, turns those two signals into the single canonical move per tier. |
| Teacher | GPT-5.5 (maximum reasoning) distills the grounded analysis into leveled coaching that obeys the system specification. |
| Quality gates | A hard filter rejects any candidate whose recommended move is not in the sound pool, leaks engine numbers, exceeds the ply cap, or breaks the output format. A non-LLM faithfulness check verifies board claims against the real position. |

Move labels are engine-verified: the recommended move is always drawn from and
checked against the Stockfish sound pool, so there are no unsound move labels. The
prose check is high-precision but low-recall: it drives verifier-detectable
board-fact violations to zero, which is not the same as certified truthfulness
(relational, threat, and evaluation claims it does not cover can still be
imperfect).

## What the data buys (measured)

The behavior is graded deterministically as tier-policy exact match, meaning
agreement with the `select_tier_move` rule computed from the engine and Maia, with
no LLM judge in the loop (this metric is labeled tier-fit in some benchmark
tables). On the held-out validation slice, the v4-trained 32B adapter reaches:

- tier-policy match 0.767, versus 0.347 for the untuned Qwen3-32B base and 0.553 for the best frontier reference (Gemini 3.1 Pro),
- distinct-moves-per-level 0.730 (73 of 100 positions where a beginner and an advanced label should differ),
- raw move-soundness 0.942.

Over all positions where v4 diverges from the best frontier, the unbiased
head-to-head is 56-24-12. The frequently cited 51-5-6 is a selection-conditioned
subset (the 62 positions where v4 already gives a distinct, sound, correctly-graded
move), so it overstates a general win rate. These numbers measure agreement with
the project's own move rule, not validated teaching quality. Because the evaluation
is periodically regenerated, refer to the live artifacts for the current field:

| Artifact | Link |
|---|---|
| Benchmark dashboard | [chess-coach-benchmark](https://huggingface.co/spaces/khoilamalphaai/chess-coach-benchmark) |
| Grand eval data | [chess-coach-grand-eval](https://huggingface.co/datasets/khoilamalphaai/chess-coach-grand-eval) |
| Live demo | [chess-coach-studio](https://huggingface.co/spaces/khoilamalphaai/chess-coach-studio) |

## Version lineage

The shipped set was reached through a documented iteration line, kept here so
results stay reproducible:

- v1 to v2: the original data intervention. v2 is the genuinely small, on-spec result; the 1.7B tune it produced reproduces the tier policy where a prompt on the same base cannot.
- v3: a 32B all-rounder that balanced move selection against prose quality.
- v4 (shipped): a 32B set tuned to maximize tier-policy match. It leads the field on move selection and is deliberately weaker on prose, which is consistent with prose being an optional layer rather than the trained objective.

A deeper-verified v6 rebuild exists as the current data frontier. It is not
published here as a config, and the shipped v4 data is left untouched. The v6
labels are re-derived with Stockfish 17 using a two-depth root search (depths 14
and 20 with agreement bands), Syzygy tablebases for endgames of seven pieces or
fewer, and Maia-2 as a human-likelihood constraint. It builds complete triads by
construction (every board carries all three tiers, with no collapsed levels),
fixes the advanced tier to the verified engine-best move, and re-derives about 45%
of the training labels (3,307 of 7,269 comparable rows). It feeds the downstream
DPO and engine-distillation retrains.

## Limitations and honest caveats

- Advanced tier is close to engine-best. By construction the advanced label is the sharp engine move, so advanced rows largely mirror the engine. The distinctive leveling signal is strongest at the Beginner and Intermediate tiers.
- The policy is learnable, not deployment-necessary. `select_tier_move` already computes the canonical move at about 1.0 from the same grounding, so a model trained on this data approximates a policy the grounded product already produces. The fine-tune becomes load-bearing only in a grounding-free, fully-local setting that was not built or measured.
- Grounding is required at inference. A model trained here reproduces the move policy reliably only when given the same Stockfish and Maia grounding in the prompt. Without the per-tier Maia signal, the three tiers collapse to a single move.
- The explanation gate is not certified truth. Move labels are engine-sound, but the prose verifier is high-precision and low-recall. Truthfulness at serving time depends on the runtime verifier, not on further fine-tuning.
- tier-policy match is fidelity to a heuristic. It measures agreement with `select_tier_move`, a project rule, not certified best teaching. Whether these moves help students learn faster is unvalidated.
- Teacher distillation. Move labels are hard-filtered against the engine, but the "instructive" judgment in the prose is distilled from an LLM teacher (GPT-5.5), not ground truth.

## Links and license

| Resource | Destination |
|---|---|
| Shipped model (v4, 32B QLoRA) | [chess-coach-32b-v4-qlora](https://huggingface.co/khoilamalphaai/chess-coach-32b-v4-qlora) |
| Small model (1.7B, MLX) | [qwen3-1.7b-chess-coach-mlx](https://huggingface.co/khoilamalphaai/qwen3-1.7b-chess-coach-mlx) |
| Benchmark data | [chess-coach-benchmark](https://huggingface.co/datasets/khoilamalphaai/chess-coach-benchmark) |

Positions derive from the public CC0 Lichess Open Database; the coaching text is
distilled from GPT-5.5. Released for research and education under CC-BY-NC-4.0.
Respect the source models' terms.
"""


def _load_token() -> str | None:
    for env_file in (_ROOT / ".env",):
        if env_file.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_file)
            except Exception:
                # Minimal fallback parser if python-dotenv is unavailable.
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _split_frontmatter(md: str):
    """Return (frontmatter_including_delimiters, body). None if not found."""
    if not md.startswith("---"):
        return None, md
    lines = md.splitlines(keepends=True)
    for i in range(1, len(lines)):
        if lines[i].rstrip("\n") == "---":
            return "".join(lines[: i + 1]), "".join(lines[i + 1 :])
    return None, md


def build_card(current_readme: str) -> str:
    frontmatter, _ = _split_frontmatter(current_readme)
    if frontmatter is None:
        raise SystemExit(
            "ERROR: could not find YAML frontmatter in the live card; refusing to "
            "push (would break the dataset viewer)."
        )
    return frontmatter.rstrip("\n") + "\n\n" + BODY.strip() + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--dry-run", action="store_true", help="print the assembled card, do not push")
    p.add_argument("--token", default=None)
    args = p.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download

    token = args.token or _load_token()
    if not token:
        print("ERROR: no HF token (set HF_TOKEN or run `hf auth login`).", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    current_path = hf_hub_download(repo_id=args.repo, repo_type="dataset", filename="README.md", token=token)
    current = Path(current_path).read_text()
    card = build_card(current)

    if args.dry_run:
        print(card)
        print(f"\n[dry-run] would push {len(card)} chars to {args.repo}", file=sys.stderr)
        return 0

    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Clarify dataset card: behavior, build, schema, lineage, caveats",
    )
    print(f"[hf] dataset card updated -> https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
