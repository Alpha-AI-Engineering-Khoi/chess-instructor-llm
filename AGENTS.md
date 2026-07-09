# AGENTS.md

## Cursor Cloud specific instructions

This repo (`chess-instructor-llm`) is a Python (FastAPI) + Next.js chess-coaching
platform ("The Analysis Room") plus offline data/eval pipelines. See `README.md`
for the product overview and the standard commands; the notes below only capture
non-obvious, durable caveats for running it on a Cloud (x86 Linux) VM. The update
script already installs all dependencies — do not re-run installs by hand.

### Services & how to run them
- **FastAPI backend** (`src.api.server:app`, port 8000) — engine grounding
  (Stockfish) + faithfulness verifier + a local LLM coach. The repo uses PEP-420
  namespace packages (no `__init__.py`), so always launch from the repo root:
  `STOCKFISH_PATH=/usr/games/stockfish .venv/bin/python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000`
- **Next.js frontend** (`web/`, port 3000) — `cd web && npm run dev`. It reads the
  backend at `NEXT_PUBLIC_API_BASE` (default `http://127.0.0.1:8000`).
- `run_platform.sh` runs both, but its defaults assume macOS: it points `PY` at
  `~/.venvs/mlx/bin/python` and `COACH_MODEL_PATH` at `models/mlx/chess-coach-v2`
  (absent here). To use it on Cloud: `PY=$PWD/.venv/bin/python STOCKFISH_PATH=/usr/games/stockfish COACH_MODEL_PATH=mlx-community/Qwen3-1.7B-4bit ./run_platform.sh`.
- Python venv lives at `.venv/` (repo root).

### Stockfish (required)
- The engine binary is required; on this VM it is at `/usr/games/stockfish`.
  `config/settings.py` / README default to the macOS Homebrew path, so you MUST
  export `STOCKFISH_PATH=/usr/games/stockfish` (env var is respected by
  `src/engine/stockfish_engine.py`).

### The MLX coach model — CPU is functional but VERY slow (key caveat)
- MLX is Apple-Silicon-first. On x86 Linux install the CPU build: `mlx[cpu]`
  (pulls `mlx-cpu`) — NOT the default `mlx` wheel, which is missing `libmlx.so`.
- `mlx-lm` 0.31.x **breaks on import with `transformers` 5.13** (`AutoTokenizer.register`
  API change). Pin `transformers==5.0.0` (still satisfies `mlx-lm`'s `>=5`).
- The backend loads its model once at startup; with no `COACH_MODEL_PATH` it
  downloads the base `mlx-community/Qwen3-1.7B-4bit` from HuggingFace (~1 GB, needs
  network). It loads fine, but **CPU generation is ~0.1 tok/s**, so a full
  `/api/coach` call takes many minutes to hours. The frontend's built-in
  cold-start budget is only ~4 min (160s/attempt), so the **in-browser live coach
  will always time out to the "coach is offline" panel on CPU** — this is a
  hardware limitation, not a bug (target hardware is Apple Silicon / a GPU
  serving endpoint).
- To demo the coaching UI without waiting on the LLM, use the **Study library**
  in the web app: it renders real, precomputed tuned-model coaching from
  `web/public/library.json` instantly (no backend call). `/api/health`,
  `/api/examples`, and the Stockfish grounding inside `/api/coach` all work.

### Model weights & Maia
- `models/` (coach + Maia nets) is gitignored and empty on a fresh clone; the
  shipped `models/mlx/chess-coach-v2` is not present. Maia (lc0) is optional and
  degrades gracefully — `/api/coach` logs "Maia unavailable" and continues.

### Tests / lint
- Backend tests: `.venv/bin/python -m pytest tests/` (42 pass; no MLX/engine
  needed — they exercise the faithfulness gate with `python-chess`).
- Web lint: `cd web && npm run lint` currently exits non-zero due to
  **pre-existing** `react-hooks` errors in `web/src/components/Showcase.tsx` and
  `Showdown.tsx` — not an environment problem.
- Offline eval/data scripts (`scripts/*.py`, `requirements-train.txt`) need cloud
  API keys / a GPU and are out of scope for running the live product.
