#!/usr/bin/env python3
"""Serve the **v6-dpo2** chess coach (Qwen3-32B + v6-dpo2 QLoRA LoRA) as a
**4-bit**, snapshot-accelerated, scale-to-zero Modal GPU endpoint using **vLLM**.

This is a VERBATIM clone of ``serve_v4_4bit_modal.py`` (the shipped v4 serving
app) with ONLY the LoRA adapter swapped v4 -> v6-dpo2 and the app name changed.
Same NF4 base, same Maia + Stockfish grounding, same verify-and-regenerate gate,
same CORS allowlist (imported verbatim from ``src.api.server``). The v6-dpo2
improvement over v4 is small and concentrated in the intermediate tier.

This is the cost/cold-start-optimized sibling of ``serve_v4_vllm_modal.py`` (the
shipped BF16-on-H100 endpoint). It serves the *identical* v4 model behavior through
the *identical* gated FastAPI pipeline, but with three deliberate changes:

1. **4-bit weights (bitsandbytes / dynamic NF4).** Instead of loading the canonical
   BF16 ``Qwen/Qwen3-32B`` (~64 GB) we load ``unsloth/Qwen3-32B-unsloth-bnb-4bit`` —
   the *exact* pre-quantized base the v4 QLoRA adapter was TRAINED on — and apply
   the same v4 LoRA on top (``quantization="bitsandbytes"``,
   ``load_format="bitsandbytes"``, ``enable_lora=True``). Serving the adapter over
   its own training-time quantized base is the most faithful reproduction of the
   trained model (the adapter learned to correct exactly these weights), and it is
   a documented vLLM path (unsloth-bnb + LoRA). This is unsloth's *dynamic* 4-bit
   quant, so it is ~38 GiB resident (it protects sensitivity-critical layers) —
   still far below the ~64 GB BF16 build.

2. **Cheaper GPU.** ~38 GiB of dynamic-4-bit weights fit an **A100-80GB** with big
   KV headroom (vs the H100-80GB the BF16 build needs). That is the GPU-cost win:
   ~$2.50/hr vs ~$3.95/hr for the H100 — roughly a **37 % cost reduction** per
   served hour. (A naive all-NF4 quant would be ~18 GiB and fit an A100-40GB for a
   larger ~47 % cut, but risks the tier moat; we prioritized fidelity.)

3. **Modal CPU memory snapshot.** Heavy CPU imports (torch/transformers) run in
   ``@modal.enter(snap=True)`` so Modal checkpoints that state; a cold restore then
   skips torch's ~20k-file import before running ``load`` (``@modal.enter(snap=False)``)
   to build the vLLM engine on the GPU. NOTE: Modal's *experimental GPU* memory
   snapshot (which would restore the whole loaded engine for a ~sub-second cold
   start) was implemented and verified to restore correctly (health in ~0.5 s), but
   its *creation* for this ~38 GiB model froze the container 8-12 min per cold boot
   (HTTP 303 throughout) — enough to break the live demo — so it is left OFF
   (``ENABLE_GPU_SNAPSHOT=False``) pending Modal GA. The reliable CPU snapshot is
   what ships. We still run vLLM in-process, single-process
   (``VLLM_ENABLE_V1_MULTIPROCESSING=0``).

What it reuses (unchanged — the whole point)
--------------------------------------------
The full gated coach pipeline lives in :mod:`src.api.server` and is entirely
backend-agnostic: Stockfish sound-pool + Maia (best-effort) + verified-facts
grounding + a VERIFY-AND-REGENERATE faithfulness gate + a deterministic
engine-derived fallback. We import that FastAPI app verbatim and only replace the
*generation backend* with a vLLM ``LLM.generate`` + a per-request ``LoRARequest``
(the v4 adapter). The tokenizer + chat template are loaded FROM THE ADAPTER REPO
(same as the BF16 serve), so prompts render byte-identically and outputs stay in
the same format. Every route and response shape (``CoachResponse`` with
``meta.attempts`` / ``meta.verified_fallback``) is inherited unchanged.

Deploy (chess-instructor-2 workspace — the funded budget; v4 stays live on
chess-instructor-3 as the fallback)::

    export MODAL_PROFILE=chess-instructor-2
    modal deploy src/serve/serve_v6dpo2_4bit_modal.py

IMPORTANT: memory snapshots only take effect for **deployed** apps (not
``modal serve``/ephemeral). The first few cold starts after deploy CREATE the
snapshot; subsequent cold starts RESTORE from it (that is when you see the
speedup).

Maia note (THE FIX): lc0 + the tier Maia nets ARE now installed on Modal, so the
per-tier human-likelihood signal is REAL — identical to the local pipeline. lc0 is
built CPU-only from source (no CUDA/OpenCL/DX backend compiled), so it uses the
Eigen BLAS backend and never contends with vLLM for the GPU; the three Maia nets
(maia-1100 / 1500 / 1900) are baked into the image at ``/root/models/maia``. The
unchanged :mod:`src.engine.maia_engine` (env ``LC0_BIN`` / ``MAIA_DIR``) then feeds
``_maia_best_effort`` -> ``render_user_prompt`` the tier-specific "Human-likelihood
at this tier (Maia)" block, which is exactly what makes the coach recommend
tier-appropriate moves (Beginner != Advanced). WITHOUT this, that block was empty
for all three tiers and the live coach collapsed to one move per position; WITH it,
live == local. lc0 engines are spawned lazily per net at request time (and warmed
once in ``load``), so no engine subprocess is captured in the CPU snapshot.
Stockfish IS installed (apt) and used for real, opened per-request (snapshot-safe).

No secrets are hardcoded: the HF token (for the base pull / adapter) comes from the
``chess-hf`` Modal secret; Modal auth comes from the ambient profile.
"""
from __future__ import annotations

from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Names / paths
# --------------------------------------------------------------------------- #
#: Maia-enabled 4-bit app for the **v6-dpo2** adapter — the EXACT clone of the
#: shipped v4 serving app (``chess-coach-v4-4bit-maia`` on chess-instructor-3),
#: with ONLY the LoRA adapter swapped from v4 -> v6-dpo2. Deployed on the funded
#: ``chess-instructor-2`` workspace ALONGSIDE the still-live v4 endpoints (which
#: stay up on chess-instructor-3 as the fallback — do NOT tear them down). This is
#: the app the live Space points at once tier differentiation is verified.
APP_NAME: str = "chess-coach-v6dpo2-4bit-maia"
VOLUME_NAME: str = "chess-coach-lora"          # shared volume (holds the v6-dpo2 adapter)
RUN_NAME: str = "chess-coach-v6-dpo2"
VOL_MOUNT: str = "/vol"
#: Volume-resident adapter: chess-coach-lora:/chess-coach-v6-dpo2/adapter (the
#: selected checkpoint — step 200). Present on chess-instructor-2's volume.
ADAPTER_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/adapter"

#: 4-bit base to serve: the EXACT NF4 base the v4 QLoRA adapter was trained on.
#: Serving the adapter over its own training-time quantized base is the most
#: faithful reproduction of the trained model AND is a documented vLLM path
#: (unsloth-bnb checkpoint + LoRA). ~18 GB on disk / in VRAM.
BASE_MODEL: str = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
#: Published HF source for the v6-dpo2 adapter — the PUBLIC fallback used only if
#: the Volume copy is missing. The adapter files live at the REPO ROOT (no
#: subfolder), so HF_ADAPTER_SUBFOLDER is empty.
HF_ADAPTER_REPO: str = "khoilamalphaai/chess-coach-32b-v6-dpo2"
HF_ADAPTER_SUBFOLDER: str = ""

#: Human-readable label surfaced as ``meta.model`` / on /api/health.
MODEL_LABEL: str = "Qwen3-32B + chess-coach-v6-dpo2 QLoRA (vLLM bitsandbytes 4-bit, Modal, Maia)"

#: LoRA rank from the v6-dpo2 adapter_config.json (r=32, same as v4). vLLM needs
#: this at engine init.
LORA_RANK: int = 32

#: GPU choice. NOTE: ``unsloth/Qwen3-32B-unsloth-bnb-4bit`` is unsloth's *dynamic*
#: 4-bit quant — it keeps sensitivity-critical layers in higher precision, so it is
#: ~37 GiB on disk / ~38 GiB resident (NOT the ~18 GiB of a naive all-NF4 quant).
#: That is the price of best quality preservation (it is the EXACT base the adapter
#: was trained on). 38 GiB does not fit a 40 GB card with KV headroom, so we use the
#: fitting cheaper card: A100-80GB (~$2.50/hr) — still ~37 % cheaper than the
#: H100-80GB the BF16 build needs (~$3.95/hr), with ~34 GiB left for the KV cache.
#: (A pure-NF4 inflight path could fit an A100-40GB but risks the tier moat, so we
#: prioritized fidelity — see the deploy report.)
GPU: str = "A100-80GB"
#: Scale-to-zero after 5 min idle. Snapshots make the subsequent cold start cheap.
SCALEDOWN_S: int = 300
#: A single request runs the gate up to COACH_MAX_ATTEMPTS x; keep the ceiling generous.
REQUEST_TIMEOUT_S: int = 900

#: --- Snapshot toggles (easy to flip + redeploy) --------------------------- #
#: CPU memory snapshot: cached torch/transformers imports so a cold start skips
#: torch's ~20k-file import + init. Reliable and safe for a live demo.
ENABLE_SNAPSHOT: bool = True
#: Experimental GPU memory snapshot (whole loaded engine restored). We TRIED this
#: (it does restore correctly — health in ~0.5 s), but Modal's alpha GPU snapshot
#: *creation* for this ~38 GiB dynamic-4-bit model froze the container for 8-12 min
#: on each cold boot (returning HTTP 303 the whole time), which would BREAK the live
#: demo far worse than the current ~3-min cold start. So we ship the reliable CPU
#: snapshot instead and leave this OFF until Modal's GPU snapshot matures / the model
#: is smaller. Flip to True + redeploy to re-test GPU snapshotting.
ENABLE_GPU_SNAPSHOT: bool = False

#: vLLM engine sizing. max_model_len covers the grounded prompt (~1.5-3k tok) + 640
#: gen with headroom; KV cache is sized by gpu_memory_utilization, not this cap.
MAX_MODEL_LEN: int = 6144
#: 4-bit weights are ~18 GB; 0.90 of a 40 GB card leaves ~18 GB for KV + activations.
GPU_MEM_UTIL: float = 0.90
#: Eager mode: bitsandbytes has no CUDA-graph/Marlin fast path and eager avoids
#: capture-time memory spikes; single-stream decode is still fine for a demo.
ENFORCE_EAGER: bool = True

CUDA_TAG: str = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION: str = "3.11"
STOCKFISH_BIN: str = "/usr/games/stockfish"    # debian/ubuntu apt install location

#: Maia (human-likelihood) grounding — the fix. lc0 is built CPU-only from this
#: release inside the image; the tier nets are baked in at ``MAIA_DIR``. Reads by
#: the UNCHANGED ``src.engine.maia_engine`` via the ``LC0_BIN`` / ``MAIA_DIR`` env.
LC0_TAG: str = "v0.31.2"                        # lc0 release built into the image
LC0_BIN: str = "/usr/local/bin/lc0"            # CPU-only build (no GPU backend)
MAIA_DIR: str = "/root/models/maia"            # baked nets: maia-1100/1500/1900.pb.gz
#: lc0's release build hard-codes ``-march=native`` (meson.build), which targets the
#: BUILD host's exact CPU and can SIGILL on a different run host (Modal builds and
#: runs on different machines). We patch that to a portable baseline that every
#: modern GPU host supports and that still provides the f16c/popcnt lc0 expects
#: (x86-64-v3 = AVX2 + F16C + BMI + POPCNT, Haswell 2013+).
LC0_ARCH: str = "x86-64-v3"

#: vLLM (brings a compatible torch/transformers/fastapi/pydantic/uvicorn) + fast HF
#: DL + bitsandbytes (required for NF4 quant at load time).
_ML_PIP: list[str] = ["vllm", "huggingface_hub", "hf_transfer", "bitsandbytes"]
#: Thin HTTP + chess layer the repo's server needs (tail layer; fastapi/uvicorn/
#: pydantic are already pulled by vLLM — listed unpinned so a missing extra can't
#: break the import while keeping vLLM's pins).
_API_PIP: list[str] = ["python-chess", "python-dotenv", "fastapi", "uvicorn", "pydantic"]

if modal.is_local():
    REPO_ROOT = Path(__file__).resolve().parents[2]
else:
    REPO_ROOT = None

#: HF token for the base + adapter pull. Created from .env before deploy:
#:   modal secret create chess-hf HF_TOKEN=... HUGGING_FACE_HUB_TOKEN=...
hf_secret = modal.Secret.from_name("chess-hf")


def _bake_base_model() -> None:
    """Download the 4-bit base into the image's HF cache (baked in => fast, and the
    weights are present offline after a snapshot restore)."""
    import os

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    snapshot_download(BASE_MODEL, token=token)
    print(f"[build] baked {BASE_MODEL} into image HF cache")


image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install(*_ML_PIP)
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            # Run the vLLM engine IN-PROCESS (no EngineCore subprocess) so the GPU
            # memory lives in the snapshotted main process — required for GPU memory
            # snapshotting to capture/restore the loaded weights.
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        }
    )
    # --- tail layers (leave the heavy cached layers above untouched) ---------
    .apt_install("stockfish")
    .pip_install(*_API_PIP)
    .env({"STOCKFISH_PATH": STOCKFISH_BIN})
    # Bake the ~37 GiB base BEFORE copying local source, so editing this serve
    # script only rebuilds the cheap source-copy layers (below) and never
    # re-triggers the large weight download. The small v4 adapter is resolved at
    # RUNTIME (Volume-first, HF fallback) — the proven pattern the BF16 serve uses —
    # which avoids a flaky build-sandbox Hub fetch; the snapshot then captures the
    # loaded LoRA so restores never re-pull it.
    .run_function(_bake_base_model, secrets=[hf_secret])
    # --- Maia grounding: lc0 (CPU-only) built from source ---------------------
    # Added as TAIL layers AFTER the ~37 GiB base bake above, so they never
    # invalidate that cached layer (only these small layers + the source copy
    # rebuild on edit). lc0 is built CPU-only: GPU backends (cudnn/plain_cuda/
    # opencl/dx) are DISABLED, so the only computation backend is BLAS, which lc0
    # always links against Eigen (`dependency('eigen3', fallback: eigen)`), so a
    # ``go nodes 1`` Maia policy pass runs on CPU and never touches the GPU vLLM
    # owns. Deps are minimal: system Eigen + zlib satisfy the two real deps (lc0's
    # protobufs are compiled by its bundled pure-python compile_proto.py — no
    # protoc/libprotobuf needed) and its net format is read from the baked nets.
    .apt_install("build-essential", "zlib1g-dev", "libeigen3-dev")
    .pip_install("meson", "ninja")
    .run_commands(
        f"git clone -b {LC0_TAG} --depth 1 --recurse-submodules --shallow-submodules "
        "https://github.com/LeelaChessZero/lc0.git /opt/lc0",
        # Replace lc0's build-host-specific ``-march=native`` with a portable
        # baseline so the binary never SIGILLs on a different Modal run host.
        f"sed -i 's/-march=native/-march={LC0_ARCH}/g' /opt/lc0/meson.build",
        "cd /opt/lc0 && meson setup build --buildtype=release "
        "-Dgtest=false -Dispc=false -Ddx=false "
        "-Dcudnn=false -Dplain_cuda=false -Dopencl=false "
        "-Dtensorflow=false -Donednn=false -Ddnnl=false "
        "-Dmkl=false -Daccelerate=false -Dopenblas=false -Dblas=true "
        "&& ninja -C build "
        f"&& install -Dm755 build/lc0 {LC0_BIN}",
        # Smoke-log the build + prove it runs on THIS arch (never fails the layer).
        f"{LC0_BIN} --version || true",
    )
    # Point the UNCHANGED maia_engine at the built binary + baked nets. Set at the
    # image level so it is present for every process before any import.
    .env({"LC0_BIN": LC0_BIN, "MAIA_DIR": MAIA_DIR})
)
if modal.is_local():
    # Copy ONLY the packages the server imports (config / src / prompts) — keep the
    # image lean (no data/, models/, node_modules, .git).
    image = (
        image
        # Maia nets (~3.8 MiB total) baked in FIRST so editing this serve script
        # (which lives under src/) never invalidates this tiny layer.
        .add_local_dir((REPO_ROOT / "models" / "maia").as_posix(), "/root/models/maia", copy=True)
        .add_local_dir((REPO_ROOT / "config").as_posix(), "/root/config", copy=True)
        .add_local_dir((REPO_ROOT / "src").as_posix(), "/root/src", copy=True)
        .add_local_dir((REPO_ROOT / "prompts").as_posix(), "/root/prompts", copy=True)
    )

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# Generation backend: a drop-in replacement for src.api.server.Coach that uses
# vLLM (NF4 base + per-request LoRA) instead of mlx_lm. Same .run(system, user) API
# and the SAME rendering/sampling as the BF16 serve, so outputs match in format.
# --------------------------------------------------------------------------- #
class _VLLMCoach:
    """Turns (system, user) into coaching text via Qwen3-32B (NF4) + the v4 LoRA.

    Mirrors ``src.api.server.Coach.run`` and the BF16 serve (chat template with
    ``enable_thinking=False``, ``<think>`` stripped, generation lock-serialized on
    the single GPU) with ONE deliberate change for tier differentiation:

    **Greedy-first decoding.** The tier-appropriate move is produced by the model's
    per-tier conditioning (the Maia "human-likelihood at this tier" block +
    tier/ply-cap in the prompt). That conditioning is a *bias*, and at the shipped
    sampling temperature (0.7) it is drowned by sampling noise — so all three tiers
    collapse onto the model's single dominant move (this is exactly why the local
    tier analysis, ``scripts/divergence_analysis.py``, decodes GREEDY: "so tier
    differences reflect genuine tier-conditioning, not sampling noise"). To make the
    LIVE coach reproduce the LOCAL/showcase differentiation, the FIRST gate attempt
    is decoded **greedily** (argmax, temperature 0) — deterministic and maximally
    tier-conditioned, matching how the showcase was generated. If that first draft
    fails the faithfulness verifier, subsequent gate attempts fall back to the
    original stochastic sampling (temp ``COACH_GEN_TEMP``, default 0.7) so the
    VERIFY-AND-REGENERATE loop still has genuinely different drafts to try. Net:
    tier differentiation (greedy first) AND faithfulness recovery (sampled retries).
    Overridable via env: ``COACH_FIRST_ATTEMPT_GREEDY=0`` (pure sampling, old
    behavior) / ``COACH_GEN_TEMP`` (retry temperature).
    """

    MAX_TOKENS: int = 640
    TEMP: float = 0.7
    TOP_P: float = 0.8
    TOP_K: int = 20
    REPETITION_PENALTY: float = 1.15

    def __init__(self, llm, tokenizer, lora_request, strip_think) -> None:
        import os
        import threading

        from vllm import SamplingParams

        self.llm = llm
        self.tok = tokenizer
        self.lora_request = lora_request
        self._strip_think = strip_think
        self._SamplingParams = SamplingParams
        self._lock = threading.Lock()

        # Retry (sampled) temperature + whether the first gate attempt is greedy.
        self._temp = float(os.environ.get("COACH_GEN_TEMP", str(self.TEMP)))
        self._first_greedy = (
            os.environ.get("COACH_FIRST_ATTEMPT_GREEDY", "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        # Per-(system,user) attempt counter so attempt 1 of each gate loop is greedy
        # and its faithfulness re-samples are stochastic. run_gate calls run() with a
        # fixed (system,user) per position/tier, sequentially; the lock serializes it.
        self._last_key = None
        self._attempt = 0

    def _render(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:  # tokenizer without the enable_thinking kwarg
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def run(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        prompt = self._render(system, user)
        with self._lock:
            # Track which attempt this is for the current (system,user): attempt 1
            # is greedy (deterministic, tier-conditioned); later gate re-samples are
            # stochastic so the faithfulness loop has new drafts to try.
            key = (system, user)
            if key != self._last_key:
                self._last_key = key
                self._attempt = 0
            self._attempt += 1
            greedy = self._first_greedy and self._attempt == 1

            if greedy:
                # vLLM treats temperature==0 as argmax/greedy (top_p/top_k ignored).
                params = self._SamplingParams(
                    temperature=0.0,
                    repetition_penalty=self.REPETITION_PENALTY,
                    max_tokens=max_tokens,
                )
            else:
                params = self._SamplingParams(
                    temperature=self._temp,
                    top_p=self.TOP_P,
                    top_k=self.TOP_K,
                    repetition_penalty=self.REPETITION_PENALTY,
                    max_tokens=max_tokens,
                )
            outputs = self.llm.generate(
                [prompt], params, lora_request=self.lora_request, use_tqdm=False,
            )
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return self._strip_think(text)


def _resolve_adapter_dir() -> str:
    """Return a local path to the v6-dpo2 adapter — Volume copy preferred (the
    selected step-200 checkpoint at chess-coach-lora:/chess-coach-v6-dpo2/adapter),
    public HF repo otherwise (offline-friendly after a snapshot restore)."""
    import os

    cfg = os.path.join(ADAPTER_DIR, "adapter_config.json")
    weights = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
    if os.path.isfile(cfg) and os.path.isfile(weights):
        print(f"[serve] using v6-dpo2 LoRA from Volume: {ADAPTER_DIR}")
        return ADAPTER_DIR

    # Fallback: pull the PUBLIC HF repo. The v6-dpo2 adapter is at the repo ROOT
    # (no subfolder), so download everything and use the snapshot dir directly.
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if HF_ADAPTER_SUBFOLDER:
        local = snapshot_download(
            HF_ADAPTER_REPO, allow_patterns=[f"{HF_ADAPTER_SUBFOLDER}/*"], token=token,
        )
        resolved = os.path.join(local, HF_ADAPTER_SUBFOLDER)
    else:
        resolved = snapshot_download(HF_ADAPTER_REPO, token=token)
    print(f"[serve] using v6-dpo2 LoRA from HF repo {HF_ADAPTER_REPO}: {resolved}")
    return resolved


# Snapshot config is assembled conditionally so the flags at the top of the file
# can toggle the (alpha) GPU snapshot without editing the decorator by hand.
_CLS_KWARGS: dict = dict(
    image=image,
    gpu=GPU,
    volumes={VOL_MOUNT: volume},
    secrets=[hf_secret],
    scaledown_window=SCALEDOWN_S,
    timeout=REQUEST_TIMEOUT_S,
    # First boot loads the model AND creates the snapshot — give it generous room.
    startup_timeout=20 * 60,
    max_containers=1,          # one GPU is plenty for a demo; caps idle/burst cost
)
if ENABLE_SNAPSHOT:
    _CLS_KWARGS["enable_memory_snapshot"] = True
    if ENABLE_GPU_SNAPSHOT:
        _CLS_KWARGS["experimental_options"] = {"enable_gpu_snapshot": True}


@app.cls(**_CLS_KWARGS)
@modal.concurrent(max_inputs=8)  # generation is lock-serialized; lets health/preflight through
class CoachV44bit:
    @modal.enter(snap=True)
    def _warm_imports(self) -> None:
        """CPU-only heavy imports, captured by the CPU memory snapshot.

        Runs BEFORE the (CPU) snapshot. Importing torch here pays the ~20k-file
        import cost ONCE at build/snapshot time; a cold restore then skips it. We do
        NOT touch CUDA here (all GPU work is in ``load`` / snap=False), which keeps
        the CPU snapshot valid and the restore reliable.
        """
        import huggingface_hub  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

    @modal.enter(snap=False)
    def load(self) -> None:
        """Load the NF4 base + v4 LoRA to the GPU and warm the engine.

        Runs AFTER a snapshot restore (snap=False), so the (freshly restored) process
        already has torch/transformers imported; here we do the GPU work: construct
        the vLLM engine (bnb 4-bit), attach the v4 LoRA, and warm it. The ~37 GiB base
        is baked into the image cache, so this is an offline, network-free load.
        """
        import os
        import shutil
        import sys
        import time

        # The repo uses namespace packages that assume the repo root is importable.
        sys.path.insert(0, "/root")

        # Set the server's env-driven config BEFORE importing it (module-level consts).
        os.environ["STOCKFISH_PATH"] = shutil.which("stockfish") or STOCKFISH_BIN
        # Maia: point the UNCHANGED src.engine.maia_engine at the CPU-only lc0 build
        # + baked nets. Must be set BEFORE ``from src.api import server`` below,
        # because maia_engine reads LC0_BIN / MAIA_DIR into module-level constants
        # at import time. (Also set at the image level, so this is belt-and-braces.)
        os.environ["LC0_BIN"] = shutil.which("lc0") or LC0_BIN
        os.environ["MAIA_DIR"] = MAIA_DIR
        os.environ["COACH_MODEL_PATH"] = MODEL_LABEL          # -> meta.model + tuned=True
        os.environ["COACH_ADAPTER_PATH"] = ADAPTER_DIR        # -> _is_tuned() True
        # Keep the faithfulness gate ON; 4 attempts gives timeout-safety margin.
        os.environ.setdefault("COACH_MAX_ATTEMPTS", "4")
        os.environ.setdefault("COACH_FAITHFULNESS_GATE", "1")

        volume.reload()  # make any committed adapter visible

        # Resolve the adapter BEFORE going offline for the (baked) base load.
        adapter_dir = _resolve_adapter_dir()

        # Base weights + adapter are baked into the image cache — load them offline
        # so neither a cold start nor a snapshot restore blocks on the Hub.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        t0 = time.time()
        # Tokenizer + chat template come FROM THE ADAPTER REPO (identical to the
        # BF16 serve) so prompt rendering is byte-identical.
        tok = AutoTokenizer.from_pretrained(adapter_dir)
        llm = LLM(
            model=BASE_MODEL,
            quantization="bitsandbytes",   # NF4 (pre-quantized unsloth base)
            load_format="bitsandbytes",    # required paired with bnb quantization
            dtype="bfloat16",              # compute dtype
            enable_lora=True,
            max_lora_rank=LORA_RANK,
            max_loras=1,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTIL,
            enforce_eager=ENFORCE_EAGER,
            tensor_parallel_size=1,
            disable_log_stats=True,
        )
        lora_request = LoRARequest("chess-coach-v6-dpo2", 1, adapter_dir)

        # Warm the engine WITH the LoRA so the adapter is resident and every code
        # path (CUDA init, eager execution) is exercised before the snapshot.
        try:
            llm.generate(
                ["<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"],
                SamplingParams(temperature=0.0, max_tokens=8),
                lora_request=lora_request,
                use_tqdm=False,
            )
        except Exception as exc:  # noqa: BLE001 - warmup must not abort startup
            print(f"[serve] warmup generate skipped: {exc}")

        print(f"[serve] loaded {BASE_MODEL} (NF4 4-bit) + v6-dpo2 LoRA via vLLM "
              f"in {time.time() - t0:.1f}s")

        # Import the repo's FastAPI app and swap ONLY the generation backend. The
        # app's lifespan calls ``Coach(COACH_MODEL_PATH, COACH_ADAPTER_PATH)`` — we
        # make that return our already-loaded vLLM coach (no mlx_lm import, no reload).
        from src.api import server

        coach = _VLLMCoach(llm, tok, lora_request, server._strip_think)
        server.Coach = lambda *a, **k: coach
        self._app = server.app

        # Warm + self-test Maia: spawns the three tier lc0 engines once (so the
        # first live request differentiates without paying the net-load latency)
        # and makes a broken build/nets LOUD in the logs. It never aborts startup:
        # if it fails, the coach simply degrades to ``maia: []`` (the old behavior).
        try:
            import chess as _chess

            from src.engine.maia_engine import human_moves as _human_moves

            t1 = time.time()
            startpos = _chess.Board().fen()
            tops = {}
            for tier in ("beginner", "intermediate", "advanced"):
                mv = _human_moves(startpos, tier, top_k=1)["moves"]
                tops[tier] = mv[0]["san"] if mv else "-"
            print(
                f"[serve] Maia ({LC0_TAG}, CPU/Eigen) ready in {time.time() - t1:.1f}s — "
                f"startpos top human move by tier: "
                f"B={tops['beginner']} I={tops['intermediate']} A={tops['advanced']}"
            )
        except Exception as exc:  # noqa: BLE001 - Maia is a signal, not required
            print(f"[serve] WARNING: Maia self-test FAILED ({exc!r}); the coach will "
                  f"degrade to maia:[] and tiers may not differentiate.")

        print("[serve] FastAPI coach app ready (v6-dpo2 gated pipeline, vLLM NF4 backend + Maia)")

    @modal.asgi_app()
    def fastapi_app(self):
        return self._app
