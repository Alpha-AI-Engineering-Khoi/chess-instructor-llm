import type { NextConfig } from "next";

// The live coach backend the static site calls at runtime (client-side fetch to
// ${NEXT_PUBLIC_API_BASE}/api/coach). This is now the v6-dpo2 product endpoint:
// Qwen3-32B + the chess-coach-v6-dpo2 QLoRA adapter (DPO successor to v4), served
// scale-to-zero via vLLM on Modal (workspace chess-instructor-2), through the SAME
// gated pipeline as the local API (Stockfish grounding + verify-and-regenerate
// faithfulness gate).
//
// This is a VERBATIM clone of the v4 MAIA-ENABLED 4-bit serving app
// (`chess-coach-v6dpo2-4bit-maia`): same NF4 base + Maia (lc0 CPU-only + tier nets)
// + Stockfish grounding + greedy-first tier-conditioned decode, with ONLY the LoRA
// adapter swapped v4 -> v6-dpo2. The v6-dpo2 improvement over v4 is small and
// concentrated in the intermediate tier (see the docs); the CORS allowlist (which
// includes this Space origin) is inherited unchanged from src/api/server.py.
//
// FALLBACK (one-line revert): the v4 endpoint is STILL DEPLOYED and alive on
// chess-instructor-3 — switch this constant back to it and rebuild:
//   https://chess-instructor-3--chess-coach-v4-4bit-maia-coachv44bit-b1deed.modal.run
// then rebuild (`npm run build`) and re-upload web/out to the chess-coach-studio Space.
//
// Baked in here (not just .env.local, which is gitignored) so the static export and
// any rebuild ship the correct endpoint. Override locally by exporting
// NEXT_PUBLIC_API_BASE before `next dev` / `next build`.
const V6DPO2_COACH_ENDPOINT =
  "https://chess-instructor-2--chess-coach-v6dpo2-4bit-maia-coachv4-513645.modal.run";

const nextConfig: NextConfig = {
  // Static HTML export: the platform ships as a static site (Hugging Face Static
  // Space). All coaching is a client-side fetch to the Modal endpoint above; the
  // Showcase/Study-library data are static JSON in public/, so no server runtime
  // is needed.
  output: "export",
  // Flat files: routes -> out/<route>.html (e.g. showcase.html). Hugging Face static
  // Spaces serve exact file paths and the root index, but NOT directory indexes
  // (/showcase/) or extensionless clean URLs (/showcase) — so the secondary pages
  // ship as /showcase.html and /showdown.html. The Studio homepage is the root ("/").
  trailingSlash: false,
  // next/image optimization needs a server; disable it for the static export.
  images: { unoptimized: true },
  env: {
    NEXT_PUBLIC_API_BASE:
      process.env.NEXT_PUBLIC_API_BASE ?? V6DPO2_COACH_ENDPOINT,
  },
};

export default nextConfig;
