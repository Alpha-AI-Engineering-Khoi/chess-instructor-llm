import type { NextConfig } from "next";

// The live coach backend the static site calls at runtime (client-side fetch to
// ${NEXT_PUBLIC_API_BASE}/api/coach). This is the shipped v4 product endpoint:
// Qwen3-32B + the chess-coach-v4 QLoRA adapter, served scale-to-zero via vLLM on
// Modal (workspace chess-instructor-3), through the SAME gated pipeline as the
// local API (Stockfish grounding + verify-and-regenerate faithfulness gate).
//
// Now points at the 4-bit (bitsandbytes NF4) build on a cheaper A100-80GB
// (`chess-coach-v4-4bit`) — verified move-for-move identical to the BF16 build in
// this Maia-less Modal environment, ~37% cheaper per GPU-hour, and more reliable
// (the LoRA is baked into the image instead of fetched from HF at boot).
//
// FALLBACK (one-line revert): switch this constant back to the BF16 endpoint
//   https://chess-instructor-3--chess-coach-v4-vllm-coachv4vllm-fastapi-app.modal.run
// then rebuild (`npm run build`) and re-upload web/out to the chess-coach-studio Space.
//
// Baked in here (not just .env.local, which is gitignored) so the static export and
// any rebuild ship the correct endpoint. Override locally by exporting
// NEXT_PUBLIC_API_BASE before `next dev` / `next build`.
const V4_COACH_ENDPOINT =
  "https://chess-instructor-3--chess-coach-v4-4bit-coachv44bit-fastapi-app.modal.run";

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
      process.env.NEXT_PUBLIC_API_BASE ?? V4_COACH_ENDPOINT,
  },
};

export default nextConfig;
