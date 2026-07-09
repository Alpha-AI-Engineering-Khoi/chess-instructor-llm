import type { NextConfig } from "next";

// The live coach backend the static site calls at runtime (client-side fetch to
// ${NEXT_PUBLIC_API_BASE}/api/coach). This is the shipped v4 product endpoint:
// Qwen3-32B (BF16) + the chess-coach-v4 QLoRA adapter, served scale-to-zero via
// vLLM on Modal (workspace chess-instructor-3), through the SAME gated pipeline as
// the local API (Stockfish grounding + verify-and-regenerate faithfulness gate).
// Baked in here (not just .env.local, which is gitignored) so the static export
// and any rebuild ship the correct endpoint. Override locally by exporting
// NEXT_PUBLIC_API_BASE before `next dev` / `next build`.
const V4_COACH_ENDPOINT =
  "https://chess-instructor-3--chess-coach-v4-vllm-coachv4vllm-fastapi-app.modal.run";

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
