// Typed client for the pre-computed "Model Showdown" slice (web/public/showdown.json).
// Built by scripts/build_showdown.py from the real benchmark artifacts — every
// model's recommended move + objective flags + coaching, per held-out position.

import type { Tier } from "@/lib/api";

export type ModelKind = "ours" | "frontier" | "base" | "open";

export interface ShowdownViolation {
  sentence: string;
  reason: string;
}

export interface ShowdownModel {
  key: string;
  name: string;
  short: string;
  kind: ModelKind;
  rec_san: string | null;
  rec_uci: string | null;
  parseable: boolean;
  sound: boolean;
  tier_appropriate: boolean;
  fabricated: boolean;
  n_violations: number;
  violations: ShowdownViolation[];
  coaching: string;
}

export interface TierTarget {
  uci: string;
  san: string;
  pool_rank: number;
  is_engine_best: boolean;
  policy: number;
  weight: number;
}

export interface ShowdownBeat {
  model: string;
  name: string;
  on: ("tier" | "faithful")[];
}

export interface ShowdownStudentMove {
  san: string | null;
  uci: string | null;
  cp_loss: number | null;
  severity: string | null;
}

export interface ShowdownPosition {
  key: string;
  benchmark: "v2" | "open";
  scenario_id: string;
  fen: string;
  tier: Tier;
  phase: string;
  severity: string;
  side_to_move: "white" | "black";
  student_move: ShowdownStudentMove | null;
  best_san: string | null;
  sound_sans: string[];
  tier_target: TierTarget | null;
  maia_top: { san: string; uci: string; policy: number }[];
  facts: string;
  ours_wins: boolean;
  ours_wins_tier: boolean;
  ours_wins_faithful: boolean;
  beats: ShowdownBeat[];
  n_beats: number;
  models: ShowdownModel[];
}

export interface ShowdownTotals {
  positions: number;
  ours_wins: number;
  ours_wins_tier: number;
  ours_wins_faithful: number;
  by_benchmark: Record<string, {
    positions: number;
    ours_wins: number;
    ours_wins_tier: number;
    ours_wins_faithful: number;
  }>;
}

export interface ShowdownMeta {
  generated_utc: string;
  condition: string;
  frontier: string[];
  tier_weight: Record<string, number>;
  model_meta: Record<string, { name: string; short: string; kind: ModelKind; family: string }>;
  benchmarks: Record<string, string>;
  definitions: Record<string, string>;
  totals: ShowdownTotals;
}

export interface ShowdownDoc {
  meta: ShowdownMeta;
  positions: ShowdownPosition[];
}

/** Fetch the static showdown slice served by Next from public/. */
export async function getShowdown(signal?: AbortSignal): Promise<ShowdownDoc | null> {
  const res = await fetch("/showdown.json", { signal, cache: "no-store" });
  if (!res.ok) return null;
  return res.json();
}
