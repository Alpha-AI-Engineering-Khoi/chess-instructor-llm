// Typed client for the multi-model "Showcase" slice.
//
// SOURCE OF TRUTH: web/public/showcase.json (array, schema in ShowcaseContract*)
// is produced by a separate worker. Until it lands, this module transparently
// falls back to the already-shipped web/public/showdown.json and adapts it into
// the same view model, so the Showcase UI works today and upgrades in place the
// moment showcase.json appears. Nothing here fabricates numbers: every value is
// read from one of those two JSON files (or derived and clearly labelled as such).

import { Chess } from "chess.js";
import type { Tier } from "@/lib/api";
import {
  getShowdown,
  type ShowdownDoc,
  type ShowdownModel,
  type ShowdownPosition,
} from "@/lib/showdown";

export type ShowcaseTier = Tier; // "beginner" | "intermediate" | "advanced"
export const TIERS: ShowcaseTier[] = ["beginner", "intermediate", "advanced"];

/**
 * Fallback OURS version label, used ONLY when the loaded data carries no OURS
 * model name to parse. The real label is derived from the data at load time
 * (see deriveOursLabel + meta.ours), so switching the pipeline from v2 → v4
 * (1.7B → 32B) updates every "OURS · vN" label automatically — nothing here is
 * hardcoded to a specific size or version.
 */
export const MODEL_VERSION = "v2";

export type DataSource = "showcase" | "interim" | "showdown";
export type Split = "train" | "test";
export type ModelKind = "ours" | "frontier" | "base" | "open";

/* ------------------------------------------------------------------ */
/* CONTRACT — exact shape of web/public/showcase.json (array)          */
/* ------------------------------------------------------------------ */

/**
 * One model's answer at one tier, as written by the showcase worker.
 *
 * Faithfulness is now a TWO-LAYER, post-GATE record (see data/showcase/gate_all_report.md):
 *  - `coaching`         — the GATED text that actually ships (the default to display).
 *  - `fabricated`       — the POST-gate deterministic verdict (0 / false for every cell
 *                         now — the fairness guarantee, not a claim of full truth).
 *  - `raw_coaching`     — the model's ORIGINAL pre-gate draft (a model-capacity artifact).
 *  - `raw_fabricated`   — whether that raw draft stated a false board fact (pre-gate).
 *  - `gate_attempts`    — how many drafts the model needed before one passed the gate.
 *  - `verified_fallback`— true when no draft passed and a deterministic engine-derived
 *                         explanation (true by construction) was substituted.
 *
 * The objective flags (`sound`/`tier_fit`/`fabricated`) may be null when no objective row
 * exists for a cell; such a cell is treated as NOT evaluated (see cellFromContract).
 */
export interface ShowcaseCell {
  move: string | null; // SAN of the recommended move
  move_uci?: string | null; // UCI written by the worker (preferred over re-deriving)
  sound: boolean | null;
  tier_fit: boolean | null;
  fabricated: boolean | null; // POST-gate deterministic verdict (false for all shipped cells)
  coaching: string; // GATED text — the default to display
  raw_coaching?: string | null; // original pre-gate model draft (capacity artifact)
  raw_fabricated?: boolean | null; // did the raw draft state a false board fact?
  gate_attempts?: number | null; // drafts sampled before one passed the gate
  verified_fallback?: boolean | null; // engine-derived explanation substituted
  council_move: number | null; // blinded council move grade
  council_instr: number | null; // blinded council instructiveness grade
  /**
   * Optional fabrication receipts (the exact false sentence + why). The full
   * showcase.json may omit these; the interim OURS re-score carries them so the
   * verifier's evidence stays visible, exactly like the showdown source.
   */
  n_violations?: number;
  violations?: { sentence: string; reason: string }[];
}

export interface ShowcaseContractModel {
  name: string;
  family: string;
  /** True for open-weight / self-hosted models (incl. OURS + BASE); false for API frontier. */
  local: boolean;
  byTier: Partial<Record<ShowcaseTier, ShowcaseCell | null>>;
}

/**
 * Pre-aggregated per-model rollup the worker attaches to a position for OURS and
 * for the best rival (`ours_summary` / `best_other_detail`). Every field is
 * optional so an older/leaner artifact still parses; the UI derives the same
 * numbers from the cells when these are absent.
 */
export interface ShowcaseModelSummary {
  key?: string | null;
  name?: string | null;
  family?: string | null;
  council_move?: number | null;
  council_instr?: number | null;
  sound_rate?: number | null;
  tier_fit_rate?: number | null;
  fabricated_rate?: number | null;
}

export interface ShowcaseContractPosition {
  id: string;
  fen: string;
  phase: string;
  split: Split;
  tier_targets: Partial<Record<ShowcaseTier, string | null>>;
  models: ShowcaseContractModel[];
  ours_wins: boolean;
  ours_loses: boolean;
  shine: boolean;
  /**
   * OURS recommends a different, level-appropriate move across the three tiers.
   * Optional: when the worker omits it we derive the same signal from the cells,
   * so the UI works before the refined showcase.json lands (see buildFromContract).
   */
  ours_tier_differentiates?: boolean;
  /**
   * The strongest non-OURS model at this position, named by the worker (matched
   * on model key / name / short). Optional: when absent we derive it from the
   * council grades + objective flags. Powers the OURS-vs-best comparison.
   */
  best_other?: string | null;
  /** Pre-aggregated rollups the worker attaches for OURS and the best rival. */
  ours_summary?: ShowcaseModelSummary | null;
  best_other_detail?: ShowcaseModelSummary | null;
  /** OURS recommends a move MIS-matched to the tier target (honest counter-signal). */
  ours_misdirected?: boolean;
  /** How many DISTINCT moves OURS gives across the three tiers (>=2 == adapts). */
  ours_distinct_moves?: number | null;
  /** How many distinct SOUND moves OURS gives across the tiers. */
  ours_distinct_sound_moves?: number | null;
  /** True when OURS was scored at all three tiers here. */
  ours_full_3tier_coverage?: boolean;
  /**
   * The curated "proof" flag: OURS adapts by level AND diverges from the best
   * rival — the headline library subset. Optional; derived when absent.
   */
  focus?: boolean;
  /** Which pool the split was drawn from (worker bookkeeping; informational). */
  split_source?: string | null;
  /**
   * Optional held-out context carried by the interim source so its view matches
   * the showdown one (student's mistake arrow, severity + benchmark chips). The
   * full showcase.json may omit them — they degrade to null.
   */
  student_move?: { san: string | null; uci: string | null; severity: string | null } | null;
  severity?: string | null;
  benchmark?: "v2" | "open" | null;
}

export type ShowcaseContract = ShowcaseContractPosition[];

/* ------------------------------------------------------------------ */
/* VIEW MODEL — what the components actually consume                    */
/* ------------------------------------------------------------------ */

export interface ViolationView {
  sentence: string;
  reason: string;
}

export interface ViewCell {
  move: string | null; // SAN
  moveUci: string | null; // for the board arrow
  sound: boolean;
  tierFit: boolean;
  fabricated: boolean; // POST-gate deterministic verdict (false for every shipped cell)
  nViolations: number;
  violations: ViolationView[];
  coaching: string; // GATED text — what actually ships / is displayed
  rawCoaching: string | null; // the model's original pre-gate draft (capacity artifact)
  rawFabricated: boolean; // did that raw draft state a false board fact (pre-gate)?
  gateAttempts: number | null; // drafts the model needed before one passed the gate
  verifiedFallback: boolean; // engine-derived explanation substituted (no draft passed)
  councilMove: number | null;
  councilInstr: number | null;
  /**
   * False when this tier has no MEASURED data for this model: either the cell is
   * absent (degraded showdown source) OR its objective flags are null/absent. A
   * non-evaluated cell must never render as a measured green "sound / faithful".
   */
  evaluated: boolean;
}

export interface ViewModel {
  key: string;
  name: string;
  short: string;
  kind: ModelKind;
  family: string;
  local: boolean;
  byTier: Record<ShowcaseTier, ViewCell | null>;
}

export interface TierTargetView {
  san: string | null;
  uci: string | null;
}

export interface ViewPosition {
  id: string;
  fen: string;
  phase: string;
  split: Split;
  sideToMove: "white" | "black";
  /** From showdown (mistake severity); null in a pure showcase source. */
  severity: string | null;
  /** Which benchmark field the row came from (showdown source only). */
  benchmark: "v2" | "open" | null;
  studentMove: { san: string | null; uci: string | null; severity: string | null } | null;
  tierTargets: Record<ShowcaseTier, TierTargetView | null>;
  /** Which tiers actually have model data (all three for a real showcase row). */
  tierEvaluated: Record<ShowcaseTier, boolean>;
  models: ViewModel[];
  oursWins: boolean;
  oursLoses: boolean;
  shine: boolean;
  /** OURS gives a different, level-appropriate move across all three tiers. */
  oursTierDifferentiates: boolean;
  /** OURS's move mis-matches the tier target somewhere (honest counter-signal). */
  oursMisdirected: boolean;
  /** Distinct OURS moves across the three tiers (>=2 == genuinely adapts). */
  oursDistinctMoves: number;
  /** Key of the strongest non-OURS model here (from contract or derived); null if none scored. */
  bestOtherKey: string | null;
  /** Key of the strongest FRONTIER rival (GPT-5.5 / Claude / Gemini); null if none scored. */
  bestFrontierKey: string | null;
  /** OURS's recommended move diverges from the best frontier model at some tier. */
  oursDiffersFromBestFrontier: boolean;
  /**
   * The curated proof subset: OURS adapts by level AND diverges from the best
   * frontier rival — the honest "our model does something the frontier doesn't"
   * library. Read from the worker's `focus` when present, else derived.
   */
  isProof: boolean;
  /** Pre-aggregated rollups from the worker (informational; may be null). */
  oursSummary: ShowcaseModelSummary | null;
  bestOtherDetail: ShowcaseModelSummary | null;
  source: DataSource;
}

export interface ShowcaseTotals {
  positions: number;
  shine: number;
  proof: number;
  differentiates: number;
  oursWins: number;
  oursLoses: number;
  train: number;
  test: number;
}

/**
 * The OURS identity, parsed from the loaded data (never hardcoded to a size /
 * version). `name` is the full model name as written by the worker; `version`
 * and `size` are pulled out of it when present (e.g. "OURS-v2 (1.7B tuned)" ->
 * version "v2", size "1.7B"). Everything that used to hardcode "v2" / "1.7B"
 * reads these instead, so a v4 / 32B artifact relabels the whole app at once.
 */
export interface OursLabel {
  name: string;
  short: string;
  version: string | null;
  size: string | null;
  /** "OURS · v4" style chip text. */
  badge: string;
}

export interface ShowcaseMeta {
  source: DataSource;
  generatedUtc: string | null;
  modelVersion: string;
  /** OURS identity parsed from the data (name / version / size / chip). */
  ours: OursLabel;
  /** Total distinct models seen anywhere in the doc (e.g. 14 with the full field). */
  modelCount: number;
  /** Observed maximum council grade across the doc (for meter scaling); >= 1. */
  councilScale: number;
  hasCouncil: boolean;
  hasTrain: boolean;
  /** True when every position carries all three tiers (real showcase behaviour). */
  perTierComplete: boolean;
  totals: ShowcaseTotals;
  notes: Record<string, string>;
}

export interface ShowcaseView {
  meta: ShowcaseMeta;
  positions: ViewPosition[];
}

/* ------------------------------------------------------------------ */
/* Ordering + naming helpers                                           */
/* ------------------------------------------------------------------ */

const KIND_RANK: Record<ModelKind, number> = { ours: 0, frontier: 1, open: 2, base: 3 };

export function orderModels<T extends { kind: ModelKind; short: string }>(models: T[]): T[] {
  return [...models].sort(
    (a, b) => KIND_RANK[a.kind] - KIND_RANK[b.kind] || a.short.localeCompare(b.short),
  );
}

const KIND_BY_FAMILY: Record<string, ModelKind> = {
  ours: "ours",
  base: "base",
  frontier: "frontier",
  open: "open",
};

function deriveKind(family: string, local: boolean, name: string): ModelKind {
  const f = (family ?? "").toLowerCase();
  if (KIND_BY_FAMILY[f]) return KIND_BY_FAMILY[f];
  if (/chess-coach|(^|\b)ours\b/i.test(name)) return "ours";
  if (/untuned|(^|\b)base\b/i.test(name)) return "base";
  return local ? "open" : "frontier";
}

function deriveKey(name: string, kind: ModelKind): string {
  if (kind === "ours") return "ours";
  if (kind === "base") return "base";
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || kind;
}

/** Strip a trailing parenthetical ("Mistral-Large-3 (675B)" -> "Mistral-Large-3"). */
function shortName(name: string): string {
  return name.replace(/\s*\([^)]*\)\s*/g, " ").trim() || name;
}

/**
 * Parse the OURS identity out of the loaded model name — the single place the
 * app learns its version/size. Handles "OURS-v2 (1.7B tuned)",
 * "OURS · chess-coach-v4 (32B)", a bare "chess-coach-v2", etc. Never hardcodes a
 * size or version; falls back to {@link MODEL_VERSION} only when the name is empty.
 */
export function deriveOursLabel(name: string | null | undefined): OursLabel {
  const raw = (name ?? "").trim();
  const version = raw.match(/v(\d+(?:\.\d+)?)/i)?.[0]?.toLowerCase() ?? null;
  const size = raw.match(/(\d+(?:\.\d+)?\s?B)\b/i)?.[1]?.replace(/\s+/g, "") ?? null;
  const badge = `OURS · ${version ?? MODEL_VERSION}`;
  return { name: raw || "OURS", short: "OURS", version, size, badge };
}

function sideToMove(fen: string): "white" | "black" {
  return fen.split(" ")[1] === "b" ? "black" : "white";
}

/**
 * A short PRINCIPLE TAG for the move hero, assembled ONLY from fields already in
 * the data — never invented. Prefers explicit `concepts_used`; otherwise a short
 * slice of the `takeaway`. Returns null when neither is present. This is the "tag"
 * half of the reframed hero ("Nf6 — develop a piece, contest e4"): the model's job
 * is the tier-appropriate MOVE, and this is the one-line reason attached to it —
 * not a paragraph of prose.
 */
export function principleTag(
  concepts: string[] | null | undefined,
  takeaway: string | null | undefined,
): string | null {
  const cs = (concepts ?? []).map((c) => c.trim()).filter(Boolean);
  if (cs.length) return cs.slice(0, 3).join(" · ");
  const t = (takeaway ?? "").replace(/\s+/g, " ").trim();
  if (!t) return null;
  const cap = 120;
  if (t.length <= cap) return t.replace(/[.\s]+$/, "");
  return t.slice(0, cap).replace(/\s+\S*$/, "") + "…";
}

/** One reusable SAN->UCI resolver per position (move + undo, no re-parse). */
function makeUci(fen: string): (san: string | null | undefined) => string | null {
  let game: Chess | null = null;
  try {
    game = new Chess(fen);
  } catch {
    game = null;
  }
  return (san) => {
    if (!san || !game) return null;
    try {
      const m = game.move(san);
      if (!m) return null;
      const uci = m.from + m.to + (m.promotion ?? "");
      game.undo();
      return uci;
    } catch {
      return null;
    }
  };
}

function emptyTierRecord<T>(value: T): Record<ShowcaseTier, T> {
  return { beginner: value, intermediate: value, advanced: value };
}

/* ------------------------------------------------------------------ */
/* Outcome descriptions (shared by cards + explorer, computed live)    */
/* ------------------------------------------------------------------ */

export interface OursOutcome {
  wins: boolean;
  loses: boolean;
  bullets: string[];
}

/**
 * Describe how OURS did against the field at a specific tier, straight from the
 * cells. Used for the "why it's a win" copy and the honest "where OURS loses"
 * section. Only reports what the cells actually say.
 *
 * Item 3 (no double standard): the comparison is against the FRONTIER references
 * (GPT-5.5 / Claude / Gemini) ONLY — exactly the rule behind the headline
 * `ours_wins` / `ours_loses` counts (see data/showcase/pipeline/assemble.py
 * `derive_wins`). That deliberately EXCLUDES OURS's own BASE baseline (beating our
 * untuned baseline is not a "win" over the field) and the wider open field, so the
 * "OURS wins here" strip can never credit a win the headline definition doesn't.
 * A per-tier win here therefore always implies the position-level `oursWins`.
 */
export function describeOursOutcome(position: ViewPosition, tier: ShowcaseTier): OursOutcome {
  const ours = position.models.find((m) => m.kind === "ours")?.byTier[tier] ?? null;
  const frontier = position.models
    .filter((m) => m.kind === "frontier")
    .map((m) => ({ name: m.short, cell: m.byTier[tier] }))
    .filter((x): x is { name: string; cell: ViewCell } => Boolean(x.cell?.evaluated));

  if (!ours || !ours.evaluated) return { wins: false, loses: false, bullets: [] };

  const oursTierFit = ours.sound && ours.tierFit;
  const oursFaithful = ours.sound && !ours.fabricated;
  const oursClean = oursTierFit && !ours.fabricated;
  const bullets: string[] = [];

  // Wins: OURS is the sound tier move where a frontier model isn't; or OURS is
  // faithful where a frontier model fabricates. (Post-gate no cell fabricates, so
  // in shipped data this reduces to the sound-tier-move comparison — as it should.)
  const tierBeaten = frontier
    .filter((o) => oursTierFit && !(o.cell.sound && o.cell.tierFit))
    .map((o) => o.name);
  const faithBeaten = frontier.filter((o) => oursFaithful && o.cell.fabricated).map((o) => o.name);
  if (tierBeaten.length) bullets.push(`Sound tier move where ${fmtList(tierBeaten)} ${wasWere(tierBeaten)}n’t`);
  if (faithBeaten.length) bullets.push(`Faithful where ${fmtList(faithBeaten)} fabricated a board fact`);

  // Loses: a frontier model is the sound tier move + faithful where OURS isn't.
  const cleanRivals = frontier
    .filter((o) => o.cell.sound && o.cell.tierFit && !o.cell.fabricated)
    .map((o) => o.name);
  const loses = !oursClean && cleanRivals.length > 0;
  const loseBullets: string[] = [];
  if (loses) {
    const why = ours.fabricated ? "fabricated a fact" : !ours.sound ? "played an unsound move" : "missed the tier move";
    loseBullets.push(`OURS ${why}; ${fmtList(cleanRivals)} ${wasWere(cleanRivals)} the sound tier move and faithful`);
  }

  const wins = tierBeaten.length > 0 || faithBeaten.length > 0;
  return { wins, loses, bullets: wins ? bullets : loseBullets };
}

function fmtList(names: string[]): string {
  const shown = names.slice(0, 3);
  const extra = names.length - shown.length;
  const base = shown.join(", ");
  return extra > 0 ? `${base} +${extra}` : base;
}

function wasWere(names: string[]): string {
  return names.length > 1 ? "were" : "was";
}

/**
 * The most informative default tier for a position. Prefer a tier where a
 * non-OURS rival is actually scored (so the head-to-head is real); fall back to
 * the first tier OURS was scored at. In the showdown source (one tier per
 * position) this is unchanged; in the interim source (OURS at all three tiers,
 * rivals only at their benchmarked tier) it lands on the tier with the rival.
 */
export function primaryTier(position: ViewPosition): ShowcaseTier {
  return (
    TIERS.find((t) => position.models.some((m) => m.kind !== "ours" && m.byTier[t]?.evaluated)) ??
    TIERS.find((t) => position.tierEvaluated[t]) ??
    "beginner"
  );
}

/**
 * The tier a compact card should describe so its blurb matches the position's
 * headline status (item 3/4 consistency): if the position is a frontier win,
 * pick a tier where OURS actually wins; if it's a loss, a tier where it loses;
 * otherwise the primary tier. Always a real, honest per-tier outcome — the card
 * labels which tier it is showing.
 */
export function representativeTier(position: ViewPosition): ShowcaseTier {
  if (position.oursWins) {
    const t = TIERS.find((tier) => describeOursOutcome(position, tier).wins);
    if (t) return t;
  }
  if (position.oursLoses) {
    const t = TIERS.find((tier) => describeOursOutcome(position, tier).loses);
    if (t) return t;
  }
  return primaryTier(position);
}

/* ------------------------------------------------------------------ */
/* OURS-vs-best comparison + tier differentiation (derived, honest)    */
/* ------------------------------------------------------------------ */

/**
 * One comparable quality score for a cell — higher is better. Used both to pick
 * the best rival at a position and to call OURS-vs-best at a tier. The objective
 * flags always count; the blinded council grades count when present (they only
 * arrive with showcase.json), so the ranking sharpens automatically then.
 */
function cellQuality(cell: ViewCell): number {
  let q = 0;
  if (cell.tierFit) q += 2;
  if (cell.sound) q += 1;
  if (cell.fabricated) q -= 2;
  if (cell.councilMove != null) q += cell.councilMove;
  if (cell.councilInstr != null) q += cell.councilInstr;
  return q;
}

/**
 * Strongest RIVAL at a position, by mean cell quality across the tiers it was
 * actually scored at. Ties break toward the more serious kind (frontier > open),
 * then name. Returns null if no rival was scored.
 *
 * Item 6: BASE (OURS's own untuned baseline) is excluded — it is a baseline, not a
 * rival, so it must never be surfaced as the "best other model". This mirrors the
 * worker's `best_other`, which is also chosen over the non-BASE field.
 */
function deriveBestOtherKey(models: ViewModel[]): string | null {
  let bestKey: string | null = null;
  let bestScore = -Infinity;
  let bestRank = Infinity;
  let bestShort = "";
  for (const m of models) {
    if (m.kind === "ours" || m.kind === "base") continue;
    let sum = 0;
    let n = 0;
    for (const t of TIERS) {
      const c = m.byTier[t];
      if (c?.evaluated) {
        sum += cellQuality(c);
        n += 1;
      }
    }
    if (n === 0) continue;
    const score = sum / n;
    const rank = KIND_RANK[m.kind];
    const better =
      score > bestScore ||
      (score === bestScore && rank < bestRank) ||
      (score === bestScore && rank === bestRank && m.short.localeCompare(bestShort) < 0);
    if (better) {
      bestKey = m.key;
      bestScore = score;
      bestRank = rank;
      bestShort = m.short;
    }
  }
  return bestKey;
}

/**
 * Match the contract's `best_other` (a key / name / short) to a rival model.
 * Item 6: OURS and BASE are never eligible — if the worker ever named BASE we drop
 * it and let the caller fall back to the derived (also BASE-excluding) rival.
 */
function resolveBestOtherKey(raw: string | null | undefined, models: ViewModel[]): string | null {
  if (!raw) return null;
  const needle = raw.trim().toLowerCase();
  if (!needle) return null;
  const hit = models.find(
    (m) =>
      m.kind !== "ours" &&
      m.kind !== "base" &&
      (m.key.toLowerCase() === needle ||
        m.name.toLowerCase() === needle ||
        m.short.toLowerCase() === needle),
  );
  return hit?.key ?? null;
}

/**
 * OURS genuinely adapts by level: scored at all three tiers, each move sound and
 * tier-appropriate, and the recommended move actually varies across the tiers.
 * This is the fallback when the worker omits `ours_tier_differentiates`. In the
 * showdown source (one tier per position) it is always false — honestly so.
 */
function deriveTierDifferentiates(oursByTier: Record<ShowcaseTier, ViewCell | null>): boolean {
  const cells = TIERS.map((t) => oursByTier[t]);
  if (cells.some((c) => !c || !c.evaluated || !c.move)) return false;
  if (!cells.every((c) => c!.tierFit && c!.sound)) return false;
  const moves = new Set(cells.map((c) => c!.move));
  return moves.size >= 2;
}

/** Resolve the pre-named / derived best non-OURS model for a position. */
export function bestOtherModel(position: ViewPosition): ViewModel | null {
  if (!position.bestOtherKey) return null;
  return position.models.find((m) => m.key === position.bestOtherKey) ?? null;
}

/** Resolve the best FRONTIER rival (GPT-5.5 / Claude / Gemini) for a position. */
export function bestFrontierModel(position: ViewPosition): ViewModel | null {
  if (!position.bestFrontierKey) return null;
  return position.models.find((m) => m.key === position.bestFrontierKey) ?? null;
}

/**
 * Strongest FRONTIER rival only (GPT-5.5 / Claude / Gemini), by mean cell quality
 * across the tiers it was scored at. This is the reference the headline
 * OURS-wins / OURS-loses definition uses, and the anchor for the "OURS diverges
 * from the best frontier" proof signal. Returns null when no frontier model was
 * scored at this position.
 */
function deriveBestFrontierKey(models: ViewModel[]): string | null {
  let bestKey: string | null = null;
  let bestScore = -Infinity;
  let bestShort = "";
  for (const m of models) {
    if (m.kind !== "frontier") continue;
    let sum = 0;
    let n = 0;
    for (const t of TIERS) {
      const c = m.byTier[t];
      if (c?.evaluated) {
        sum += cellQuality(c);
        n += 1;
      }
    }
    if (n === 0) continue;
    const score = sum / n;
    const better = score > bestScore || (score === bestScore && m.short.localeCompare(bestShort) < 0);
    if (better) {
      bestKey = m.key;
      bestScore = score;
      bestShort = m.short;
    }
  }
  return bestKey;
}

/**
 * True when OURS recommends a genuinely different move from the best frontier
 * rival at some tier both were scored at — the "our move isn't just the frontier
 * move" half of the proof. Only counts tiers where both actually have a move.
 */
function deriveDiffersFromBest(
  oursByTier: Record<ShowcaseTier, ViewCell | null>,
  frontier: ViewModel | null,
): boolean {
  if (!frontier) return false;
  let comparable = false;
  for (const t of TIERS) {
    const oc = oursByTier[t];
    const fc = frontier.byTier[t];
    if (oc?.evaluated && oc.move && fc?.evaluated && fc.move) {
      comparable = true;
      if (oc.move !== fc.move) return true;
    }
  }
  // No overlapping tier to compare → not a demonstrated divergence.
  return comparable ? false : false;
}

/** Distinct recommended moves OURS gives across the tiers it was scored at. */
function countDistinctOursMoves(oursByTier: Record<ShowcaseTier, ViewCell | null>): number {
  const moves = new Set<string>();
  for (const t of TIERS) {
    const c = oursByTier[t];
    if (c?.evaluated && c.move) moves.add(c.move);
  }
  return moves.size;
}

/** Normalize an optional worker rollup into the view summary (all fields nullable). */
function summaryFromContract(raw: ShowcaseModelSummary | null | undefined): ShowcaseModelSummary | null {
  if (!raw) return null;
  const num = (v: unknown): number | null => (typeof v === "number" ? v : null);
  return {
    key: raw.key ?? null,
    name: raw.name ?? null,
    family: raw.family ?? null,
    council_move: num(raw.council_move),
    council_instr: num(raw.council_instr),
    sound_rate: num(raw.sound_rate),
    tier_fit_rate: num(raw.tier_fit_rate),
    fabricated_rate: num(raw.fabricated_rate),
  };
}

export type DuelVerdict = "beats" | "trails" | "even" | "na";

export interface TierDuelRow {
  tier: ShowcaseTier;
  ours: ViewCell | null;
  other: ViewCell | null;
  /** "na" where either side has no answer at this tier (e.g. the showdown fallback). */
  verdict: DuelVerdict;
}

/**
 * OURS vs the position's best-other model, tier by tier — the honest indicator
 * of whether OURS beats or trails the strongest rival at each level.
 */
export function tierDuel(position: ViewPosition, otherKey: string | null): TierDuelRow[] {
  const ours = position.models.find((m) => m.kind === "ours") ?? null;
  const other = otherKey ? position.models.find((m) => m.key === otherKey) ?? null : null;
  return TIERS.map((tier) => {
    const oc = ours?.byTier[tier] ?? null;
    const xc = other?.byTier[tier] ?? null;
    let verdict: DuelVerdict = "na";
    if (oc?.evaluated && xc?.evaluated) {
      const d = cellQuality(oc) - cellQuality(xc);
      verdict = d > 0 ? "beats" : d < 0 ? "trails" : "even";
    }
    return { tier, ours: oc, other: xc, verdict };
  });
}

/* ------------------------------------------------------------------ */
/* Build from the real contract (showcase.json)                        */
/* ------------------------------------------------------------------ */

function cellFromContract(
  raw: ShowcaseCell | null | undefined,
  uci: (san: string | null | undefined) => string | null,
): ViewCell | null {
  if (!raw) return null;
  // Item 8 (defensive): a cell is only "evaluated" when the objective verdict is
  // actually present. If any of sound/tier_fit/fabricated is null or absent we treat
  // the cell as NOT measured, so incomplete future data can never masquerade as a
  // green "sound / faithful" result. (All shipped cells carry the full triple.)
  const evaluated = raw.sound != null && raw.tier_fit != null && raw.fabricated != null;
  return {
    move: raw.move ?? null,
    // Prefer the worker-written UCI; only re-derive from SAN when it is missing.
    moveUci: raw.move_uci ?? uci(raw.move),
    sound: raw.sound === true,
    tierFit: raw.tier_fit === true,
    fabricated: raw.fabricated === true,
    nViolations:
      typeof raw.n_violations === "number" ? raw.n_violations : raw.violations?.length ?? 0,
    violations: (raw.violations ?? []).map((v) => ({ sentence: v.sentence, reason: v.reason })),
    coaching: raw.coaching ?? "",
    rawCoaching: raw.raw_coaching ?? null,
    rawFabricated: raw.raw_fabricated === true,
    gateAttempts: typeof raw.gate_attempts === "number" ? raw.gate_attempts : null,
    verifiedFallback: raw.verified_fallback === true,
    councilMove: typeof raw.council_move === "number" ? raw.council_move : null,
    councilInstr: typeof raw.council_instr === "number" ? raw.council_instr : null,
    evaluated,
  };
}

function buildFromContract(doc: ShowcaseContract, source: DataSource = "showcase"): ShowcaseView {
  let councilScale = 1;
  let hasCouncil = false;
  let hasTrain = false;
  let perTierComplete = true;

  const positions: ViewPosition[] = doc.map((p) => {
    const uci = makeUci(p.fen);

    const models: ViewModel[] = p.models.map((m) => {
      const kind = deriveKind(m.family, Boolean(m.local), m.name);
      const byTier = emptyTierRecord<ViewCell | null>(null);
      for (const t of TIERS) {
        const cell = cellFromContract(m.byTier?.[t], uci);
        byTier[t] = cell;
        if (cell) {
          if (cell.councilMove != null) {
            hasCouncil = true;
            councilScale = Math.max(councilScale, cell.councilMove);
          }
          if (cell.councilInstr != null) {
            hasCouncil = true;
            councilScale = Math.max(councilScale, cell.councilInstr);
          }
        }
      }
      return {
        key: deriveKey(m.name, kind),
        name: m.name,
        short: shortName(m.name),
        kind,
        family: m.family,
        local: Boolean(m.local),
        byTier,
      };
    });

    const tierEvaluated = emptyTierRecord(false);
    const tierTargets = emptyTierRecord<TierTargetView | null>(null);
    for (const t of TIERS) {
      const has = models.some((m) => m.byTier[t]?.evaluated);
      tierEvaluated[t] = has;
      if (!has) perTierComplete = false;
      const san = p.tier_targets?.[t] ?? null;
      tierTargets[t] = san ? { san, uci: uci(san) } : null;
    }

    if (p.split === "train") hasTrain = true;

    const oursModel = models.find((m) => m.kind === "ours") ?? null;
    const bestOtherKey = resolveBestOtherKey(p.best_other, models) ?? deriveBestOtherKey(models);
    const bestFrontierKey = deriveBestFrontierKey(models);
    const bestFrontier = bestFrontierKey ? models.find((m) => m.key === bestFrontierKey) ?? null : null;
    const oursTierDifferentiates =
      typeof p.ours_tier_differentiates === "boolean"
        ? p.ours_tier_differentiates
        : oursModel
          ? deriveTierDifferentiates(oursModel.byTier)
          : false;
    const oursDistinctMoves =
      typeof p.ours_distinct_moves === "number"
        ? p.ours_distinct_moves
        : oursModel
          ? countDistinctOursMoves(oursModel.byTier)
          : 0;
    const oursDiffersFromBestFrontier = oursModel
      ? deriveDiffersFromBest(oursModel.byTier, bestFrontier)
      : false;
    // The proof subset (spec definition): OURS adapts by level AND its move
    // diverges from the best frontier rival. The worker's `focus` flag currently
    // tracks only the tier-differentiation half, so we compute the frontier-
    // divergence half here from the cells rather than aliasing `focus`.
    const isProof = oursTierDifferentiates && oursDiffersFromBestFrontier;

    return {
      id: p.id,
      fen: p.fen,
      phase: p.phase,
      split: p.split,
      sideToMove: sideToMove(p.fen),
      severity: p.severity ?? null,
      benchmark: p.benchmark ?? null,
      studentMove: p.student_move
        ? {
            san: p.student_move.san,
            uci: p.student_move.uci,
            severity: p.student_move.severity,
          }
        : null,
      tierTargets,
      tierEvaluated,
      models: orderModels(models),
      oursWins: Boolean(p.ours_wins),
      oursLoses: Boolean(p.ours_loses),
      shine: Boolean(p.shine),
      oursTierDifferentiates,
      oursMisdirected: Boolean(p.ours_misdirected),
      oursDistinctMoves,
      bestOtherKey,
      bestFrontierKey,
      oursDiffersFromBestFrontier,
      isProof,
      oursSummary: summaryFromContract(p.ours_summary),
      bestOtherDetail: summaryFromContract(p.best_other_detail),
      source,
    };
  });

  const info = metaModelInfo(positions);
  return {
    meta: {
      source,
      generatedUtc: null,
      modelVersion: info.ours.version ?? MODEL_VERSION,
      ours: info.ours,
      modelCount: info.modelCount,
      councilScale: Math.max(1, councilScale),
      hasCouncil,
      hasTrain,
      perTierComplete,
      totals: tally(positions),
      notes: NOTES,
    },
    positions,
  };
}

/* ------------------------------------------------------------------ */
/* Build from showdown.json (graceful fallback before showcase lands)  */
/* ------------------------------------------------------------------ */

function cellFromShowdown(m: ShowdownModel): ViewCell {
  return {
    move: m.rec_san,
    moveUci: m.rec_uci,
    sound: Boolean(m.sound),
    tierFit: Boolean(m.tier_appropriate),
    fabricated: Boolean(m.fabricated),
    nViolations: m.n_violations ?? m.violations?.length ?? 0,
    violations: (m.violations ?? []).map((v) => ({ sentence: v.sentence, reason: v.reason })),
    coaching: m.coaching ?? "",
    // The showdown source predates the two-layer gate: the shown text IS the raw
    // text, and there is no gate/attempt metadata, so we surface no gate badge.
    rawCoaching: m.coaching ?? null,
    rawFabricated: Boolean(m.fabricated),
    gateAttempts: null,
    verifiedFallback: false,
    councilMove: null, // council grades only arrive with showcase.json
    councilInstr: null,
    evaluated: true,
  };
}

function buildFromShowdown(doc: ShowdownDoc): ShowcaseView {
  const positions: ViewPosition[] = doc.positions.map((p: ShowdownPosition) => {
    const evalTier = p.tier as ShowcaseTier;

    const models: ViewModel[] = p.models.map((m) => {
      const byTier = emptyTierRecord<ViewCell | null>(null);
      byTier[evalTier] = cellFromShowdown(m);
      return {
        key: m.key,
        name: m.name,
        short: m.short,
        kind: m.kind,
        family: doc.meta.model_meta[m.key]?.family ?? m.kind,
        local: m.kind !== "frontier",
        byTier,
      };
    });

    const tierEvaluated = emptyTierRecord(false);
    tierEvaluated[evalTier] = true;

    const tierTargets = emptyTierRecord<TierTargetView | null>(null);
    if (p.tier_target) {
      tierTargets[evalTier] = { san: p.tier_target.san, uci: p.tier_target.uci };
    }

    // OURS loses (derived, honest): a competitor is tier-fit + faithful at this
    // tier where OURS isn't. Surfaced so the "where OURS loses" lens isn't faked.
    const ours = p.models.find((m) => m.key === "ours");
    const oursClean = Boolean(ours?.tier_appropriate && !ours?.fabricated);
    const rivalClean = p.models.some(
      (m) => m.key !== "ours" && m.tier_appropriate && !m.fabricated,
    );
    const oursLoses = !oursClean && rivalClean;

    // Best-other is still meaningful in the fallback (the strongest rival at the
    // single scored tier); tier-differentiation is not (one tier per position),
    // so it is honestly false until showcase.json supplies all three tiers.
    const oursModel = models.find((m) => m.kind === "ours") ?? null;
    const bestOtherKey = deriveBestOtherKey(models);
    const bestFrontierKey = deriveBestFrontierKey(models);
    const oursTierDifferentiates = oursModel
      ? deriveTierDifferentiates(oursModel.byTier)
      : false;

    return {
      id: p.key,
      fen: p.fen,
      phase: p.phase,
      split: "test", // every showdown position is held-out — the honest measure
      sideToMove: p.side_to_move,
      severity: p.severity ?? null,
      benchmark: p.benchmark,
      studentMove: p.student_move
        ? { san: p.student_move.san, uci: p.student_move.uci, severity: p.student_move.severity }
        : null,
      tierTargets,
      tierEvaluated,
      models: orderModels(models),
      oursWins: Boolean(p.ours_wins),
      oursLoses,
      // Item 4: "shine" is the CLEAN tier-differentiator subset — it requires all
      // three tiers scored per position. The showdown source scores one tier per
      // position, so shine is honestly false here (it must NOT be aliased to wins,
      // which would make the Shine lens a duplicate of "OURS wins").
      shine: false,
      oursTierDifferentiates,
      oursMisdirected: false,
      oursDistinctMoves: oursModel ? countDistinctOursMoves(oursModel.byTier) : 0,
      bestOtherKey,
      bestFrontierKey,
      // One tier per position in the fallback, so a cross-tier "proof" can't be
      // demonstrated — honestly false until showcase.json lands.
      oursDiffersFromBestFrontier: false,
      isProof: false,
      oursSummary: null,
      bestOtherDetail: null,
      source: "showdown",
    };
  });

  const info = metaModelInfo(positions);
  return {
    meta: {
      source: "showdown",
      generatedUtc: doc.meta.generated_utc ?? null,
      modelVersion: info.ours.version ?? MODEL_VERSION,
      ours: info.ours,
      modelCount: info.modelCount,
      councilScale: 2, // council.jsonl axis scale, for meter geometry only
      hasCouncil: false,
      hasTrain: false,
      perTierComplete: false,
      totals: tally(positions),
      notes: NOTES,
    },
    positions,
  };
}

function tally(positions: ViewPosition[]): ShowcaseTotals {
  return {
    positions: positions.length,
    shine: positions.filter((p) => p.shine).length,
    proof: positions.filter((p) => p.isProof).length,
    differentiates: positions.filter((p) => p.oursTierDifferentiates).length,
    oursWins: positions.filter((p) => p.oursWins).length,
    oursLoses: positions.filter((p) => p.oursLoses).length,
    train: positions.filter((p) => p.split === "train").length,
    test: positions.filter((p) => p.split === "test").length,
  };
}

/** Derive the OURS identity + distinct-model count straight from the built view. */
function metaModelInfo(positions: ViewPosition[]): { ours: OursLabel; modelCount: number } {
  const keys = new Set<string>();
  let oursName: string | null = null;
  for (const p of positions) {
    for (const m of p.models) {
      keys.add(m.key);
      if (!oursName && m.kind === "ours") oursName = m.name;
    }
  }
  return { ours: deriveOursLabel(oursName), modelCount: keys.size };
}

const NOTES: Record<string, string> = {
  proof: "The proof set — positions where OURS gives a genuinely different, level-appropriate move across the three tiers AND diverges from the best frontier model’s move. The one-screen case that the tuned coach adapts by level in a way the frontier doesn’t.",
  differentiates: "Distinct-tier — every position where OURS recommends a different move across Beginner / Intermediate / Advanced (each move sound and tier-fit). The broader adaptation set the proof subset is drawn from.",
  shine: "Clean tier-differentiators — the subset where OURS gives a different, level-appropriate move across all three tiers, doesn’t lose to the frontier, and isn’t mis-directed.",
  train: "Training sample — positions in-distribution for the tuned model. Expected to be strong; NOT a generalization test.",
  test: "Test sample — held-out positions the model never trained on. This is the honest measure of the coach.",
  // Item 7: make explicit that tier-fit is a TARGET-MATCH, not an emergent judgment.
  tier_fit: "tier-fit = the move matches the canonical tier-appropriate target OURS is trained to produce for that rating band — a trained target match, not an emergent capability.",
  fabricated: "Every model ships 0% user-visible fabrication after the verify-and-regenerate gate — a fairness floor applied equally to all, not a per-model differentiator. Where models actually differ is the semantic-judge truthfulness residual below.",
  council: "Blinded council of judges grades each answer for move correctness and instructiveness, models anonymized.",
};

/* ------------------------------------------------------------------ */
/* Per-model aggregate leaderboard — computed live from the loaded data */
/* ------------------------------------------------------------------ */

/**
 * One model's rolled-up eval metrics over the evaluated cells in scope. Every
 * number is computed from the loaded showcase data, so it always matches the
 * OURS version on screen (no hardcoded training figures anywhere).
 */
export interface ModelAggregate {
  key: string;
  name: string;
  short: string;
  kind: ModelKind;
  /** Evaluated cells counted for this model in scope. */
  cells: number;
  tierFitRate: number; // 0..1
  soundRate: number; // 0..1
  fabricatedRate: number; // 0..1 (post-gate; 0 across shipped data)
  councilMove: number | null; // mean 0..scale, or null when uncouncilled
  councilInstr: number | null; // mean 0..scale
  /** Fewer cells than the fullest model in scope (a partial / throttled model). */
  partial: boolean;
}

export interface Leaderboard {
  rows: ModelAggregate[];
  councilScale: number;
  maxCells: number;
  hasCouncil: boolean;
  oursKey: string | null;
  baseKey: string | null;
}

interface Accum {
  key: string;
  name: string;
  short: string;
  kind: ModelKind;
  n: number;
  tierFit: number;
  sound: number;
  fabricated: number;
  cmSum: number;
  cmN: number;
  ciSum: number;
  ciN: number;
}

/**
 * Aggregate every model's evaluated cells across the positions in scope
 * (optionally one split) into a comparable leaderboard. Ordered OURS → BASE →
 * frontier → open by mean council-move then name, so the tuned model and its own
 * untuned baseline sit adjacent for the honest before/after read.
 */
export function computeLeaderboard(view: ShowcaseView, split?: Split): Leaderboard {
  const acc = new Map<string, Accum>();
  for (const p of view.positions) {
    if (split && p.split !== split) continue;
    for (const m of p.models) {
      for (const t of TIERS) {
        const c = m.byTier[t];
        if (!c || !c.evaluated) continue;
        let a = acc.get(m.key);
        if (!a) {
          a = {
            key: m.key,
            name: m.name,
            short: m.short,
            kind: m.kind,
            n: 0,
            tierFit: 0,
            sound: 0,
            fabricated: 0,
            cmSum: 0,
            cmN: 0,
            ciSum: 0,
            ciN: 0,
          };
          acc.set(m.key, a);
        }
        a.n += 1;
        if (c.tierFit) a.tierFit += 1;
        if (c.sound) a.sound += 1;
        if (c.fabricated) a.fabricated += 1;
        if (c.councilMove != null) {
          a.cmSum += c.councilMove;
          a.cmN += 1;
        }
        if (c.councilInstr != null) {
          a.ciSum += c.councilInstr;
          a.ciN += 1;
        }
      }
    }
  }

  const maxCells = Math.max(0, ...[...acc.values()].map((a) => a.n));
  let hasCouncil = false;
  const rows: ModelAggregate[] = [...acc.values()]
    .filter((a) => a.n > 0)
    .map((a) => {
      if (a.cmN > 0 || a.ciN > 0) hasCouncil = true;
      return {
        key: a.key,
        name: a.name,
        short: a.short,
        kind: a.kind,
        cells: a.n,
        tierFitRate: a.tierFit / a.n,
        soundRate: a.sound / a.n,
        fabricatedRate: a.fabricated / a.n,
        councilMove: a.cmN > 0 ? a.cmSum / a.cmN : null,
        councilInstr: a.ciN > 0 ? a.ciSum / a.ciN : null,
        partial: maxCells > 0 && a.n < maxCells,
      };
    })
    .sort(
      (x, y) =>
        KIND_RANK[x.kind] - KIND_RANK[y.kind] ||
        (y.councilMove ?? -1) - (x.councilMove ?? -1) ||
        x.short.localeCompare(y.short),
    );

  return {
    rows,
    councilScale: view.meta.councilScale,
    maxCells,
    hasCouncil,
    oursKey: rows.find((r) => r.kind === "ours")?.key ?? null,
    baseKey: rows.find((r) => r.kind === "base")?.key ?? null,
  };
}

/* ------------------------------------------------------------------ */
/* The ONE-BEHAVIOR headline: tier-appropriate move selection          */
/* ------------------------------------------------------------------ */

/** Mean number of DISTINCT moves a model gives across the three tiers, over the
 *  positions (in scope) where it was scored at all three — the deterministic
 *  "does it adapt the move to the level?" signal. */
function meanDistinctMoves(view: ShowcaseView, kind: ModelKind, split: Split): { mean: number; n: number } {
  let sum = 0;
  let n = 0;
  for (const p of view.positions) {
    if (p.split !== split) continue;
    for (const m of p.models) {
      if (m.kind !== kind) continue;
      const cells = TIERS.map((t) => m.byTier[t]).filter(
        (c): c is ViewCell => Boolean(c?.evaluated && c.move),
      );
      if (cells.length < TIERS.length) continue;
      sum += new Set(cells.map((c) => c.move)).size;
      n += 1;
    }
  }
  return { mean: n > 0 ? sum / n : 0, n };
}

/**
 * The single honest headline the whole product now competes on: the tuned model
 * SELECTS THE TIER-APPROPRIATE MOVE where its own base — and the frontier — can't.
 * Everything here is computed live from the loaded cells (held-out split), so it
 * always matches the OURS version on screen and never hardcodes a training figure.
 *   - tier-fit: OURS vs its own untuned BASE vs the best frontier model.
 *   - adaptation: mean distinct moves OURS gives across the three levels, on the
 *     SAME positions where the frontier repeats essentially one move.
 * Deliberately carries NO instructiveness "lift" — fine-tuning did not improve the
 * prose (it regressed), so claiming it would be dishonest; the council grades stay
 * available in the leaderboard as context, not as the headline.
 */
export interface MoveSelectionHeadline {
  ours: ModelAggregate;
  base: ModelAggregate;
  bestFrontier: ModelAggregate | null;
  oursDistinct: number;
  frontierDistinct: number;
  nPositions: number;
}

export function moveSelectionHeadline(
  view: ShowcaseView | null,
  split: Split = "test",
): MoveSelectionHeadline | null {
  if (!view) return null;
  const lb = computeLeaderboard(view, split);
  const ours = lb.rows.find((r) => r.kind === "ours") ?? null;
  const base = lb.rows.find((r) => r.kind === "base") ?? null;
  if (!ours || !base) return null;
  const frontiers = lb.rows.filter((r) => r.kind === "frontier");
  const bestFrontier = frontiers.length
    ? frontiers.reduce((a, b) => (b.tierFitRate > a.tierFitRate ? b : a))
    : null;
  const oursAdapt = meanDistinctMoves(view, "ours", split);
  const frontierAdapt = meanDistinctMoves(view, "frontier", split);
  return {
    ours,
    base,
    bestFrontier,
    oursDistinct: oursAdapt.mean,
    frontierDistinct: frontierAdapt.mean,
    nPositions: oursAdapt.n,
  };
}

/* ------------------------------------------------------------------ */
/* Gate provenance badge (per cell) — honest, from the two-layer gate   */
/* ------------------------------------------------------------------ */

export type GateBadgeTone = "good" | "muted" | "caution";

export interface GateBadgeInfo {
  label: string;
  tone: GateBadgeTone;
  detail: string;
}

/**
 * A small, honest provenance badge for a shipped coaching cell, derived only from
 * the gate record (`verified_fallback` / `gate_attempts`). Returns null when the
 * source carries no gate metadata (e.g. the legacy showdown fallback), so we never
 * imply a gate ran when it didn't.
 */
export function gateBadge(cell: ViewCell): GateBadgeInfo | null {
  if (cell.verifiedFallback) {
    return {
      label: "engine-derived fallback",
      tone: "caution",
      detail:
        "No draft from this model passed the deterministic board-fact gate, so a deterministic engine-derived explanation (true by construction) is shown instead of model prose.",
    };
  }
  if (cell.gateAttempts == null) return null;
  if (cell.gateAttempts <= 1) {
    return {
      label: "verifier: clean on draft 1",
      tone: "good",
      detail: "The model’s first draft passed the deterministic board-fact faithfulness gate unchanged.",
    };
  }
  return {
    label: `verifier: re-sampled → draft ${cell.gateAttempts}`,
    tone: "muted",
    detail: `The model’s first draft was flagged by the board-fact gate; it was re-sampled and draft ${cell.gateAttempts} passed.`,
  };
}

/* ------------------------------------------------------------------ */
/* Two-layer truthfulness residual (per-model), copied from            */
/* data/showcase/truthfulness.json + data/showcase/gate_all_report.md.  */
/* These are STATIC, final numbers (the app can't read data/ at         */
/* runtime), shown verbatim so the residual gap is never hidden.        */
/* ------------------------------------------------------------------ */

/** One aggregation's truthful-rate point estimate + 95% CI (percent units). */
export interface TruthRate {
  pct: number;
  ciLoPct: number;
  ciHiPct: number;
}

export interface TruthfulnessRow {
  name: string;
  short: string;
  kind: ModelKind;
  cells: number;
  /** ⚠ partial model — only a subset of cells exist (Bedrock throttling); rates are over the cells it DOES have. */
  partial: boolean;
  /** Sampled cells fact-checked by the panel for this model. */
  judgeN: number;
  /** STRICT lower bound — truthful iff all 3 judges agree (a single objection sinks the cell). */
  any: TruthRate;
  /** Majority — truthful iff >=2 of 3 judges found it truthful. */
  majority: TruthRate;
  /** LENIENT upper bound — truthful unless all 3 judges object (only a unanimous objection sinks it). */
  unanimous: TruthRate;
}

export interface TruthfulnessData {
  /** Deterministic post-gate residual, pooled over every shipped cell — the fairness guarantee. */
  determOverall: { n: number; pct: number };
  /** Semantic-judge residual pooled over sampled cells, under all three aggregations. */
  overall: { n: number; any: TruthRate; majority: TruthRate; unanimous: TruthRate };
  judgePanel: string[];
  judgeCalls: number;
  judgeCostUsd: number;
  rows: TruthfulnessRow[];
}

/**
 * The honest truthfulness residual (source: data/showcase/truthfulness.json).
 * Faithfulness is a fairness FLOOR, not a differentiator: after the
 * verify-and-regenerate gate, deterministic board-fact fabrication is 0% for
 * EVERY model (determOverall). The real, non-zero differentiator is the
 * cross-family semantic-judge residual, reported under three nested panel rules
 * — any (strict lower bound) / majority / unanimous (lenient upper bound), each
 * with a 95% CI. OURS trails the frontier here; that gap is shown, not hidden.
 */
export const TRUTHFULNESS: TruthfulnessData = {
  determOverall: { n: 18135, pct: 0.0 },
  overall: {
    n: 378,
    any: { pct: 35.5, ciLoPct: 30.8, ciHiPct: 40.4 },
    majority: { pct: 57.1, ciLoPct: 52.1, ciHiPct: 62.0 },
    unanimous: { pct: 73.8, ciLoPct: 69.2, ciHiPct: 78.0 },
  },
  judgePanel: ["GPT-5.5", "Claude", "Gemini"],
  judgeCalls: 1134,
  judgeCostUsd: 13.05,
  rows: [
    { name: "OURS-v2 (1.7B tuned)", short: "OURS", kind: "ours", cells: 1476, partial: false, judgeN: 39, any: { pct: 23.1, ciLoPct: 12.7, ciHiPct: 38.3 }, majority: { pct: 25.6, ciLoPct: 14.6, ciHiPct: 41.1 }, unanimous: { pct: 30.8, ciLoPct: 18.6, ciHiPct: 46.4 } },
    { name: "BASE (Qwen3-1.7B-4bit, untuned)", short: "BASE", kind: "base", cells: 1476, partial: false, judgeN: 18, any: { pct: 0.0, ciLoPct: 0.0, ciHiPct: 17.6 }, majority: { pct: 16.7, ciLoPct: 5.8, ciHiPct: 39.2 }, unanimous: { pct: 33.3, ciLoPct: 16.3, ciHiPct: 56.3 } },
    { name: "GPT-5.5", short: "GPT-5.5", kind: "frontier", cells: 1476, partial: false, judgeN: 39, any: { pct: 79.5, ciLoPct: 64.5, ciHiPct: 89.2 }, majority: { pct: 97.4, ciLoPct: 86.8, ciHiPct: 99.5 }, unanimous: { pct: 100.0, ciLoPct: 91.0, ciHiPct: 100.0 } },
    { name: "Claude Opus 4.8", short: "Claude Opus 4.8", kind: "frontier", cells: 1476, partial: false, judgeN: 39, any: { pct: 25.6, ciLoPct: 14.6, ciHiPct: 41.1 }, majority: { pct: 56.4, ciLoPct: 41.0, ciHiPct: 70.7 }, unanimous: { pct: 84.6, ciLoPct: 70.3, ciHiPct: 92.8 } },
    { name: "Gemini 3.1 Pro", short: "Gemini 3.1 Pro", kind: "frontier", cells: 1476, partial: false, judgeN: 39, any: { pct: 33.3, ciLoPct: 20.6, ciHiPct: 49.0 }, majority: { pct: 64.1, ciLoPct: 48.4, ciHiPct: 77.3 }, unanimous: { pct: 92.3, ciLoPct: 79.7, ciHiPct: 97.3 } },
    { name: "Llama-3.3-70B", short: "Llama-3.3-70B", kind: "open", cells: 1476, partial: false, judgeN: 18, any: { pct: 72.2, ciLoPct: 49.1, ciHiPct: 87.5 }, majority: { pct: 83.3, ciLoPct: 60.8, ciHiPct: 94.2 }, unanimous: { pct: 94.4, ciLoPct: 74.2, ciHiPct: 99.0 } },
    { name: "Qwen3-32B", short: "Qwen3-32B", kind: "open", cells: 1476, partial: false, judgeN: 39, any: { pct: 46.2, ciLoPct: 31.6, ciHiPct: 61.4 }, majority: { pct: 64.1, ciLoPct: 48.4, ciHiPct: 77.3 }, unanimous: { pct: 74.4, ciLoPct: 58.9, ciHiPct: 85.4 } },
    { name: "GLM-5", short: "GLM-5", kind: "open", cells: 1476, partial: false, judgeN: 18, any: { pct: 44.4, ciLoPct: 24.6, ciHiPct: 66.3 }, majority: { pct: 66.7, ciLoPct: 43.7, ciHiPct: 83.7 }, unanimous: { pct: 88.9, ciLoPct: 67.2, ciHiPct: 96.9 } },
    { name: "Kimi-K2.5", short: "Kimi-K2.5", kind: "open", cells: 653, partial: true, judgeN: 18, any: { pct: 44.4, ciLoPct: 24.6, ciHiPct: 66.3 }, majority: { pct: 72.2, ciLoPct: 49.1, ciHiPct: 87.5 }, unanimous: { pct: 88.9, ciLoPct: 67.2, ciHiPct: 96.9 } },
    { name: "Mistral-Large-3 (675B)", short: "Mistral-Large-3", kind: "open", cells: 623, partial: true, judgeN: 18, any: { pct: 38.9, ciLoPct: 20.3, ciHiPct: 61.4 }, majority: { pct: 55.6, ciLoPct: 33.7, ciHiPct: 75.4 }, unanimous: { pct: 66.7, ciLoPct: 43.7, ciHiPct: 83.7 } },
    { name: "Gemma-3-27B-it", short: "Gemma-3-27B-it", kind: "open", cells: 623, partial: true, judgeN: 39, any: { pct: 25.6, ciLoPct: 14.6, ciHiPct: 41.1 }, majority: { pct: 53.8, ciLoPct: 38.6, ciHiPct: 68.4 }, unanimous: { pct: 71.8, ciLoPct: 56.2, ciHiPct: 83.5 } },
    { name: "DeepSeek-V3.2", short: "DeepSeek-V3.2", kind: "open", cells: 1476, partial: false, judgeN: 18, any: { pct: 22.2, ciLoPct: 9.0, ciHiPct: 45.2 }, majority: { pct: 38.9, ciLoPct: 20.3, ciHiPct: 61.4 }, unanimous: { pct: 66.7, ciLoPct: 43.7, ciHiPct: 83.7 } },
    { name: "DeepSeek-R1 (reasoning)", short: "DeepSeek-R1", kind: "open", cells: 1476, partial: false, judgeN: 18, any: { pct: 11.1, ciLoPct: 3.1, ciHiPct: 32.8 }, majority: { pct: 50.0, ciLoPct: 29.0, ciHiPct: 71.0 }, unanimous: { pct: 61.1, ciLoPct: 38.6, ciHiPct: 79.7 } },
    { name: "Qwen3-Next-80B-A3B", short: "Qwen3-Next-80B-A3B", kind: "open", cells: 1476, partial: false, judgeN: 18, any: { pct: 5.6, ciLoPct: 1.0, ciHiPct: 25.8 }, majority: { pct: 33.3, ciLoPct: 16.3, ciHiPct: 56.3 }, unanimous: { pct: 66.7, ciLoPct: 43.7, ciHiPct: 83.7 } },
  ],
};

/* ------------------------------------------------------------------ */
/* Loader                                                              */
/* ------------------------------------------------------------------ */

async function fetchContract(url: string, signal?: AbortSignal): Promise<ShowcaseContract | null> {
  try {
    const res = await fetch(url, { signal, cache: "no-store" });
    if (!res.ok) return null;
    const data: unknown = await res.json();
    if (!Array.isArray(data) || data.length === 0) return null;
    return data as ShowcaseContract;
  } catch {
    return null; // absent / malformed → try the next source
  }
}

/**
 * Load the Showcase view, preferring the richest source available:
 *   1. showcase.json  — the full curated slice (all models × 3 tiers + council),
 *      owned by the separate worker.
 *   2. showcase_interim.json — the same array contract, but only OURS is scored
 *      at all three tiers (re-run locally); rivals stay at their benchmarked tier
 *      and there are no council grades yet. Lets the tier-differentiation moat and
 *      the per-tier OURS-vs-best duel work before the full slice lands.
 *   3. showdown.json — the shipped held-out benchmark (one tier per position).
 * Returns null only if none exists. Never throws for a missing file.
 */
export async function loadShowcaseView(signal?: AbortSignal): Promise<ShowcaseView | null> {
  const full = await fetchContract("/showcase.json", signal);
  if (full) return buildFromContract(full, "showcase");

  const interim = await fetchContract("/showcase_interim.json", signal);
  if (interim) return buildFromContract(interim, "interim");

  const showdown = await getShowdown(signal);
  if (showdown && showdown.positions?.length) return buildFromShowdown(showdown);

  return null;
}
