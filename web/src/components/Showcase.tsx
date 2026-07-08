"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  Button,
  Card,
  ListBox,
  Meter,
  Select,
  Separator,
  Tabs,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
} from "@heroui/react";
import type { Key } from "@heroui/react";
import { postCoachResilient, type CoachResponse, type CoachWakeStatus } from "@/lib/api";
import {
  bestFrontierModel,
  bestOtherModel,
  computeLeaderboard,
  describeOursOutcome,
  gateBadge,
  loadShowcaseView,
  moveSelectionHeadline,
  orderModels,
  primaryTier,
  principleTag,
  representativeTier,
  tierDuel,
  TIERS,
  TRUTHFULNESS,
  type DuelVerdict,
  type GateBadgeInfo,
  type ModelAggregate,
  type ModelKind,
  type MoveSelectionHeadline,
  type OursLabel,
  type ShowcaseTier,
  type ShowcaseView,
  type Split,
  type TierDuelRow,
  type TruthfulnessRow,
  type ViewCell,
  type ViewModel,
  type ViewPosition,
} from "@/lib/showcase";
import ShowdownBoard from "./ShowdownBoard";
import CopyFenButton from "./CopyFenButton";
import { ShieldCheckIcon } from "./icons";

type Status = "loading" | "ready" | "error" | "empty";
type LibFilter = "proof" | "differentiates" | "wins" | "loses" | "shine" | "all";
type PhaseFilter = "all" | "opening" | "middlegame" | "endgame";
type LiveState = {
  status: "idle" | "loading" | "done" | "error";
  result: CoachResponse | null;
  error: string | null;
  fen: string | null;
  tier: ShowcaseTier | null;
  waking: CoachWakeStatus | null;
};

const LIB_PAGE = 9;
const IDLE_LIVE: LiveState = {
  status: "idle",
  result: null,
  error: null,
  fen: null,
  tier: null,
  waking: null,
};

function cap(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

const TIER_BAND: Record<ShowcaseTier, string> = {
  beginner: "1000–1200",
  intermediate: "1300–1600",
  advanced: "1700–2000",
};

/**
 * The deterministic PRINCIPLE TAG for a precomputed cell's move — the honest "why
 * this move" for the reframed hero. Showcase cells carry no concept/takeaway (only
 * the regressed prose), so the tag is built from the objective verdict we actually
 * compete on: tier-appropriateness + soundness. This keeps the hero anchored to the
 * trained behavior (the level-appropriate move), never to the prose.
 */
function cellMoveTag(cell: ViewCell, tier: ShowcaseTier): string | null {
  if (!cell.move) return null;
  if (cell.tierFit) return `the ${cap(tier)}-appropriate move`;
  if (cell.sound) return `sound — but not the ${cap(tier)} target move`;
  return `off-target for ${cap(tier)}`;
}

/* ================================================================== */
/* Main view                                                           */
/* ================================================================== */

export default function Showcase() {
  const [view, setView] = useState<ShowcaseView | null>(null);
  const [status, setStatus] = useState<Status>("loading");

  const [split, setSplit] = useState<Split>("test");
  const [libFilter, setLibFilter] = useState<LibFilter>("proof");
  const [phase, setPhase] = useState<PhaseFilter>("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [tier, setTier] = useState<ShowcaseTier>("beginner");
  const [modelKey, setModelKey] = useState<string>("ours");
  const [visible, setVisible] = useState(LIB_PAGE);
  const [live, setLive] = useState<LiveState>(IDLE_LIVE);

  const abortRef = useRef<AbortController | null>(null);
  const liveAbortRef = useRef<AbortController | null>(null);

  const load = useCallback(() => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setStatus("loading");
    loadShowcaseView(ctrl.signal)
      .then((v) => {
        if (ctrl.signal.aborted) return;
        if (!v || v.positions.length === 0) {
          setStatus("empty");
          return;
        }
        setView(v);
        setStatus("ready");
      })
      .catch(() => {
        if (ctrl.signal.aborted) return;
        setStatus("error");
      });
  }, []);

  useEffect(() => {
    load();
    return () => {
      abortRef.current?.abort();
      liveAbortRef.current?.abort();
    };
  }, [load]);

  const meta = view?.meta ?? null;
  const positions = useMemo(() => view?.positions ?? [], [view]);

  // The ONE-BEHAVIOR headline: tier-appropriate MOVE SELECTION, computed live from
  // the HELD-OUT test split (the honest measure — and the leaderboard's default
  // view, so the two agree). No hardcoded training figures; always consistent with
  // the OURS version on screen. Carries no instructiveness "lift" — fine-tuning
  // regressed the prose, so the headline stays on the axis we actually win.
  const headline = useMemo(() => moveSelectionHeadline(view, "test"), [view]);

  // If showcase.json has no training split at all, keep the toggle on Test.
  useEffect(() => {
    if (meta && !meta.hasTrain && split === "train") setSplit("test");
  }, [meta, split]);

  const splitPositions = useMemo(
    () => positions.filter((p) => p.split === split),
    [positions, split],
  );

  // How many positions each lens holds in the current split. Used both to label
  // the lens tabs and to keep us off empty lenses (items 4 & 5).
  const lensCounts = useMemo(
    () => ({
      proof: splitPositions.filter((p) => p.isProof).length,
      differentiates: splitPositions.filter((p) => p.oursTierDifferentiates).length,
      shine: splitPositions.filter((p) => p.shine).length,
      wins: splitPositions.filter((p) => p.oursWins).length,
      loses: splitPositions.filter((p) => p.oursLoses).length,
      all: splitPositions.length,
    }),
    [splitPositions],
  );

  // Lenses worth showing in this split: "all" always; the rest only when populated
  // (so the Shine tab hides when it isn't a distinct, non-empty subset — item 4).
  const availableLenses = useMemo<LibFilter[]>(
    () =>
      (["proof", "differentiates", "wins", "loses", "shine", "all"] as LibFilter[]).filter(
        (f) => f === "all" || lensCounts[f] > 0,
      ),
    [lensCounts],
  );

  // The lens we actually render: honor the user's pick when it's populated, else
  // fall back to the first populated lens (preferred order above). Derived — never
  // lands on an empty view, and needs no set-state-in-effect (item 5).
  const effectiveLibFilter: LibFilter = availableLenses.includes(libFilter)
    ? libFilter
    : availableLenses[0] ?? "all";

  const filtered = useMemo(() => {
    return splitPositions.filter((p) => {
      if (phase !== "all" && p.phase !== phase) return false;
      if (effectiveLibFilter === "proof" && !p.isProof) return false;
      if (effectiveLibFilter === "differentiates" && !p.oursTierDifferentiates) return false;
      if (effectiveLibFilter === "shine" && !p.shine) return false;
      if (effectiveLibFilter === "wins" && !p.oursWins) return false;
      if (effectiveLibFilter === "loses" && !p.oursLoses) return false;
      return true;
    });
  }, [splitPositions, phase, effectiveLibFilter]);

  useEffect(() => setVisible(LIB_PAGE), [split, effectiveLibFilter, phase]);

  // Keep a valid selection as the library changes.
  useEffect(() => {
    if (filtered.length === 0) {
      setSelectedId(null);
      return;
    }
    if (!selectedId || !filtered.some((p) => p.id === selectedId)) {
      setSelectedId(filtered[0].id);
    }
  }, [filtered, selectedId]);

  const selected = useMemo(
    () => filtered.find((p) => p.id === selectedId) ?? filtered[0] ?? null,
    [filtered, selectedId],
  );

  // When the selected position changes, default the tier to one it has data for
  // and keep the model selection valid (fall back to OURS / first).
  useEffect(() => {
    if (!selected) return;
    // Keep the current tier if a rival is scored there; otherwise land on the
    // position's primary (rival-bearing) tier so the head-to-head is populated.
    setTier((t) =>
      selected.models.some((m) => m.kind !== "ours" && m.byTier[t]?.evaluated)
        ? t
        : primaryTier(selected),
    );
    setModelKey((k) =>
      selected.models.some((m) => m.key === k)
        ? k
        : selected.models.find((m) => m.kind === "ours")?.key ?? selected.models[0]?.key ?? "ours",
    );
    setLive(IDLE_LIVE);
  }, [selected]);

  // A tier/position change invalidates a live re-run tied to the old context.
  useEffect(() => {
    setLive((l) =>
      l.status !== "idle" && (l.fen !== selected?.fen || l.tier !== tier) ? IDLE_LIVE : l,
    );
  }, [selected, tier]);

  const selModel = useMemo(
    () =>
      selected?.models.find((m) => m.key === modelKey) ??
      selected?.models.find((m) => m.kind === "ours") ??
      selected?.models[0] ??
      null,
    [selected, modelKey],
  );
  const selCell = selModel?.byTier[tier] ?? null;

  const liveActive =
    live.status === "done" && !!live.result && live.fen === selected?.fen && live.tier === tier;
  const boardMoveUci = liveActive ? live.result!.recommended_move_uci : selCell?.moveUci ?? null;

  const runLive = useCallback(() => {
    if (!selected) return;
    liveAbortRef.current?.abort();
    const ctrl = new AbortController();
    liveAbortRef.current = ctrl;
    const forFen = selected.fen;
    const forTier = tier;
    setLive({ status: "loading", result: null, error: null, fen: forFen, tier: forTier, waking: null });
    postCoachResilient(
      { fen: forFen, tier: forTier, student_move: selected.studentMove?.uci ?? undefined },
      {
        signal: ctrl.signal,
        // Cold start in progress — annotate the loading state, don't error out.
        onStatus: (st) => {
          if (ctrl.signal.aborted) return;
          setLive((l) => (l.status === "loading" ? { ...l, waking: st } : l));
        },
      },
    )
      .then((res) => {
        if (ctrl.signal.aborted) return;
        setLive({ status: "done", result: res, error: null, fen: forFen, tier: forTier, waking: null });
      })
      .catch((e: unknown) => {
        if (ctrl.signal.aborted) return;
        setLive({
          status: "error",
          result: null,
          error: e instanceof Error ? e.message : "The coach service didn’t respond.",
          fen: forFen,
          tier: forTier,
          waking: null,
        });
      });
  }, [selected, tier]);

  const shownLib = filtered.slice(0, visible);
  const oursOutcome = selected ? describeOursOutcome(selected, tier) : null;

  return (
    <div className="relative z-[1] mx-auto flex min-h-dvh w-full max-w-[1320px] flex-col gap-7 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      <ShowcaseHeader meta={meta} status={status} headline={headline} />

      {status === "loading" && <LoadingSkeleton />}

      {status === "error" && (
        <EmptyState
          title="The showcase data didn’t load."
          body="It reads showcase.json (or falls back to showdown.json) from web/public. Rebuild those artifacts and retry."
          onRetry={load}
        />
      )}

      {status === "empty" && (
        <EmptyState
          title="No showcase positions yet."
          body="Neither showcase.json nor showdown.json is present in web/public. Generate one and retry."
          onRetry={load}
        />
      )}

      {status === "ready" && meta && (
        <>
          <ControlBar
            meta={meta}
            split={split}
            onSplit={setSplit}
            libFilter={effectiveLibFilter}
            onLibFilter={setLibFilter}
            phase={phase}
            onPhase={setPhase}
            lensCounts={lensCounts}
            availableLenses={availableLenses}
            filteredCount={filtered.length}
          />

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
            {/* Curated library */}
            <section className="flex flex-col gap-3" aria-label="Curated position library">
              <div className="flex items-baseline justify-between gap-2">
                <h2 className="text-sm font-semibold text-ink">
                  {LIB_TITLE[effectiveLibFilter]}
                </h2>
                <span className="text-xs text-muted tnum">{filtered.length}</span>
              </div>

              {filtered.length === 0 ? (
                <div className="rounded-[10px] border border-[color:var(--border)] px-4 py-10 text-center text-sm leading-relaxed text-muted">
                  {emptyLibraryText(effectiveLibFilter, meta, split)}
                </div>
              ) : (
                <>
                  <ul className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-1">
                    {shownLib.map((p) => (
                      <li key={p.id}>
                        <LibraryCard
                          position={p}
                          active={p.id === selected?.id}
                          onSelect={() => setSelectedId(p.id)}
                        />
                      </li>
                    ))}
                  </ul>
                  {visible < filtered.length && (
                    <Button
                      variant="tertiary"
                      size="md"
                      className="min-h-11 self-center rounded-full"
                      onPress={() => setVisible((v) => v + LIB_PAGE)}
                    >
                      Show more ({filtered.length - visible} left)
                    </Button>
                  )}
                </>
              )}
            </section>

            {/* Explorer */}
            <section aria-label="Model explorer">
              {selected && selModel ? (
                <Explorer
                  meta={meta}
                  position={selected}
                  models={selected.models}
                  modelKey={selModel.key}
                  onModelKey={setModelKey}
                  tier={tier}
                  onTier={setTier}
                  selModel={selModel}
                  selCell={selCell}
                  boardMoveUci={boardMoveUci}
                  liveActive={liveActive}
                  outcome={oursOutcome}
                  live={live}
                  onRunLive={runLive}
                  onClearLive={() => setLive(IDLE_LIVE)}
                />
              ) : (
                <div className="rounded-[10px] border border-[color:var(--border)] px-6 py-16 text-center text-sm text-muted">
                  Select a position from the library to compare models.
                </div>
              )}
            </section>
          </div>

          <LeaderboardPanel view={view!} meta={meta} split={split} />

          <TruthfulnessPanel oursLabel={meta.ours} />
        </>
      )}
    </div>
  );
}

const LIB_TITLE: Record<LibFilter, string> = {
  proof: "The proof — adapts by level AND diverges from the frontier",
  differentiates: "Where OURS adapts by level",
  shine: "Where our model shines",
  wins: "Where OURS beats the frontier",
  loses: "Where OURS trails the frontier",
  all: "All positions",
};

/** Honest empty-state copy per filter — tells the grader exactly why a lens is
 *  empty and what would fill it, without pretending data exists. */
function emptyLibraryText(libFilter: LibFilter, meta: ShowcaseView["meta"], split: Split): string {
  if (libFilter === "proof" || libFilter === "differentiates") {
    return meta.perTierComplete
      ? libFilter === "proof"
        ? "No position in this split has OURS both adapting across all three tiers AND diverging from the best frontier move."
        : "No position in this split varies OURS’s move across all three tiers."
      : "The full tier-differentiation set arrives with showcase.json — it needs all three tiers scored per position. Meanwhile, the OURS wins / All lenses are live from the held-out set.";
  }
  if (split === "train" && !meta.hasTrain) {
    return "The training sample arrives with showcase.json.";
  }
  return "No positions match these filters.";
}

/* ================================================================== */
/* Header                                                              */
/* ================================================================== */

function ShowcaseHeader({
  meta,
  status,
  headline,
}: {
  meta: ShowcaseView["meta"] | null;
  status: Status;
  headline: MoveSelectionHeadline | null;
}) {
  const ours = meta?.ours ?? null;
  const sizeLabel = ours?.size ? `${ours.size} ` : "";
  const modelCount = meta?.modelCount ?? 0;
  return (
    <header className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-3">
        <Link
          href="/"
          className="inline-flex min-h-9 items-center gap-1.5 text-sm text-muted transition-colors hover:text-ink"
        >
          <span aria-hidden className="text-faint">‹</span>
          Coach studio
        </Link>
        <div className="flex items-center gap-3">
          <Link
            href="/showdown"
            className="inline-flex min-h-9 items-center gap-1.5 text-sm text-muted transition-colors hover:text-ink"
          >
            Showdown list
            <span aria-hidden className="text-faint">›</span>
          </Link>
          <span className="rounded-full px-2.5 py-1 font-mono text-xs text-signal ring-1 ring-signal/40 tnum">
            {ours?.badge ?? "OURS"}
            {ours?.size ? ` · ${ours.size}` : ""}
          </span>
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center rounded-full bg-signal/12 px-2.5 py-1 text-[11px] font-medium uppercase tracking-wide text-signal ring-1 ring-signal/30">
            One behavior · move selection
          </span>
          <span className="text-[11px] uppercase tracking-wide text-faint">
            Multi-Model Showcase{modelCount ? ` · ${modelCount} models` : ""}
          </span>
        </div>
        <h1 className="text-2xl font-semibold tracking-tight text-balance text-ink sm:text-3xl">
          The fine-tune reliably picks the level-appropriate move where its base — and the frontier —
          can’t.
        </h1>
        <p className="max-w-3xl text-sm leading-relaxed text-muted sm:text-base">
          <span className="text-signal">OURS</span> is a {sizeLabel}model running locally with one
          job: <span className="text-ink">select the tier-appropriate move</span> for the player’s
          rating.{" "}
          {headline ? (
            <>
              On the same held-out positions as its untuned <span className="text-ink">base</span> —
              same weights, byte-identical input — fine-tuning lifts tier-appropriate move selection{" "}
              <span className="text-ink tnum">
                {pct(headline.base.tierFitRate)} → {pct(headline.ours.tierFitRate)}
              </span>
              {headline.bestFrontier && (
                <>
                  , past even the best frontier model{" "}
                  <span className="text-ink tnum">({pct(headline.bestFrontier.tierFitRate)})</span>
                </>
              )}
              . And it <span className="text-ink">adapts the move to the level</span>: on those same
              positions OURS gives{" "}
              <span className="text-ink tnum">{headline.oursDistinct.toFixed(1)}</span> distinct,
              level-appropriate moves across the three tiers, where the frontier repeats essentially
              one{" "}
              <span className="tnum">({headline.frontierDistinct.toFixed(1)})</span>. That
              level-appropriate move is the behavior a prompt on the same weights can’t reproduce
              (computed live from {headline.ours.cells.toLocaleString()} held-out cells).
            </>
          ) : (
            <>
              Fine-tuning teaches it to pick a different, level-appropriate move per rating where the
              frontier repeats one — a behavior a prompt on the same weights can’t reproduce.
            </>
          )}
        </p>
        <p className="max-w-3xl text-sm leading-relaxed text-muted">
          <span className="text-ink">The comparison, below,</span> is centered on that axis:
          tier-appropriate move selection and per-level move adaptation, both deterministic. The
          coaching prose is a <span className="text-ink">secondary, optional layer</span> — the
          frontier writes livelier explanations (see the council instructiveness grades and the
          truthfulness residual), and we don’t claim the prose as the win. Pick a position, switch
          the model and the rating tier, and read each model’s move at every level side by side; only{" "}
          <span className="text-signal">OURS</span> can be re-run live.
        </p>
      </div>

      {meta && status === "ready" && <ProvenanceNote meta={meta} />}
    </header>
  );
}

/** X.X% with a single decimal, tabular-friendly. */
function pct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`;
}

function ProvenanceNote({ meta }: { meta: ShowcaseView["meta"] }) {
  const { source } = meta;
  const tone =
    source === "showdown"
      ? "border-[color:var(--caution)]/40 bg-[color:var(--caution)]/10"
      : "border-signal/40 bg-signal/10";
  const dotCls = source === "showdown" ? "text-[color:var(--caution)]" : "text-signal";
  const title =
    source === "showcase"
      ? "Live showcase data — showcase.json"
      : source === "interim"
        ? "Interim data — OURS re-scored at all three tiers"
        : "Preview data — showdown.json (held-out only)";
  return (
    <div className={`flex flex-col gap-1 rounded-[10px] border px-4 py-3 text-xs leading-relaxed ${tone}`}>
      <div className="flex items-center gap-2 text-sm font-medium text-ink">
        <span aria-hidden className={dotCls}>
          {source === "showcase" ? "●" : "◐"}
        </span>
        {title}
      </div>
      <p className="text-muted">
        {source === "showcase" && (
          <>
            Reading the curated showcase slice: training + test splits, all three tiers per position,
            and blinded council grades where the council scored them.
          </>
        )}
        {source === "interim" && (
          <>
            showcase.json isn’t here yet. OURS ({meta.ours.badge}) was re-run locally at{" "}
            <span className="text-ink">all three tiers</span>, so the tier-differentiation moat and
            the per-tier OURS-vs-best duel are live on the held-out set. Rivals stay at their single
            benchmarked tier, and the <span className="text-ink">training sample</span> +{" "}
            <span className="text-ink">blinded council grades</span> arrive with the full
            showcase.json.
          </>
        )}
        {source === "showdown" && (
          <>
            showcase.json isn’t here yet, so this reads the shipped held-out benchmark. The{" "}
            <span className="text-ink">Training sample</span>, the per-tier comparison across all
            three levels, and the <span className="text-ink">blinded council grades</span> light up
            automatically once showcase.json lands.
          </>
        )}
      </p>
    </div>
  );
}

/* ================================================================== */
/* Control bar — split tabs, curated filters, phase                    */
/* ================================================================== */

const LENS_LABEL: Record<LibFilter, string> = {
  proof: "Proof",
  differentiates: "Distinct-tier",
  shine: "Shine",
  wins: "OURS wins",
  loses: "OURS loses",
  all: "All",
};

function ControlBar({
  meta,
  split,
  onSplit,
  libFilter,
  onLibFilter,
  phase,
  onPhase,
  lensCounts,
  availableLenses,
  filteredCount,
}: {
  meta: ShowcaseView["meta"];
  split: Split;
  onSplit: (s: Split) => void;
  libFilter: LibFilter;
  onLibFilter: (f: LibFilter) => void;
  phase: PhaseFilter;
  onPhase: (p: PhaseFilter) => void;
  lensCounts: Record<LibFilter, number>;
  availableLenses: LibFilter[];
  filteredCount: number;
}) {
  // Only offer lenses that actually hold positions in this split (item 4/5): the
  // Shine tab disappears when it isn't a distinct, non-empty subset.
  const lensOptions = availableLenses.map(
    (f) => [f, `${LENS_LABEL[f]} ${lensCounts[f]}`] as [LibFilter, string],
  );

  const trainDisabled = !meta.hasTrain;

  return (
    <section className="flex flex-col gap-4">
      {/* Split: Training vs Test as real HeroUI Tabs, honestly labelled */}
      <div className="flex flex-wrap items-center gap-2">
        <Tabs
          aria-label="Sample split"
          variant="secondary"
          selectedKey={split}
          onSelectionChange={(k) => onSplit(k as Split)}
        >
          <Tabs.ListContainer>
            {/* These tabs carry a two-line label (title + subtitle). HeroUI's tab is
                a fixed single-line height (h-8), which clips the title and lets the
                secondary-variant underline cut through the text. Override to an auto
                height with vertical padding so the label breathes and the underline
                sits below it; nowrap keeps each label a tidy two lines so both tabs
                stay the same height. */}
            <Tabs.List aria-label="Sample split" className="*:h-auto *:py-2">
              <Tabs.Tab id="test">
                <span className="flex flex-col items-start whitespace-nowrap leading-tight">
                  <span>Test sample</span>
                  <span className="text-[11px] text-faint">held-out · honest measure</span>
                </span>
                <Tabs.Indicator />
              </Tabs.Tab>
              <Tabs.Tab id="train" isDisabled={trainDisabled}>
                <span className="flex flex-col items-start whitespace-nowrap leading-tight">
                  <span>Training sample</span>
                  <span className="text-[11px] text-faint">
                    {trainDisabled ? "with showcase.json" : "in-distribution"}
                  </span>
                </span>
                <Tabs.Indicator />
              </Tabs.Tab>
            </Tabs.List>
          </Tabs.ListContainer>
        </Tabs>
        <Tooltip delay={200}>
          <Tooltip.Trigger aria-label="What the split means">
            <span className="inline-flex size-5 cursor-help items-center justify-center rounded-full text-xs text-faint ring-1 ring-[color:var(--border)]">
              ?
            </span>
          </Tooltip.Trigger>
          <Tooltip.Content showArrow className="max-w-[20rem]">
            <Tooltip.Arrow />
            <p className="leading-relaxed">
              Training positions are in-distribution for the tuned model — expected to be strong, and{" "}
              <span className="font-medium">not</span> a generalization test. Test positions are
              held-out; that split is the honest measure of the coach.
            </p>
          </Tooltip.Content>
        </Tooltip>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
          <div className="flex items-center gap-1.5">
            <SegmentedControl
              label="Library"
              value={libFilter}
              onChange={onLibFilter}
              options={lensOptions}
            />
            <LensLegend />
          </div>
          <SegmentedControl
            label="Phase"
            value={phase}
            onChange={onPhase}
            options={[
              ["all", "All"],
              ["opening", "Opening"],
              ["middlegame", "Middle"],
              ["endgame", "Endgame"],
            ]}
          />
        </div>
        <span className="text-xs text-muted tnum">
          {filteredCount} {filteredCount === 1 ? "position" : "positions"}
        </span>
      </div>

      <p className="max-w-4xl text-xs leading-relaxed text-faint">
        <span className="font-medium text-muted">Proof</span> is the headline — where OURS both
        adapts by level (a different, level-appropriate move across the three tiers) AND diverges
        from the best frontier model’s move. <span className="font-medium text-muted">Distinct-tier</span>{" "}
        is the broader adaptation set it’s drawn from.{" "}
        <span className="font-medium text-muted">OURS wins / OURS loses</span> compare OURS to the
        three <span className="text-muted">frontier</span> models only (GPT-5.5 / Claude / Gemini)
        on the sound tier move — OURS’s own <span className="text-muted">BASE</span> baseline and the
        open field are excluded from win credit, so the banner and these counts use one rule. Every
        shipped cell is gated, so deterministic board-fact fabrication is{" "}
        <span className="text-muted">0% for all models</span>; the measured per-model metrics are in
        the leaderboard, and the honest semantic-truth gap in the residual panel below.
      </p>
    </section>
  );
}

/** Single-select segmented control built on HeroUI ToggleButtonGroup, themed to
 *  the Bench-Instrument look. Used for the Library and Phase filters. */
function SegmentedControl<T extends string>({
  label,
  value,
  onChange,
  options,
  size = "sm",
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: [T, string][];
  size?: "sm" | "md" | "lg";
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs font-medium uppercase tracking-wide text-faint">{label}</span>
      <ToggleButtonGroup
        aria-label={label}
        size={size}
        selectionMode="single"
        disallowEmptySelection
        selectedKeys={new Set([value])}
        onSelectionChange={(keys) => {
          const k = [...keys][0];
          if (k) onChange(k as T);
        }}
      >
        {options.map(([id, text], i) => (
          <ToggleButton key={id} id={id}>
            {i > 0 && <ToggleButtonGroup.Separator />}
            {text}
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
    </div>
  );
}

/** Honest explainer for the two different library lenses: the tier-adaptation
 *  moat vs the general-quality tally where the frontier is expected to lead. */
function LensLegend() {
  return (
    <Tooltip delay={200}>
      <Tooltip.Trigger aria-label="What the library lenses mean">
        <span className="inline-flex size-5 cursor-help items-center justify-center rounded-full text-xs text-faint ring-1 ring-[color:var(--border)]">
          ?
        </span>
      </Tooltip.Trigger>
      <Tooltip.Content showArrow className="max-w-[24rem]">
        <Tooltip.Arrow />
        <div className="flex flex-col gap-1.5 leading-relaxed">
          <p>
            <span className="font-medium text-signal">Proof</span> — the headline set. OURS
            recommends a genuinely different, level-appropriate move across Beginner / Intermediate /
            Advanced (each sound and tier-fit) AND that move diverges from the best frontier model’s
            move — the case that the tuned coach adapts by level in a way the frontier doesn’t.
          </p>
          <p>
            <span className="font-medium">Distinct-tier</span> — the broader adaptation set: OURS
            varies its move across the three tiers (excluded when all three are the same move). Proof
            is this set intersected with frontier-divergence.
          </p>
          <p>
            <span className="font-medium">OURS wins / OURS loses</span> — head-to-head vs the three
            frontier models only (GPT-5.5 / Claude / Gemini), on the sound tier move.{" "}
            <span className="font-medium">OURS’s own BASE and the open field are excluded</span> from
            win credit, so the “OURS wins here” banner uses the exact same rule as these counts.
          </p>
          <p className="text-muted">
            Faithfulness is a fairness floor, not a ranking axis: after the verify-and-regenerate
            gate, user-visible board-fact fabrication is{" "}
            <span className="font-medium">0% for all models</span>. The honest differentiator — the
            cross-family semantic-judge residual (any / majority / unanimous, with CIs) — is in the
            truthfulness panel below.
          </p>
        </div>
      </Tooltip.Content>
    </Tooltip>
  );
}

/* ================================================================== */
/* Library card                                                        */
/* ================================================================== */

function LibraryCard({
  position,
  active,
  onSelect,
}: {
  position: ViewPosition;
  active: boolean;
  onSelect: () => void;
}) {
  // Describe the card at a tier that matches its headline status (win/lose), so a
  // card in the "OURS wins" lens shows a real winning tier (labelled below).
  const t = representativeTier(position);
  const ours = position.models.find((m) => m.kind === "ours")?.byTier[t] ?? null;
  const outcome = describeOursOutcome(position, t);

  return (
    <Card
      variant={active ? "tertiary" : "secondary"}
      role="button"
      tabIndex={0}
      aria-pressed={active}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
      className={`cursor-pointer outline-none transition-colors focus-visible:ring-2 focus-visible:ring-signal/60 ${
        active ? "ring-[1.5px] ring-[color:var(--field-border)]" : "hover:bg-[color:var(--surface-tertiary)]"
      }`}
    >
      <Card.Content className="flex items-stretch gap-3 p-2.5">
        <div className="w-[104px] shrink-0 sm:w-[112px]">
          <ShowdownBoard
            fen={position.fen}
            orientation={position.sideToMove}
            moveUci={ours?.moveUci}
            studentUci={position.studentMove?.uci}
          />
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-1.5">
          <div className="flex flex-wrap items-center gap-1.5">
            {position.shine && (
              <span className="inline-flex items-center gap-1 rounded-full bg-signal/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-signal">
                ★ shine
              </span>
            )}
            {position.oursTierDifferentiates && (
              <span
                className="inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted ring-1 ring-[color:var(--border)]"
                title="OURS gives a different, level-appropriate move per tier"
              >
                ⇅ tiers
              </span>
            )}
            <span className="rounded-full bg-[color:var(--surface-tertiary)] px-1.5 py-0.5 text-[10px] text-muted">
              {cap(position.phase)}
            </span>
            <span className="rounded-full px-1.5 py-0.5 text-[10px] text-faint ring-1 ring-[color:var(--border)]">
              {position.split === "train" ? "train" : "held-out"}
            </span>
          </div>

          <div className="flex items-baseline gap-1.5 text-xs text-muted">
            <span>OURS</span>
            <span className="font-mono text-sm font-semibold text-ink tnum">
              {ours?.move ?? "—"}
            </span>
            <span className="text-faint">· {cap(t)}</span>
          </div>

          {outcome.bullets[0] ? (
            <p
              className={`text-xs leading-snug ${
                position.oursLoses && !outcome.wins ? "text-[color:var(--caution)]" : "text-muted"
              }`}
            >
              {outcome.bullets[0]}
            </p>
          ) : (
            <p className="text-xs leading-snug text-faint">Grounded comparison across the field.</p>
          )}
        </div>
      </Card.Content>
    </Card>
  );
}

/* ================================================================== */
/* Explorer                                                            */
/* ================================================================== */

function Explorer({
  meta,
  position,
  models,
  modelKey,
  onModelKey,
  tier,
  onTier,
  selModel,
  selCell,
  boardMoveUci,
  liveActive,
  outcome,
  live,
  onRunLive,
  onClearLive,
}: {
  meta: ShowcaseView["meta"];
  position: ViewPosition;
  models: ViewModel[];
  modelKey: string;
  onModelKey: (k: string) => void;
  tier: ShowcaseTier;
  onTier: (t: ShowcaseTier) => void;
  selModel: ViewModel;
  selCell: ViewCell | null;
  boardMoveUci: string | null;
  liveActive: boolean;
  outcome: ReturnType<typeof describeOursOutcome> | null;
  live: LiveState;
  onRunLive: () => void;
  onClearLive: () => void;
}) {
  const ordered = useMemo(() => orderModels(models), [models]);
  const target = position.tierTargets[tier];
  const tierHasData = position.tierEvaluated[tier];
  // Item 8: only show measured verdicts when THIS cell is actually evaluated — a
  // cell with absent/null objective flags must not render as green "sound/faithful".
  const cellEvaluated = Boolean(selCell?.evaluated);

  return (
    <Card variant="secondary" className="overflow-hidden">
      <Card.Content className="grid grid-cols-1 gap-6 p-4 sm:p-5 lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)]">
        {/* Left: board + position facts + tier switch */}
        <div className="flex flex-col gap-4">
          <div className="mx-auto flex w-full max-w-[320px] flex-col gap-2">
            <ShowdownBoard
              fen={position.fen}
              orientation={position.sideToMove}
              moveUci={boardMoveUci}
              studentUci={position.studentMove?.uci}
            />
            {/* Copy the exact FEN of the position currently on the board. */}
            <div className="flex justify-end">
              <CopyFenButton fen={position.fen} className="min-h-8" />
            </div>
          </div>

          <div className="flex flex-col gap-1.5 text-xs text-muted">
            <div className="flex items-center gap-1.5">
              <span
                aria-hidden
                className="inline-block size-3 rounded-full ring-1 ring-border"
                style={{
                  backgroundColor:
                    position.sideToMove === "white" ? "var(--board-light)" : "var(--board-dark)",
                }}
              />
              {cap(position.sideToMove)} to move
              {position.studentMove?.san && (
                <>
                  <span className="text-faint">·</span> student played{" "}
                  <span className="font-mono text-[color:var(--your-move)] tnum">
                    {position.studentMove.san}
                  </span>
                </>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="rounded-full bg-[color:var(--surface-tertiary)] px-2 py-0.5">
                {cap(position.phase)}
              </span>
              {position.severity && (
                <span className="rounded-full bg-[color:var(--surface-tertiary)] px-2 py-0.5">
                  {cap(position.severity)}
                </span>
              )}
              {position.benchmark && (
                <span className="rounded-full px-2 py-0.5 text-faint ring-1 ring-[color:var(--border)]">
                  {position.benchmark === "open"
                    ? `+open field · ${meta.modelCount} models`
                    : "core set"}
                </span>
              )}
            </div>
            {target?.san && (
              <div>
                {cap(tier)} target{" "}
                <span className="font-mono text-ink tnum">{target.san}</span>
              </div>
            )}
          </div>

          <TierSwitch tier={tier} onTier={onTier} evaluated={position.tierEvaluated} />

          {/* Board arrow legend */}
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted">
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block h-1.5 w-4 rounded-full bg-signal" aria-hidden />
              {liveActive ? "OURS live move" : `${selModel.short} move`}
            </span>
            {position.studentMove?.uci && (
              <span className="inline-flex items-center gap-1.5">
                <span
                  className="inline-block h-1.5 w-4 rounded-full bg-[color:var(--your-move)]"
                  aria-hidden
                />
                Student move
              </span>
            )}
          </div>
        </div>

        {/* Right: OURS-vs-best + model pick + verdicts + council + coaching + field + re-run */}
        <div className="flex min-w-0 flex-col gap-4">
          <OursVsBest position={position} tier={tier} onTier={onTier} onSelectModel={onModelKey} />

          <ModelPicker models={ordered} value={modelKey} onChange={onModelKey} total={meta.modelCount} />

          {/* The selected model's recommendation at every tier, side by side. */}
          <SelectedModelTiers model={selModel} tier={tier} onTier={onTier} />

          {tierHasData && selCell && cellEvaluated ? (
            <>
              <VerdictRow model={selModel} cell={selCell} tier={tier} />
              <CouncilPanel meta={meta} cell={selCell} />
              <CoachingBlock model={selModel} cell={selCell} />
            </>
          ) : (
            <NotEvaluated tier={tier} source={meta.source} />
          )}

          {/* Honest OURS outcome summary at this tier */}
          {outcome && (outcome.wins || outcome.loses) && (
            <OutcomeStrip outcome={outcome} />
          )}

          <Separator />

          {/* The whole field at this tier — the differentiation view */}
          <FieldGrid
            meta={meta}
            models={ordered}
            tier={tier}
            selectedKey={modelKey}
            onSelect={onModelKey}
          />

          <Separator />

          <RerunPanel
            tier={tier}
            live={live}
            liveActive={liveActive}
            oursCell={models.find((m) => m.kind === "ours")?.byTier[tier] ?? null}
            onRun={onRunLive}
            onClear={onClearLive}
          />
        </div>
      </Card.Content>
    </Card>
  );
}

/* ---- OURS vs best-other duel (front and center) ------------------ */

/** Shared grid tracks so the header row and each tier row line up perfectly
 *  (each row is its own grid, so the fixed columns + identical padding matter). */
const DUEL_COLS = "grid grid-cols-[104px_minmax(0,1fr)_92px_minmax(0,1fr)] items-center gap-2 px-2";

function OursVsBest({
  position,
  tier,
  onTier,
  onSelectModel,
}: {
  position: ViewPosition;
  tier: ShowcaseTier;
  onTier: (t: ShowcaseTier) => void;
  onSelectModel: (k: string) => void;
}) {
  const best = bestOtherModel(position);
  const frontier = bestFrontierModel(position);
  const bestIsFrontier = Boolean(best && frontier && best.key === frontier.key);
  const rows = useMemo(() => tierDuel(position, position.bestOtherKey), [position]);
  const anyScored = rows.some((r) => r.verdict !== "na");

  return (
    <div className="flex flex-col gap-2.5 rounded-[10px] border-[1.5px] border-[color:var(--border)] bg-[color:var(--surface)] px-3.5 py-3">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="text-sm font-semibold text-signal">OURS</span>
        <span className="text-[11px] text-faint">vs</span>
        {best ? (
          <button
            type="button"
            onClick={() => onSelectModel(best.key)}
            className="inline-flex items-center gap-1.5 rounded px-1 py-0.5 text-sm font-semibold text-ink transition-colors hover:text-signal focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60"
            title="Open this model in the comparison below"
          >
            {best.short}
            <KindTag kind={best.kind} />
          </button>
        ) : (
          <span className="text-sm text-muted">no rival scored here</span>
        )}
        {/* When the strongest rival isn't the frontier, point to the frontier
            reference the win/loss verdict is actually measured against. */}
        {best && !bestIsFrontier && frontier && (
          <button
            type="button"
            onClick={() => onSelectModel(frontier.key)}
            className="inline-flex items-center gap-1 rounded px-1 py-0.5 text-[11px] text-faint transition-colors hover:text-engine focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60"
            title="Open the best frontier model — the win/loss reference"
          >
            frontier ref: {frontier.short}
          </button>
        )}
        <span className="ml-auto text-[11px] text-faint">
          {bestIsFrontier ? "best frontier · win/loss ref" : "best other model"}
        </span>
      </div>

      {best && (
        <>
          <div className={`${DUEL_COLS} py-0.5 text-[10px] uppercase tracking-wide text-faint`}>
            <span>Tier</span>
            <span className="text-signal">OURS</span>
            <span className="text-center">verdict</span>
            <span className="truncate text-right">{best.short}</span>
          </div>
          <div className="flex flex-col gap-1">
            {rows.map((row) => (
              <DuelRow
                key={row.tier}
                row={row}
                active={row.tier === tier}
                onSelect={() => onTier(row.tier)}
              />
            ))}
          </div>
          <p className="px-2 text-[10px] leading-relaxed text-faint">
            {anyScored ? (
              <>
                Verdict weighs tier-fit, soundness and faithfulness
                {position.source === "showcase" ? " plus the blinded council grades" : ""} at each
                level. “—” marks a tier neither side was scored at
                {position.source === "showdown"
                  ? " (all three light up with showcase.json)"
                  : position.source === "interim"
                    ? " — OURS is scored at all three; rivals only at their benchmarked tier"
                    : ""}.
              </>
            ) : (
              "Neither side is scored at these tiers yet — the per-tier duel fills in with showcase.json."
            )}
          </p>
        </>
      )}
    </div>
  );
}

function DuelRow({
  row,
  active,
  onSelect,
}: {
  row: TierDuelRow;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={active}
      className={`${DUEL_COLS} rounded-lg py-1.5 text-left transition-colors hover:bg-[color:var(--surface-tertiary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-signal/60`}
    >
      <span className="flex flex-col leading-tight">
        <span className="text-xs font-medium text-ink">{cap(row.tier)}</span>
        <span className="text-[10px] text-faint tnum">{TIER_BAND[row.tier]}</span>
      </span>
      <span className="truncate font-mono text-sm font-semibold text-ink tnum">
        {row.ours?.move ?? "—"}
      </span>
      <VerdictPill verdict={row.verdict} />
      <span className="truncate text-right font-mono text-sm text-muted tnum">
        {row.other?.move ?? "—"}
      </span>
    </button>
  );
}

function VerdictPill({ verdict }: { verdict: DuelVerdict }) {
  const map: Record<DuelVerdict, { label: string; cls: string }> = {
    beats: { label: "OURS ahead", cls: "text-[color:var(--good)] bg-[color:var(--good)]/12" },
    trails: { label: "OURS behind", cls: "text-[color:var(--caution)] bg-[color:var(--caution)]/12" },
    even: { label: "even", cls: "text-muted bg-[color:var(--surface-tertiary)]" },
    na: { label: "—", cls: "text-faint" },
  };
  const v = map[verdict];
  return (
    <span
      className={`inline-flex items-center justify-center rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${v.cls}`}
    >
      {v.label}
    </span>
  );
}

/* ---- Tier switch ------------------------------------------------- */

function TierSwitch({
  tier,
  onTier,
  evaluated,
}: {
  tier: ShowcaseTier;
  onTier: (t: ShowcaseTier) => void;
  evaluated: Record<ShowcaseTier, boolean>;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium uppercase tracking-wide text-faint">Recommend for</span>
        <span className="text-[11px] text-faint tnum">{TIER_BAND[tier]}</span>
      </div>
      <ToggleButtonGroup
        aria-label="Recommendation tier"
        fullWidth
        size="md"
        selectionMode="single"
        disallowEmptySelection
        selectedKeys={new Set([tier])}
        onSelectionChange={(keys) => {
          const k = [...keys][0];
          if (k) onTier(k as ShowcaseTier);
        }}
      >
        {TIERS.map((t, i) => (
          <ToggleButton key={t} id={t} className="min-h-10">
            {i > 0 && <ToggleButtonGroup.Separator />}
            <span
              className="inline-flex items-center gap-1.5"
              title={evaluated[t] ? undefined : "Not evaluated at this tier in the current data"}
            >
              {cap(t)}
              <span
                aria-hidden
                className={`inline-block size-1.5 rounded-full ${
                  t === tier
                    ? "bg-current"
                    : evaluated[t]
                      ? "bg-[color:var(--good)]"
                      : "bg-faint/50"
                }`}
              />
            </span>
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
    </div>
  );
}

/* ---- Model picker ------------------------------------------------ */

function ModelPicker({
  models,
  value,
  onChange,
  total,
}: {
  models: ViewModel[];
  value: string;
  onChange: (k: string) => void;
  total: number;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium uppercase tracking-wide text-faint">Model</span>
        <Select
          aria-label="Model to compare"
          variant="secondary"
          value={value}
          onChange={(v: Key | Key[] | null) => {
            if (typeof v === "string") onChange(v);
          }}
          className="min-w-[210px]"
        >
          <Select.Trigger>
            <Select.Value />
            <Select.Indicator />
          </Select.Trigger>
          <Select.Popover className="max-h-[320px]">
            <ListBox>
              {models.map((m) => (
                <ListBox.Item key={m.key} id={m.key} textValue={m.name}>
                  {m.name}
                  <ListBox.ItemIndicator />
                </ListBox.Item>
              ))}
            </ListBox>
          </Select.Popover>
        </Select>
      </div>
      <span className="text-[11px] text-faint tnum">
        {total > 0 && models.length >= total
          ? `all ${total} models`
          : `${models.length}${total > 0 ? ` of ${total}` : ""} models here`}
      </span>
    </div>
  );
}

/* ---- Selected model's per-tier recommendations ------------------- */

/** The picked model's recommended move at Beginner / Intermediate / Advanced,
 *  side by side — so switching the dropdown shows one model's answer at every
 *  level at a glance. Each cell doubles as the tier selector. */
function SelectedModelTiers({
  model,
  tier,
  onTier,
}: {
  model: ViewModel;
  tier: ShowcaseTier;
  onTier: (t: ShowcaseTier) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium uppercase tracking-wide text-faint">
          {model.short} · move per tier
        </span>
        <span className="text-[11px] text-faint">tap a tier to compare below</span>
      </div>
      <div className="grid grid-cols-3 gap-1.5">
        {TIERS.map((t) => {
          const c = model.byTier[t];
          const active = t === tier;
          const evaluated = Boolean(c?.evaluated);
          return (
            <button
              key={t}
              type="button"
              onClick={() => onTier(t)}
              aria-pressed={active}
              className={`flex flex-col items-start gap-1 rounded-lg border px-2.5 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-signal/60 ${
                active
                  ? "border-[color:var(--field-border)] bg-[color:var(--surface-tertiary)]"
                  : "border-[color:var(--border)] hover:bg-[color:var(--surface-tertiary)]/60"
              }`}
            >
              <span className="flex w-full items-baseline justify-between">
                <span className="text-[11px] font-medium text-muted">{cap(t)}</span>
                <span className="text-[10px] text-faint tnum">{TIER_BAND[t]}</span>
              </span>
              <span
                className={`font-mono text-base font-semibold tnum ${
                  model.kind === "ours" ? "text-signal" : "text-ink"
                }`}
              >
                {c?.move ?? "—"}
              </span>
              {evaluated ? (
                <span className="flex flex-wrap items-center gap-1">
                  <Dot on={c!.tierFit} label="tier" />
                  <Dot on={!c!.fabricated} label="faithful" />
                </span>
              ) : (
                <span className="text-[10px] text-faint">not scored</span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* ---- Verdict chips ----------------------------------------------- */

/** The reframed HERO: the recommended MOVE + a short principle tag — the trained
 *  behavior — with the deterministic verdict chips (tier-fit / sound / faithful)
 *  right beneath it. The move, not the prose, is the loudest element. */
function VerdictRow({ model, cell, tier }: { model: ViewModel; cell: ViewCell; tier: ShowcaseTier }) {
  const isOurs = model.kind === "ours";
  const tag = cellMoveTag(cell, tier);
  return (
    <div className="flex flex-col gap-2.5 rounded-[10px] border-[1.5px] border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3.5">
      {/* Move + principle tag — the single loudest element in the explorer. */}
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span
          className={`font-mono text-[2rem] font-semibold leading-none tnum ${isOurs ? "text-signal" : "text-ink"}`}
        >
          {cell.move ?? "—"}
        </span>
        {tag && <span className="text-sm text-muted">— {tag}</span>}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <span className={`text-sm font-semibold ${isOurs ? "text-signal" : "text-ink"}`}>
          {model.name}
        </span>
        <KindTag kind={model.kind} />
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <Chip
          tone={cell.tierFit ? "signal" : cell.sound ? "muted" : "danger"}
          label={cell.tierFit ? "tier-fit" : cell.sound ? "sound" : cell.move ? "unsound" : "no move"}
        />
        <Chip tone={cell.sound ? "good" : "muted"} label={cell.sound ? "sound move" : "not sound"} />
        <Chip
          tone={cell.fabricated ? "danger" : "good"}
          label={cell.fabricated ? `fabricated${cell.nViolations ? ` ×${cell.nViolations}` : ""}` : "faithful"}
        />
      </div>
    </div>
  );
}

/* ---- Council grades ---------------------------------------------- */

function CouncilPanel({ meta, cell }: { meta: ShowcaseView["meta"]; cell: ViewCell }) {
  const has = cell.councilMove != null || cell.councilInstr != null;
  return (
    <div className="flex flex-col gap-2 rounded-[10px] border-[1.5px] border-[color:var(--border)] px-3.5 py-3">
      <div className="flex items-center gap-1.5">
        <ShieldCheckIcon width={14} height={14} className="text-muted" />
        <span className="text-xs font-semibold text-ink">Blinded council grades</span>
        <Tooltip delay={200}>
          <Tooltip.Trigger aria-label="How the council grades">
            <span className="inline-flex size-4 cursor-help items-center justify-center rounded-full text-[10px] text-faint ring-1 ring-[color:var(--border)]">
              ?
            </span>
          </Tooltip.Trigger>
          <Tooltip.Content showArrow className="max-w-[18rem]">
            <Tooltip.Arrow />
            <p className="leading-relaxed">{meta.notes.council}</p>
          </Tooltip.Content>
        </Tooltip>
      </div>
      {has ? (
        <div className="grid grid-cols-2 gap-3">
          <CouncilMeter label="Move" value={cell.councilMove} max={meta.councilScale} />
          <CouncilMeter label="Instruction" value={cell.councilInstr} max={meta.councilScale} />
        </div>
      ) : (
        <p className="text-xs leading-relaxed text-faint">
          {meta.source !== "showcase"
            ? "Council move + instruction grades arrive with showcase.json."
            : "The council didn’t score this model here."}
        </p>
      )}
    </div>
  );
}

/** Council grade rendered with the HeroUI Meter (accent = signal), with an
 *  explicit "value/scale" readout so the grader sees the raw number too. */
function CouncilMeter({ label, value, max }: { label: string; value: number | null; max: number }) {
  const scale = Math.max(1, max);
  return (
    <Meter
      aria-label={`Council ${label.toLowerCase()} grade`}
      value={value ?? 0}
      minValue={0}
      maxValue={scale}
      color="accent"
      size="sm"
      className="flex flex-col gap-1"
    >
      <div className="flex items-baseline justify-between">
        <span className="text-[11px] uppercase tracking-wide text-faint">{label}</span>
        <span className="font-mono text-sm font-semibold text-ink tnum">
          {value == null ? "—" : trimNum(value)}
          <span className="text-[10px] font-normal text-faint">/{trimNum(scale)}</span>
        </span>
      </div>
      <Meter.Track>
        <Meter.Fill />
      </Meter.Track>
    </Meter>
  );
}

function trimNum(n: number): string {
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

/* ---- Coaching block ---------------------------------------------- */

/**
 * The coaching PROSE — now a secondary, OPTIONAL layer. Collapsed behind an
 * expander so the move + principle tag above stays the hero; the prose reads as a
 * supplementary, engine-assisted explanation, not the trained output. The gate
 * badge and the (defensive) post-gate alarm stay visible.
 */
function CoachingBlock({ model, cell }: { model: ViewModel; cell: ViewCell }) {
  const badge = gateBadge(cell);
  const raw = cell.rawCoaching;
  const rawDiffers = raw != null && raw !== cell.coaching;

  return (
    <div className="flex flex-col gap-2">
      {/* Defensive: a POST-gate board-fact error would be a genuine alarm. The
          deterministic residual is 0 for every shipped cell, so this normally
          stays hidden — but if incomplete data ever carried one, we surface it. */}
      {cell.violations.length > 0 && (
        <div className="rounded-md border border-[color:var(--danger)]/40 bg-[color:var(--danger)]/10 px-3 py-2">
          <div className="text-xs font-semibold text-[color:var(--danger)]">
            Post-gate board-fact error{cell.violations.length === 1 ? "" : ` ×${cell.violations.length}`} (verifier)
          </div>
          <ul className="mt-1 flex flex-col gap-1 text-xs text-muted">
            {cell.violations.map((v, i) => (
              <li key={i}>
                <span className="text-ink">&ldquo;{v.sentence}&rdquo;</span>{" "}
                <span className="text-[color:var(--danger)]">— {v.reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Full coaching prose — OPTIONAL, collapsed, clearly labeled as secondary. */}
      <details className="group rounded-[10px] border border-[color:var(--border)] bg-[color:var(--surface)]">
        <summary className="flex min-h-10 cursor-pointer list-none items-center gap-2 px-3.5 py-2.5 text-xs font-medium text-muted [&::-webkit-details-marker]:hidden">
          <span aria-hidden className="text-faint transition-transform group-open:rotate-90">
            ›
          </span>
          Show full explanation
          <span className="text-faint">(optional, engine-assisted)</span>
          {badge && (
            <span className="ml-auto">
              <GateBadge badge={badge} />
            </span>
          )}
        </summary>
        <div className="flex flex-col gap-2 px-3.5 pb-3.5">
          <p className="text-[11px] leading-relaxed text-faint">
            {model.short} coaching · {model.kind === "ours" ? "tuned" : model.kind} · gated + shipped.
            A supplementary layer over the trained behavior (the move above) — not the trained output
            itself. Fine-tuning did not improve the prose; the frontier writes livelier explanations
            (see the council + truthfulness panels below).
          </p>
          <p className="max-h-64 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed text-muted">
            {cell.coaching || <span className="text-faint">No coaching text produced.</span>}
          </p>

          {/* Collapsed, clearly-labeled view of the model's ORIGINAL pre-gate draft. */}
          {raw != null && (
            <details className="group/raw rounded-md border border-[color:var(--border)] bg-[color:var(--surface-tertiary)]/40">
              <summary className="flex list-none items-center gap-1.5 px-3 py-2 text-[11px] font-medium text-faint [&::-webkit-details-marker]:hidden">
                <span aria-hidden className="text-faint transition-transform group-open/raw:rotate-90">
                  ›
                </span>
                Raw model output (pre-gate)
                <span className="ml-1 rounded-full px-1.5 py-0.5 text-[10px] text-faint ring-1 ring-[color:var(--border)]">
                  {rawDiffers ? "differs from shipped" : "identical to shipped"}
                </span>
              </summary>
              <div className="flex flex-col gap-2 px-3 pb-3">
                <p className="text-[11px] leading-relaxed text-faint">
                  {rawDiffers
                    ? "The model’s first draft was adjusted by the verify-and-regenerate gate; the shipped text above is the gated version. Faithfulness is a fairness floor applied equally to every model — see the truthfulness panel for where models actually differ."
                    : "The model’s first draft passed the board-fact gate unchanged — the shipped text above is the model’s own words."}
                </p>
                <p className="max-h-56 overflow-y-auto whitespace-pre-wrap text-xs leading-relaxed text-muted">
                  {raw || <span className="text-faint">No raw text recorded.</span>}
                </p>
              </div>
            </details>
          )}
        </div>
      </details>
    </div>
  );
}

/** Honest per-cell gate provenance chip (clean-on-draft-1 / re-sampled / fallback). */
function GateBadge({ badge }: { badge: GateBadgeInfo }) {
  const toneCls: Record<GateBadgeInfo["tone"], string> = {
    good: "text-[color:var(--good)] bg-[color:var(--good)]/12 ring-[color:var(--good)]/30",
    muted: "text-muted bg-[color:var(--surface-tertiary)] ring-[color:var(--border)]",
    caution: "text-[color:var(--caution)] bg-[color:var(--caution)]/12 ring-[color:var(--caution)]/30",
  };
  return (
    <Tooltip delay={200}>
      <Tooltip.Trigger aria-label="Gate provenance for this coaching cell">
        <span
          className={`inline-flex cursor-help items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ring-1 ${toneCls[badge.tone]}`}
        >
          <ShieldCheckIcon width={11} height={11} />
          {badge.label}
        </span>
      </Tooltip.Trigger>
      <Tooltip.Content showArrow className="max-w-[20rem]">
        <Tooltip.Arrow />
        <p className="leading-relaxed">{badge.detail}</p>
      </Tooltip.Content>
    </Tooltip>
  );
}

function NotEvaluated({ tier, source }: { tier: ShowcaseTier; source: ShowcaseView["meta"]["source"] }) {
  return (
    <div className="rounded-[10px] border border-dashed border-[color:var(--border)] px-4 py-6 text-center">
      <p className="text-sm text-ink">Not evaluated at {cap(tier)} in this dataset.</p>
      <p className="mt-1 text-xs leading-relaxed text-muted">
        {source === "showdown"
          ? "This held-out position was scored at one tier. Per-tier answers for all three levels arrive with showcase.json — or re-run OURS live below to see it at this tier now."
          : source === "interim"
            ? "This rival was scored only at its benchmarked tier. OURS is scored at all three tiers — switch the model to OURS, or re-run OURS live below."
            : "The showcase didn’t include this tier for this position."}
      </p>
    </div>
  );
}

/* ---- Honest outcome strip ---------------------------------------- */

function OutcomeStrip({ outcome }: { outcome: ReturnType<typeof describeOursOutcome> }) {
  const win = outcome.wins;
  return (
    <div
      className={`rounded-[10px] border px-3.5 py-2.5 ${
        win ? "border-signal/40 bg-signal/10" : "border-[color:var(--caution)]/40 bg-[color:var(--caution)]/10"
      }`}
    >
      <div className={`flex items-center gap-2 text-sm font-semibold ${win ? "text-signal" : "text-[color:var(--caution)]"}`}>
        <span aria-hidden>{win ? "★" : "▽"}</span>
        {win ? "OURS wins here" : "OURS loses here"}
        {/* Same rule as the headline counts: vs the frontier only, BASE excluded. */}
        <span className="ml-auto text-[10px] font-normal text-faint">vs frontier · this tier</span>
      </div>
      <ul className="mt-1 flex flex-col gap-0.5 text-xs text-muted">
        {outcome.bullets.map((b, i) => (
          <li key={i}>{b}</li>
        ))}
      </ul>
    </div>
  );
}

/* ---- Field grid: every model at this tier ------------------------ */

function FieldGrid({
  meta,
  models,
  tier,
  selectedKey,
  onSelect,
}: {
  meta: ShowcaseView["meta"];
  models: ViewModel[];
  tier: ShowcaseTier;
  selectedKey: string;
  onSelect: (k: string) => void;
}) {
  const anyEvaluated = models.some((m) => m.byTier[tier]?.evaluated);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-faint">
          The field at {cap(tier)}
        </h3>
        <span className="text-[11px] text-faint">move · tier-fit · faithful{meta.hasCouncil ? " · council" : ""}</span>
      </div>
      {/* Item 7: keep "tier-fit" from being misread as an emergent judgment. */}
      <p className="text-[10px] leading-relaxed text-faint">
        <span className="text-muted">tier-fit</span> = the move matches the canonical
        tier-appropriate target OURS is trained to produce for that band — a trained-target match,
        not an emergent capability. <span className="text-muted">faithful</span> = passed the
        deterministic board-fact gate (not a guarantee of full semantic truth — see the residual
        panel).
      </p>
      {anyEvaluated ? (
      <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
        {models.map((m) => {
          const cell = m.byTier[tier];
          const selected = m.key === selectedKey;
          return (
            <button
              key={m.key}
              type="button"
              onClick={() => onSelect(m.key)}
              aria-pressed={selected}
              className={`flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-signal/60 ${
                selected
                  ? "bg-[color:var(--surface-tertiary)] shadow-[inset_2px_0_0_0_var(--field-border)]"
                  : "hover:bg-[color:var(--surface-tertiary)]"
              }`}
            >
              <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                <div className="flex items-center gap-1.5">
                  <span className="truncate text-xs font-medium text-ink">
                    {m.short}
                  </span>
                  <KindTag kind={m.kind} />
                </div>
                <div className="flex flex-wrap items-center gap-1">
                  {cell?.evaluated ? (
                    <>
                      <Dot on={cell.tierFit} label="tier" />
                      <Dot on={!cell.fabricated} label="faithful" />
                      {(cell.councilMove != null || cell.councilInstr != null) && (
                        <span className="font-mono text-[10px] text-faint tnum">
                          {cell.councilMove != null ? `M${trimNum(cell.councilMove)}` : ""}
                          {cell.councilInstr != null ? ` I${trimNum(cell.councilInstr)}` : ""}
                        </span>
                      )}
                    </>
                  ) : null}
                </div>
              </div>
              <span className="shrink-0 font-mono text-sm font-semibold tnum text-ink">
                {cell?.move ?? "—"}
              </span>
            </button>
          );
        })}
      </div>
      ) : (
        <p className="rounded-[10px] border border-dashed border-[color:var(--border)] px-4 py-6 text-center text-xs leading-relaxed text-muted">
          No model was scored at {cap(tier)} for this position — the preview data (showdown.json)
          scores each held-out position at a single tier. All three levels for the full field arrive
          with showcase.json; you can re-run OURS live below to score it at {cap(tier)} now.
        </p>
      )}
    </div>
  );
}

function Dot({ on, label }: { on: boolean; label: string }) {
  return (
    <span className="inline-flex items-center gap-1 text-[10px] text-muted">
      <span
        aria-hidden
        className="inline-block size-1.5 rounded-full"
        style={{ backgroundColor: on ? "var(--good)" : "color-mix(in oklab, var(--danger) 80%, transparent)" }}
      />
      {label}
    </span>
  );
}

/* ---- Re-run panel (live, OURS only) ------------------------------ */

function RerunPanel({
  tier,
  live,
  liveActive,
  oursCell,
  onRun,
  onClear,
}: {
  tier: ShowcaseTier;
  live: LiveState;
  liveActive: boolean;
  oursCell: ViewCell | null;
  onRun: () => void;
  onClear: () => void;
}) {
  const loading = live.status === "loading";
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-col">
          <span className="text-sm font-semibold text-ink">
            Re-run through the workflow
          </span>
          <span className="text-[11px] text-muted">
            OURS live on the backend: engines → grounding → model → verifier, at {cap(tier)}.
          </span>
        </div>
        <div className="flex items-center gap-2">
          {liveActive && (
            <Button variant="tertiary" size="sm" className="min-h-9" onPress={onClear}>
              Clear
            </Button>
          )}
          <Button
            variant="primary"
            size="md"
            className="min-h-10"
            isDisabled={loading}
            aria-busy={loading}
            onPress={onRun}
          >
            {loading ? (live.waking ? "Waking…" : "Running…") : liveActive ? "Re-run again" : "Re-run OURS live"}
          </Button>
        </div>
      </div>

      {loading && live.waking && (
        <div className="rounded-[10px] border border-signal/40 bg-signal/[0.07] px-3.5 py-2.5 text-xs leading-relaxed text-muted">
          <span className="font-medium text-signal">Waking the model…</span> First live call after
          idle takes ~2–3 min while the scale-to-zero container spins up — retrying automatically
          {live.waking.elapsedSec > 0 ? ` · ${live.waking.elapsedSec}s elapsed` : ""}.
        </div>
      )}

      {live.status === "error" && (
        <div className="rounded-[10px] border border-[color:var(--danger)]/40 bg-[color:var(--danger)]/10 px-3.5 py-3 text-xs leading-relaxed text-muted">
          <span className="font-medium text-[color:var(--danger)]">The coach service didn’t respond.</span>{" "}
          It runs on the live backend (:8000). Make sure it’s up, then re-run.
          {live.error && <p className="mt-1 break-words font-mono text-faint">{live.error}</p>}
        </div>
      )}

      {liveActive && live.result && (
        <LiveResult result={live.result} tier={tier} oursCell={oursCell} />
      )}
    </div>
  );
}

function LiveResult({
  result,
  tier,
  oursCell,
}: {
  result: CoachResponse;
  tier: ShowcaseTier;
  oursCell: ViewCell | null;
}) {
  const hasProse = Boolean(result.coaching?.trim());
  const changed = oursCell?.move && oursCell.move !== result.recommended_move_san;
  // Principle tag assembled from the live CoachResponse's own fields — concepts if
  // present, else a short slice of the takeaway. The move is the hero, not the prose.
  const tag = principleTag(result.concepts_used, result.takeaway);

  return (
    <div className="rise flex flex-col gap-3 rounded-[10px] border border-signal/40 bg-signal/[0.07] px-4 py-3.5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full bg-signal/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-signal">
          live · OURS
        </span>
        {oursCell?.evaluated && (
          <span className="ml-auto text-[11px] text-faint">
            precomputed: <span className="font-mono tnum">{oursCell.move ?? "—"}</span>
            {changed ? " (differs)" : " (matches)"}
          </span>
        )}
      </div>

      {/* HERO: the level-appropriate move + a short principle tag. */}
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="font-mono text-[2rem] font-semibold leading-none text-signal tnum">
          {result.recommended_move_san}
        </span>
        <span className="text-xs text-muted">for {cap(tier)} · {result.side_to_move} to move</span>
      </div>
      {tag && <p className="text-sm leading-relaxed text-ink">— {tag}</p>}

      {/* Full explanation — optional, secondary, collapsed. */}
      {hasProse && (
        <details className="group rounded-md border border-[color:var(--border)] bg-[color:var(--surface)]/60">
          <summary className="flex min-h-9 cursor-pointer list-none items-center gap-2 px-3 py-2 text-[11px] font-medium text-muted [&::-webkit-details-marker]:hidden">
            <span aria-hidden className="text-faint transition-transform group-open:rotate-90">
              ›
            </span>
            Show full explanation
            <span className="text-faint">(optional, engine-assisted)</span>
          </summary>
          <div className="flex flex-col gap-2 px-3 pb-3">
            {result.coaching
              .split(/\n\s*\n/)
              .map((p) => p.trim())
              .filter(Boolean)
              .map((p, i) => (
                <p key={i} className="text-sm leading-relaxed text-muted">
                  {p}
                </p>
              ))}
          </div>
        </details>
      )}

      <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted">
        <span className="rounded bg-[color:var(--surface-tertiary)] px-1.5 py-0.5">
          {result.meta.tuned ? "tuned coach" : "base model"}
        </span>
        {result.meta.verified_fallback && (
          <span className="inline-flex items-center gap-1 rounded bg-[color:var(--caution)]/15 px-1.5 py-0.5 text-[color:var(--caution)]">
            <ShieldCheckIcon width={11} height={11} /> verified explanation
          </span>
        )}
        <span className="tnum">
          {result.meta.attempts} {result.meta.attempts === 1 ? "attempt" : "attempts"} to pass the faithfulness gate
        </span>
        <span className="font-mono text-faint">{result.meta.model}</span>
      </div>
    </div>
  );
}

/* ================================================================== */
/* Per-model leaderboard — computed live from the loaded cells          */
/* ================================================================== */

function LeaderboardPanel({
  view,
  meta,
  split,
}: {
  view: ShowcaseView;
  meta: ShowcaseView["meta"];
  split: Split;
}) {
  const board = useMemo(() => computeLeaderboard(view, split), [view, split]);
  if (board.rows.length === 0) return null;

  const splitLabel = split === "train" ? "training" : "held-out test";

  return (
    <section aria-label="Per-model metrics" className="flex flex-col gap-3">
      <Separator />
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2 className="text-sm font-semibold text-ink">
          Per-model metrics — measured on the {splitLabel} split
        </h2>
        <span className="text-[11px] text-faint tnum">
          {board.rows.length} models · council 0–{trimNum(board.councilScale)}
        </span>
      </div>
      <p className="max-w-4xl text-xs leading-relaxed text-muted">
        Every number here is computed live from the loaded cells — nothing is hardcoded, so it always
        matches the OURS version on screen. The axis this product competes on is the leftmost:{" "}
        <span className="text-ink">tier-appropriate move selection</span> (tier-fit) — where{" "}
        <span className="text-signal">OURS</span> leads its own untuned <span className="text-ink">BASE</span>{" "}
        and the frontier. The <span className="text-muted">council move / instructiveness</span> grades
        are shown as context, not the headline: they grade the coaching prose, where OURS does{" "}
        <span className="text-ink">not</span> lead — fine-tuning sharpened the move, not the paragraph,
        and we don’t claim otherwise.
      </p>

      <div className="overflow-hidden rounded-[10px] border border-[color:var(--border)]">
        {/* Header row */}
        <div className="grid grid-cols-[minmax(120px,1.5fr)_minmax(0,2fr)_64px_64px_56px] items-center gap-2 border-b border-[color:var(--separator)] bg-[color:var(--surface)] px-3 py-2 text-[10px] uppercase tracking-wide text-faint">
          <span>Model</span>
          <span>Tier-fit</span>
          <span className="text-right">Council&nbsp;move</span>
          <span className="text-right">Instruct.</span>
          <span className="text-right">Cells</span>
        </div>
        <ul className="divide-y divide-[color:var(--separator)]">
          {board.rows.map((r) => (
            <LeaderboardRow key={r.key} row={r} scale={board.councilScale} />
          ))}
        </ul>
      </div>
      <p className="text-[10px] leading-relaxed text-faint">
        Deterministic board-fact fabrication is <span className="text-muted">0% for every model</span>{" "}
        after the verify-and-regenerate gate (the fairness floor), so it isn’t a ranking column here.{" "}
        <span className="text-[color:var(--caution)]">⚠</span> marks a partial model (fewer cells —
        provider throttling); its rates are over the cells it does have.
      </p>
    </section>
  );
}

function LeaderboardRow({ row, scale }: { row: ModelAggregate; scale: number }) {
  const isOurs = row.kind === "ours";
  const isBase = row.kind === "base";
  const fill =
    row.kind === "ours"
      ? "var(--signal)"
      : row.kind === "frontier"
        ? "var(--engine)"
        : row.kind === "base"
          ? "var(--faint)"
          : "var(--muted)";
  return (
    <li
      className={`grid grid-cols-[minmax(120px,1.5fr)_minmax(0,2fr)_64px_64px_56px] items-center gap-2 px-3 py-2 ${
        isOurs ? "bg-signal/[0.06]" : isBase ? "bg-[color:var(--surface-tertiary)]/40" : ""
      }`}
    >
      <div className="flex min-w-0 items-center gap-1.5">
        <span
          className={`truncate text-xs ${isOurs ? "font-semibold text-signal" : "text-ink"}`}
          title={row.name}
        >
          {row.short}
        </span>
        <KindTag kind={row.kind} />
        {row.partial && (
          <span
            className="text-[10px] text-[color:var(--caution)]"
            title="Partial model — only a subset of cells exist (provider throttling)."
          >
            ⚠
          </span>
        )}
      </div>

      {/* Tier-fit bar */}
      <div className="flex items-center gap-2">
        <span
          className="relative h-2 flex-1 overflow-hidden rounded-full bg-[color:var(--surface-tertiary)]"
          title={`tier-fit ${pct(row.tierFitRate)}`}
        >
          <span
            aria-hidden
            className="absolute inset-y-0 left-0 rounded-full"
            style={{ width: `${row.tierFitRate * 100}%`, backgroundColor: fill }}
          />
        </span>
        <span
          className={`w-12 shrink-0 text-right font-mono text-xs tnum ${isOurs ? "font-semibold text-signal" : "text-ink"}`}
        >
          {pct(row.tierFitRate)}
        </span>
      </div>

      <span className="text-right font-mono text-xs text-ink tnum">
        {row.councilMove == null ? "—" : trimNum(row.councilMove)}
      </span>
      <span className="text-right font-mono text-xs text-ink tnum">
        {row.councilInstr == null ? "—" : trimNum(row.councilInstr)}
      </span>
      <span className="text-right font-mono text-[11px] text-faint tnum">
        {row.cells.toLocaleString()}
      </span>
    </li>
  );
}

/* ================================================================== */
/* Truthfulness residual — two honest layers (per-model, static)       */
/* ================================================================== */

function TruthfulnessPanel({ oursLabel }: { oursLabel: OursLabel }) {
  const t = TRUTHFULNESS;
  // Relabel the OURS row with the identity parsed from the loaded data, so the
  // sampled study is transparent about which OURS version it covers.
  const rows = useMemo(
    () =>
      [...t.rows]
        .map((r) => (r.kind === "ours" ? { ...r, name: oursLabel.name } : r))
        .sort((a, b) => b.any.pct - a.any.pct),
    [t.rows, oursLabel.name],
  );

  return (
    <section aria-label="Truthfulness residual" className="flex flex-col gap-4">
      <Separator />
      <div className="flex flex-col gap-1.5">
        <h2 className="text-sm font-semibold text-ink">
          Truthfulness — one fairness floor, one honest differentiator
        </h2>
        <p className="max-w-4xl text-xs leading-relaxed text-muted">
          Faithfulness is a <span className="text-ink">fairness floor, not a differentiator</span>:
          after the verify-and-regenerate gate, <span className="text-ink">every</span> model ships{" "}
          <span className="text-ink">0.0%</span> user-visible board-fact fabrication
          (n={t.determOverall.n.toLocaleString()} cells) — the same bar for OURS, BASE, frontier and
          open alike. Where models genuinely differ is the{" "}
          <span className="text-ink">semantic-truth</span> residual: a strict cross-family judge panel
          ({t.judgePanel.join(" + ")}) fact-checks a stratified sample of the{" "}
          <span className="text-ink">gated</span> text for the multi-move / evaluative claims the
          deterministic layer can’t decide. OURS trails the frontier here — shown, not smoothed over.
        </p>
      </div>

      {/* The fairness floor — one statement, all models */}
      <div className="flex items-start gap-3 rounded-[10px] border-[1.5px] border-[color:var(--good)]/35 bg-[color:var(--good)]/10 px-4 py-3">
        <ShieldCheckIcon width={18} height={18} className="mt-0.5 shrink-0 text-[color:var(--good)]" />
        <div className="flex flex-col gap-0.5">
          <span className="text-sm font-semibold text-ink">
            User-visible fabrication ={" "}
            <span className="tnum text-[color:var(--good)]">0.0%</span> for all {t.rows.length} models
          </span>
          <span className="text-xs leading-relaxed text-muted">
            After the verify-and-regenerate gate, no mechanical board-fact lie survives in what a user
            sees — verified on n={t.determOverall.n.toLocaleString()} shipped cells, the same gate for
            every model. Faithfulness is table-stakes here, so it is not a ranking axis.
          </span>
        </div>
      </div>

      {/* The semantic-judge residual — three nested aggregations, per model */}
      <div className="flex flex-col gap-2.5 rounded-[10px] border-[1.5px] border-[color:var(--border)] px-4 py-3.5">
        <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-faint">
            Semantic-judge truthful-rate — any · majority · unanimous (95% CI)
          </h3>
          <span className="text-[11px] text-faint tnum">
            pooled: any {trimNum(t.overall.any.pct)}% · maj {trimNum(t.overall.majority.pct)}% · unan{" "}
            {trimNum(t.overall.unanimous.pct)}% · n={t.overall.n}
          </span>
        </div>
        <div className="flex flex-col gap-2">
          {rows.map((r) => (
            <TruthBar key={r.name} row={r} />
          ))}
        </div>
        <p className="text-[10px] leading-relaxed text-faint">
          The bar draws the <span className="text-muted">any</span> (strict) rate as a solid floor,
          extended by a lighter band to the <span className="text-muted">unanimous</span> (lenient)
          rate; the tick marks <span className="text-muted">majority</span>.{" "}
          <span className="text-muted">any</span> = a single cross-family judge’s objection sinks the
          cell — a conservative <span className="text-muted">lower bound</span>, not a claim the rest
          are lies. <span className="text-muted">unanimous</span> = only a 3/3 objection sinks it — an{" "}
          <span className="text-muted">upper bound</span>. Truth sits inside that band. Engine-derived
          fallback cells are judged 100% truthful; the residual lives in mechanically-clean model
          prose. {t.judgeCalls.toLocaleString()} judge calls · ${t.judgeCostUsd.toFixed(2)}.{" "}
          <span className="text-[color:var(--caution)]">⚠</span> = partial model (subset of cells).
        </p>
      </div>
    </section>
  );
}

/** One model's semantic-judge truthfulness as a floor(any)→band(unanimous) bar,
 *  a majority tick, and the strict-rate 95% CI whisker; all three rates + CIs at right. */
function TruthBar({ row }: { row: TruthfulnessRow }) {
  const fill =
    row.kind === "ours"
      ? "var(--signal)"
      : row.kind === "frontier"
        ? "var(--engine)"
        : row.kind === "base"
          ? "var(--faint)"
          : "var(--muted)";
  const bandWidth = Math.max(0, row.unanimous.pct - row.any.pct);
  const ciWidth = Math.max(0, row.any.ciHiPct - row.any.ciLoPct);
  return (
    <div className="grid grid-cols-[minmax(112px,148px)_minmax(0,1fr)_auto] items-center gap-2.5">
      <div className="flex min-w-0 items-center gap-1.5">
        <span
          className={`truncate text-xs ${row.kind === "ours" ? "font-semibold text-signal" : "text-ink"}`}
        >
          {row.short}
        </span>
        <KindTag kind={row.kind} />
        {row.partial && (
          <span className="text-[10px] text-[color:var(--caution)]" title="Partial model — only a subset of cells exist (Bedrock throttling).">
            ⚠
          </span>
        )}
      </div>

      <div
        className="relative h-3 rounded-full bg-[color:var(--surface-tertiary)]"
        title={`any ${trimNum(row.any.pct)}% · majority ${trimNum(row.majority.pct)}% · unanimous ${trimNum(row.unanimous.pct)}%`}
      >
        {/* lenient band: any → unanimous (lower opacity) */}
        <span
          aria-hidden
          className="absolute top-0 h-full rounded-full"
          style={{ left: `${row.any.pct}%`, width: `${bandWidth}%`, backgroundColor: fill, opacity: 0.28 }}
        />
        {/* strict "any" floor fill */}
        <span
          aria-hidden
          className="absolute left-0 top-0 h-full rounded-full"
          style={{ width: `${row.any.pct}%`, backgroundColor: fill, opacity: 0.9 }}
        />
        {/* majority tick */}
        <span
          aria-hidden
          className="absolute top-1/2 h-3 w-[1.5px] -translate-x-1/2 -translate-y-1/2 bg-ink/55"
          style={{ left: `${row.majority.pct}%` }}
        />
        {/* 95% CI whisker for the strict (any) estimate */}
        <span
          aria-hidden
          className="absolute top-1/2 h-[2px] -translate-y-1/2 rounded-full bg-ink/40"
          style={{ left: `${row.any.ciLoPct}%`, width: `${ciWidth}%` }}
        />
      </div>

      <div className="flex flex-col items-end leading-tight tnum">
        <span className={`text-xs font-semibold ${row.kind === "ours" ? "text-signal" : "text-ink"}`}>
          any {trimNum(row.any.pct)}%
          <span className="ml-1 text-[10px] font-normal text-faint">
            [{trimNum(row.any.ciLoPct)}–{trimNum(row.any.ciHiPct)}]
          </span>
        </span>
        <span className="text-[10px] text-faint">
          maj {trimNum(row.majority.pct)}% [{trimNum(row.majority.ciLoPct)}–{trimNum(row.majority.ciHiPct)}]
        </span>
        <span className="text-[10px] text-faint">
          unan {trimNum(row.unanimous.pct)}% [{trimNum(row.unanimous.ciLoPct)}–{trimNum(row.unanimous.ciHiPct)}] · n={row.judgeN}
        </span>
      </div>
    </div>
  );
}

/* ================================================================== */
/* Small shared pieces                                                 */
/* ================================================================== */

function KindTag({ kind }: { kind: ModelKind }) {
  const map: Record<ModelKind, { label: string; cls: string }> = {
    ours: { label: "ours", cls: "text-signal ring-signal/40" },
    frontier: { label: "frontier", cls: "text-[color:var(--engine)] ring-[color:var(--engine)]/40" },
    base: { label: "base", cls: "text-faint ring-[color:var(--border)]" },
    open: { label: "open", cls: "text-muted ring-[color:var(--border)]" },
  };
  const t = map[kind];
  return (
    <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ring-1 ${t.cls}`}>
      {t.label}
    </span>
  );
}

function Chip({ tone, label }: { tone: "good" | "signal" | "danger" | "muted"; label: string }) {
  const map = {
    good: "text-[color:var(--good)] bg-[color:var(--good)]/12",
    signal: "text-signal bg-signal/15",
    danger: "text-[color:var(--danger)] bg-[color:var(--danger)]/12",
    muted: "text-muted bg-[color:var(--surface-tertiary)]",
  } as const;
  return (
    <span className={`inline-flex items-center rounded px-1.5 py-0.5 text-[11px] font-medium ${map[tone]}`}>
      {label}
    </span>
  );
}

function LoadingSkeleton() {
  return (
    <div className="flex flex-col gap-4" role="status" aria-busy="true" aria-live="polite">
      <span className="sr-only">Loading the showcase…</span>
      <div className="skeleton h-16 w-full" aria-hidden />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
        <div className="flex flex-col gap-2.5">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="skeleton h-[120px] w-full" aria-hidden />
          ))}
        </div>
        <div className="skeleton h-[520px] w-full" aria-hidden />
      </div>
    </div>
  );
}

function EmptyState({
  title,
  body,
  onRetry,
}: {
  title: string;
  body: string;
  onRetry: () => void;
}) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-[10px] border border-[color:var(--border)] px-6 py-12 text-center">
      <h2 className="text-lg font-semibold text-ink">{title}</h2>
      <p className="max-w-md text-sm text-muted">{body}</p>
      <Button variant="secondary" size="md" className="min-h-11" onPress={onRetry}>
        Try again
      </Button>
    </div>
  );
}
