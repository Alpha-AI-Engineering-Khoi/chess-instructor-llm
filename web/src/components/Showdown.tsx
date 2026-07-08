"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Card } from "@heroui/react";
import {
  getShowdown,
  type ShowdownDoc,
  type ShowdownModel,
  type ShowdownPosition,
} from "@/lib/showdown";
import type { Orientation } from "@/lib/chess";
import ShowdownBoard from "./ShowdownBoard";

type Status = "loading" | "ready" | "error" | "empty";
type TierFilter = "all" | "beginner" | "intermediate" | "advanced";
type PhaseFilter = "all" | "opening" | "middlegame" | "endgame";
type BenchFilter = "all" | "v2" | "open";

const PAGE = 24;

function cap(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

function sevDot(sev: string): string {
  switch (sev) {
    case "blunder":
      return "var(--danger)";
    case "mistake":
      return "oklch(0.7 0.16 47)";
    case "inaccuracy":
      return "var(--caution)";
    default:
      return "var(--engine)";
  }
}

export default function Showdown() {
  const [doc, setDoc] = useState<ShowdownDoc | null>(null);
  const [status, setStatus] = useState<Status>("loading");

  const [tier, setTier] = useState<TierFilter>("all");
  const [phase, setPhase] = useState<PhaseFilter>("all");
  const [bench, setBench] = useState<BenchFilter>("all");
  const [model, setModel] = useState<string>("all");
  const [winsOnly, setWinsOnly] = useState(true);
  const [visible, setVisible] = useState(PAGE);

  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(() => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setStatus("loading");
    getShowdown(ctrl.signal)
      .then((d) => {
        if (ctrl.signal.aborted) return;
        if (!d || !d.positions?.length) {
          setStatus("empty");
          return;
        }
        setDoc(d);
        setStatus("ready");
      })
      .catch(() => {
        if (ctrl.signal.aborted) return;
        setStatus("error");
      });
  }, []);

  useEffect(() => {
    load();
    return () => abortRef.current?.abort();
  }, [load]);

  const positions = doc?.positions ?? [];

  // Model dropdown options, ordered ours → frontier → base → open.
  const modelOptions = useMemo(() => {
    if (!doc) return [] as { key: string; name: string }[];
    const order: Record<string, number> = { ours: 0, gpt: 1, claude: 2, gemini: 3, base: 4 };
    const seen = new Map<string, string>();
    for (const p of positions) for (const m of p.models) if (!seen.has(m.key)) seen.set(m.key, m.short);
    return [...seen.entries()]
      .map(([key, name]) => ({ key, name }))
      .sort((a, b) => (order[a.key] ?? 5) - (order[b.key] ?? 5) || a.name.localeCompare(b.name));
  }, [doc, positions]);

  const filtered = useMemo(() => {
    return positions.filter((p) => {
      if (tier !== "all" && p.tier !== tier) return false;
      if (phase !== "all" && p.phase !== phase) return false;
      if (bench !== "all" && p.benchmark !== bench) return false;
      if (winsOnly && !p.ours_wins) return false;
      if (model !== "all" && !p.models.some((m) => m.key === model)) return false;
      return true;
    });
  }, [positions, tier, phase, bench, winsOnly, model]);

  // Reset pagination whenever the filter set changes.
  useEffect(() => setVisible(PAGE), [tier, phase, bench, model, winsOnly]);

  const shown = filtered.slice(0, visible);
  const totals = doc?.meta.totals;
  const focusModel = model === "all" ? "ours" : model;

  return (
    <div className="relative z-[1] mx-auto flex min-h-dvh w-full max-w-[1240px] flex-col gap-8 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      {/* Header */}
      <header className="flex flex-col gap-4">
        <div className="flex items-center justify-between gap-3">
          <Link
            href="/"
            className="inline-flex min-h-9 items-center gap-1.5 text-sm text-muted transition-colors hover:text-ink"
          >
            <span aria-hidden className="text-faint">
              ‹
            </span>
            Coach studio
          </Link>
          <div className="flex items-center gap-3">
            {doc && (
              <span className="hidden font-mono text-xs text-faint tnum sm:inline">
                grounded · {doc.meta.condition === "grounded" ? "same input for every model" : doc.meta.condition}
              </span>
            )}
            <Link
              href="/showcase"
              className="inline-flex min-h-9 items-center gap-1.5 rounded-full bg-signal/12 px-3.5 text-sm font-medium text-signal ring-1 ring-signal/40 transition-colors hover:bg-signal/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60"
            >
              Showcase
              <span aria-hidden>★</span>
              <span aria-hidden className="text-signal/70">›</span>
            </Link>
          </div>
        </div>
        <div className="flex flex-col gap-2">
          <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
            Model Showdown — where OURS beats the frontier
          </h1>
          <p className="max-w-3xl text-sm leading-relaxed text-muted sm:text-base">
            Every held-out position, with each model&rsquo;s recommended move on the same grounded
            input. A move is <span className="text-ink">tier-fit</span> when it is the human-findable
            sound move for that tier; <span className="text-ink">fabricated</span> when the
            faithfulness verifier caught a false board fact. Rows where{" "}
            <span className="text-signal">OURS wins</span> — sound + tier-fit where a frontier model
            isn&rsquo;t, or faithful where a frontier model invents a fact — are surfaced first.
          </p>
        </div>

        {/* Summary tiles */}
        {totals && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatTile label="Positions" value={totals.positions} sub="held-out" />
            <StatTile label="OURS wins" value={totals.ours_wins} sub="beats ≥1 frontier" accent />
            <StatTile label="Tier-fit wins" value={totals.ours_wins_tier} sub="right move for the level" />
            <StatTile label="Faithfulness wins" value={totals.ours_wins_faithful} sub="honest where frontier fabricates" />
          </div>
        )}

        {/* Honest methodology note */}
        {doc && (
          <details className="group rounded-[10px] border border-[color:var(--border)] px-4 py-3">
            <summary className="flex cursor-pointer list-none items-center gap-2 text-sm font-medium text-muted transition-colors hover:text-ink">
              <span className="text-faint transition-transform group-open:rotate-90">›</span>
              How &ldquo;OURS wins&rdquo; is defined (kept honest)
            </summary>
            <dl className="mt-3 flex flex-col gap-2 text-xs leading-relaxed text-muted">
              {Object.entries(doc.meta.definitions).map(([k, v]) => (
                <div key={k} className="flex flex-col gap-0.5 sm:flex-row sm:gap-2">
                  <dt className="shrink-0 font-mono text-faint sm:w-36">{k}</dt>
                  <dd>{v}</dd>
                </div>
              ))}
              <div className="mt-1 flex flex-col gap-0.5 border-t border-[color:var(--separator)] pt-2 sm:flex-row sm:gap-2">
                <dt className="shrink-0 font-mono text-faint sm:w-36">benchmarks</dt>
                <dd>
                  <span className="text-ink">v2</span> — {doc.meta.benchmarks.v2}{" "}
                  <span className="text-ink">open</span> — {doc.meta.benchmarks.open}
                </dd>
              </div>
            </dl>
          </details>
        )}
      </header>

      {/* Filter bar */}
      {status === "ready" && (
        <section className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
            <PillGroup
              label="Tier"
              value={tier}
              onChange={(v) => setTier(v as TierFilter)}
              options={[
                ["all", "All"],
                ["beginner", "Beginner"],
                ["intermediate", "Intermediate"],
                ["advanced", "Advanced"],
              ]}
            />
            <PillGroup
              label="Phase"
              value={phase}
              onChange={(v) => setPhase(v as PhaseFilter)}
              options={[
                ["all", "All"],
                ["opening", "Opening"],
                ["middlegame", "Middlegame"],
                ["endgame", "Endgame"],
              ]}
            />
            <PillGroup
              label="Set"
              value={bench}
              onChange={(v) => setBench(v as BenchFilter)}
              options={[
                ["all", "All"],
                ["v2", "5-model"],
                ["open", "+open"],
              ]}
            />
          </div>

          <div className="flex flex-wrap items-center justify-between gap-x-5 gap-y-3">
            <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
              <label className="flex items-center gap-2 text-xs font-medium text-muted">
                <span className="uppercase tracking-wide text-faint">Focus model</span>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="min-h-9 rounded-md border border-[color:var(--field-border)] bg-[color:var(--field-background)] px-2.5 py-1 text-sm text-ink transition-colors hover:border-[color:var(--border)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60"
                >
                  <option value="all">All models (focus OURS)</option>
                  {modelOptions.map((m) => (
                    <option key={m.key} value={m.key}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </label>

              <button
                type="button"
                role="switch"
                aria-checked={winsOnly}
                onClick={() => setWinsOnly((w) => !w)}
                className={`inline-flex min-h-9 items-center gap-2 rounded-full px-3.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60 ${
                  winsOnly
                    ? "bg-signal text-[color:var(--signal-ink)]"
                    : "text-muted ring-1 ring-[color:var(--border)] hover:text-ink"
                }`}
              >
                <span
                  aria-hidden
                  className={`inline-block size-2 rounded-full ${
                    winsOnly ? "bg-[color:var(--signal-ink)]" : "bg-faint"
                  }`}
                />
                OURS wins only
              </button>
            </div>

            <span className="text-xs text-muted tnum">
              {filtered.length} {filtered.length === 1 ? "position" : "positions"}
              {winsOnly ? " where OURS wins" : ""}
            </span>
          </div>
        </section>
      )}

      {/* Body */}
      <main className="flex flex-1 flex-col gap-4">
        {status === "loading" && (
          <div className="flex flex-col gap-4" role="status" aria-busy="true" aria-live="polite">
            <span className="sr-only">Loading the model showdown…</span>
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="skeleton h-[280px] w-full" aria-hidden />
            ))}
          </div>
        )}

        {status === "error" && (
          <EmptyState
            title="The showdown data didn’t load."
            body="showdown.json is served from web/public — rebuild it with scripts/build_showdown.py, then retry."
            onRetry={load}
          />
        )}

        {status === "empty" && (
          <EmptyState
            title="No showdown positions yet."
            body="Run scripts/build_showdown.py to generate web/public/showdown.json from the benchmark artifacts."
            onRetry={load}
          />
        )}

        {status === "ready" && (
          <>
            {shown.length === 0 ? (
              <div className="rounded-[10px] border border-[color:var(--border)] px-4 py-10 text-center text-sm text-muted">
                No positions match these filters.
              </div>
            ) : (
              <ul className="flex flex-col gap-4">
                {shown.map((p) => (
                  <li key={p.key}>
                    <PositionCard position={p} focusModel={focusModel} />
                  </li>
                ))}
              </ul>
            )}

            {visible < filtered.length && (
              <div className="flex justify-center pt-2">
                <button
                  type="button"
                  onClick={() => setVisible((v) => v + PAGE)}
                  className="min-h-11 rounded-full px-5 text-sm font-medium text-ink ring-1 ring-[color:var(--border)] transition-colors hover:bg-[color:var(--surface-tertiary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60"
                >
                  Show more ({filtered.length - visible} left)
                </button>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Position card                                                       */
/* ------------------------------------------------------------------ */

function orderedModels(models: ShowdownModel[]): ShowdownModel[] {
  const order: Record<string, number> = { ours: 0, gpt: 1, claude: 2, gemini: 3, base: 4 };
  return [...models].sort(
    (a, b) => (order[a.key] ?? 5) - (order[b.key] ?? 5) || a.short.localeCompare(b.short),
  );
}

function PositionCard({ position, focusModel }: { position: ShowdownPosition; focusModel: string }) {
  const models = useMemo(() => orderedModels(position.models), [position.models]);
  const defaultKey = models.some((m) => m.key === focusModel) ? focusModel : "ours";
  const [sel, setSel] = useState<string>(defaultKey);
  useEffect(() => setSel(defaultKey), [defaultKey]);

  const selModel = models.find((m) => m.key === sel) ?? models.find((m) => m.key === "ours") ?? models[0];
  const orientation: Orientation = position.side_to_move;

  return (
    <Card variant="secondary" className="overflow-hidden">
      <Card.Content className="grid grid-cols-1 gap-5 p-4 sm:p-5 lg:grid-cols-[minmax(0,300px)_minmax(0,1fr)]">
        {/* Left: board + position meta */}
        <div className="flex flex-col gap-3">
          <div className="mx-auto w-full max-w-[300px]">
            <ShowdownBoard
              fen={position.fen}
              orientation={orientation}
              moveUci={selModel?.rec_uci}
              studentUci={position.student_move?.uci}
            />
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="inline-flex items-center gap-1.5 rounded-full bg-[color:var(--surface-tertiary)] px-2.5 py-1 text-muted">
              <span aria-hidden className="size-2 rounded-full" style={{ backgroundColor: sevDot(position.severity) }} />
              {cap(position.severity)}
            </span>
            <span className="rounded-full bg-[color:var(--surface-tertiary)] px-2.5 py-1 text-muted">
              {cap(position.tier)}
            </span>
            <span className="rounded-full bg-[color:var(--surface-tertiary)] px-2.5 py-1 text-muted">
              {cap(position.phase)}
            </span>
            <span className="rounded-full px-2.5 py-1 text-faint ring-1 ring-[color:var(--border)]">
              {position.benchmark === "open" ? "+open field" : "5-model"}
            </span>
          </div>

          <div className="flex flex-col gap-1 text-xs text-muted">
            <div className="flex items-center gap-1.5">
              <span
                aria-hidden
                className="inline-block size-3 rounded-full ring-1 ring-border"
                style={{ backgroundColor: position.side_to_move === "white" ? "var(--board-light)" : "var(--board-dark)" }}
              />
              {cap(position.side_to_move)} to move
              {position.student_move?.san && (
                <>
                  <span className="text-faint">·</span> student played{" "}
                  <span className="font-mono text-[color:var(--your-move)] tnum">{position.student_move.san}</span>
                </>
              )}
            </div>
            {position.tier_target && (
              <div>
                {cap(position.tier)} should find{" "}
                <span className="font-mono text-ink tnum">{position.tier_target.san}</span>{" "}
                <span className="text-faint">
                  ({position.tier_target.is_engine_best ? "engine best" : `pool #${position.tier_target.pool_rank + 1}`}
                  {position.tier_target.policy > 0 ? ` · ${Math.round(position.tier_target.policy * 100)}% human` : ""})
                </span>
              </div>
            )}
          </div>

          {position.ours_wins && <WinBadge position={position} />}
        </div>

        {/* Right: model verdicts + focused coaching */}
        <div className="flex min-w-0 flex-col gap-4">
          <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {models.map((m) => (
              <ModelRow
                key={m.key}
                model={m}
                selected={m.key === sel}
                onSelect={() => setSel(m.key)}
              />
            ))}
          </div>

          {selModel && <CoachingPanel model={selModel} />}
        </div>
      </Card.Content>
    </Card>
  );
}

function WinBadge({ position }: { position: ShowdownPosition }) {
  return (
    <div className="rounded-[10px] border border-signal/40 bg-signal/10 px-3 py-2.5">
      <div className="flex items-center gap-2 text-sm font-semibold text-signal">
        <span aria-hidden>★</span> OURS wins here
      </div>
      <ul className="mt-1 flex flex-col gap-0.5 text-xs text-muted">
        {position.beats.map((b) => (
          <li key={b.model}>
            beats <span className="text-ink">{b.name}</span> on{" "}
            {b.on
              .map((o) => (o === "tier" ? "tier-fit" : "faithfulness"))
              .join(" + ")}
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Model row + coaching panel                                          */
/* ------------------------------------------------------------------ */

function ModelRow({
  model,
  selected,
  onSelect,
}: {
  model: ShowdownModel;
  selected: boolean;
  onSelect: () => void;
}) {
  const isOurs = model.kind === "ours";
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={`flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-signal/60 ${
        selected
          ? "bg-signal/12 shadow-[inset_2px_0_0_0_var(--signal)]"
          : "hover:bg-[color:var(--surface-tertiary)]"
      } ${isOurs ? "ring-1 ring-signal/30" : ""}`}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <div className="flex items-center gap-1.5">
          <span className={`truncate text-sm font-medium ${isOurs ? "text-signal" : "text-ink"}`}>
            {model.short}
          </span>
          <KindTag kind={model.kind} />
        </div>
        <div className="flex flex-wrap items-center gap-1">
          <Chip
            tone={model.tier_appropriate ? "signal" : model.sound ? "muted" : "danger"}
            label={
              model.tier_appropriate ? "tier-fit" : model.sound ? "sound" : model.parseable ? "unsound" : "no move"
            }
          />
          <Chip
            tone={model.fabricated ? "danger" : "good"}
            label={model.fabricated ? `fabricated ×${model.n_violations}` : "faithful"}
          />
        </div>
      </div>
      <span className={`shrink-0 font-mono text-base font-semibold tnum ${isOurs ? "text-signal" : "text-ink"}`}>
        {model.rec_san ?? "—"}
      </span>
    </button>
  );
}

function CoachingPanel({ model }: { model: ShowdownModel }) {
  const hasProse = Boolean(model.coaching?.trim());
  return (
    <div className="flex flex-col gap-2 rounded-[10px] border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className={`text-sm font-semibold ${model.kind === "ours" ? "text-signal" : "text-ink"}`}>
            {model.name}
          </span>
          <KindTag kind={model.kind} />
        </div>
        <span className="font-mono text-sm font-semibold text-ink tnum">{model.rec_san ?? "—"}</span>
      </div>

      {model.violations.length > 0 && (
        <div className="rounded-md border border-[color:var(--danger)]/40 bg-[color:var(--danger)]/10 px-3 py-2">
          <div className="text-xs font-semibold text-[color:var(--danger)]">
            Fabricated {model.violations.length === 1 ? "fact" : "facts"} (verifier)
          </div>
          <ul className="mt-1 flex flex-col gap-1 text-xs text-muted">
            {model.violations.map((v, i) => (
              <li key={i}>
                <span className="text-ink">&ldquo;{v.sentence}&rdquo;</span>{" "}
                <span className="text-[color:var(--danger)]">— {v.reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Coaching prose — a secondary, OPTIONAL layer, collapsed by default so the
          model's recommended move (in the header above) stays the hero, matching
          the Studio + Showcase reveal. The fabrication receipts above stay visible. */}
      {hasProse ? (
        <details className="group rounded-md border border-[color:var(--border)] bg-[color:var(--surface-tertiary)]/40">
          <summary className="flex min-h-9 cursor-pointer list-none items-center gap-2 px-3 py-2 text-[11px] font-medium text-muted [&::-webkit-details-marker]:hidden">
            <span aria-hidden className="text-faint transition-transform group-open:rotate-90">
              ›
            </span>
            Show full explanation
            <span className="text-faint">(optional, engine-assisted)</span>
          </summary>
          <div className="px-3 pb-3">
            <p className="max-h-56 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed text-muted">
              {model.coaching}
            </p>
          </div>
        </details>
      ) : (
        <p className="text-sm leading-relaxed text-faint">No coaching text produced.</p>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Small pieces                                                        */
/* ------------------------------------------------------------------ */

function KindTag({ kind }: { kind: ShowdownModel["kind"] }) {
  const map: Record<ShowdownModel["kind"], { label: string; cls: string }> = {
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

function StatTile({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: number;
  sub: string;
  accent?: boolean;
}) {
  return (
    <div
      className={`flex flex-col gap-0.5 rounded-[10px] border px-3.5 py-3 ${
        accent ? "border-signal/40 bg-signal/10" : "border-[color:var(--border)]"
      }`}
    >
      <span className={`font-mono text-2xl font-semibold tnum ${accent ? "text-signal" : "text-ink"}`}>
        {value}
      </span>
      <span className="text-xs font-medium text-ink">{label}</span>
      <span className="text-[11px] text-faint">{sub}</span>
    </div>
  );
}

function PillGroup({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: [string, string][];
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs font-medium uppercase tracking-wide text-faint">{label}</span>
      <div className="flex flex-wrap gap-1.5">
        {options.map(([id, text]) => {
          const active = value === id;
          return (
            <button
              key={id}
              type="button"
              onClick={() => onChange(id)}
              aria-pressed={active}
              className={`inline-flex min-h-9 items-center rounded-full px-3 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60 ${
                active
                  ? "bg-signal text-[color:var(--signal-ink)]"
                  : "text-muted ring-1 ring-[color:var(--border)] hover:text-ink hover:ring-[color:var(--field-border)]"
              }`}
            >
              {text}
            </button>
          );
        })}
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
      <button
        type="button"
        onClick={onRetry}
        className="min-h-11 rounded-full px-5 text-sm font-medium text-signal ring-1 ring-signal/40 transition-colors hover:bg-signal/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60"
      >
        Try again
      </button>
    </div>
  );
}
