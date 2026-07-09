"use client";

import { useMemo, useState } from "react";
import type { LibraryEntry, Tier } from "@/lib/api";

type Filter = Tier | "all";
export type LibStatus = "loading" | "ready" | "error";

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "beginner", label: "Beginner" },
  { id: "intermediate", label: "Intermediate" },
  { id: "advanced", label: "Advanced" },
];

// Severity → a legible dot color + label. The label always renders in ink/muted
// (high contrast); color is a secondary cue via the dot, never the sole signal.
// A red → orange → amber → cool severity ramp (each hue distinct). The text label
// always carries the meaning; the dot is a scannable secondary cue.
function sevStyle(sev: string | null): { label: string; dot: string } {
  switch (sev) {
    case "blunder":
      return { label: "Blunder", dot: "var(--danger)" };
    case "mistake":
      return { label: "Mistake", dot: "oklch(0.7 0.16 47)" };
    case "inaccuracy":
      return { label: "Inaccuracy", dot: "var(--caution)" };
    default:
      return { label: sev ? cap(sev) : "To plan", dot: "var(--engine)" };
  }
}

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

interface Props {
  entries: LibraryEntry[];
  status: LibStatus;
  activeId: string | null;
  disabled: boolean;
  onSelect: (entry: LibraryEntry) => void;
  onRetry: () => void;
}

export default function PositionLibrary({
  entries,
  status,
  activeId,
  disabled,
  onSelect,
  onRetry,
}: Props) {
  const [filter, setFilter] = useState<Filter>("all");

  const shown = useMemo(
    () => (filter === "all" ? entries : entries.filter((e) => e.tier === filter)),
    [entries, filter],
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-medium text-ink">Study library</h3>
        {status === "ready" && entries.length > 0 && (
          <span className="text-xs text-muted tnum">{entries.length} coached positions</span>
        )}
      </div>

      {/* Tier filter */}
      <div className="flex flex-wrap gap-1.5">
        {FILTERS.map((f) => {
          const active = filter === f.id;
          const count =
            f.id === "all" ? entries.length : entries.filter((e) => e.tier === f.id).length;
          return (
            <button
              key={f.id}
              type="button"
              onClick={() => setFilter(f.id)}
              aria-pressed={active}
              className={`inline-flex min-h-11 items-center gap-1.5 rounded-full px-3.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60 ${
                active
                  ? "bg-signal text-[color:var(--signal-ink)]"
                  : "text-muted ring-1 ring-[color:var(--border)] hover:text-ink hover:ring-[color:var(--field-border)]"
              }`}
            >
              {f.label}
              <span
                className={
                  active ? "text-[color:var(--signal-ink)] tnum" : "text-muted tnum"
                }
              >
                {count}
              </span>
            </button>
          );
        })}
      </div>

      {/* List / states */}
      {status === "loading" ? (
        <div className="flex flex-col gap-1.5" role="status" aria-busy="true" aria-live="polite">
          <span className="sr-only">Loading study library…</span>
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="skeleton h-[52px] w-full" aria-hidden />
          ))}
        </div>
      ) : status === "error" ? (
        <div className="flex flex-col items-start gap-2 rounded-[10px] border border-[color:var(--border)] px-3.5 py-4">
          <p className="text-sm text-ink">The study library didn&rsquo;t load.</p>
          <p className="text-xs text-muted">You can still set a position by hand below.</p>
          <button
            type="button"
            onClick={onRetry}
            className="min-h-11 rounded-md px-3.5 text-xs font-medium text-signal ring-1 ring-signal/40 transition-colors hover:bg-signal/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60"
          >
            Try again
          </button>
        </div>
      ) : (
        <div className="relative">
            <ul className="flex max-h-[340px] flex-col divide-y divide-[color:var(--separator)] overflow-y-auto rounded-[10px] border border-[color:var(--border)]">
              {shown.map((e) => {
                const sev = sevStyle(e.severity);
                const active = e.id === activeId;
                return (
                  <li key={e.id}>
                    <button
                      type="button"
                      disabled={disabled}
                      onClick={() => onSelect(e)}
                      aria-pressed={active}
                      className={`group flex w-full cursor-pointer items-center gap-3 px-3.5 py-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-signal/60 disabled:cursor-default disabled:opacity-50 ${
                        active
                          ? "bg-signal/12"
                          : "hover:bg-[color:var(--surface-tertiary)]"
                      }`}
                    >
                      <span
                        aria-hidden
                        className={`size-2 shrink-0 rounded-full ${active ? "ring-2 ring-signal/40" : ""}`}
                        style={{ backgroundColor: sev.dot }}
                      />
                      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                        <span className="text-sm text-muted">
                          {e.student_move ? (
                            <>
                              You played{" "}
                              <span className="font-serif text-[color:var(--your-move)] tnum">
                                {e.student_move}
                              </span>
                            </>
                          ) : (
                            "Position to plan"
                          )}
                        </span>
                        <span className="text-xs text-muted">
                          {sev.label} · {cap(e.phase)}
                        </span>
                      </div>
                      <div className="flex shrink-0 items-baseline gap-1.5">
                        <span className="text-xs text-muted">coach</span>
                        <span className="font-serif text-base font-semibold text-signal tnum">
                          {e.coach.recommended_move_san}
                        </span>
                      </div>
                    </button>
                  </li>
                );
              })}
            {shown.length === 0 && (
              <li className="px-3.5 py-6 text-center text-xs text-muted">
                No {filter === "all" ? "" : filter} positions in the library yet.
              </li>
            )}
          </ul>
          {/* Continuation cue when the list overflows. */}
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-px bottom-px h-8 rounded-b-[10px] bg-gradient-to-t from-[color:var(--surface-secondary)] to-transparent"
          />
        </div>
      )}
    </div>
  );
}
