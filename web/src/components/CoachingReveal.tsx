"use client";

import { Chip, Separator, Tooltip } from "@heroui/react";
import type { CoachResponse, Tier } from "@/lib/api";
import { principleTag } from "@/lib/showcase";
import AnalysisRail from "./AnalysisRail";
import EngineLines from "./EngineLines";
import { ShieldCheckIcon } from "./icons";

function severityChip(severity: string) {
  const s = severity.toLowerCase();
  if (s === "blunder" || s === "mistake") return { color: "danger" as const, label: s };
  if (s === "inaccuracy") return { color: "warning" as const, label: s };
  return { color: "default" as const, label: s === "none" ? "reasonable" : s };
}

function Block({
  delay,
  children,
  className,
}: {
  delay: number;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`rise ${className ?? ""}`} style={{ animationDelay: `${delay}s` }}>
      {children}
    </div>
  );
}

export default function CoachingReveal({
  result,
  tier,
  fen,
  precomputed = false,
  onPlayLine,
}: {
  result: CoachResponse;
  tier: Tier;
  fen: string;
  /** True when this answer is cached from the benchmark (not a live model call). */
  precomputed?: boolean;
  onPlayLine?: (pv: string[], count: number) => void;
}) {
  const paragraphs = result.coaching
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean);
  const hasProse = paragraphs.length > 0;

  // The PRINCIPLE TAG for the move: assembled from existing CoachResponse fields
  // (concepts if any, else a short slice of the takeaway). This is the hero's
  // one-line reason; the full prose drops to an optional, secondary expander.
  const tag = principleTag(result.concepts_used, result.takeaway);

  const student = result.engine.student_move;
  const sev = student ? severityChip(student.severity) : null;

  // The faithfulness gate's internal notes ("...stated a false board fact...") read
  // as alarming raw prose in a demo. Drop them here: when the answer is a verified
  // fallback we surface a clean, honest "Verified explanation" chip instead, and
  // when a retry eventually verified clean there's no caveat worth showing.
  const fallback = result.meta.verified_fallback;
  const notes = result.meta.notes.filter((n) => !/false board fact/i.test(n));

  return (
    <div className="flex flex-col gap-7">
      {/* HERO: the recommended move + a short principle tag: the trained behavior
          (the tier-appropriate move), and the one-line reason attached to it. */}
      <Block delay={0}>
        <h2 className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <span
            className="font-serif font-semibold text-signal tnum"
            style={{ fontSize: "var(--text-verdict)", lineHeight: 0.95, letterSpacing: "-0.02em" }}
          >
            {result.recommended_move_san}
          </span>
          <span className="inline-flex flex-wrap items-center gap-x-1.5 text-sm text-muted">
            <Tooltip delay={200}>
              <Tooltip.Trigger aria-label={`Why this is the move for ${aOrAn(tier)} ${tier} player`}>
                <span className="cursor-help underline decoration-dotted underline-offset-2">
                  the move for {aOrAn(tier)} {tier} player
                </span>
              </Tooltip.Trigger>
              <Tooltip.Content showArrow className="max-w-[18rem]">
                <Tooltip.Arrow />
                <p className="leading-relaxed">
                  The tuned model&apos;s one job: pick the move that fits this rating band. Switch levels
                  to watch the pick adapt.
                </p>
              </Tooltip.Content>
            </Tooltip>
            <span aria-hidden className="text-faint">·</span>
            <span>{result.side_to_move} to move</span>
          </span>
        </h2>
        {precomputed && (
          <div className="mt-2.5">
            <Tooltip delay={200}>
              <Tooltip.Trigger aria-label="What “precomputed” means">
                <span className="inline-flex cursor-help items-center gap-1.5 rounded-full bg-[color:var(--surface-tertiary)] px-2.5 py-1 text-[11px] font-medium text-muted ring-1 ring-[color:var(--border)]">
                  <span aria-hidden className="size-1.5 rounded-full bg-signal" />
                  precomputed
                </span>
              </Tooltip.Trigger>
              <Tooltip.Content showArrow className="max-w-[18rem]">
                <Tooltip.Arrow />
                <p className="leading-relaxed">
                  This answer is precomputed (cached from the held-out benchmark) and shown
                  instantly. Use “Run live” to recompute it on the hosted coach.
                </p>
              </Tooltip.Content>
            </Tooltip>
          </div>
        )}
        {tag && (
          <p className="mt-3 max-w-[60ch] text-lg leading-relaxed text-ink text-pretty">{tag}</p>
        )}
        {student && sev && (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Chip color={sev.color} variant="soft" size="sm">
              {sev.label}
            </Chip>
            <span className="text-sm text-muted">
              You played{" "}
              <span className="font-serif text-[color:var(--your-move)] tnum">{student.san}</span>.
            </span>
          </div>
        )}
      </Block>

      {/* Full coaching prose for the currently selected tier: a supplementary,
          engine-assisted explanation, always shown in a plain card (no toggle). */}
      {hasProse && (
        <Block delay={0.04}>
          <div className="rounded-[10px] border border-[color:var(--border)] bg-[color:var(--surface)]">
            <div className="flex max-w-[66ch] flex-col gap-3.5 px-4 py-4">
              {paragraphs.map((p, i) => (
                <p key={i} className="text-base leading-relaxed text-muted text-pretty">
                  {p}
                </p>
              ))}
            </div>
          </div>
        </Block>
      )}

      {/* Under the board: the measured evidence. */}
      <Separator />
      <Block delay={0.08}>
        <AnalysisRail
          engine={result.engine}
          maia={result.maia}
          recommendedUci={result.recommended_move_uci}
          studentUci={result.engine.student_move?.uci ?? null}
        />
      </Block>

      <Block delay={0.12}>
        <EngineLines
          engine={result.engine}
          fen={fen}
          recommendedUci={result.recommended_move_uci}
          onPlayLine={onPlayLine}
        />
      </Block>

      {/* Provenance */}
      <Block delay={0.16} className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Chip variant="soft" color="default" size="sm">
            {result.meta.tuned ? "Tuned coach" : "Base model"}
          </Chip>
          {fallback && (
            <Tooltip delay={200}>
              <Tooltip.Trigger aria-label="What “Verified explanation” means">
                <Chip variant="soft" color="warning" size="sm" className="cursor-help">
                  <ShieldCheckIcon width={13} height={13} />
                  <Chip.Label>Verified explanation</Chip.Label>
                </Chip>
              </Tooltip.Trigger>
              <Tooltip.Content showArrow className="max-w-[18rem]">
                <Tooltip.Arrow />
                <p className="leading-relaxed">
                  The model’s wording didn’t pass the board-fact faithfulness check, so a verified
                  explanation of a sound, approved move is shown instead.
                </p>
              </Tooltip.Content>
            </Tooltip>
          )}
          <span className="font-mono text-xs text-muted">{result.meta.model}</span>
        </div>
        {notes.map((n) => (
          <p key={n} className="text-xs leading-relaxed text-muted">
            {n}
          </p>
        ))}
      </Block>
    </div>
  );
}

function aOrAn(word: string): string {
  return /^[aeiou]/i.test(word) ? "an" : "a";
}
