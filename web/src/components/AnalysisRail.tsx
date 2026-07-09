"use client";

import type { EngineBlock, MaiaInfo } from "@/lib/api";

// Pawn-unit eval (chess.com style), consistent with the Top-engine-lines panel.
function fmtEval(cp: number): string {
  if (cp >= 20000) return "#";
  if (cp <= -20000) return "-#";
  if (Math.abs(cp) < 5) return "0.0"; // avoid a misleading "-0.0"
  const p = cp / 100;
  return (p >= 0 ? "+" : "") + p.toFixed(1);
}

type Tone = "rec" | "student" | "neutral";

const TONE_FILL: Record<Tone, string> = {
  rec: "var(--signal)",
  student: "var(--your-move)",
  neutral: "color-mix(in oklab, var(--engine) 60%, transparent)",
};

const TONE_TAG: Record<Tone, string | null> = {
  rec: "coach",
  student: "you",
  neutral: null,
};

function Row({
  san,
  tone,
  fillPct,
  value,
}: {
  san: string;
  tone: Tone;
  fillPct: number;
  value: string;
}) {
  const tag = TONE_TAG[tone];
  return (
    <li
      className="flex h-7 items-center gap-2.5"
      aria-label={`${san}, ${value}${tag ? `, ${tone === "rec" ? "coach's move" : "your move"}` : ""}`}
    >
      <span
        aria-hidden
        className="w-11 shrink-0 font-serif text-sm tnum"
        style={{ color: tone === "neutral" ? "var(--muted)" : "var(--ink)" }}
      >
        {san}
      </span>
      <span
        className="relative h-2 flex-1 overflow-hidden rounded-full"
        style={{ backgroundColor: "color-mix(in oklab, var(--ink) 9%, transparent)" }}
      >
        <span
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ width: `${fillPct}%`, backgroundColor: TONE_FILL[tone] }}
        />
      </span>
      {tag && (
        <span
          aria-hidden
          className="w-10 shrink-0 text-right text-xs font-medium lowercase"
          style={{ color: tone === "rec" ? "var(--signal)" : "var(--your-move)" }}
        >
          {tag}
        </span>
      )}
      {!tag && <span aria-hidden className="w-10 shrink-0" />}
      <span className="w-11 shrink-0 text-right font-mono text-xs text-muted tnum">{value}</span>
    </li>
  );
}

export default function AnalysisRail({
  engine,
  maia,
  recommendedUci,
  studentUci,
}: {
  engine: EngineBlock;
  maia: MaiaInfo[];
  recommendedUci: string;
  studentUci: string | null;
}) {
  const pool = engine.sound_pool.slice(0, 6);
  // The best move in the pool anchors a fixed cp scale, so a bar shows how close
  // each move is to best (honest magnitude), not a min–max stretch of a tiny gap.
  const bestCp = pool.length ? Math.max(...pool.map((m) => m.cp)) : 0;

  const maiaTop = maia.slice(0, 6);

  const toneFor = (uci: string): Tone =>
    uci === recommendedUci ? "rec" : uci === studentUci ? "student" : "neutral";

  return (
    <div className="flex flex-col gap-5">
      {/* Legend — the color language, stated once (not color-alone). */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2 rounded-full bg-signal" aria-hidden />
          Coach&rsquo;s move
        </span>
        {studentUci && (
          <span className="inline-flex items-center gap-1.5">
            <span className="size-2 rounded-full bg-[color:var(--your-move)]" aria-hidden />
            Your move
          </span>
        )}
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2 rounded-full bg-[color:var(--engine)]" aria-hidden />
          Reference
        </span>
      </div>

      <div className="grid gap-6 sm:grid-cols-2">
        <section aria-label="Strong moves">
          <h3 className="mb-2 text-sm font-medium text-ink">Strong moves</h3>
          <ul className="flex flex-col gap-1.5">
            {pool.map((m) => {
              const tone = toneFor(m.uci);
              // Fixed scale: best move fills the bar; ~300cp worse empties it.
              const pct = Math.max(12, Math.min(100, 100 - (bestCp - m.cp) / 3));
              return <Row key={m.uci} san={m.san} tone={tone} fillPct={pct} value={fmtEval(m.cp)} />;
            })}
          </ul>
        </section>

        <section aria-label="What players usually play">
          <h3 className="mb-2 text-sm font-medium text-ink">
            What players usually play
          </h3>
          {maiaTop.length === 0 ? (
            <p className="text-xs leading-relaxed text-muted">
              No player-frequency data for this position.
            </p>
          ) : (
            <ul className="flex flex-col gap-1.5">
              {maiaTop.map((m) => {
                const tone = toneFor(m.uci);
                // Human odds are true probabilities — bars use the absolute 0–100%
                // scale (with a small floor so tiny odds stay visible).
                const pct = Math.max(6, m.policy * 100);
                return (
                  <Row
                    key={m.uci}
                    san={m.san}
                    tone={tone}
                    fillPct={pct}
                    value={`${Math.round(m.policy * 100)}%`}
                  />
                );
              })}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}
