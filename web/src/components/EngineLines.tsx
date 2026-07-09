"use client";

import type { EngineBlock } from "@/lib/api";

/** Centipawns (side-to-move POV) → chess.com-style pawn eval, e.g. +0.5 / -1.2 / #. */
function fmtEval(cp: number): string {
  if (cp >= 20000) return "#";
  if (cp <= -20000) return "-#";
  if (Math.abs(cp) < 5) return "0.0"; // avoid a misleading "-0.0"
  const p = cp / 100;
  return (p >= 0 ? "+" : "") + p.toFixed(1);
}

interface Token {
  num?: string;
  san: string;
}

/** Format a SAN principal variation with real move numbers from the FEN. */
function lineTokens(pv: string[], startNo: number, whiteToMove: boolean): Token[] {
  return pv.map((san, i) => {
    const whiteAtPly = whiteToMove ? i % 2 === 0 : i % 2 === 1;
    const moveNo = startNo + Math.floor((i + (whiteToMove ? 0 : 1)) / 2);
    let num: string | undefined;
    if (whiteAtPly) num = `${moveNo}.`;
    else if (i === 0) num = `${moveNo}…`;
    return { num, san };
  });
}

export default function EngineLines({
  engine,
  fen,
  recommendedUci,
  onPlayLine,
  disabled,
}: {
  engine: EngineBlock;
  fen: string;
  recommendedUci: string;
  /** Play the first `count` half-moves of a line onto the board (from `fen`). */
  onPlayLine?: (pv: string[], count: number) => void;
  disabled?: boolean;
}) {
  const lines = engine.sound_pool.filter((m) => m.pv && m.pv.length > 0).slice(0, 3);
  if (lines.length === 0) return null;

  const parts = fen.split(" ");
  const whiteToMove = parts[1] !== "b";
  const startNo = parseInt(parts[5] || "1", 10) || 1;
  const playable = Boolean(onPlayLine) && !disabled;

  return (
    <section aria-label="Likely continuations">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-medium text-ink">How it could continue</h3>
        <span className="text-xs text-muted">Click a line to play it — or a move to go deeper</span>
      </div>

      <ul className="flex flex-col gap-1.5">
        {lines.map((m, idx) => {
          const toks = lineTokens(m.pv, startNo, whiteToMove);
          const isRec = m.uci === recommendedUci;
          const top = idx === 0;
          // The whole row is a click target that plays this variation's move; the
          // per-move buttons inside stop propagation so they can jump deeper into
          // the same line. (List semantics stay on <li>; the row is a role=button.)
          const playLine = (count: number) => onPlayLine?.(m.pv, count);
          // The best line is distinguished by a slightly stronger surface + bolder
          // move weight (no accent border). Amber stays reserved for the coach's
          // recommended move (the eval pill + first token).
          return (
            <li key={m.uci}>
              <div
                role={playable ? "button" : undefined}
                tabIndex={playable ? 0 : undefined}
                aria-label={
                  playable
                    ? `${isRec ? "Recommended line. " : ""}Play this line, starting ${
                        m.pv[0]
                      }, evaluation ${fmtEval(m.cp)}`
                    : undefined
                }
                onClick={playable ? () => playLine(1) : undefined}
                onKeyDown={
                  playable
                    ? (e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          playLine(1);
                        }
                      }
                    : undefined
                }
                className={`flex items-start gap-2.5 rounded-[8px] border border-[color:var(--separator)] px-2.5 py-2 transition-colors ${
                  top ? "bg-[color:var(--surface-tertiary)]/70" : "bg-[color:var(--surface-tertiary)]/40"
                } ${
                  playable
                    ? "cursor-pointer hover:border-[color:var(--border)] hover:bg-[color:var(--surface-tertiary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/50"
                    : ""
                }`}
              >
                <span
                  className="mt-px shrink-0 rounded px-1.5 py-0.5 font-mono text-xs font-semibold tnum"
                  style={{
                    backgroundColor: isRec
                      ? "var(--signal)"
                      : "color-mix(in oklab, var(--ink) 8%, transparent)",
                    color: isRec ? "var(--signal-ink)" : "var(--muted)",
                  }}
                >
                  {fmtEval(m.cp)}
                </span>
                <div className="min-w-0 flex-1 font-serif text-[14px] leading-relaxed">
                  {toks.map((t, i) => {
                    const tokenColor =
                      i === 0
                        ? isRec
                          ? "font-semibold text-signal"
                          : "font-semibold text-ink"
                        : "text-muted hover:text-ink";
                    return (
                      <span key={i} className="whitespace-nowrap">
                        {t.num && <span className="text-faint">{t.num} </span>}
                        <button
                          type="button"
                          disabled={!playable}
                          onClick={(e) => {
                            e.stopPropagation();
                            playLine(i + 1);
                          }}
                          aria-label={`Play this line up to ${t.san}`}
                          className={`-mx-0.5 cursor-pointer rounded px-0.5 transition-colors hover:bg-[color:var(--surface-tertiary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/50 disabled:cursor-default disabled:hover:bg-transparent ${tokenColor}`}
                        >
                          {t.san}
                        </button>{" "}
                      </span>
                    );
                  })}
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
