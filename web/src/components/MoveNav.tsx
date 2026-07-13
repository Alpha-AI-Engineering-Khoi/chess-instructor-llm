"use client";

import { Button } from "@heroui/react";
import type { MainlinePly } from "@/lib/chess";
import {
  ChevronDoubleLeftIcon,
  ChevronDoubleRightIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from "./icons";

/**
 * Lichess-style move navigator for the coached position's mainline: first / prev /
 * next / last controls plus a compact, clickable SAN move list. Ply 0 is the
 * loaded position; plies 1..N are the coach's recommended move + engine PV. The
 * parent owns the current ply and applies it to the board.
 */
export default function MoveNav({
  plies,
  currentPly,
  onJump,
}: {
  plies: MainlinePly[];
  currentPly: number;
  onJump: (ply: number) => void;
}) {
  const last = plies.length - 1;
  if (last < 1) return null; // nothing to navigate
  const atStart = currentPly <= 0;
  const atEnd = currentPly >= last;

  const navBtn = (
    label: string,
    icon: React.ReactNode,
    to: number,
    disabled: boolean,
  ) => (
    <Button
      isIconOnly
      variant="tertiary"
      size="sm"
      className="min-h-9 min-w-9"
      aria-label={label}
      isDisabled={disabled}
      onPress={() => onJump(to)}
    >
      {icon}
    </Button>
  );

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-1">
        {navBtn("First move", <ChevronDoubleLeftIcon width={16} height={16} />, 0, atStart)}
        {navBtn("Previous move", <ChevronLeftIcon width={16} height={16} />, currentPly - 1, atStart)}
        {navBtn("Next move", <ChevronRightIcon width={16} height={16} />, currentPly + 1, atEnd)}
        {navBtn("Last move", <ChevronDoubleRightIcon width={16} height={16} />, last, atEnd)}
        <span className="ml-1 text-xs text-muted tnum" aria-hidden>
          {currentPly}/{last}
        </span>
      </div>

      {/* Clickable SAN move list; clicking a move jumps the board to that ply. */}
      <ol
        aria-label="Coach mainline moves"
        className="flex flex-wrap items-center gap-x-1.5 gap-y-1 rounded-[10px] border border-[color:var(--border)] bg-[color:var(--surface)] px-3 py-2 text-sm"
      >
        {plies.slice(1).map((ply, i) => {
          const idx = i + 1;
          const showNum = ply.whiteMoved || idx === 1;
          const numLabel = ply.whiteMoved ? `${ply.moveNumber}.` : `${ply.moveNumber}\u2026`;
          const active = idx === currentPly;
          return (
            <li key={idx} className="inline-flex items-center gap-1">
              {showNum && <span className="text-faint tnum">{numLabel}</span>}
              <button
                type="button"
                onClick={() => onJump(idx)}
                aria-current={active ? "true" : undefined}
                className={`cursor-pointer rounded px-1 font-serif tnum transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60 ${
                  active
                    ? "bg-signal/20 font-semibold text-ink"
                    : "text-muted hover:bg-[color:var(--surface-tertiary)] hover:text-ink"
                }`}
              >
                {ply.san}
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
