"use client";

// A board for the Showcase/Showdown views. Reuses the same ChessgroundBoard
// (Lichess board) the Studio uses, but the PIECES are non-movable (no legal-move
// targets): it renders a position plus up to two annotation arrows — the selected
// model's recommended move (amber signal) and the student's mistake (coral).
//
// Drawing IS enabled: like on Lichess, the user can right-click-drag (or
// shift-drag) to draw their own arrows and circles for analysis. Those user
// shapes are a separate layer from the annotation autoShapes, so both coexist —
// and ChessgroundBoard preserves user drawings across annotation/tier/model
// changes (it only resets them when the position itself changes). Cheap enough to
// render many at once.

import { useMemo } from "react";
import type { DrawShape } from "chessground/draw";
import type * as cg from "chessground/types";
import ChessgroundBoard from "./ChessgroundBoard";
import { uciToSquares, type Orientation } from "@/lib/chess";

const NO_DESTS: cg.Dests = new Map();

interface ShowdownBoardProps {
  fen: string;
  orientation: Orientation;
  /** The recommended move to spotlight (amber). */
  moveUci?: string | null;
  /** The student's move under review (coral, contextual). */
  studentUci?: string | null;
}

export default function ShowdownBoard({ fen, orientation, moveUci, studentUci }: ShowdownBoardProps) {
  const turnColor: cg.Color = fen.split(" ")[1] === "b" ? "black" : "white";

  const shapes = useMemo<DrawShape[]>(() => {
    const out: DrawShape[] = [];
    if (studentUci) {
      const s = uciToSquares(studentUci);
      if (s) out.push({ orig: s.from as cg.Key, dest: s.to as cg.Key, brush: "yourmove" });
    }
    if (moveUci) {
      const s = uciToSquares(moveUci);
      if (s) out.push({ orig: s.from as cg.Key, dest: s.to as cg.Key, brush: "signal" });
    }
    return out;
  }, [moveUci, studentUci]);

  // Accessible description: the position plus a hint that arrows can be drawn.
  const boardLabel = `Chess position — ${
    turnColor === "white" ? "White" : "Black"
  } to move. Right-click and drag to draw your own arrows.`;

  return (
    <div className="relative aspect-square w-full select-none">
      <ChessgroundBoard
        fen={fen}
        orientation={orientation}
        turnColor={turnColor}
        dests={NO_DESTS}
        movableColor={undefined}
        autoShapes={shapes}
        // Pieces stay locked (dests empty, no movable color) but drawing is on,
        // so the user gets Lichess-style right-click-drag arrows/circles that
        // sit alongside the recommended-move + student-move annotations.
        drawable
        coordinates
        label={boardLabel}
      />
    </div>
  );
}
