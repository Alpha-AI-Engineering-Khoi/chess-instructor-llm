"use client";

// A static, non-interactive board for the Showdown rows. Reuses the same
// ChessgroundBoard (Lichess board) the Studio uses, but with no legal-move
// targets: it only renders a position plus up to two annotation arrows — the
// selected model's recommended move (amber signal) and the student's mistake
// (coral). Cheap enough to render many at once.

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

  return (
    <div className="relative aspect-square w-full select-none">
      <ChessgroundBoard
        fen={fen}
        orientation={orientation}
        turnColor={turnColor}
        dests={NO_DESTS}
        movableColor={undefined}
        autoShapes={shapes}
        drawable={false}
        coordinates
      />
    </div>
  );
}
