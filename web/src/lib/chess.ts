// Thin client-side chess helpers built on chess.js (validation + geometry).
import { Chess } from "chess.js";

export type Orientation = "white" | "black";

export interface ValidatedFen {
  ok: boolean;
  sideToMove: Orientation;
  gameOver: boolean;
  error?: string;
}

export function validateFen(fen: string): ValidatedFen {
  try {
    const game = new Chess(fen);
    return {
      ok: true,
      sideToMove: game.turn() === "w" ? "white" : "black",
      gameOver: game.isGameOver(),
    };
  } catch (err) {
    return {
      ok: false,
      sideToMove: "white",
      gameOver: false,
      error: err instanceof Error ? err.message : "Invalid FEN",
    };
  }
}

export function sideToMove(fen: string): Orientation {
  return validateFen(fen).sideToMove;
}

/**
 * Validate a drag (source -> target) against the position and return its UCI
 * (with a queen promotion when a pawn reaches the last rank), or null if the
 * move is not legal. The board itself is never mutated: the position stays put
 * so the drag becomes an annotation ("the move you are unsure about").
 */
export function legalDragUci(fen: string, from: string, to: string): string | null {
  try {
    const game = new Chess(fen);
    const piece = game.get(from as never);
    const isPawn = piece && piece.type === "p";
    const lastRank = to.endsWith("8") || to.endsWith("1");
    const promotion = isPawn && lastRank ? "q" : undefined;
    const move = game.move({ from, to, promotion });
    if (!move) return null;
    return move.from + move.to + (move.promotion ?? "");
  } catch {
    return null;
  }
}

/** Accept a SAN or UCI move on a position and return its UCI, or null. */
export function moveToUci(fen: string, move: string): string | null {
  const text = move.trim();
  if (!text) return null;
  try {
    const game = new Chess(fen);
    const m = game.move(text); // chess.js accepts SAN and long-algebraic
    return m ? m.from + m.to + (m.promotion ?? "") : null;
  } catch {
    return null;
  }
}

/** SAN for a UCI (or SAN) move on a position, for display. */
export function uciToSan(fen: string, move: string): string | null {
  const text = move.trim();
  if (!text) return null;
  try {
    const game = new Chess(fen);
    const from = text.slice(0, 2);
    const to = text.slice(2, 4);
    const promotion = text.length > 4 ? text[4] : undefined;
    const m = game.move({ from, to, promotion } as never) ?? game.move(text);
    return m ? m.san : null;
  } catch {
    try {
      const game = new Chess(fen);
      const m = game.move(text);
      return m ? m.san : null;
    } catch {
      return null;
    }
  }
}

/** Apply a UCI move to a FEN and return the resulting FEN, or null if illegal. */
export function applyUciMove(fen: string, uci: string): string | null {
  const text = uci.trim();
  if (text.length < 4) return null;
  try {
    const game = new Chess(fen);
    const from = text.slice(0, 2);
    const to = text.slice(2, 4);
    const promotion = text.length > 4 ? text[4] : "q";
    const m = game.move({ from, to, promotion });
    return m ? game.fen() : null;
  } catch {
    return null;
  }
}

export interface SteppedLine {
  /** Positions BEFORE the board's current one, oldest → newest (for take-back). */
  history: string[];
  /** The resulting board position after applying `applied` moves. */
  boardFen: string;
  /** Squares of the last applied move, for the board's last-move highlight. */
  lastMove: [string, string] | null;
  /** Whether the last applied move was a capture (chooses the sound). */
  captured: boolean;
  /** How many half-moves were actually applied (stops early on an illegal move). */
  applied: number;
}

/**
 * Play the first `count` SAN moves of a principal variation from `startFen`.
 * Returns the resulting board position plus the chain of prior positions so the
 * existing take-back model can walk back move by move (Lichess/chess.com style).
 */
export function stepSanLine(startFen: string, sanMoves: string[], count: number): SteppedLine | null {
  try {
    const game = new Chess(startFen);
    const history: string[] = [startFen];
    let lastMove: [string, string] | null = null;
    let captured = false;
    let applied = 0;
    const n = Math.min(count, sanMoves.length);
    for (let i = 0; i < n; i++) {
      const m = game.move(sanMoves[i]);
      if (!m) break;
      lastMove = [m.from, m.to];
      captured = Boolean(m.captured);
      applied++;
      // Record every position except the final board (which is returned separately).
      if (i < n - 1) history.push(game.fen());
    }
    if (applied === 0) return null;
    return { history, boardFen: game.fen(), lastMove, captured, applied };
  } catch {
    return null;
  }
}

export interface Squares {
  from: string;
  to: string;
}

export function uciToSquares(uci: string): Squares | null {
  if (!uci || uci.length < 4) return null;
  return { from: uci.slice(0, 2), to: uci.slice(2, 4) };
}

/** Center of a square in a normalized 0..8 board space, honoring orientation. */
export function squareCenter(square: string, orientation: Orientation): { x: number; y: number } {
  const file = square.charCodeAt(0) - 97; // a=0 .. h=7
  const rank = parseInt(square[1], 10) - 1; // 1=0 .. 8=7
  if (orientation === "white") {
    return { x: file + 0.5, y: 7 - rank + 0.5 };
  }
  return { x: 7 - file + 0.5, y: rank + 0.5 };
}
