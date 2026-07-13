"use client";

// Lichess' own board library. CSS is global (board layout + brown theme + the
// cburnett piece set, whose SVGs are embedded as data-URIs: no asset hosting).
import "chessground/assets/chessground.base.css";
import "chessground/assets/chessground.brown.css";
import "chessground/assets/chessground.cburnett.css";

import { Chessground } from "chessground";
import type { Api } from "chessground/api";
import type { DrawShape, DrawBrushes } from "chessground/draw";
import type * as cg from "chessground/types";
import { useCallback, useEffect, useRef } from "react";
import { playCapture, playMove } from "@/lib/sound";

// Arrow brushes: default set (for the user's own right-click arrows) + the coach's
// signal (recommended move) and your-move brushes used by autoShapes. Colors match
// the Bench Instrument palette: amber signal for the verdict, coral for your move.
const BRUSHES: DrawBrushes = {
  green: { key: "green", color: "#15781B", opacity: 0.9, lineWidth: 10 },
  red: { key: "red", color: "#882020", opacity: 0.9, lineWidth: 10 },
  blue: { key: "blue", color: "#003088", opacity: 0.9, lineWidth: 10 },
  yellow: { key: "yellow", color: "#e68f00", opacity: 0.9, lineWidth: 10 },
  signal: { key: "signal", color: "#eaa93a", opacity: 0.98, lineWidth: 13 },
  yourmove: { key: "yourmove", color: "#d47a54", opacity: 0.8, lineWidth: 11 },
};

export interface ChessgroundBoardProps {
  fen: string;
  orientation: cg.Color;
  turnColor: cg.Color;
  dests: cg.Dests;
  movableColor?: cg.Color;
  lastMove?: cg.Key[];
  check?: cg.Color | boolean;
  autoShapes?: DrawShape[];
  drawable?: boolean;
  coordinates?: boolean;
  /** Accessible name for the board as a whole. When set, the board wrapper is
   *  exposed to assistive tech as role="img" with this label (the chessground
   *  DOM underneath is decorative to a screen reader). */
  label?: string;
  /** Fired when the user drags/clicks a legal move (the move sticks; the parent
   *  advances the position and coaches the move). */
  onMove?: (orig: cg.Key, dest: cg.Key) => void;
}

export default function ChessgroundBoard(props: ChessgroundBoardProps) {
  const {
    fen,
    orientation,
    turnColor,
    dests,
    movableColor,
    lastMove,
    check = false,
    autoShapes = [],
    drawable = true,
    coordinates = true,
    label,
    onMove,
  } = props;

  const elRef = useRef<HTMLDivElement>(null);
  const apiRef = useRef<Api | null>(null);
  // Last fen we handed to chessground. Used so the sync effect only re-sends the
  // fen when the position actually changed: chessground's configure() resets
  // user-drawn shapes whenever a fen is present, so re-sending it on every
  // annotation/orientation update would wipe the user's arrows (see sync effect).
  const prevFenRef = useRef(fen);
  const onMoveRef = useRef(onMove);
  // Latest fen, read by handleAfter so it can SNAP THE PIECE BACK to the reviewed
  // (pre-move) position after a drag: the drag is an annotation, not a board move.
  const fenRef = useRef(fen);
  // Latest annotation arrows, read by the bounds-guarded drawer below so a resize
  // (or a late first layout) always repaints the current shapes.
  const shapesRef = useRef(autoShapes);
  // Keep the latest onMove + shapes + fen without re-running the mount effect
  // (updated post-render, before any board interaction can fire handleAfter).
  useEffect(() => {
    onMoveRef.current = onMove;
    shapesRef.current = autoShapes;
    fenRef.current = fen;
  });

  // Apply the coach's auto-shapes (the pinned recommended + your-move arrows).
  // These are chessground AUTOSHAPES: app-controlled and redrawn by us, and are
  // NEVER touched by the user's own right-click arrows or left-click clears (those
  // only mutate drawable.shapes), so the two coaching arrows are permanent. We
  // pass the shapes EXPLICITLY from the current render (not via shapesRef) so the
  // recommended arrow can never lag a tier behind when the user switches levels.
  // Bounds-guarded: chessground's pos2user divides board width by height, so a 0×0
  // board would produce 0/0 = NaN <line> coordinates; we bail until it has real
  // size (the ResizeObserver re-invokes the draw the moment the board sizes).
  const drawArrows = useCallback((shapes: DrawShape[] = shapesRef.current) => {
    const cg = apiRef.current;
    const el = elRef.current;
    if (!cg || !el) return;
    const b = el.getBoundingClientRect();
    if (b.width === 0 || b.height === 0) return;
    cg.setAutoShapes(shapes);
  }, []);

  // Resize / first-layout path: the board bounds changed, so refresh chessground's
  // bounds memo (redrawAll) before repainting the current shapes. Kept out of the
  // per-update path so a plain tier/prop change is a light setAutoShapes, not a
  // full DOM rebuild.
  const redrawForResize = useCallback(() => {
    const cg = apiRef.current;
    const el = elRef.current;
    if (!cg || !el) return;
    const b = el.getBoundingClientRect();
    if (b.width === 0 || b.height === 0) return;
    cg.redrawAll();
    cg.setAutoShapes(shapesRef.current);
  }, []);

  // Stable move handler: play the sound, report the move, then SNAP THE PIECE
  // BACK. A drag on this board is a MOVE-REVIEW ANNOTATION, not a board advance:
  // the parent keeps the reviewed (pre-move) position and only records the move as
  // "your move" (student arrow), so we revert chessground's own optimistic drop by
  // re-setting the current (pre-move) fen. The piece animates back to its origin
  // and the position stays put, so the coach's arrows never contradict the board.
  // (When no onMove consumer is wired — the read-only Showcase boards — nothing to
  // revert, so we skip it.)
  const handleAfter = useCallback((orig: cg.Key, dest: cg.Key, meta: cg.MoveMetadata) => {
    if (meta?.captured) playCapture();
    else playMove();
    const cb = onMoveRef.current;
    if (!cb) return;
    cb(orig, dest);
    apiRef.current?.set({ fen: fenRef.current });
  }, []);

  // Mount once.
  useEffect(() => {
    if (!elRef.current) return;
    // Respect the user's reduced-motion preference: no piece-slide animation.
    const prefersReducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const api = Chessground(elRef.current, {
      fen,
      orientation,
      turnColor,
      coordinates,
      check,
      lastMove,
      highlight: { lastMove: true, check: true },
      animation: { enabled: !prefersReducedMotion, duration: 200 },
      movable: {
        free: false,
        color: movableColor,
        dests,
        showDests: true,
        events: { after: handleAfter },
      },
      // Start with no annotations; the sync effect applies them once the board has
      // real bounds (drawing arrows at 0x0 yields NaN line coordinates).
      drawable: { enabled: drawable, visible: true, autoShapes: [], brushes: BRUSHES },
    });
    apiRef.current = api;
    // A ResizeObserver fires on first layout and on every resize; each time we
    // (re)draw the annotation arrows. drawArrows() is bounds-guarded, so it
    // repositions/repaints only when the board has real size and never emits the
    // NaN arrow coordinates a 0×0 board would produce. This replaces the old
    // requestAnimationFrame(redrawAll/bounds) loop that thrashed layout every
    // frame until the board measured.
    const ro = new ResizeObserver(() => redrawForResize());
    ro.observe(elRef.current);
    return () => {
      ro.disconnect();
      api.destroy();
      apiRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync on prop changes.
  useEffect(() => {
    const api = apiRef.current;
    if (!api) return;
    // Only include `fen` in the update when the position actually changed.
    // chessground's configure() wipes user-drawn shapes (arrows/circles) any time
    // a fen is present in the config, so re-sending an unchanged fen on an
    // annotation/orientation/turn update would erase the user's own drawings.
    // Omitting it preserves them; a genuine position change still resets them to
    // a clean slate (matching Lichess). Piece-move animation still runs because a
    // real move always carries a new fen.
    const fenChanged = prevFenRef.current !== fen;
    prevFenRef.current = fen;
    api.set({
      ...(fenChanged ? { fen } : {}),
      orientation,
      turnColor,
      check,
      lastMove,
      movable: {
        free: false,
        color: movableColor,
        dests,
        showDests: true,
        events: { after: handleAfter },
      },
    });
    // Re-apply the coach arrows from THIS render's shapes (bounds-guarded) so a
    // tier switch updates the recommended arrow immediately, and so a fen change
    // (which clears user-drawn shapes) never drops the pinned coach arrows.
    drawArrows(autoShapes);
  }, [fen, orientation, turnColor, movableColor, lastMove, check, dests, autoShapes, handleAfter, drawArrows]);

  return (
    <div
      ref={elRef}
      className="cg-wrap"
      style={{ width: "100%", aspectRatio: "1 / 1" }}
      role={label ? "img" : undefined}
      aria-label={label}
    />
  );
}
