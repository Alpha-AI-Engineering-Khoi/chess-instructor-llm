"use client";

// Lichess' own board library. CSS is global (board layout + brown theme + the
// cburnett piece set, whose SVGs are embedded as data-URIs — no asset hosting).
import "chessground/assets/chessground.base.css";
import "chessground/assets/chessground.brown.css";
import "chessground/assets/chessground.cburnett.css";

import { Chessground } from "chessground";
import type { Api } from "chessground/api";
import type { DrawShape, DrawBrushes } from "chessground/draw";
import type * as cg from "chessground/types";
import { useCallback, useEffect, useRef, useState } from "react";
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
  // fen when the position actually changed — chessground's configure() resets
  // user-drawn shapes whenever a fen is present, so re-sending it on every
  // annotation/orientation update would wipe the user's arrows (see sync effect).
  const prevFenRef = useRef(fen);
  // The board is "ready" once it has been laid out (double-rAF after mount); only
  // then do we draw annotation arrows, so chessground never computes NaN geometry
  // from a 0×0 board (a mobile-first-paint issue that logs console errors).
  const [ready, setReady] = useState(false);
  const didFirstDraw = useRef(false);
  const rafRef = useRef(0);
  const onMoveRef = useRef(onMove);
  // Keep the latest onMove without re-running the mount effect (updated post-render,
  // before any board interaction can fire handleAfter).
  useEffect(() => {
    onMoveRef.current = onMove;
  });

  // Stable move handler: play the sound and report the move. The move "sticks"
  // (the board keeps it); the parent advances its state and the sync effect
  // confirms the resulting position (a no-op diff, so no flicker).
  const handleAfter = useCallback((orig: cg.Key, dest: cg.Key, meta: cg.MoveMetadata) => {
    if (meta?.captured) playCapture();
    else playMove();
    onMoveRef.current?.(orig, dest);
  }, []);

  // Mount once.
  useEffect(() => {
    if (!elRef.current) return;
    const api = Chessground(elRef.current, {
      fen,
      orientation,
      turnColor,
      coordinates,
      check,
      lastMove,
      highlight: { lastMove: true, check: true },
      animation: { enabled: true, duration: 200 },
      movable: {
        free: false,
        color: movableColor,
        dests,
        showDests: true,
        events: { after: handleAfter },
      },
      // Start with no annotations; the sync effect applies them once the board has
      // real bounds (drawing arrows at 0×0 yields NaN line coordinates).
      drawable: { enabled: drawable, visible: true, autoShapes: [], brushes: BRUSHES },
    });
    apiRef.current = api;
    const ro = new ResizeObserver(() => api.redrawAll());
    ro.observe(elRef.current);
    // Poll redrawAll (which rebuilds chessground's bounds memo) until the board's
    // OWN measured bounds are non-zero, then flip ready. Chessground computes arrow
    // geometry as min(1, bounds.width / bounds.height); a 0×0 board yields NaN
    // <line> coordinates, so we never draw arrows until its bounds are real.
    const waitForLayout = () => {
      const cg = apiRef.current;
      if (!cg) return;
      cg.redrawAll();
      const b = cg.state.dom.bounds();
      if (b.width > 0 && b.height > 0) {
        setReady(true);
      } else {
        rafRef.current = requestAnimationFrame(waitForLayout);
      }
    };
    rafRef.current = requestAnimationFrame(waitForLayout);
    return () => {
      cancelAnimationFrame(rafRef.current);
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
    // Draw arrows only after the board is laid out. Refresh chessground's bounds
    // memo once (redrawAll) before the first arrow draw so geometry is never NaN.
    if (ready) {
      if (!didFirstDraw.current) {
        api.redrawAll();
        didFirstDraw.current = true;
      }
      api.setAutoShapes(autoShapes);
    }
  }, [fen, orientation, turnColor, movableColor, lastMove, check, dests, autoShapes, handleAfter, ready]);

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
