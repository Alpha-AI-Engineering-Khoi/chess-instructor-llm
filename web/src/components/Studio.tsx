"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Button, FieldError, Input, Label, Separator, TextArea, TextField, Tooltip } from "@heroui/react";
import {
  getLibrary,
  postCoachResilient,
  warmupCoach,
  type CoachResponse,
  type CoachWakeStatus,
  type LibraryEntry,
  type Tier,
} from "@/lib/api";
import {
  legalDragUci,
  moveToUci,
  sideToMove,
  stepSanLine,
  uciToSan,
  uciToSquares,
  validateFen,
  type Orientation,
} from "@/lib/chess";
import { deriveOursLabel } from "@/lib/showcase";
import { STUDIO_DEFAULT_TIERS } from "@/lib/studioDefault";
import { playCapture, playMove } from "@/lib/sound";
import BoardStage, { type StageArrow } from "./BoardStage";
import TierControl from "./TierControl";
import CoachingReveal from "./CoachingReveal";
import PositionLibrary, { type LibStatus } from "./PositionLibrary";
import CopyFenButton from "./CopyFenButton";
import { FlipVerticalIcon, ResetIcon, UndoIcon } from "./icons";

type Status = "idle" | "loading" | "done" | "error";

// DEFAULT is a GENUINE per-tier fork, regenerated live through the v6-dpo2 endpoint
// (the same tuned model the demo serves) — the precomputed answers live in
// web/src/lib/studioDefault.ts. On this king-and-pawn endgame the student's Ne2 was
// fine but passive, and the coach hands back an engine-sound, tier-appropriate move
// per level: Ne6+ (beginner — a forcing check) vs h6 (intermediate / advanced — the
// passed-pawn push). Beginner forks away from the stronger tiers, so switching to
// Beginner changes the move — the one-screen proof that the fine-tune adapts to the
// player's level. (v6-dpo2's gain over v4 is small and concentrated in the
// intermediate tier, so this position reads as a 2-move fork rather than v4's
// 3-move split — kept honest rather than cherry-picked.)
const DEFAULT = {
  fen: "8/7b/5p2/P1kp3P/2pN1P2/4K3/8/8 w - - 1 39",
  tier: "beginner" as Tier,
  move: "Ne2",
};
const DEFAULT_STUDENT_UCI = moveToUci(DEFAULT.fen, DEFAULT.move);
const TIER_LABEL: Record<Tier, string> = {
  beginner: "Beginner",
  intermediate: "Intermediate",
  advanced: "Advanced",
};

interface Preset {
  label: string;
  hint: string;
  fen: string;
  move?: string;
}

// A few one-click positions: three "moat" demos (held-out positions where the
// tuned model gives a GENUINELY distinct move per level — verified on the v4
// generations with the avoid-framing-aware extractor, one per game phase), a
// handful of recognizable openings, and the classic beginner blunder.
const PRESETS: Preset[] = [
  {
    label: "Moat · opening",
    hint: "Held-out opening: Beginner/Intermediate keep the recapture Nxd4, but the tuned model steers Advanced to Bxf6 first. Switch tiers to watch the move change.",
    fen: "r2q1rk1/ppp2ppp/5nb1/2b2NB1/2Bn4/2NP4/PPP3PP/R2QKR2 w Q - 3 13",
    move: "Nxd4",
  },
  {
    label: "Moat · middlegame",
    hint: "Your Qh5+ is right for Beginner/Intermediate; Advanced prefers the quiet f6. A verified per-level fork on a held-out position.",
    fen: "r4rk1/1p3ppp/pQp1n2q/4P3/2P2n2/P3BK2/7P/R4B1R b - - 0 21",
    move: "Qh5",
  },
  {
    label: "Moat · rook endgame",
    hint: "Beginner/Intermediate get the simpler Ng3; Advanced endorses the student's own Nf2. A verified per-level fork.",
    fen: "8/8/8/4k3/4NRK1/7p/8/7r w - - 0 52",
    move: "Nf2",
  },
  {
    label: "Italian Game",
    hint: "1.e4 e5 2.Nf3 Nc6 3.Bc4: Black to choose a developing move.",
    fen: "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
  },
  {
    label: "Ruy López",
    hint: "1.e4 e5 2.Nf3 Nc6 3.Bb5: the Spanish, Black to respond.",
    fen: "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
  },
  {
    label: "Sicilian Defence",
    hint: "1.e4 c5: White to develop against the Sicilian.",
    fen: "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
  },
  {
    label: "Early queen 2.Qh5?!",
    hint: "The classic beginner queen sortie: sound pressure, or just a target?",
    fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    move: "Qh5",
  },
  {
    label: "Starting position",
    hint: "A fresh game: White to make the first move.",
    fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  },
];

export default function Studio() {
  const [library, setLibrary] = useState<LibraryEntry[]>([]);
  const [libStatus, setLibStatus] = useState<LibStatus>("loading");
  const [activeLibId, setActiveLibId] = useState<string | null>(null);

  const [fen, setFen] = useState(DEFAULT.fen); // what the board shows
  const [coachedFen, setCoachedFen] = useState(DEFAULT.fen); // position the result is about
  const [history, setHistory] = useState<string[]>([]); // prior board positions (for take-back)
  const [lastMove, setLastMove] = useState<string[] | null>(
    DEFAULT_STUDENT_UCI ? [DEFAULT_STUDENT_UCI.slice(0, 2), DEFAULT_STUDENT_UCI.slice(2, 4)] : null,
  );
  const [tier, setTier] = useState<Tier>(DEFAULT.tier);
  const [studentUci, setStudentUci] = useState<string | null>(DEFAULT_STUDENT_UCI);
  const [orientation, setOrientation] = useState<Orientation>(sideToMove(DEFAULT.fen));

  const [fenDraft, setFenDraft] = useState(DEFAULT.fen);
  const [moveDraft, setMoveDraft] = useState(DEFAULT.move);

  // Per-tier cache for the CURRENT coaching session (the position + student move
  // in `coachedFen`/`studentUci`). Switching Beginner/Intermediate/Advanced just
  // swaps the shown answer with no new network call. On mount this is SEEDED from
  // PRECOMPUTED static data for the default position (all three tiers), so the
  // tier-adaptive move renders instantly with the coach cold. Live inference is
  // then strictly user-initiated (per tier); a new position resets these.
  const [tierResults, setTierResults] = useState<Record<Tier, CoachResponse | undefined>>(() => ({
    beginner: STUDIO_DEFAULT_TIERS.beginner,
    intermediate: STUDIO_DEFAULT_TIERS.intermediate,
    advanced: STUDIO_DEFAULT_TIERS.advanced,
  }));
  const [tierStatus, setTierStatus] = useState<Record<Tier, Status>>(() => ({
    beginner: "done",
    intermediate: "done",
    advanced: "done",
  }));
  // Whether the shown answer for each tier is PRECOMPUTED (cached from the
  // benchmark) rather than a fresh live model call, so the UI can label it.
  const [tierPrecomputed, setTierPrecomputed] = useState<Record<Tier, boolean>>(() => ({
    beginner: true,
    intermediate: true,
    advanced: true,
  }));
  const [tierError, setTierError] = useState<Record<Tier, string | null>>(() => ({
    beginner: null,
    intermediate: null,
    advanced: null,
  }));
  const [wakingCoach, setWakingCoach] = useState<CoachWakeStatus | null>(null);
  // Which tier currently has a live model call in flight (drives Cancel + status).
  const [liveTier, setLiveTier] = useState<Tier | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const didInit = useRef(false);

  // Start a fresh coaching SESSION for a position + student move. Any tier we
  // already have a PRECOMPUTED answer for (the default's three tiers on mount, or
  // a library entry's cached tier) is seeded so it shows INSTANTLY; every other
  // tier goes idle and is coached only on an explicit, user-initiated "Run live".
  // Nothing here fires a model call, so changing positions never blocks the board
  // on a cold start. Aborts any live call still in flight for the old session.
  const resetSession = useCallback(
    (f: string, s: string | null, seed?: Partial<Record<Tier, CoachResponse>>) => {
      abortRef.current?.abort();
      abortRef.current = null;
      setCoachedFen(f); // every cached/live tier answer is about position f
      setWakingCoach(null);
      setLiveTier(null);
      const seeded = seed ?? {};
      setTierResults({
        beginner: seeded.beginner,
        intermediate: seeded.intermediate,
        advanced: seeded.advanced,
      });
      setTierStatus({
        beginner: seeded.beginner ? "done" : "idle",
        intermediate: seeded.intermediate ? "done" : "idle",
        advanced: seeded.advanced ? "done" : "idle",
      });
      setTierPrecomputed({
        beginner: Boolean(seeded.beginner),
        intermediate: Boolean(seeded.intermediate),
        advanced: Boolean(seeded.advanced),
      });
      setTierError({ beginner: null, intermediate: null, advanced: null });
    },
    [],
  );

  // User-initiated LIVE inference for a SINGLE tier (the active one; the others
  // are already cached). Sends the exact position the console is about
  // (coachedFen) + the student move through the hosted coach-gate pipeline, with
  // cold-start-resilient progress. On success it swaps that tier's cached answer
  // for the live one; on FAILURE it KEEPS any cached answer (never overwrites a
  // good precomputed tier with an error panel) and only shows an error panel when
  // there was nothing cached to fall back to.
  const runLiveTier = useCallback(
    (t: Tier) => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      const f = coachedFen;
      const s = studentUci;
      const hadCached = tierResults[t] != null;
      setLiveTier(t);
      setWakingCoach(null);
      setTierStatus((prev) => ({ ...prev, [t]: "loading" }));
      setTierError((prev) => ({ ...prev, [t]: null }));

      void postCoachResilient(
        { fen: f, tier: t, student_move: s ?? undefined },
        {
          signal: ctrl.signal,
          onStatus: (st: CoachWakeStatus) => {
            if (!ctrl.signal.aborted) setWakingCoach(st);
          },
        },
      )
        .then((res) => {
          if (ctrl.signal.aborted) return;
          setTierResults((prev) => ({ ...prev, [t]: res }));
          setTierStatus((prev) => ({ ...prev, [t]: "done" }));
          setTierPrecomputed((prev) => ({ ...prev, [t]: false }));
          setWakingCoach(null);
          setLiveTier(null);
        })
        .catch((e: unknown) => {
          if (ctrl.signal.aborted) return;
          setWakingCoach(null);
          setLiveTier(null);
          const msg = e instanceof Error ? e.message : "Something went wrong.";
          if (hadCached) {
            // Keep the cached answer on screen; report the miss as a soft note.
            setTierStatus((prev) => ({ ...prev, [t]: "done" }));
            setTierError((prev) => ({ ...prev, [t]: msg }));
          } else {
            setTierStatus((prev) => ({ ...prev, [t]: "error" }));
            setTierError((prev) => ({ ...prev, [t]: msg }));
          }
        });
    },
    [coachedFen, studentUci, tierResults],
  );

  // Cancel an in-flight live call: abort, drop the "waking" state, and fall back
  // to the tier's cached answer if it has one (else idle) — never an error panel.
  const cancelLive = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    const t = liveTier;
    setWakingCoach(null);
    setLiveTier(null);
    if (t) {
      setTierStatus((prev) => ({ ...prev, [t]: tierResults[t] != null ? "done" : "idle" }));
    }
  }, [liveTier, tierResults]);

  const loadLibrary = useCallback(() => {
    setLibStatus("loading");
    getLibrary()
      .then((lib) => {
        setLibrary(lib);
        setLibStatus("ready");
      })
      .catch(() => setLibStatus("error"));
  }, []);

  useEffect(() => {
    if (didInit.current) return;
    didInit.current = true;
    warmupCoach(); // silently nudge the scaled-to-zero coach awake — no auto-coach
    loadLibrary();
    // The default position's three tiers are already SEEDED from precomputed data
    // (see the tier* useState initializers), so the board + coaching render
    // instantly with no live call. Live inference is user-initiated from here.
  }, [loadLibrary]);

  // The displayed answer is whatever the ACTIVE tier holds in the cache; a tier
  // switch just re-points these: no network call when a tier is already cached.
  const activeStatus = tierStatus[tier];
  const activeResult = tierResults[tier] ?? null;
  const activeError = tierError[tier];
  const activePrecomputed = tierPrecomputed[tier];
  // The active tier has a live model call in flight (the console shows a
  // skeleton). The BOARD is NEVER gated by this — the board, presets, the study
  // library, and tier switching all stay live while a call runs.
  const activeLoading = activeStatus === "loading";

  const toMove = useMemo(() => sideToMove(fen), [fen]);
  // OURS identity parsed from the live coach response (never hardcoded), so the
  // "· vN" chip tracks whatever model the backend is actually serving.
  const oursLabel = useMemo(
    () => (activeResult ? deriveOursLabel(activeResult.meta.model) : null),
    [activeResult],
  );
  const studentSan = useMemo(
    () => (studentUci ? uciToSan(coachedFen, studentUci) : null),
    [coachedFen, studentUci],
  );
  const draftUci = useMemo(
    () => (moveDraft.trim() ? moveToUci(fen, moveDraft) : null),
    [fen, moveDraft],
  );
  const fenDraftState = useMemo(() => validateFen(fenDraft), [fenDraft]);

  const arrows: StageArrow[] = useMemo(() => {
    // Only draw the coach's arrows when the board is on the position being coached.
    if (fen !== coachedFen) return [];
    if (activeStatus === "done" && activeResult) {
      const list: StageArrow[] = [];
      const su = activeResult.engine.student_move?.uci;
      if (su) {
        const sq = uciToSquares(su);
        if (sq) list.push({ from: sq.from, to: sq.to, kind: "student", delay: 0.45, draw: true });
      }
      const rq = uciToSquares(activeResult.recommended_move_uci);
      if (rq) list.push({ from: rq.from, to: rq.to, kind: "rec", delay: 0.15, draw: true });
      return list;
    }
    if (studentUci) {
      const sq = uciToSquares(studentUci);
      if (sq) return [{ from: sq.from, to: sq.to, kind: "student", delay: 0, draw: false }];
    }
    return [];
  }, [activeStatus, activeResult, studentUci, fen, coachedFen]);

  // Dragging a legal move REVIEWS it in place: the board does NOT advance
  // (ChessgroundBoard snaps the piece back to the reviewed position). We record it
  // as "your move" on the CURRENT position and start a fresh session; the coach
  // then runs only when the user clicks "Run live". Typed moves take the identical
  // path (see setMoveFromDraft), so drag and type behave the same. This is the
  // documented annotation behavior: the coach reviews (before, uci) and both the
  // student and coach arrows render on the SAME (pre-move) position.
  const onBoardMove = useCallback(
    (uci: string) => {
      const su = legalDragUci(fen, uci.slice(0, 2), uci.slice(2, 4));
      if (!su) return;
      setActiveLibId(null);
      setStudentUci(su);
      setLastMove([su.slice(0, 2), su.slice(2, 4)]);
      setMoveDraft("");
      resetSession(fen, su);
    },
    [fen, resetSession],
  );

  // Take back the last stepped engine-line move: return to the previous position.
  const undo = useCallback(() => {
    if (history.length === 0) return;
    const prev = history[history.length - 1];
    setHistory((h) => h.slice(0, -1));
    setFen(prev);
    setLastMove(null);
    setStudentUci(null);
    setMoveDraft("");
    setActiveLibId(null);
    resetSession(prev, null);
  }, [history, resetSession]);

  // Click a move in "Top engine lines" to play that variation onto the board:
  // apply the PV up to the clicked move from the analyzed position (coachedFen),
  // animate with the move sound, push history so take-back walks it back, then
  // coach the resulting position. Reuses the fen/lastMove/history board model.
  const playEngineLine = useCallback(
    (pv: string[], count: number) => {
      const stepped = stepSanLine(coachedFen, pv, count);
      if (!stepped) return;
      if (stepped.captured) playCapture();
      else playMove();
      setActiveLibId(null);
      setHistory(stepped.history);
      setFen(stepped.boardFen);
      setLastMove(stepped.lastMove);
      setStudentUci(null);
      setMoveDraft("");
      resetSession(stepped.boardFen, null);
    },
    [coachedFen, resetSession],
  );

  const flip = useCallback(() => setOrientation((o) => (o === "white" ? "black" : "white")), []);

  // Tier switching is INSTANT: all three bands were fetched together for this
  // position, so we just swap which cached answer is displayed: no new model
  // call. (A tier still in flight shows a lightweight per-tier skeleton via the
  // console until its result lands.) Dropping the library pin keeps the highlight
  // honest once the shown level differs from the pinned entry.
  const changeTier = useCallback(
    (t: Tier) => {
      if (t === tier) return;
      setTier(t);
      setActiveLibId(null);
    },
    [tier],
  );

  const clearMove = () => {
    setStudentUci(null);
    setMoveDraft("");
    setLastMove(null);
    resetSession(fen, null);
  };

  // Coach the ACTIVE tier LIVE on the current position. The single user-initiated
  // live entry point: it runs only the shown level (the other tiers are already
  // cached, or run on demand when you switch to them), so a click costs exactly
  // one model call.
  const runLiveActive = useCallback(() => {
    setActiveLibId(null);
    runLiveTier(tier);
  }, [tier, runLiveTier]);

  const selectLibraryItem = (e: LibraryEntry) => {
    const su = e.student_move ? moveToUci(e.fen, e.student_move) : null;
    setFen(e.fen);
    setHistory([]);
    setFenDraft(e.fen);
    setTier(e.tier);
    setStudentUci(su);
    setMoveDraft(e.student_move ?? "");
    setLastMove(su ? [su.slice(0, 2), su.slice(2, 4)] : null);
    setOrientation((e.coach.side_to_move as Orientation) ?? sideToMove(e.fen));
    setActiveLibId(e.id);
    // Seed the entry's tier with its cached coaching (precomputed, instant); the
    // other two tiers stay idle until the user runs them live.
    resetSession(e.fen, su, { [e.tier]: e.coach });
  };

  const loadFen = () => {
    const v = validateFen(fenDraft);
    if (!v.ok) return;
    const f = fenDraft.trim();
    setFen(f);
    setHistory([]);
    setStudentUci(null);
    setMoveDraft("");
    setLastMove(null);
    setOrientation(v.sideToMove);
    setActiveLibId(null);
    resetSession(f, null);
  };

  // One-click preset: load the position (and its illustrative move, if any). The
  // board is usable immediately; coaching for the position is user-initiated via
  // "Run live" (no auto-fire, so a cold coach never blocks the load).
  const selectPreset = (p: Preset) => {
    const su = p.move ? moveToUci(p.fen, p.move) : null;
    setFen(p.fen);
    setHistory([]);
    setFenDraft(p.fen);
    setMoveDraft(p.move ?? "");
    setStudentUci(su);
    setLastMove(su ? [su.slice(0, 2), su.slice(2, 4)] : null);
    setOrientation(sideToMove(p.fen));
    setActiveLibId(null);
    resetSession(p.fen, su);
  };

  // Review a typed move on the CURRENT position without advancing the board:
  // identical to a drag (see onBoardMove).
  const setMoveFromDraft = () => {
    if (!draftUci) return;
    setStudentUci(draftUci);
    setLastMove([draftUci.slice(0, 2), draftUci.slice(2, 4)]);
    resetSession(fen, draftUci);
  };

  // Keyboard accelerators for the primary board flow (ignored while typing).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        flip();
      } else if ((e.key === "z" || e.key === "Z") && history.length > 0) {
        e.preventDefault();
        undo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [flip, undo, history.length]);

  // Label for the user-initiated live button: run the shown level on the position.
  const runLiveLabel = `Run live · ${TIER_LABEL[tier]}`;

  return (
    <div className="relative z-[1] mx-auto flex min-h-dvh w-full max-w-[1240px] flex-col gap-8 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      {/* Top-of-page intro. The hero is the ONE behavior: the tuned model selects
          the tier-appropriate MOVE (+ a short principle tag) substantially more
          reliably than its base or the frontier; the coaching prose is a
          secondary, optional layer, and the cross-model views are the bonus
          comparison. */}
      <header className="flex flex-col gap-5">
        <div className="flex items-center justify-end gap-2">
          <Link
            href="/showdown"
            className="inline-flex min-h-9 items-center gap-1.5 rounded-full px-3.5 text-sm font-medium text-muted ring-1 ring-[color:var(--border)] transition-colors hover:text-ink hover:ring-[color:var(--field-border)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60"
          >
            Showdown list
            <span aria-hidden className="text-faint">›</span>
          </Link>
          <Link
            href="/showcase"
            className="inline-flex min-h-9 items-center gap-1.5 rounded-full bg-signal/12 px-3.5 text-sm font-medium text-signal ring-1 ring-signal/40 transition-colors hover:bg-signal/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60"
          >
            Multi-Model Showcase
            <span aria-hidden>★</span>
            <span aria-hidden className="text-signal/70">›</span>
          </Link>
        </div>

        <div className="flex max-w-3xl flex-col gap-2.5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex w-fit items-center rounded-full bg-signal/12 px-2.5 py-1 text-[11px] font-medium text-signal ring-1 ring-signal/30">
              One behavior: move selection
            </span>
            {oursLabel?.version && (
              <span className="rounded-full px-2.5 py-1 font-mono text-[11px] text-signal ring-1 ring-signal/40 tnum">
                {oursLabel.badge}
              </span>
            )}
          </div>
          <h1 className="text-2xl font-semibold leading-tight tracking-tight text-balance text-ink sm:text-[2rem]">
            The fine-tune picks the level-appropriate move far more reliably than its base or the
            frontier.
          </h1>
          <p className="text-pretty text-sm leading-relaxed text-muted sm:text-base">
            A fine-tuned chess coach served on a hosted endpoint. Set a position, mark the move
            you&apos;re unsure about, and pick your rating. It hands back <span className="text-ink">one
            move chosen for your level</span> and a short <span className="text-ink">principle
            tag</span> for why. Its one job is selecting the tier-appropriate move; the full
            explanation is an <span className="text-ink">optional layer</span> underneath, not the
            headline. On the held-out benchmark it is substantially more reliable at this{" "}
            <span className="text-ink tnum">(76.7% tuned vs 34.7% base / 55.3% best frontier)</span>.{" "}
            <span className="text-faint">
              See the{" "}
              <Link
                href="/showcase"
                className="text-muted underline decoration-dotted underline-offset-2 transition-colors hover:text-ink"
              >
                multi-model comparison
              </Link>{" "}
              (OURS against frontier and open models on tier-appropriate move selection, with the
              measured per-model metrics).
            </span>
          </p>
        </div>

        {/* GRADER 30-SECOND ORIENTATION: the one behavior + the three-step loop,
            so a first-time viewer knows exactly what to do and what to read. */}
        <div className="flex flex-col gap-3 rounded-xl border border-[color:var(--border)] bg-[color:var(--surface)] p-4 sm:flex-row sm:items-center sm:gap-5">
          <p className="text-sm leading-relaxed text-muted sm:max-w-[15rem]">
            <span className="font-medium text-ink">How it works.</span> The model&apos;s one job is to
            pick the move that fits your rating, not to lecture.
          </p>
          <ol className="grid flex-1 grid-cols-1 gap-2 sm:grid-cols-3">
            {[
              ["Set a position", "Pick a preset or study, or paste a FEN."],
              ["Pick your level", "Beginner, Intermediate, or Advanced."],
              ["Read the move + tag", "One move for that level, one-line reason."],
            ].map(([title, body], i) => (
              <li
                key={title}
                className="flex items-start gap-2.5 rounded-lg bg-[color:var(--surface-tertiary)] px-3 py-2"
              >
                <span className="mt-0.5 inline-flex size-5 shrink-0 items-center justify-center rounded-full bg-signal/15 font-mono text-[11px] font-semibold text-signal tnum">
                  {i + 1}
                </span>
                <span className="flex flex-col gap-0.5">
                  <span className="text-xs font-medium text-ink">{title}</span>
                  <span className="text-[11px] leading-snug text-muted">{body}</span>
                </span>
              </li>
            ))}
          </ol>
        </div>
      </header>

      {/* Board-centric console. Desktop: board + controls in the left column, the
          coaching console tall on the right. Mobile: board → console → controls. */}
      <main className="grid flex-1 grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.02fr)]">
        {/* Board + toolbar */}
        <section className="order-1 flex flex-col gap-4 lg:col-start-1 lg:row-start-1">
          {/* The board is NEVER gated by a coaching call — it stays interactive
              through cold starts and live runs so the position is always usable. */}
          <BoardStage
            fen={fen}
            orientation={orientation}
            arrows={arrows}
            lastMove={lastMove}
            loading={false}
            interactive
            onMove={onBoardMove}
          />

          {/* Board toolbar */}
          <div className="flex min-h-11 flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span
                aria-hidden
                className="inline-block size-3 rounded-full ring-1 ring-border"
                style={{
                  backgroundColor: toMove === "white" ? "var(--board-light)" : "var(--board-dark)",
                }}
              />
              <span className="text-sm text-muted">
                {toMove === "white" ? "White" : "Black"} to move
              </span>
              {studentSan && (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-[color:var(--surface-tertiary)] px-2.5 py-1 text-xs text-muted">
                  your move
                  <span className="font-mono text-[color:var(--your-move)] tnum">{studentSan}</span>
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {/* Copy the FEN of the position currently on the board. */}
              <CopyFenButton fen={fen} className="min-h-11" />
              <Tooltip delay={300}>
                <Button
                  isIconOnly
                  variant="tertiary"
                  size="md"
                  className="min-h-11 min-w-11"
                  aria-label="Take back move (Z)"
                  isDisabled={history.length === 0}
                  onPress={undo}
                >
                  <UndoIcon />
                </Button>
                <Tooltip.Content>Take back (Z)</Tooltip.Content>
              </Tooltip>
              <Tooltip delay={300}>
                <Button
                  isIconOnly
                  variant="tertiary"
                  size="md"
                  className="min-h-11 min-w-11"
                  aria-label={`Flip the board (F). Viewing from ${
                    orientation === "white" ? "White" : "Black"
                  }.`}
                  aria-pressed={orientation === "black"}
                  onPress={flip}
                >
                  {/* Rotate the glyph a half-turn when viewing from Black so the
                      control visibly reflects the board's current orientation. */}
                  <span
                    className="inline-flex transition-transform duration-200 motion-reduce:transition-none"
                    style={{ transform: orientation === "black" ? "rotate(180deg)" : "none" }}
                  >
                    <FlipVerticalIcon />
                  </span>
                </Button>
                <Tooltip.Content>
                  Flip the board (F): viewing from {orientation === "white" ? "White" : "Black"}
                </Tooltip.Content>
              </Tooltip>
            </div>
          </div>
        </section>

        {/* Coaching console: unboxed onto the felt. On desktop a 1px divider (not
            a card) separates it from the board column. */}
        <section className="order-2 flex lg:col-start-2 lg:row-span-2 lg:row-start-1">
          <div className="flex flex-1 flex-col lg:min-h-[660px] lg:border-l lg:border-[color:var(--separator)] lg:pl-8">
            <div className="flex flex-1 flex-col pt-1" aria-live="polite">
              {activeStatus === "done" && activeResult ? (
                // Key by position+tier so an INSTANT tier switch (done→done) still
                // replays the reveal animation, while a background tier landing for
                // a non-active band never disturbs what's on screen.
                <div key={`${coachedFen}-${tier}`}>
                  {/* A live run that failed but had a cached answer keeps the
                      cached answer on screen; note the miss softly (never an
                      error panel over a good precomputed tier). */}
                  {activePrecomputed && activeError && (
                    <p className="mb-4 rounded-[10px] border border-[color:var(--caution)]/40 bg-[color:var(--caution)]/10 px-3.5 py-2.5 text-xs leading-relaxed text-muted">
                      <span className="font-medium text-[color:var(--caution)]">
                        The live coach didn&apos;t respond.
                      </span>{" "}
                      Showing the precomputed answer for this level instead.
                    </p>
                  )}
                  <CoachingReveal
                    result={activeResult}
                    tier={tier}
                    fen={coachedFen}
                    precomputed={activePrecomputed}
                    onPlayLine={playEngineLine}
                  />
                </div>
              ) : activeStatus === "loading" ? (
                <CoachingSkeleton waking={wakingCoach} onCancel={cancelLive} />
              ) : activeStatus === "error" ? (
                <ErrorPanel error={activeError} onRetry={() => runLiveTier(tier)} />
              ) : (
                <IdlePanel
                  toMove={toMove}
                  studentSan={studentSan}
                  tierLabel={TIER_LABEL[tier]}
                  onRunLive={runLiveActive}
                />
              )}
            </div>
          </div>
        </section>

        {/* Controls: unboxed onto the felt, sections divided by 1px rules. */}
        <section className="order-3 lg:col-start-1 lg:row-start-2">
          <div className="flex flex-col">
            <div className="flex flex-col gap-5">
              {/* Always enabled: switching bands reads from the per-tier cache, so
                  the user can click around the three levels any time. */}
              <TierControl tier={tier} onChange={changeTier} />

              {/* LIVE inference is OPTIONAL and user-initiated. The move above is
                  precomputed and always on screen; this runs the shown level live
                  on the hosted coach (first call after idle ~2–3 min) and never
                  blocks the board. While a call runs, a Cancel appears. */}
              <div className="flex flex-col gap-1.5">
                {activeLoading ? (
                  <div className="flex gap-2">
                    <Button
                      variant="primary"
                      size="lg"
                      className="min-h-12 flex-1 font-medium"
                      isDisabled
                      aria-busy
                    >
                      <span className="inline-flex items-center justify-center gap-2">
                        <Spinner />
                        {wakingCoach ? "Waking the coach…" : "Running live…"}
                      </span>
                    </Button>
                    <Button
                      variant="tertiary"
                      size="lg"
                      className="min-h-12 font-medium"
                      onPress={cancelLive}
                    >
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <Tooltip delay={300}>
                    <Button
                      variant="primary"
                      size="lg"
                      className="min-h-12 w-full gap-2 font-medium"
                      onPress={runLiveActive}
                    >
                      <ResetIcon width={16} height={16} />
                      {runLiveLabel}
                    </Button>
                    <Tooltip.Content className="max-w-[16rem]">
                      Optional. Runs the shown level live on the hosted coach-gate pipeline
                      (engines → grounding → model → verifier). The move above is precomputed and
                      stays put if the live call fails.
                    </Tooltip.Content>
                  </Tooltip>
                )}
                <p className="text-[11px] leading-relaxed text-faint" aria-live="polite">
                  {activeLoading
                    ? wakingCoach
                      ? `Waking the hosted coach — first call after idle takes ~2–3 min${
                          wakingCoach.elapsedSec > 0 ? ` · ${wakingCoach.elapsedSec}s elapsed` : ""
                        }.`
                      : "Running the shown level through the live coach…"
                    : "Optional live pass on the hosted coach (Modal, scale-to-zero). The move shown is precomputed; a live run replaces just this level."}
                </p>
              </div>

              <Separator />

              {/* Position library: real dataset positions, each with the tuned
                  model's precomputed coaching (never disabled by a live call). */}
              <PositionLibrary
                entries={library}
                status={libStatus}
                activeId={activeLibId}
                disabled={false}
                onSelect={selectLibraryItem}
                onRetry={loadLibrary}
              />

              <Separator />

              {/* Quick positions: one-click openings + "moat" demos, each a fresh
                  full-tier coach so switching levels stays instant afterwards. */}
              <div className="flex flex-col gap-2.5">
                <span className="text-sm font-medium text-ink">Jump to a position</span>
                <div className="flex flex-wrap gap-2">
                  {PRESETS.map((p) => {
                    const active = fen === p.fen;
                    return (
                      <Tooltip key={`${p.fen}|${p.move ?? ""}`} delay={300}>
                        <Button
                          variant={active ? "secondary" : "tertiary"}
                          size="sm"
                          className="min-h-11 transition-transform hover:-translate-y-px active:translate-y-0 motion-reduce:transition-none motion-reduce:hover:translate-y-0"
                          aria-pressed={active}
                          onPress={() => selectPreset(p)}
                        >
                          {p.label}
                        </Button>
                        <Tooltip.Content className="max-w-[17rem]">{p.hint}</Tooltip.Content>
                      </Tooltip>
                    );
                  })}
                </div>
                <p className="text-[11px] leading-relaxed text-faint">
                  Recognizable openings plus three “moat” demos (one per game phase) where the tuned
                  model adapts the move by level. Loads instantly; press “Run live” to coach it.
                </p>
              </div>

              <Separator />

              {/* Advanced: paste any FEN or set a move */}
              <details className="group">
                <summary className="flex min-h-11 cursor-pointer list-none items-center gap-2 text-sm font-medium text-muted transition-colors hover:text-ink">
                  <span className="text-faint transition-transform group-open:rotate-90">›</span>
                  Paste a FEN or set a move
                </summary>
                <div className="mt-3 flex flex-col gap-3">
                  <TextField
                    className="flex flex-col gap-1.5"
                    aria-label="Position (FEN)"
                    value={fenDraft}
                    onChange={setFenDraft}
                    isInvalid={!fenDraftState.ok}
                  >
                    <Label className="text-xs font-normal text-muted">Position (FEN)</Label>
                    <TextArea
                      rows={2}
                      spellCheck={false}
                      className="w-full resize-none font-mono leading-snug"
                    />
                    {!fenDraftState.ok && (
                      <FieldError className="text-xs text-[color:var(--caution)]">
                        {fenDraftState.error
                          ? `Not a valid FEN: ${fenDraftState.error}`
                          : "That doesn’t look like a valid FEN. Paste the full board string."}
                      </FieldError>
                    )}
                  </TextField>
                  {/* Valid FEN, but a finished game has no move to coach. */}
                  {fenDraftState.ok && fenDraftState.gameOver && (
                    <p className="text-xs leading-relaxed text-[color:var(--caution)]">
                      That position is already game over: there’s no move to coach. Try another FEN.
                    </p>
                  )}
                  <TextField
                    className="flex flex-col gap-1.5"
                    aria-label="Your move (SAN or UCI)"
                    value={moveDraft}
                    onChange={setMoveDraft}
                    isInvalid={Boolean(moveDraft.trim()) && !draftUci}
                  >
                    <Label className="text-xs font-normal text-muted">Your move (SAN or UCI)</Label>
                    <Input
                      spellCheck={false}
                      placeholder="Qh5 or d1h5"
                      className="w-full font-mono"
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && draftUci) {
                          e.preventDefault();
                          setMoveFromDraft();
                        }
                      }}
                    />
                    {moveDraft.trim() && !draftUci && (
                      <FieldError className="text-xs text-[color:var(--caution)]">
                        That isn’t a legal move here. Try SAN (Qh5) or UCI (d1h5).
                      </FieldError>
                    )}
                  </TextField>
                  <div className="flex flex-wrap gap-2">
                    <Tooltip delay={300}>
                      <Button
                        variant="secondary"
                        size="md"
                        className="min-h-11"
                        isDisabled={!fenDraftState.ok || fenDraftState.gameOver}
                        onPress={loadFen}
                      >
                        Load position
                      </Button>
                      <Tooltip.Content className="max-w-[15rem]">
                        Put the pasted FEN on the board. Press “Run live” to coach it.
                      </Tooltip.Content>
                    </Tooltip>
                    <Tooltip delay={300}>
                      <Button
                        variant="tertiary"
                        size="md"
                        className="min-h-11"
                        isDisabled={!draftUci}
                        onPress={setMoveFromDraft}
                      >
                        Set move
                      </Button>
                      <Tooltip.Content className="max-w-[15rem]">
                        Mark this move on the current position without advancing the board.
                      </Tooltip.Content>
                    </Tooltip>
                    {studentUci && (
                      <Button
                        isIconOnly
                        variant="tertiary"
                        size="md"
                        className="min-h-11 min-w-11"
                        aria-label="Clear your move"
                        onPress={clearMove}
                      >
                        <ResetIcon />
                      </Button>
                    )}
                  </div>
                </div>
              </details>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function CoachingSkeleton({
  waking,
  onCancel,
}: {
  waking?: CoachWakeStatus | null;
  onCancel?: () => void;
}) {
  return (
    <div className="flex flex-1 flex-col gap-7" role="status" aria-live="polite" aria-busy="true">
      <div className="flex flex-col gap-3">
        <div className="skeleton h-12 w-40" />
        <div className="skeleton h-4 w-56" />
      </div>
      <div className="flex flex-col gap-2.5">
        <div className="skeleton h-4 w-full" />
        <div className="skeleton h-4 w-full" />
        <div className="skeleton h-4 w-11/12" />
        <div className="skeleton h-4 w-4/5" />
      </div>
      <div className="skeleton h-20 w-full" />
      <div className="grid grid-cols-2 gap-6">
        <div className="flex flex-col gap-2">
          <div className="skeleton h-4 w-24" />
          <div className="skeleton h-3 w-full" />
          <div className="skeleton h-3 w-full" />
          <div className="skeleton h-3 w-5/6" />
        </div>
        <div className="flex flex-col gap-2">
          <div className="skeleton h-4 w-24" />
          <div className="skeleton h-3 w-full" />
          <div className="skeleton h-3 w-5/6" />
          <div className="skeleton h-3 w-4/6" />
        </div>
      </div>
      {waking ? (
        <div className="flex flex-col gap-1.5">
          <p className="flex items-center gap-2 text-sm font-medium text-ink">
            <span aria-hidden className="inline-block size-2 shrink-0 animate-pulse rounded-full bg-signal" />
            Waking the hosted coach: first call after idle takes ~2–3 min…
          </p>
          <p className="text-xs leading-relaxed text-muted">
            The model scales to zero when idle.{" "}
            {waking.attempt >= 1 ? "Retrying automatically" : "Hang tight"}
            {waking.elapsedSec > 0 ? ` · ${waking.elapsedSec}s elapsed` : ""}.
          </p>
        </div>
      ) : (
        <p className="text-sm text-muted">Reading the sound moves and the human odds…</p>
      )}
      {onCancel && (
        <div>
          <Button variant="tertiary" size="sm" className="min-h-9" onPress={onCancel}>
            Cancel live run
          </Button>
        </div>
      )}
    </div>
  );
}

function ErrorPanel({ error, onRetry }: { error: string | null; onRetry: () => void }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 text-center">
      <h2 className="text-xl font-semibold text-ink">The coach is offline for a moment.</h2>
      <p className="max-w-sm text-sm leading-relaxed text-muted">
        Your position is saved. This usually means the coaching service is restarting: give it a
        moment and try again.
      </p>
      <Button variant="primary" size="md" className="min-h-11" onPress={onRetry}>
        Try again
      </Button>
      {error && (
        <details className="w-full max-w-sm text-left">
          <summary className="cursor-pointer text-xs text-faint transition-colors hover:text-muted">
            Technical details
          </summary>
          <p className="mt-2 break-words font-mono text-xs leading-relaxed text-muted">{error}</p>
        </details>
      )}
    </div>
  );
}

function IdlePanel({
  toMove,
  studentSan,
  tierLabel,
  onRunLive,
}: {
  toMove: Orientation;
  studentSan: string | null;
  tierLabel: string;
  onRunLive?: () => void;
}) {
  const steps = [
    "Set a position, or pick one from the study library.",
    "Drag your move onto the board to mark what you’d play.",
    "Choose your rating, then run the coach live.",
  ];
  return (
    <div className="flex flex-1 flex-col justify-center gap-7">
      <div className="flex flex-col gap-3">
        <h2 className="max-w-md text-2xl font-semibold leading-tight text-balance text-ink sm:text-3xl">
          Play the move you are unsure about.
        </h2>
        <p className="max-w-md text-base leading-relaxed text-pretty text-muted">
          Pick a study position or set your own, drag the piece for the move you are second-guessing,
          choose a rating, then run the coach. You get one move to focus on and a plain reason why.
          Coaching this position is a live call to the hosted model (optional; the first call after
          idle takes ~2–3 min).
        </p>
      </div>
      <ol className="flex flex-col gap-3">
        {steps.map((s, i) => (
          <li key={i} className="flex items-baseline gap-3 text-sm leading-relaxed text-muted">
            <span className="font-mono text-xs font-semibold text-signal tnum">{i + 1}</span>
            {s}
          </li>
        ))}
      </ol>
      {onRunLive && (
        <div>
          <Button variant="primary" size="lg" className="min-h-12" onPress={onRunLive}>
            Run live · {tierLabel}
          </Button>
        </div>
      )}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-[color:var(--separator)] pt-5 text-sm text-muted">
        <span
          aria-hidden
          className="inline-block size-3 rounded-full ring-1 ring-border"
          style={{ backgroundColor: toMove === "white" ? "var(--board-light)" : "var(--board-dark)" }}
        />
        {toMove === "white" ? "White" : "Black"} to move
        {studentSan && (
          <>
            <span className="text-faint">·</span>
            <span>
              your move <span className="font-mono text-ink tnum">{studentSan}</span>
            </span>
          </>
        )}
      </div>
    </div>
  );
}

/** A small currentColor spinner for in-button loading states. */
function Spinner() {
  return (
    <span
      aria-hidden
      className="inline-block size-4 shrink-0 animate-spin rounded-full border-2 motion-reduce:animate-none"
      style={{ borderColor: "currentColor", borderTopColor: "transparent" }}
    />
  );
}
