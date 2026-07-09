"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Button, Card, FieldError, Input, Label, Separator, TextArea, TextField, Tooltip } from "@heroui/react";
import {
  getLibrary,
  postCoachAll,
  postCoachResilient,
  warmupCoach,
  type CoachResponse,
  type CoachWakeStatus,
  type LibraryEntry,
  type Tier,
} from "@/lib/api";
import {
  applyUciMove,
  moveToUci,
  sideToMove,
  stepSanLine,
  uciToSan,
  uciToSquares,
  validateFen,
  type Orientation,
} from "@/lib/chess";
import { deriveOursLabel } from "@/lib/showcase";
import { playCapture, playMove } from "@/lib/sound";
import BoardStage, { type StageArrow } from "./BoardStage";
import TierControl from "./TierControl";
import CoachingReveal from "./CoachingReveal";
import PositionLibrary, { type LibStatus } from "./PositionLibrary";
import CopyFenButton from "./CopyFenButton";
import { FlipVerticalIcon, ResetIcon, UndoIcon } from "./icons";

type Status = "idle" | "loading" | "done" | "error";

// DEFAULT is a moat demo verified on the LIVE served model: on this held-out
// rook endgame the coach gives a DIFFERENT, level-appropriate move for each tier
// — g5 (beginner) · Ra1 (intermediate) · h5 (advanced). Switching tiers on load
// is the one-screen proof that the fine-tune adapts the move to the player's
// level. No student move — the point is the recommendation per tier.
const DEFAULT = {
  fen: "r5kr/6pp/p7/8/6PP/8/8/5RK1 w - - 1 39",
  tier: "beginner" as Tier,
  move: "",
};
const DEFAULT_STUDENT_UCI = moveToUci(DEFAULT.fen, DEFAULT.move);
const TIERS: Tier[] = ["beginner", "intermediate", "advanced"];

interface Preset {
  label: string;
  hint: string;
  fen: string;
  move?: string;
}

// A few one-click positions: a "moat" demo (a held-out position where the live
// served model gives a distinct move per level), a handful of recognizable
// openings, and the classic beginner blunder.
const PRESETS: Preset[] = [
  {
    label: "Moat · rook endgame",
    hint: "The default demo — the live model adapts g5 / Ra1 / h5 by level; switch tiers to watch the move change.",
    fen: "r5kr/6pp/p7/8/6PP/8/8/5RK1 w - - 1 39",
  },
  {
    label: "Italian Game",
    hint: "1.e4 e5 2.Nf3 Nc6 3.Bc4 — Black to choose a developing move.",
    fen: "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
  },
  {
    label: "Ruy López",
    hint: "1.e4 e5 2.Nf3 Nc6 3.Bb5 — the Spanish, Black to respond.",
    fen: "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
  },
  {
    label: "Sicilian Defence",
    hint: "1.e4 c5 — White to develop against the Sicilian.",
    fen: "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
  },
  {
    label: "Early queen 2.Qh5?!",
    hint: "The classic beginner queen sortie — sound pressure, or just a target?",
    fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    move: "Qh5",
  },
  {
    label: "Starting position",
    hint: "A fresh game — White to make the first move.",
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
  // in `coachedFen`/`studentUci`). All three bands are fetched together so the
  // Beginner/Intermediate/Advanced buttons can swap the shown answer with no new
  // network call. A new session (position/move change or explicit re-run) resets
  // all three; each entry lands independently for progressive, per-tier loading.
  const [tierResults, setTierResults] = useState<Record<Tier, CoachResponse | undefined>>(() => ({
    beginner: undefined,
    intermediate: undefined,
    advanced: undefined,
  }));
  const [tierStatus, setTierStatus] = useState<Record<Tier, Status>>(() => ({
    beginner: "idle",
    intermediate: "idle",
    advanced: "idle",
  }));
  const [tierError, setTierError] = useState<Record<Tier, string | null>>(() => ({
    beginner: null,
    intermediate: null,
    advanced: null,
  }));
  const [wakingCoach, setWakingCoach] = useState<CoachWakeStatus | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const didInit = useRef(false);

  // Coach the position at EVERY rating tier at once and cache the answers, so the
  // tier buttons can switch the shown result with zero new calls. Fired on the
  // initial load, "Coach this move/position", any position/move change, and the
  // explicit re-run. Optionally `seed` a tier that already has an answer (a
  // library entry's cached coaching) so it shows instantly while the other tiers
  // prefetch live.
  //
  // A FRESH full fetch (no seed) uses the batch endpoint POST /api/coach_all: the
  // server computes the engine facts (Stockfish sound pool + student severity)
  // ONCE and returns all three tiers, so the initial load costs a single engine
  // pass, not three. If that route is missing (404 before the backend is
  // restarted) or fails, it GRACEFULLY FALLS BACK to the proven 3-parallel
  // `postCoachResilient` path — each tier landing independently, with the existing
  // cold-start "waking" handling — so the Studio never breaks. A seeded run always
  // takes the per-tier path so the seeded band stays instant while the others
  // prefetch. Either way the answers land in the same per-tier cache, so switching
  // levels afterward is instant.
  const runCoachAllTiers = useCallback(
    (f: string, s: string | null, seed?: Partial<Record<Tier, CoachResponse>>) => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setCoachedFen(f); // every cached tier answer is about position f
      setWakingCoach(null);

      const seeded = seed ?? {};
      setTierResults({
        beginner: seeded.beginner,
        intermediate: seeded.intermediate,
        advanced: seeded.advanced,
      });
      setTierStatus({
        beginner: seeded.beginner ? "done" : "loading",
        intermediate: seeded.intermediate ? "done" : "loading",
        advanced: seeded.advanced ? "done" : "loading",
      });
      setTierError({ beginner: null, intermediate: null, advanced: null });

      // Cold start in progress — surface "waking" without flipping to error.
      const onStatus = (st: CoachWakeStatus) => {
        if (!ctrl.signal.aborted) setWakingCoach(st);
      };
      const landTier = (t: Tier, res: CoachResponse) => {
        if (ctrl.signal.aborted) return;
        setTierResults((prev) => {
          const next = { ...prev };
          next[t] = res;
          return next;
        });
        setTierStatus((prev) => {
          const next = { ...prev };
          next[t] = "done";
          return next;
        });
        setWakingCoach(null); // a landed tier means the container is warm
      };
      const failTier = (t: Tier, e: unknown) => {
        if (ctrl.signal.aborted) return;
        // Only reached once the resilient call gives up (retries exhausted or a
        // hard, non-cold error) — so "offline" never flashes mid-wake.
        setTierStatus((prev) => {
          const next = { ...prev };
          next[t] = "error";
          return next;
        });
        setTierError((prev) => {
          const next = { ...prev };
          next[t] = e instanceof Error ? e.message : "Something went wrong.";
          return next;
        });
      };

      // The proven path: coach each not-yet-seeded tier in parallel, each landing
      // independently (instant for a ready tier, a per-tier skeleton for one still
      // in flight) with the existing cold-start resilience.
      const firePerTier = () => {
        void Promise.all(
          TIERS.map((t) => {
            if (seeded[t]) return Promise.resolve();
            return postCoachResilient(
              { fen: f, tier: t, student_move: s ?? undefined },
              { signal: ctrl.signal, onStatus },
            )
              .then((res) => landTier(t, res))
              .catch((e: unknown) => failTier(t, e));
          }),
        );
      };

      // A seeded run keeps the progressive per-tier path so the seeded band shows
      // instantly while the other two prefetch live.
      if (seeded.beginner || seeded.intermediate || seeded.advanced) {
        firePerTier();
        return;
      }

      // Fresh full fetch → one batch call, with a graceful fallback to per-tier.
      void postCoachAll(
        { fen: f, student_move: s ?? undefined },
        { signal: ctrl.signal, onStatus },
      )
        .then((all) => {
          if (ctrl.signal.aborted) return;
          setTierResults({
            beginner: all.beginner,
            intermediate: all.intermediate,
            advanced: all.advanced,
          });
          setTierStatus({ beginner: "done", intermediate: "done", advanced: "done" });
          setWakingCoach(null);
        })
        .catch(() => {
          if (ctrl.signal.aborted) return;
          // Batch route missing (pre-restart 404) or failed → degrade to the
          // resilient 3-parallel path so the platform keeps working.
          firePerTier();
        });
    },
    [],
  );

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
    warmupCoach(); // nudge a scaled-to-zero container awake before the first call
    loadLibrary();
    runCoachAllTiers(DEFAULT.fen, DEFAULT_STUDENT_UCI);
  }, [runCoachAllTiers, loadLibrary]);

  // The displayed answer is whatever the ACTIVE tier holds in the cache; a tier
  // switch just re-points these — no network call when a tier is already cached.
  const activeStatus = tierStatus[tier];
  const activeResult = tierResults[tier] ?? null;
  const activeError = tierError[tier];
  const anyLoading = TIERS.some((t) => tierStatus[t] === "loading");
  const hasAnyResult = TIERS.some((t) => tierResults[t] != null);
  const anySettled = TIERS.some((t) => tierStatus[t] === "done" || tierStatus[t] === "error");
  // The one board-blocking loading state is the INITIAL batch fetch: still working
  // with nothing cached yet. Once any tier lands the board unlocks; a tier still
  // in flight then shows only a lightweight per-tier skeleton in the console.
  const initialLoading = anyLoading && !hasAnyResult;

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

  // Dragging a legal move makes it "stick": the board advances and the coach
  // reviews the move you just played (position BEFORE it + that move).
  const onBoardMove = useCallback(
    (uci: string) => {
      const before = fen;
      const after = applyUciMove(before, uci);
      if (!after) return;
      setActiveLibId(null);
      setHistory((h) => [...h, before]);
      setFen(after);
      setLastMove([uci.slice(0, 2), uci.slice(2, 4)]);
      setStudentUci(uci);
      setMoveDraft("");
      runCoachAllTiers(before, uci);
    },
    [fen, runCoachAllTiers],
  );

  // Take back the last move — return to the previous position and ask "what now?".
  const undo = useCallback(() => {
    if (history.length === 0) return;
    const prev = history[history.length - 1];
    setHistory((h) => h.slice(0, -1));
    setFen(prev);
    setLastMove(null);
    setStudentUci(null);
    setMoveDraft("");
    setActiveLibId(null);
    runCoachAllTiers(prev, null);
  }, [history, runCoachAllTiers]);

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
      runCoachAllTiers(stepped.boardFen, null);
    },
    [coachedFen, runCoachAllTiers],
  );

  const flip = useCallback(() => setOrientation((o) => (o === "white" ? "black" : "white")), []);

  // Tier switching is INSTANT: all three bands were fetched together for this
  // position, so we just swap which cached answer is displayed — no new model
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
    runCoachAllTiers(fen, null);
  };

  // Explicit "re-run through the workflow": send the exact position the current
  // answer is about back through the coach for a fresh LIVE pass across ALL THREE
  // tiers (not the cached library text), reusing the same student move so the
  // board and the answers stay aligned.
  const rerunWorkflow = useCallback(() => {
    const su = activeResult?.engine.student_move?.uci ?? (fen === coachedFen ? studentUci : null);
    setActiveLibId(null);
    runCoachAllTiers(coachedFen, su);
  }, [activeResult, fen, coachedFen, studentUci, runCoachAllTiers]);

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
    // Seed the entry's tier with its cached coaching (instant), then prefetch the
    // other two tiers live so switching levels is also instant.
    const seed: Partial<Record<Tier, CoachResponse>> = {};
    seed[e.tier] = e.coach;
    runCoachAllTiers(e.fen, su, seed);
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
    runCoachAllTiers(f, null);
  };

  // One-click preset: load the position (and its illustrative move, if any) and
  // coach all three tiers via the same fresh-fetch path as a hand-entered FEN, so
  // it uses the batch endpoint and stays instant to switch between levels after.
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
    runCoachAllTiers(p.fen, su);
  };

  // Review a move on the CURRENT position without advancing the board.
  const setMoveFromDraft = () => {
    if (!draftUci) return;
    setStudentUci(draftUci);
    setLastMove([draftUci.slice(0, 2), draftUci.slice(2, 4)]);
    runCoachAllTiers(fen, draftUci);
  };

  // `loading` gates the board + primary actions on the INITIAL fetch only; tier
  // switching stays live, and an in-flight tier shows its own console skeleton.
  const loading = initialLoading;

  // Keyboard accelerators for the primary board flow (ignored while typing).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        flip();
      } else if ((e.key === "z" || e.key === "Z") && history.length > 0 && !loading) {
        e.preventDefault();
        undo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [flip, undo, history.length, loading]);

  const coachLabel = loading
    ? "Coaching…"
    : fen === coachedFen && studentUci
      ? "Coach this move"
      : "Coach this position";

  return (
    <div className="relative z-[1] mx-auto flex min-h-dvh w-full max-w-[1240px] flex-col gap-8 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      {/* Top-of-page intro. The hero is the ONE behavior — the tuned model selects
          the tier-appropriate MOVE (+ a short principle tag) where its base can't;
          the coaching prose is a secondary, optional layer, and the cross-model
          views are the bonus comparison. */}
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
            The fine-tune reliably picks the level-appropriate move where its base can't.
          </h1>
          <p className="text-pretty text-sm leading-relaxed text-muted sm:text-base">
            A fine-tuned model running locally. Set a position, mark the move you're unsure about, and
            pick your rating. It hands back <span className="text-ink">one move chosen for your
            level</span> and a short <span className="text-ink">principle tag</span> for why. Its one
            job is selecting the tier-appropriate move; the full explanation is an{" "}
            <span className="text-ink">optional layer</span> underneath, not the headline. On the same
            grounded position and rating, its untuned base picks the level-appropriate move far less
            reliably.{" "}
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
            <span className="font-medium text-ink">How it works.</span> The model's one job is to
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
          <BoardStage
            fen={fen}
            orientation={orientation}
            arrows={arrows}
            lastMove={lastMove}
            loading={loading}
            interactive={!loading}
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
                  Flip the board (F) — viewing from {orientation === "white" ? "White" : "Black"}
                </Tooltip.Content>
              </Tooltip>
            </div>
          </div>
        </section>

        {/* Coaching console */}
        <section className="order-2 flex lg:col-start-2 lg:row-span-2 lg:row-start-1">
          <Card variant="secondary" className="flex flex-1 flex-col lg:min-h-[660px]">
            <Card.Content className="flex flex-1 flex-col p-5 sm:p-7" aria-live="polite">
              {activeStatus === "done" && activeResult ? (
                // Key by position+tier so an INSTANT tier switch (done→done) still
                // replays the reveal animation, while a background tier landing for
                // a non-active band never disturbs what's on screen.
                <div key={`${coachedFen}-${tier}`}>
                  <CoachingReveal
                    result={activeResult}
                    tier={tier}
                    fen={coachedFen}
                    onPlayLine={playEngineLine}
                  />
                </div>
              ) : activeStatus === "loading" ? (
                <CoachingSkeleton waking={wakingCoach} />
              ) : activeStatus === "error" ? (
                <ErrorPanel
                  error={activeError}
                  onRetry={() => runCoachAllTiers(coachedFen, studentUci)}
                />
              ) : (
                <IdlePanel toMove={toMove} studentSan={studentSan} />
              )}
            </Card.Content>
          </Card>
        </section>

        {/* Controls */}
        <section className="order-3 lg:col-start-1 lg:row-start-2">
          <Card variant="secondary">
            <Card.Content className="flex flex-col gap-5 p-4 sm:p-5">
              {/* Always enabled — switching bands reads from the per-tier cache, so
                  the user can click around the three levels even mid-fetch. */}
              <TierControl tier={tier} onChange={changeTier} />

              {/* Loading shows a subtle spinner + the "Coaching…" label; the
                  button also dims (disabled) so the state reads two ways. */}
              <Tooltip delay={300}>
                <Button
                  variant="primary"
                  size="lg"
                  className="min-h-12 w-full font-medium"
                  isDisabled={loading}
                  aria-busy={loading}
                  onPress={() =>
                    runCoachAllTiers(fen, fen === coachedFen ? studentUci : null)
                  }
                >
                  <span className="inline-flex items-center justify-center gap-2">
                    {loading && <Spinner />}
                    {coachLabel}
                  </span>
                </Button>
                <Tooltip.Content className="max-w-[16rem]">
                  Read the position and hand back the move for your selected level, with a one-line
                  principle tag. All three levels are fetched together, so switching is instant.
                </Tooltip.Content>
              </Tooltip>

              {/* Explicit live re-run through the full coaching workflow. */}
              <div className="flex flex-col gap-1.5">
                <Tooltip delay={300}>
                  <Button
                    variant="tertiary"
                    size="md"
                    className="min-h-11 w-full gap-2 font-medium"
                    isDisabled={loading || !anySettled}
                    onPress={rerunWorkflow}
                  >
                    <ResetIcon width={16} height={16} />
                    Re-run through the workflow
                  </Button>
                  <Tooltip.Content className="max-w-[16rem]">
                    Re-runs all three levels live through the coach-gate pipeline instead of reusing
                    the cached answers.
                  </Tooltip.Content>
                </Tooltip>
                <p className="text-[11px] leading-relaxed text-faint">
                  A fresh live pass over this exact position — re-reads it and re-picks the
                  level-appropriate move, rather than reusing a cached answer.
                </p>
              </div>

              <Separator />

              {/* Position library — real dataset positions, each coached by the tuned model */}
              <PositionLibrary
                entries={library}
                status={libStatus}
                activeId={activeLibId}
                disabled={loading}
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
                          className="min-h-9 transition-transform hover:-translate-y-px active:translate-y-0 motion-reduce:transition-none motion-reduce:hover:translate-y-0"
                          isDisabled={loading}
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
                  Recognizable openings plus two “moat” demos where the tuned model adapts the move
                  by level. Loads instantly and re-coaches all three tiers.
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
                    isDisabled={loading}
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
                          ? `Not a valid FEN — ${fenDraftState.error}`
                          : "That doesn’t look like a valid FEN. Paste the full board string."}
                      </FieldError>
                    )}
                  </TextField>
                  {/* Valid FEN, but a finished game has no move to coach. */}
                  {fenDraftState.ok && fenDraftState.gameOver && (
                    <p className="text-xs leading-relaxed text-[color:var(--caution)]">
                      That position is already game over — there’s no move to coach. Try another FEN.
                    </p>
                  )}
                  <TextField
                    className="flex flex-col gap-1.5"
                    aria-label="Your move (SAN or UCI)"
                    value={moveDraft}
                    onChange={setMoveDraft}
                    isDisabled={loading}
                    isInvalid={Boolean(moveDraft.trim()) && !draftUci}
                  >
                    <Label className="text-xs font-normal text-muted">Your move (SAN or UCI)</Label>
                    <Input
                      spellCheck={false}
                      placeholder="Qh5 or d1h5"
                      className="w-full font-mono"
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && draftUci && !loading) {
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
                        isDisabled={!fenDraftState.ok || fenDraftState.gameOver || loading}
                        onPress={loadFen}
                      >
                        Load position
                      </Button>
                      <Tooltip.Content className="max-w-[15rem]">
                        Put the pasted FEN on the board and coach all three levels.
                      </Tooltip.Content>
                    </Tooltip>
                    <Tooltip delay={300}>
                      <Button
                        variant="tertiary"
                        size="md"
                        className="min-h-11"
                        isDisabled={!draftUci || loading}
                        onPress={setMoveFromDraft}
                      >
                        Set move
                      </Button>
                      <Tooltip.Content className="max-w-[15rem]">
                        Mark this move on the current position without advancing the board, then
                        re-coach.
                      </Tooltip.Content>
                    </Tooltip>
                    {studentUci && (
                      <Button
                        isIconOnly
                        variant="tertiary"
                        size="md"
                        className="min-h-11 min-w-11"
                        aria-label="Clear your move"
                        isDisabled={loading}
                        onPress={clearMove}
                      >
                        <ResetIcon />
                      </Button>
                    )}
                  </div>
                </div>
              </details>
            </Card.Content>
          </Card>
        </section>
      </main>
    </div>
  );
}

function CoachingSkeleton({ waking }: { waking?: CoachWakeStatus | null }) {
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
            Waking the coach model — first call after idle takes ~2–3 min…
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
    </div>
  );
}

function ErrorPanel({ error, onRetry }: { error: string | null; onRetry: () => void }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 text-center">
      <h2 className="text-xl font-semibold text-ink">The coach is offline for a moment.</h2>
      <p className="max-w-sm text-sm leading-relaxed text-muted">
        Your position is saved. This usually means the coaching service is restarting — give it a
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
}: {
  toMove: Orientation;
  studentSan: string | null;
}) {
  const steps = [
    "Set a position, or pick one from the study library.",
    "Drag your move onto the board to mark what you’d play.",
    "Choose your rating and ask the coach.",
  ];
  return (
    <div className="flex flex-1 flex-col justify-center gap-7">
      <div className="flex flex-col gap-3">
        <h2 className="max-w-md text-2xl font-semibold leading-tight text-balance text-ink sm:text-3xl">
          Play the move you are unsure about.
        </h2>
        <p className="max-w-md text-base leading-relaxed text-pretty text-muted">
          Pick a study position or set your own, drag the piece for the move you are second-guessing,
          choose a rating, then ask the coach. You get one move to focus on and a plain reason why.
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
