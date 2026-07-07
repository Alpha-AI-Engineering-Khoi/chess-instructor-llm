"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Button, Card, FieldError, Input, Label, Separator, TextArea, TextField, Tooltip } from "@heroui/react";
import {
  getLibrary,
  postCoach,
  type CoachResponse,
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
import { playCapture, playMove } from "@/lib/sound";
import BoardStage, { type StageArrow } from "./BoardStage";
import TierControl from "./TierControl";
import CoachingReveal from "./CoachingReveal";
import PositionLibrary, { type LibStatus } from "./PositionLibrary";
import { FlipVerticalIcon, ResetIcon, UndoIcon } from "./icons";

type Status = "idle" | "loading" | "done" | "error";

const DEFAULT = {
  fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
  tier: "beginner" as Tier,
  move: "Qh5",
};
const DEFAULT_STUDENT_UCI = moveToUci(DEFAULT.fen, DEFAULT.move);

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

  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<CoachResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revealKey, setRevealKey] = useState(0);

  const abortRef = useRef<AbortController | null>(null);
  const didInit = useRef(false);

  const runCoach = useCallback((f: string, t: Tier, s: string | null) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setCoachedFen(f); // the result will be about position f
    setStatus("loading");
    setError(null);
    postCoach({ fen: f, tier: t, student_move: s ?? undefined }, ctrl.signal)
      .then((res) => {
        setResult(res);
        setStatus("done");
        setRevealKey((k) => k + 1);
      })
      .catch((e: unknown) => {
        if (ctrl.signal.aborted) return;
        setError(e instanceof Error ? e.message : "Something went wrong.");
        setStatus("error");
      });
  }, []);

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
    loadLibrary();
    runCoach(DEFAULT.fen, DEFAULT.tier, DEFAULT_STUDENT_UCI);
  }, [runCoach, loadLibrary]);

  const toMove = useMemo(() => sideToMove(fen), [fen]);
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
    if (status === "done" && result) {
      const list: StageArrow[] = [];
      const su = result.engine.student_move?.uci;
      if (su) {
        const sq = uciToSquares(su);
        if (sq) list.push({ from: sq.from, to: sq.to, kind: "student", delay: 0.45, draw: true });
      }
      const rq = uciToSquares(result.recommended_move_uci);
      if (rq) list.push({ from: rq.from, to: rq.to, kind: "rec", delay: 0.15, draw: true });
      return list;
    }
    if (studentUci) {
      const sq = uciToSquares(studentUci);
      if (sq) return [{ from: sq.from, to: sq.to, kind: "student", delay: 0, draw: false }];
    }
    return [];
  }, [status, result, studentUci, fen, coachedFen]);

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
      setError(null);
      runCoach(before, tier, uci);
    },
    [fen, tier, runCoach],
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
    runCoach(prev, tier, null);
  }, [history, tier, runCoach]);

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
      setError(null);
      runCoach(stepped.boardFen, tier, null);
    },
    [coachedFen, tier, runCoach],
  );

  const flip = useCallback(() => setOrientation((o) => (o === "white" ? "black" : "white")), []);

  // Changing the tier must RE-COACH the position under review with the new level,
  // not just relabel the heading: the coaching prose is written for a specific
  // tier, so a tier switch has to re-fetch or it leaves stale, wrong-level text.
  // Library entries carry tier-specific *cached* prose, so we also drop the active
  // library pin and fetch fresh, tier-specific coaching from the model.
  const changeTier = useCallback(
    (t: Tier) => {
      if (t === tier) return;
      setTier(t);
      setActiveLibId(null);
      // Re-coach the exact position/move the current result is about (coachedFen),
      // reusing that result's student move so the board doesn't jump.
      const su = result?.engine.student_move?.uci ?? (fen === coachedFen ? studentUci : null);
      runCoach(coachedFen, t, su);
    },
    [tier, result, fen, coachedFen, studentUci, runCoach],
  );

  const clearMove = () => {
    setStudentUci(null);
    setMoveDraft("");
    setLastMove(null);
    runCoach(fen, tier, null);
  };

  const selectLibraryItem = (e: LibraryEntry) => {
    abortRef.current?.abort();
    const su = e.student_move ? moveToUci(e.fen, e.student_move) : null;
    setFen(e.fen);
    setCoachedFen(e.fen);
    setHistory([]);
    setFenDraft(e.fen);
    setTier(e.tier);
    setStudentUci(su);
    setMoveDraft(e.student_move ?? "");
    setLastMove(su ? [su.slice(0, 2), su.slice(2, 4)] : null);
    setOrientation((e.coach.side_to_move as Orientation) ?? sideToMove(e.fen));
    setActiveLibId(e.id);
    setError(null);
    setResult(e.coach); // instant: the tuned model's cached coaching
    setStatus("done");
    setRevealKey((k) => k + 1);
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
    runCoach(f, tier, null);
  };

  // Review a move on the CURRENT position without advancing the board.
  const setMoveFromDraft = () => {
    if (!draftUci) return;
    setStudentUci(draftUci);
    setLastMove([draftUci.slice(0, 2), draftUci.slice(2, 4)]);
    setError(null);
    runCoach(fen, tier, draftUci);
  };

  const loading = status === "loading";

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
      {/* Slim bar: jump to the cross-model comparison view. */}
      <div className="flex items-center justify-end">
        <Link
          href="/showdown"
          className="inline-flex min-h-9 items-center gap-1.5 rounded-full px-3.5 text-sm font-medium text-muted ring-1 ring-[color:var(--border)] transition-colors hover:text-ink hover:ring-[color:var(--field-border)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60"
        >
          Model Showdown
          <span aria-hidden className="text-signal">★</span>
          <span aria-hidden className="text-faint">›</span>
        </Link>
      </div>

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
              {status === "done" && result ? (
                <div key={revealKey}>
                  <CoachingReveal
                    result={result}
                    tier={tier}
                    fen={coachedFen}
                    onPlayLine={playEngineLine}
                  />
                </div>
              ) : status === "loading" ? (
                <CoachingSkeleton />
              ) : status === "error" ? (
                <ErrorPanel error={error} onRetry={() => runCoach(fen, tier, studentUci)} />
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
              <TierControl tier={tier} onChange={changeTier} disabled={loading} />

              {/* No leading icon or spinner — loading reads through the dimmed
                  (disabled) label, which also changes to "Coaching…". */}
              <Button
                variant="primary"
                size="lg"
                className="min-h-12 w-full font-medium"
                isDisabled={loading}
                aria-busy={loading}
                onPress={() =>
                  runCoach(fen, tier, fen === coachedFen ? studentUci : null)
                }
              >
                {coachLabel}
              </Button>

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

              {/* Advanced: paste a FEN or set a move */}
              <details className="group">
                <summary className="flex min-h-11 cursor-pointer list-none items-center gap-2 text-sm font-medium text-muted transition-colors hover:text-ink">
                  <span className="text-faint transition-transform group-open:rotate-90">›</span>
                  Set a position by hand
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
                    <Button
                      variant="secondary"
                      size="md"
                      className="min-h-11"
                      isDisabled={!fenDraftState.ok || fenDraftState.gameOver || loading}
                      onPress={loadFen}
                    >
                      Load position
                    </Button>
                    <Button
                      variant="tertiary"
                      size="md"
                      className="min-h-11"
                      isDisabled={!draftUci || loading}
                      onPress={setMoveFromDraft}
                    >
                      Set move
                    </Button>
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

function CoachingSkeleton() {
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
      <p className="text-sm text-muted">Reading the sound moves and the human odds…</p>
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
