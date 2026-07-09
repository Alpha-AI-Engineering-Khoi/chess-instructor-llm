// Typed client for the FastAPI coach backend.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export type Tier = "beginner" | "intermediate" | "advanced";

export interface Example {
  label: string;
  fen: string;
  tier: Tier;
  student_move: string | null;
}

export interface EngineMove {
  san: string;
  uci: string;
  cp: number;
  pv: string[];
}

export interface StudentInfo {
  san: string;
  uci: string;
  cp_loss: number;
  severity: string;
}

export interface EngineBlock {
  best_san: string;
  best_cp: number;
  sound_pool: EngineMove[];
  student_move: StudentInfo | null;
}

export interface MaiaInfo {
  san: string;
  uci: string;
  policy: number;
}

export interface CoachMeta {
  model: string;
  tuned: boolean;
  notes: string[];
  /** How many generations the faithfulness gate needed (1 = clean first try). */
  attempts: number;
  /** True when every attempt failed and a verified, engine-derived reply was used. */
  verified_fallback: boolean;
}

export interface CoachResponse {
  recommended_move_san: string;
  recommended_move_uci: string;
  coaching: string;
  takeaway: string;
  concepts_used: string[];
  side_to_move: "white" | "black";
  engine: EngineBlock;
  maia: MaiaInfo[];
  meta: CoachMeta;
}

export interface CoachRequest {
  fen: string;
  tier: Tier;
  student_move?: string | null;
}

/** Batch request: coach ONE position, get every tier back (no `tier` field). */
export interface CoachAllRequest {
  fen: string;
  student_move?: string | null;
}

/** All three tiers from a single {@link postCoachAll} call, each the standard
 *  {@link CoachResponse} shape: the server computes the engine facts once and
 *  runs the identical coach-gate pipeline per tier. */
export interface CoachAllResponse {
  beginner: CoachResponse;
  intermediate: CoachResponse;
  advanced: CoachResponse;
}

/** A pre-computed library entry: a real dataset position + the tuned model's
 *  cached coaching, so the gallery renders instantly without a model call. */
export interface LibraryEntry {
  id: string;
  label: string;
  fen: string;
  tier: Tier;
  phase: string;
  severity: string | null;
  student_move: string | null;
  coach: CoachResponse;
}

async function readError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
    if (Array.isArray(data?.detail) && data.detail[0]?.msg) return data.detail[0].msg;
  } catch {
    /* fall through */
  }
  return `Request failed (${res.status})`;
}

export async function getExamples(signal?: AbortSignal): Promise<Example[]> {
  const res = await fetch(`${API_BASE}/api/examples`, { signal });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function getHealth(
  signal?: AbortSignal,
): Promise<{ status: string; model: string; tuned: boolean; ready: boolean }> {
  const res = await fetch(`${API_BASE}/api/health`, { signal });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

/** The pre-generated position library (static asset served by Next from public/). */
export async function getLibrary(signal?: AbortSignal): Promise<LibraryEntry[]> {
  const res = await fetch("/library.json", { signal, cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

export async function postCoach(req: CoachRequest, signal?: AbortSignal): Promise<CoachResponse> {
  const res = await fetch(`${API_BASE}/api/coach`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) throw new CoachHttpError(res.status, await readError(res));
  return res.json();
}

/* ------------------------------------------------------------------ *
 * Cold-start-aware coach calls                                        *
 *                                                                     *
 * The coach backend (Modal vLLM, scale-to-zero) answers warm calls in *
 * ~15–50s, but the FIRST call after idle triggers a ~2.5–3 min cold   *
 * start. During it Modal 303s at its 150s synchronous cap and the     *
 * browser's redirect-follow eventually drops the request. These       *
 * helpers wake the container early and retry across the cold start so  *
 * the UI can show progress instead of hanging or flashing "offline".  *
 * ------------------------------------------------------------------ */

/** HTTP error from the coach API that carries the status code, so callers can
 *  tell a transient/cold failure (retry) from a hard client error (give up). */
export class CoachHttpError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "CoachHttpError";
    this.status = status;
  }
}

export type CoachWakePhase = "waking" | "retrying";

/** Progress signal emitted by {@link postCoachResilient} while it waits out a
 *  cold start, so the UI can show a "waking the model" message with progress. */
export interface CoachWakeStatus {
  phase: CoachWakePhase;
  /** 0 for the early "still waking" hint, then the 1-based retry count. */
  attempt: number;
  /** Whole seconds elapsed since the resilient call began. */
  elapsedSec: number;
}

export interface PostCoachResilientOptions {
  signal?: AbortSignal;
  onStatus?: (status: CoachWakeStatus) => void;
}

const COLD_START = {
  /** Total time to keep retrying before giving up (~4 min). */
  totalBudgetMs: 240_000,
  /** Per-attempt ceiling: just past Modal's 150s synchronous 303 cap. */
  perAttemptMs: 160_000,
  /** If the first attempt outlives this, assume a cold start and hint the UI. */
  earlyWakingMs: 30_000,
  firstBackoffMs: 10_000,
  nextBackoffMs: 22_000,
} as const;

function abortError(): DOMException {
  return new DOMException("The coach request was cancelled.", "AbortError");
}

/** A hard client error (4xx other than 408/425/429) won't fix itself by waiting,
 *  so it should surface immediately rather than burn the whole retry budget. */
function isNonColdError(err: unknown): boolean {
  if (err instanceof CoachHttpError) {
    if (err.status === 408 || err.status === 425 || err.status === 429) return false;
    return err.status >= 400 && err.status < 500;
  }
  return false;
}

/** Abortable sleep that rejects with an AbortError the moment `signal` fires. */
function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(abortError());
      },
      { once: true },
    );
  });
}

/** A child signal that aborts when EITHER `outer` fires or `ms` elapses, so a
 *  hung attempt trips the timeout while the caller's signal still cancels all. */
function linkTimeout(
  outer: AbortSignal | undefined,
  ms: number,
): { signal: AbortSignal; cleanup: () => void } {
  const ctrl = new AbortController();
  const onAbort = () => ctrl.abort();
  const timer = setTimeout(() => ctrl.abort(), ms);
  if (outer) {
    if (outer.aborted) ctrl.abort();
    else outer.addEventListener("abort", onAbort, { once: true });
  }
  const cleanup = () => {
    clearTimeout(timer);
    outer?.removeEventListener("abort", onAbort);
  };
  return { signal: ctrl.signal, cleanup };
}

/** Fire-and-forget GET /api/health to start waking a scaled-to-zero container
 *  early. Best-effort: nothing is awaited and all errors are swallowed. */
export function warmupCoach(): void {
  try {
    void fetch(`${API_BASE}/api/health`, { method: "GET" }).catch(() => {
      /* a cold container may drop the first ping: that's expected */
    });
  } catch {
    /* ignore: warmup is a nudge, not a dependency */
  }
}

/** {@link postCoach} with cold-start resilience: retries with backoff across a
 *  Modal scale-to-zero cold start (network error / non-ok / 303 redirect
 *  exhaustion / per-attempt timeout) for up to ~4 min, reporting progress via
 *  `onStatus`. The passed `signal` cancels everything immediately (rethrowing
 *  AbortError); a hard client 4xx is surfaced without retrying. */
export async function postCoachResilient(
  req: CoachRequest,
  opts: PostCoachResilientOptions = {},
): Promise<CoachResponse> {
  const { signal, onStatus } = opts;
  const start = Date.now();
  const elapsedSec = (): number => Math.round((Date.now() - start) / 1000);

  // Early hint: warm calls return well before earlyWakingMs. If the first
  // attempt is still outstanding past it, we're almost certainly cold-starting.
  let earlyTimer: ReturnType<typeof setTimeout> | null = setTimeout(() => {
    earlyTimer = null;
    onStatus?.({ phase: "waking", attempt: 0, elapsedSec: elapsedSec() });
  }, COLD_START.earlyWakingMs);
  const clearEarly = (): void => {
    if (earlyTimer !== null) {
      clearTimeout(earlyTimer);
      earlyTimer = null;
    }
  };

  let attempt = 0;
  let lastError: unknown = null;

  try {
    while (Date.now() - start < COLD_START.totalBudgetMs) {
      if (signal?.aborted) throw abortError();

      const remaining = COLD_START.totalBudgetMs - (Date.now() - start);
      const link = linkTimeout(signal, Math.min(COLD_START.perAttemptMs, remaining));
      try {
        return await postCoach(req, link.signal);
      } catch (err) {
        lastError = err;
        // Outer cancellation → stop everything immediately.
        if (signal?.aborted) throw abortError();
        // A hard client error won't resolve by waiting: surface it now.
        if (isNonColdError(err)) throw err;
        attempt += 1;
      } finally {
        link.cleanup();
      }

      const elapsed = Date.now() - start;
      if (elapsed >= COLD_START.totalBudgetMs) break;

      clearEarly();
      onStatus?.({
        phase: attempt <= 1 ? "waking" : "retrying",
        attempt,
        elapsedSec: elapsedSec(),
      });

      const backoff = Math.min(
        attempt === 1 ? COLD_START.firstBackoffMs : COLD_START.nextBackoffMs,
        COLD_START.totalBudgetMs - elapsed,
      );
      await delay(backoff, signal);
    }
  } finally {
    clearEarly();
  }

  throw lastError instanceof Error ? lastError : new Error("The coach didn’t respond in time.");
}

/** One batch attempt is generous: it generates all three tiers server-side, off
 *  a single Stockfish pass: so we give it well past a warm three-tier run. */
const COACH_ALL_TIMEOUT_MS = 200_000;

/** POST /api/coach_all: coach a position at every tier in one round-trip.
 *
 * A SINGLE attempt with a generous timeout plus the same early-"waking" hint the
 * per-tier path uses, so the cold-start UI still shows progress. It deliberately
 * does NOT retry: any failure: a 404 before this endpoint is deployed, a
 * cold-start timeout, a network error, a hard 4xx: is thrown so the caller can
 * fall back to the proven per-tier {@link postCoachResilient} path, which keeps
 * its own progressive, cold-start-resilient loading. That fallback is why the
 * Studio never breaks when the batch route is missing or slow.
 */
export async function postCoachAll(
  req: CoachAllRequest,
  opts: PostCoachResilientOptions = {},
): Promise<CoachAllResponse> {
  const { signal, onStatus } = opts;
  const start = Date.now();
  const elapsedSec = (): number => Math.round((Date.now() - start) / 1000);

  // Warm calls return well before this fires; if we're still waiting past it,
  // surface the same "waking the model" hint the per-tier path shows.
  let earlyTimer: ReturnType<typeof setTimeout> | null = setTimeout(() => {
    earlyTimer = null;
    onStatus?.({ phase: "waking", attempt: 0, elapsedSec: elapsedSec() });
  }, COLD_START.earlyWakingMs);
  const clearEarly = (): void => {
    if (earlyTimer !== null) {
      clearTimeout(earlyTimer);
      earlyTimer = null;
    }
  };

  const link = linkTimeout(signal, COACH_ALL_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE}/api/coach_all`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal: link.signal,
    });
    if (!res.ok) throw new CoachHttpError(res.status, await readError(res));
    return (await res.json()) as CoachAllResponse;
  } finally {
    clearEarly();
    link.cleanup();
  }
}
