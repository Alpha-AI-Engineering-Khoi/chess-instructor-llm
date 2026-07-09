# UI Revamp Notes — AI Chess Instructor

A full UI/UX revamp run with the in-repo **impeccable** + **impeccable-swarm** skills
(partition → parallel QA swarm → fix → re-verify loop). Frontend only; the backend,
API contract (`src/lib/api.ts`), coaching/engine logic, and chessground move/coach
behavior were left untouched. `npx tsc --noEmit` stays clean.

## The verdict that started this

> "a lot of the app is UI AI slop; fully revamp the UI; fix the font."

The shipped app was the **"Analysis Room"**: a warm green-ink dark theme with a
literary serif (Hedvig) for the wordmark *and* the coaching prose, IBM Plex Mono
labels, a `feTurbulence` paper-grain overlay, "parchment" warm tokens, and a
tiny-uppercase-tracked mono eyebrow over nearly every block (~10 of them). That is
two AI-slop reflexes at once: the **first-order** chess cliché (wooden board + fancy
serif) and the **second-order** escape-slop reflex (warm-paper / editorial-mono
study aesthetic). Baseline swarm QA: **AI-slop FAIL on all 7 surfaces**, 21 P1s.

## The design direction chosen

**Scene sentence (the physical scene that forces the choices):**
> A focused club player at a dark desk late at night, replaying the game they just
> lost on one calm glowing screen — a precision instrument that *measures* the
> position and hands back a single, confident verdict; the mood is quiet
> concentration, not play.

That scene forces **dark** (night, single screen) and **cool** (a bench instrument,
not a cozy lamp). Aesthetic lane: a **precision measuring instrument / analysis
console** — deliberately not wooden-and-serif and not cream-editorial.

**Color strategy — Restrained (product floor), with a load-bearing signal.**
The palette *encodes the product thesis*: cool = measured engine truth; one warm
signal = the human verdict.
- Canvas: cool graphite, blue-tinted (OKLCH hue ~258), never pure black/white.
- Ink: cool near-white; `--muted` / `--faint` tuned so the Studio's primary flow targets WCAG AA (dense comparison views still in progress).
- `--signal` (luminous amber, hue ~66): **reserved** for the recommended move,
  current selection, focus ring, and the primary action — never decoration.
- `--engine` (cool blue, hue ~236): raw Stockfish/Maia data (the measured lane).
- `--your-move` (coral) for the move the user played; a red→orange→amber→cool
  severity ramp; meaning is never carried by color alone (labels + legend + tags).

**Type system (the explicit complaint — fully reworked).** Two families, paired on a
real structural axis, mono strictly *earned* (chess notation = tabular data):
- **Archivo** (variable grotesque) — UI, labels, headings, **coaching prose**
  (kills the serif chess voice), wordmark. Hierarchy via weight/size/tracking.
- **Spline Sans Mono** — the notation face: the big recommended move (notation-as-hero),
  evals, FEN, engine principal variations. Replaces IBM Plex Mono (a reflex-reject default).
- Fixed rem scale (~1.25 heading ratio) + one fluid `--text-verdict` clamp for the move;
  prose measure ≤66ch; 12px type floor (killed all the 9–11px labels).

## Biggest slop fixes

- **Typography:** serif chess prose → grotesque; IBM Plex Mono → Spline Sans Mono;
  deleted the ~10 tiny-uppercase-tracked mono eyebrows; enforced a 12px floor.
- **Absolute bans removed:** `feTurbulence` paper-grain; warm cream/parchment canvas;
  over-decorated board bezel (warm gradient + inset bevel + 60px glow → flat cool
  surface + hairline + one defined shadow); nested Takeaway card → single tinted inset.
- **Real bug fixed:** the Maia "Human odds" bars rendered invisible (an undefined
  `var(--parchment)` collapsed the fill to transparent) → now visible, absolute 0–100%.
- **Color semantics:** signal reserved for the move (removed amber from the wordmark
  mark, provenance chip, takeaway label, eyebrows); "your move" no longer painted as
  a danger/error before the coach speaks; a legend + per-row tags so nothing is color-only.
- **A11y / WCAG AA (primary flow):** the Studio's primary flow targets WCAG AA
  (dense comparison views still in progress); fixed the contrast failures (rust
  error text, translucent severity chips, faint captions, amber-on-amber eval chip);
  ≥44px tap targets (tier toggle, filter pills, retry); focus-visible signal rings;
  recommended move promoted to the panel `h2`.
- **States:** spinner-in-content → measured skeleton (console) + a status bar
  (board); plain-language error with retry + collapsible technical details;
  a real loading / error / empty distinction for the study library.
- **Mobile:** stack reordered to **board → coaching verdict → controls** so the
  product's promise isn't buried below the 45-item library; grounding chip kept visible.
- **Data honesty (instrument):** unified pawn-unit evals across both panels; Sound-move
  bars anchored to a fixed cp scale (no min–max exaggeration); no "-0.0".

## Before → After

| | Before (Run 1) | After (final) |
|---|---|---|
| AI-slop | FAIL ×7 | pass ×7 |
| Slices passing bar | 0/7 | 6/7 |
| Avg critique | 26.7/40 | 33.4/40 |
| Avg audit | 11.7/20 | 18.3/20 |
| Open P0 / P1 | 0 / 21 | 0 / 0 |

Evidence (headless inspector, desktop + mobile):
- Before: `.impeccable-swarm/run-1/s1/root/desktop.png`, `.impeccable-swarm/run-1/s7/root/mobile.png`
- After:  `.impeccable-swarm/run-3/_populated_live/desktop.png`, `.impeccable-swarm/run-4/_check/desktop.png`
- Full per-slice reports + scoreboards: `.impeccable-swarm/run-{1,2,3,4}/`

## Known residuals (non-blocking)

- **s6 analysis console** clears AI-slop, audit **20/20**, and zero P0/P1, but its
  **critique holds at 31/40** (bar 32) on a deliberate scope choice: it's a read-only
  "verify the position + one move" console, so Flexibility is 2/4 (no clickable /
  hoverable PV stepping). Making PVs interactive would change chessground board
  behavior — out of scope and guard-railed. Suggested if ever in scope: `$impeccable shape`.
- **Transient console noise:** ~8× chessground `<line> attribute NaN` are emitted only
  while the inspector takes its full-page screenshot (chessground redraws arrows at a
  transient 0×0 during the resize; `pos2user` does `min(1, w/h)` → `0/0`). Arrows render
  correctly and real users on a stable viewport never see them; the board component gates
  its own arrow draws behind a real-bounds check. Non-fatal.
- Minor polish carried in the scoreboard: `background-attachment: fixed` repaint on
  scroll; re-coach-on-tier-change; a couple of aria-live scoping refinements.
