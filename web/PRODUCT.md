# Product

## Register

product

## Users

Chess improvers rated roughly 1000–2000 who have just finished a game (or are
reviewing one) and want to understand a single position better. They are
comfortable with a board and notation but find raw engine output — a wall of
principal variations and centipawn numbers — unhelpful on its own. Their context
is focused, often solitary study: "I felt unsure here, what should I have played,
and why, at my level?" They trust the tool because it is grounded in real engine
truth, not vibes.

## Product Purpose

An engine-grounded chess coach. The user sets a position (or picks from a
45-position study library), optionally marks the move they were unsure about,
picks their rating tier, and the coach returns **one** leveled, jargon-free
teaching move with a plain-language explanation. Every recommendation is grounded
in Stockfish (a pool of sound moves + principal variations) and Maia (how likely
a human of that level is to find each move), and the raw engine top-lines are
shown chess.com-style for the user who wants to verify. Success is the user
leaving with one clear idea they can actually apply in their next game —
not a dump of analysis they have to decode themselves.

## Brand Personality

Credible, calm, focused. The feeling of sitting down next to a strong coach who
has already done the analysis and now tells you the one thing that matters, in
your language. Trustworthy because it is measured against real engine truth.
Three words: **measured, grounded, unhurried.** The voice is confident and
plainspoken, never chatty, never hyped.

## Anti-references

- **Generic AI-slop SaaS.** Identical icon-card grids, hero-metric templates,
  gradient text, decorative glassmorphism, side-stripe accents, flashy dashboards.
  This is a study instrument, not a growth-metrics dashboard.
- **Gamified / childish chess apps.** No confetti, no XP, no cartoon mascots, no
  "streak" theatrics. The user is a serious improver, not a kid.
- **The first-order chess cliché:** wooden board + fancy serif + "royal game"
  gravitas. Avoid literary/editorial serif as the voice.
- **The second-order escape-slop reflex:** editorial-mono-on-cream, magazine
  restraint, italic display serif + ruled columns. Avoid the parchment/warm-paper
  study aesthetic (which the previous version fell into).

## Design Principles

1. **Grounded in truth, legible as language.** The interface must always make the
   engine evidence visible and verifiable, while the coaching itself reads as
   plain human advice. Show the measurement; speak the verdict.
2. **One move to focus on.** The product's whole promise is singular focus. The UI
   should make the recommended move unmistakably the loudest thing on screen and
   subordinate everything else to it.
3. **An instrument, not a toy.** Every control is precise and does exactly one
   thing. Calm, quiet, and confident beats decorated and eager. The tool
   disappears into the task.
4. **Respect the improver's intelligence.** No hand-holding tone, no jargon walls.
   Meet the user at their tier and explain, don't lecture or dumb down.
5. **Distinctive by intent, not by decoration.** Escape the category reflexes
   (wooden/serif AND cream/editorial) through a committed, coherent visual system,
   not through surface ornament.

## Accessibility & Inclusion

- Target **WCAG 2.1 AA** (the Studio's primary flow targets AA; dense comparison
  views still in progress): body/UI text ≥ 4.5:1 contrast, large text ≥ 3:1,
  interactive controls with visible `:focus-visible` rings and ≥ 44×44px hit areas.
- Never encode meaning by color alone (recommended vs. your move vs. engine data
  carry a label or shape, not just a hue) — important for color-vision deficiency.
- Full keyboard operability for the primary flow (pick tier, coach, take back,
  flip). Board interaction is pointer-first (chessground) but every surrounding
  control is reachable and labeled.
- Honor `prefers-reduced-motion`: the coaching reveal degrades to an instant,
  already-visible state.
