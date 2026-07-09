# Tournament Hall Design Identity

**Subject:** A local-first AI chess coach: a small fine-tuned model picks the tier-appropriate, instructive move for a player's level (Beginner/Intermediate/Advanced) + a principle tag, grounded by Stockfish + Maia + a truthfulness verifier.
**Audience:** Chess learners ~1000-2500 + a technical grader.
**Vibe:** A trustworthy, premium chess instrument. Tactile physical-chess. Calm, focused, and disciplined. NOT a generic AI dev tool.

## Color Palette (OKLCH)

The color strategy is **Committed**: a drenched deep tournament-felt green surface carries the identity, with a single warm brass/gold signal color for the recommended move and key CTAs.

*   **Background (Felt Green):** `oklch(0.28 0.045 155)` - The deep, drenched tournament felt. Never flat #000/#fff.
*   **Panels (Lighter Felt):** `oklch(0.32 0.04 155)` - For elevated surfaces, cards, and dense rows on the felt.
*   **Ink (Cream/Parchment):** `oklch(0.93 0.02 90)` - Primary text color. Warm, legible, premium.
*   **Muted Ink:** `oklch(0.78 0.02 90)` - Secondary text, subtle borders, inactive states.
*   **Signal (Warm Brass/Gold):** `oklch(0.78 0.13 85)` - The coach's arrow, recommended move, and primary CTAs. Luminous and distinct.
*   **Eval Beats (Win):** Calm win-green (distinct from felt).
*   **Eval Beats (Loss):** Muted clay-red.

## Board Tokens (Chessground)

*   **Light Squares (Chalk):** `oklch(0.90 0.02 90)`
*   **Dark Squares (Walnut):** `oklch(0.42 0.05 60)`
*   **Piece Lift:** Tactile piece-lift (scale + shadow) on `.cg-wrap piece.dragging`.
*   **Arrow:** Luminous brass drop-shadow on the coach's recommended-move arrow.
*   **Coordinates:** Contrast contextually per square + board orientation.

## Typography

*   **Display:** Archivo (Keep) - Strong, structured headings.
*   **Data/Mono:** Spline Sans Mono (Keep) - For engine lines, raw data, and technical details.
*   **Notation (Signature):** A chess-publishing SERIF (e.g., Merriweather, Playfair Display, or similar available Google Font) for SAN move notation. Evokes the "Informant" feel.
*   **Hierarchy:** >=1.25 scale + weight contrast.
*   **Body:** Cap line length at <=75ch for readability.

## Layout & Structure

*   **No Side-Stripe Borders:** Remove ALL side-stripe borders (e.g., `shadow-[inset_2px_0_0_0_var(--signal)]`). Use bg tint + severity dot / full ring instead.
*   **No Eyebrows:** Remove uppercase-tracked "eyebrow" kickers. Use sentence-case headings with size+weight contrast.
*   **No Nested Cards / Identical Grids:** Break identical card grids into dense rows on the felt surface with hairline rules. Native `<table>` for leaderboards/matrices.
*   **No Hero-Metric KPI Grids:** Remove generic SaaS KPI grids.

## Motion

*   **S-Tier Reveals:** Keep disciplined S-tier reveals.
*   **Easing:** Ease-out only. No bounce, no elastic.
*   **Reduced Motion:** Respect `prefers-reduced-motion`.
*   **No Layout Thrashing:** Remove rAF layout-thrash redrawAll/bounds loop in ChessgroundBoard (use ResizeObserver).
*   **Shimmer:** Upgrade C-tier skeleton shimmer to S-tier via `transform: translateX()`.

## Copy & Voice

*   **Voice:** Confident calm coach voice, no filler. Honest framing.
*   **Punctuation:** Purge ALL em dashes. Use colons, parentheses, or short sentences.
*   **Clutter:** Strip emoji clutter and over-bolding.

## Signature Element

The **Notation Serif** combined with the **Luminous Brass Arrow** on the **Chalk/Walnut Board** resting on **Deep Felt Green**. This instantly signals "premium chess instruction" rather than "generic AI dashboard."
