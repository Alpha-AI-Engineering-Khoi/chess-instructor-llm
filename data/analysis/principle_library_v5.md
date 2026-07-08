# v5 Correctness-Checked Coaching Principle Library

> DE-DUPLICATED, chess-correct principle library distilled from the GothamChess /
> Naroditsky transcripts + standard chess pedagogy, **fact-checked against
> Stockfish logic**. Each principle carries **when-it-applies**, a **board cue**,
> a **correctness caveat**, and the **tier(s)** it belongs to. Wrong or
> context-dependent commentary heuristics were dropped/fixed — see the last
> section. Intended to REPLACE `prompts/principles.md` for v5 generation (drop-in
> as `{PRINCIPLES}`), and to seed the controlled takeaway-principle vocabulary.

## A. Named transferable principles (the `{PRINCIPLE}` vocabulary)

Each entry: **Principle** — *when it applies* / board cue / **caveat** [tiers].

1. **King safety first — get the king out of the center.**
   *Early game, king still on e1/e8, no forcing tactic.* Cue: open/semi-open
   center, uncastled king. **Caveat:** not absolute — if there is a concrete
   tactic, or the center is locked, castling can wait; never castle *into* an
   attack. [beginner, intermediate]

2. **Develop before you attack; develop every piece once.**
   *Opening/early middlegame with pieces on the back rank.* **Caveat:** develop
   *with a purpose* (ideally making a threat); don't move the same piece twice
   without a concrete reason, and don't launch a one-piece attack. [all]

3. **Don't leave pieces hanging — count attackers vs defenders.**
   *Every move (the safety check).* Cue: an undefended piece is attacked.
   Universal, no caveat. [beginner core]

4. **Make a threat while you develop (gain time).**
   *When a developing move can also attack something.* **Caveat (beginner
   wording):** say "gain time / for free," NOT "tempo." Don't chase tempo for its
   own sake. [all; beginner-safe if worded plainly]

5. **Don't bring the queen out too early.**
   *Opening.* Cue: early queen sortie that can be chased with development.
   **Caveat:** fine once it does a concrete job that can't be punished. [beginner]

6. **Control the center.**
   *Opening/middlegame.* **Caveat:** center can be controlled by pieces
   (hypermodern), not only pawns; a big pawn center is a liability if it can't be
   defended. [all]

7. **Put rooks on open (or half-open) files.**
   *Middlegame with open files.* **Caveat:** only if the file leads to a target or
   a penetration square — a rook on an open file with no entry point is not
   automatically good. [intermediate, advanced]

8. **A rook on the 7th (2nd) rank is powerful; double rooks for a battering ram.**
   *Late middlegame/endgame when a rook can reach the 7th.* **Caveat:** strong
   when it *hits pawns* or *cuts off the king*; a 7th-rank rook doing nothing (or
   trapped) is not. "Pigs on the 7th" (doubled) is very strong. [intermediate, advanced]

9. **Connected/doubled rooks support each other on a file.**
   *When contesting or owning a file.* Correct. [intermediate, advanced]

10. **Create and push a passed pawn — when it is safe.**
    *Endgames, or when you can make a passer.* **Caveat (FIX the slogan):**
    "passed pawns must be pushed" is an oversimplification. Push when it *gains
    ground safely*; otherwise **support or blockade first** — a premature push
    drops the pawn. Outside passed pawns are especially valuable. [intermediate, advanced]

11. **Blockade the opponent's passed pawn (a knight is the ideal blockader).**
    *Opponent has a passer.* Correct (Nimzowitsch). [advanced]

12. **Rook behind the passed pawn (yours or the opponent's).**
    *Rook endgames.* Correct (Tarrasch rule). [advanced]

13. **Improve your worst-placed piece.**
    *Quiet positions with no forcing move.* High-value; "find your saddest piece
    and give it a job." Correct. [intermediate, advanced]

14. **Trade pieces when you are AHEAD in material; keep pieces when BEHIND.**
    *When a trade is offered/available.* **Caveats (all correct chess):**
    (a) trade **pieces, not pawns**, when ahead (you want pawns to promote);
    (b) it is **not automatic** — evaluate what each trade changes (Naroditsky:
    "don't automatically simplify, look around"); (c) when *attacking* you may
    keep the queens on. The **inverse** ("trade when behind to reach an endgame")
    is a **beginner mistake** — see rejected list. [beginner (simple form), all]

15. **Prophylaxis — stop the opponent's plan before pursuing your own.**
    *Opponent has a concrete break/threat.* **Caveat:** advanced vocabulary; for
    lower tiers phrase as "take away their idea first." [advanced]

16. **Don't release central tension automatically; keep moves flexible.**
    *Middlegame with pawn tension.* Cue: an immediate capture that helps the
    opponent as much as you. Correct. [intermediate, advanced]

17. **Don't grab pawns at the cost of development / king safety, and don't open
    lines for the opponent.** *When a pawn grab is tempting.* **Caveat:** material
    still matters — take it if it's safe and there's no compensation. [all]

18. **Attack the king by opening lines and removing its pawn cover; bring enough
    attackers.** *Opposite-side castling / exposed king.* **Caveat:** only with
    the initiative and sufficient force; opening lines near *your own* king without
    activity backfires. [advanced]

19. **Outpost:** a square in/near enemy territory that your piece occupies and a
    pawn defends, that no enemy pawn can chase. Knights love outposts.
    *Middlegame with a hole in the opponent's camp.* Correct. [intermediate, advanced]

20. **Good vs bad bishop — improve, trade, or reroute your bad bishop.**
    *When your bishop is blocked by its own pawns.* Correct. [intermediate, advanced]

21. **The bishop pair is an asset — in OPEN positions.**
    **Caveat:** context-dependent; in closed positions knights can be better.
    [advanced]

22. **Activate the king in the endgame.**
    *Endgame, no mating danger.* Correct. [intermediate, advanced]

23. **Space advantage — gain it, but you need a break to use it.**
    **Caveat:** overextended space becomes a target; space with no pawn break is
    inert. [advanced]

24. **Fianchetto trade-offs:** it costs a tempo and leaves holes on the squares
    the bishop vacated. *When considering g3/b3 setups.* Correct. [advanced]

## B. Beginner-first "safety habits" (most human-findable, always sound)

These are the transferable principles that should dominate **beginner** takeaways
(the audit shows beginner strategic-principle naming is only ~69%, and only ~42%
in the takeaway; these fill that gap with correct, simple, named habits):

- **King safety first.**
- **Don't hang pieces — count attackers and defenders.**
- **Develop a new piece before pushing side pawns.**
- **Make a threat while you develop.**
- **Look at all checks, captures, and threats before you move.**
- **A move that does two jobs at once is usually best.**
- **When you're ahead, trade pieces (not pawns) to make winning easier.**
- **Don't bring the queen out early to be chased.**

## C. Tier assignment (what to teach where)

- **beginner (1000–1200):** B-list habits + #1,2,3,4,5,6,14(simple),17.
- **intermediate (1300–1600):** +#7,8,9,10,13,16,19,20,22.
- **advanced (1700–2000):** +#11,12,15,18,21,23,24 and nuanced forms of the rest.

## D. REJECTED or FIXED commentary heuristics (do NOT parrot)

| Heuristic as often stated | Verdict | Correct version |
|---|---|---|
| "When you're losing, **trade/simplify** and pray for the endgame." | **REJECT (inverted)** — the transcripts themselves frame this as a confessed *bad habit*. | Trade when **AHEAD**; keep pieces when **BEHIND** to retain counterplay. |
| "**Passed pawns must be pushed**" (as absolute). | **FIX** | Push only when it gains ground safely; otherwise **support/blockade** first. |
| "**Always castle early**, castling is always the priority." | **FIX** | Usually yes, but not into an attack and not when a concrete tactic or a locked center makes it wait. |
| "**Always trade when a trade is available**" (auto-simplify). | **REJECT** — Naroditsky: "don't automatically simplify, look around." | Evaluate each trade by what it changes (files opened, activity, structure). |
| "**Space advantage is always good.**" | **FIX** | Only if you have a break to use it; overextension = target. |
| "**The bishop pair is always better.**" | **FIX** | Only in **open** positions; closed → knights can be better. |
| "**Grab the free pawn.**" | **FIX** | Not if it costs development/king safety or opens lines for the opponent. |
| "Bring the queen out early to be **active**." | **REJECT** | It gets chased and loses time; develop minor pieces first. |
| "**Sacrifice / attack** looks strong, go for it." | **FIX** | Only with concrete compensation or a real initiative. |
| Rote opening move-orders ("knights before bishops always"). | **DROP as hard rules** | Keep only the transferable core (develop, don't move a piece twice without reason). |
| Centipawns / "the engine says" / eval numbers (from commentary). | **REJECT (engine-speak)** | Translate to a board effect: which piece/plan/square improves. |

## E. Why this matters for truthfulness (weak axis)

Replacing `principles.md` with this fact-checked set removes the *strategic*
claims most likely to be judged false by the cross-family semantic panel (the
"trade when behind", "passed pawns must be pushed", "space is always good"
heuristics). Combined with the existing narrow-faithfulness board-fact gate and
the wide-checker → LLM-judge exclusion, it targets the semantic-truth residual
where OURS trails the frontier.
