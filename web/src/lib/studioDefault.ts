// AUTO-GENERATED cached-first seed for the Studio homepage.
//
// The DEFAULT king-and-pawn endgame (id vaLVwTHK_77) with the tuned coach's
// PRECOMPUTED answer at all three rating tiers, so the Studio renders the
// tier-adaptive move INSTANTLY on mount without any live call to the
// scale-to-zero Modal endpoint (which cold-starts in ~2-3 min).
//
// Provenance (all values are precomputed, never fabricated at runtime):
//   - per-tier recommended move + coaching prose + takeaway: regenerated LIVE
//     through the v6-dpo2 endpoint (/api/coach_all) -- the same tuned model the
//     live demo serves (Qwen3-32B + chess-coach-v6-dpo2 QLoRA).
//   - position-level engine facts (sound pool, best move, the student's Ne2 +
//     severity) and Maia human-frequency come back from the SAME gated pipeline,
//     computed once per position exactly as the live /api/coach_all does.
//
// Regenerate with: python scripts/reseed_studio_default.py --endpoint <url>
import type { CoachResponse, Tier } from "@/lib/api";

export const STUDIO_DEFAULT_FEN = "8/7b/5p2/P1kp3P/2pN1P2/4K3/8/8 w - - 1 39";
export const STUDIO_DEFAULT_STUDENT_UCI = "d4e2";

/** Precomputed tuned-coach answers for the default position, one per tier. */
export const STUDIO_DEFAULT_TIERS: Record<Tier, CoachResponse> = {
  "beginner": {
    "recommended_move_san": "Ne6+",
    "recommended_move_uci": "d4e6",
    "coaching": "Move: Ne6+. Play Ne6+. Your Ne2 was understandable because it defended the pawn on f4, but it missed a more active way to do that same defensive job. Ne6+ uses your knight with tempo: it gives check by attacking the black king on c5, and it still defends the pawn on f4. So you improve your knight while making Black answer your check right away. How to find it: Before choosing a quiet defending move, run this simple checklist: first look for checks, then ask if any of those checks also helps defend something. Here, Ne6+ is easy to spot once you notice both jobs: the knight gives check and protects f4.",
    "takeaway": "Look for moves that do two useful things at once, especially checks that also defend an important pawn.",
    "concepts_used": [],
    "side_to_move": "white",
    "engine": {
      "best_san": "h6",
      "best_cp": 362,
      "sound_pool": [
        {
          "san": "h6",
          "uci": "h5h6",
          "cp": 362,
          "pv": [
            "h6",
            "c3",
            "f5",
            "c2",
            "Nxc2",
            "Bxf5"
          ]
        },
        {
          "san": "Kd2",
          "uci": "e3d2",
          "cp": 361,
          "pv": [
            "Kd2",
            "Bg8",
            "Kc3",
            "Bf7",
            "a6",
            "Kb6"
          ]
        },
        {
          "san": "Ne2",
          "uci": "d4e2",
          "cp": 253,
          "pv": [
            "Ne2"
          ]
        },
        {
          "san": "f5",
          "uci": "f4f5",
          "cp": 235,
          "pv": [
            "f5",
            "Bg8",
            "Kd2",
            "Bh7",
            "Kc3",
            "Bg8"
          ]
        },
        {
          "san": "Ne6+",
          "uci": "d4e6",
          "cp": 214,
          "pv": [
            "Ne6+",
            "Kb5",
            "Kd4",
            "Kxa5",
            "Nc7",
            "Kb4"
          ]
        }
      ],
      "student_move": {
        "san": "Ne2",
        "uci": "d4e2",
        "cp_loss": 147,
        "severity": "mistake"
      }
    },
    "maia": [
      {
        "san": "a6",
        "uci": "a5a6",
        "policy": 0.4177
      },
      {
        "san": "Ne6+",
        "uci": "d4e6",
        "policy": 0.1454
      },
      {
        "san": "f5",
        "uci": "f4f5",
        "policy": 0.1251
      },
      {
        "san": "h6",
        "uci": "h5h6",
        "policy": 0.1241
      },
      {
        "san": "Kd2",
        "uci": "e3d2",
        "policy": 0.0646
      },
      {
        "san": "Ne2",
        "uci": "d4e2",
        "policy": 0.0351
      }
    ],
    "meta": {
      "model": "OURS-v6-dpo2 (Qwen3-32B tuned)",
      "tuned": true,
      "notes": [],
      "attempts": 1,
      "verified_fallback": false
    }
  },
  "intermediate": {
    "recommended_move_san": "h6",
    "recommended_move_uci": "h5h6",
    "coaching": "Play h6. Your Ne2 was understandable because it defends the pawn on f4, but it spends the move only improving defense. h6 uses your advanced kingside pawn more actively while keeping your other options available. In this kind of endgame, an active passed pawn matters: if Black plays ...c3, you have the simple follow-up f5, pushing forward instead of letting Black’s pawns take over. How to find it: Use this routine in practical endgames: first ask, 'Do I have any urgent hanging-piece problem?' Here there isn’t one. Then compare quiet defensive moves like Ne2 with active pawn pushes: 'Does my move improve something immediately, or does it keep pressure by advancing a passer?' Since Ne2 mainly defends f4, look next at your farthest advanced pawns and calculate one useful reply. After h6, if ...c3 happens, f5 shows up as your clear answer.",
    "takeaway": "When nothing is hanging in an endgame, do not automatically defend—look for a safe way to make your passed pawns work together.",
    "concepts_used": [],
    "side_to_move": "white",
    "engine": {
      "best_san": "h6",
      "best_cp": 362,
      "sound_pool": [
        {
          "san": "h6",
          "uci": "h5h6",
          "cp": 362,
          "pv": [
            "h6",
            "c3",
            "f5",
            "c2",
            "Nxc2",
            "Bxf5"
          ]
        },
        {
          "san": "Kd2",
          "uci": "e3d2",
          "cp": 361,
          "pv": [
            "Kd2",
            "Bg8",
            "Kc3",
            "Bf7",
            "a6",
            "Kb6"
          ]
        },
        {
          "san": "Ne2",
          "uci": "d4e2",
          "cp": 253,
          "pv": [
            "Ne2"
          ]
        },
        {
          "san": "f5",
          "uci": "f4f5",
          "cp": 235,
          "pv": [
            "f5",
            "Bg8",
            "Kd2",
            "Bh7",
            "Kc3",
            "Bg8"
          ]
        },
        {
          "san": "Ne6+",
          "uci": "d4e6",
          "cp": 214,
          "pv": [
            "Ne6+",
            "Kb5",
            "Kd4",
            "Kxa5",
            "Nc7",
            "Kb4"
          ]
        }
      ],
      "student_move": {
        "san": "Ne2",
        "uci": "d4e2",
        "cp_loss": 147,
        "severity": "mistake"
      }
    },
    "maia": [],
    "meta": {
      "model": "OURS-v6-dpo2 (Qwen3-32B tuned)",
      "tuned": true,
      "notes": [
        "Human-likelihood (Maia) analysis was unavailable for this position."
      ],
      "attempts": 1,
      "verified_fallback": false
    }
  },
  "advanced": {
    "recommended_move_san": "h6",
    "recommended_move_uci": "h5h6",
    "coaching": "Move to learn: h6  \nYour Ne2 was understandable because it defends the pawn on f4, but it also makes the knight less active. In this endgame, activity matters more than simply guarding every pawn. h6 advances the h-pawn while keeping your other options flexible: if ...c3 happens, you have the simple follow-up f5. The point is practical conversion — improve the structure first, keep the kingside space, and only then decide where the knight belongs. How to find it: Run an endgame priority scan before choosing a defensive-looking move like Ne2: 1) Are any pieces actually hanging? Here, no. 2) Which side has fixed weaknesses I need to avoid letting expand freely? With black’s pawns already on c4 and d5, watch ...c3. 3) Can my passed or farthest advanced pawn make progress without dropping something? Yes — h6 pushes the h-pawn forward. 4) After the opponent gains time by pushing, do I still have a useful next lever? Yes — after ...c3, look for f5. That routine leads you to h6 rather than automatically defending f4.",
    "takeaway": "In quiet endgames, don’t defend passively unless there is real danger; first ask whether your best-placed pawn can advance safely and create a clear counterplan.",
    "concepts_used": [],
    "side_to_move": "white",
    "engine": {
      "best_san": "h6",
      "best_cp": 362,
      "sound_pool": [
        {
          "san": "h6",
          "uci": "h5h6",
          "cp": 362,
          "pv": [
            "h6",
            "c3",
            "f5",
            "c2",
            "Nxc2",
            "Bxf5"
          ]
        },
        {
          "san": "Kd2",
          "uci": "e3d2",
          "cp": 361,
          "pv": [
            "Kd2",
            "Bg8",
            "Kc3",
            "Bf7",
            "a6",
            "Kb6"
          ]
        },
        {
          "san": "Ne2",
          "uci": "d4e2",
          "cp": 253,
          "pv": [
            "Ne2"
          ]
        },
        {
          "san": "f5",
          "uci": "f4f5",
          "cp": 235,
          "pv": [
            "f5",
            "Bg8",
            "Kd2",
            "Bh7",
            "Kc3",
            "Bg8"
          ]
        },
        {
          "san": "Ne6+",
          "uci": "d4e6",
          "cp": 214,
          "pv": [
            "Ne6+",
            "Kb5",
            "Kd4",
            "Kxa5",
            "Nc7",
            "Kb4"
          ]
        }
      ],
      "student_move": {
        "san": "Ne2",
        "uci": "d4e2",
        "cp_loss": 147,
        "severity": "mistake"
      }
    },
    "maia": [
      {
        "san": "a6",
        "uci": "a5a6",
        "policy": 0.2981
      },
      {
        "san": "h6",
        "uci": "h5h6",
        "policy": 0.20620000000000002
      },
      {
        "san": "f5",
        "uci": "f4f5",
        "policy": 0.193
      },
      {
        "san": "Ne6+",
        "uci": "d4e6",
        "policy": 0.1396
      },
      {
        "san": "Kd2",
        "uci": "e3d2",
        "policy": 0.0591
      },
      {
        "san": "Ne2",
        "uci": "d4e2",
        "policy": 0.0404
      }
    ],
    "meta": {
      "model": "OURS-v6-dpo2 (Qwen3-32B tuned)",
      "tuned": true,
      "notes": [],
      "attempts": 1,
      "verified_fallback": false
    }
  }
};
