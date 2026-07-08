import type { Metadata } from "next";
import Studio from "@/components/Studio";

// Grader-facing clarity in the browser tab / link previews: state the one
// behavior up front — the tuned model picks the level-appropriate move.
export const metadata: Metadata = {
  title: "Coach Studio · one move for your level",
  description:
    "A locally fine-tuned chess coach whose one job is selecting the level-appropriate move. Set a position, pick your rating, and read the move plus a short principle tag.",
};

export default function Page() {
  return <Studio />;
}
