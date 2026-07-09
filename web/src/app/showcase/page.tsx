import type { Metadata } from "next";
import Showcase from "@/components/Showcase";

export const metadata: Metadata = {
  title: "Multi-Model Showcase · AI Chess Instructor",
  description:
    "Compare every model on the same grounded position: the tier-appropriate move it selects per rating, tier-fit and faithfulness verdicts, blinded council grades, and optional coaching text: with OURS re-runnable live.",
};

export default function ShowcasePage() {
  return <Showcase />;
}
