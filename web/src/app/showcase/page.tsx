import type { Metadata } from "next";
import Showcase from "@/components/Showcase";

export const metadata: Metadata = {
  title: "Multi-Model Showcase · AI Chess Instructor",
  description:
    "Compare all 14 models on the same grounded position: recommended move, tier-fit and faithfulness verdicts, blinded council grades, and coaching per rating tier — with OURS re-runnable live.",
};

export default function ShowcasePage() {
  return <Showcase />;
}
