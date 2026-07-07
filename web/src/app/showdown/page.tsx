import type { Metadata } from "next";
import Showdown from "@/components/Showdown";

export const metadata: Metadata = {
  title: "Model Showdown · AI Chess Instructor",
  description:
    "Per held-out position, every model's recommended move on the same grounded input — highlighting where OURS is tier-appropriate and faithful while the frontier isn't.",
};

export default function ShowdownPage() {
  return <Showdown />;
}
