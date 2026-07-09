// EVIDENCE + TRUST BAR
// A global footer, rendered once in the root layout, that surfaces the ONE
// deterministic claim this product is built on (tier-appropriate move selection)
// as a measured, non-hand-wavy proof, next to the artifacts anyone can check:
// the model weights, the benchmark dataset, the live Space, the source, and the
// written thesis. On-theme (Bench-Instrument dark tokens, one warm signal accent);
// no server state, so it's a plain server component. Links open in a new tab and
// inherit the global pointer + focus-visible affordances from globals.css.

import {
  BookIcon,
  DatabaseIcon,
  ExternalLinkIcon,
  GitHubIcon,
  LayersIcon,
  RocketIcon,
  ShieldCheckIcon,
} from "./icons";

type EvidenceLink = {
  href: string;
  label: string;
  hint: string;
  Icon: (props: React.SVGProps<SVGSVGElement>) => React.ReactElement;
};

// The verifiable artifacts. Order goes from "the thing itself" (weights) outward
// to the write-up, so the proof reads left-to-right as: here is the model, here is
// what it was measured on, try it, read the code, read the reasoning.
const LINKS: EvidenceLink[] = [
  {
    href: "https://huggingface.co/khoilamalphaai/chess-coach-32b-v4-qlora",
    label: "Model",
    hint: "Qwen3-32B chess-coach v4 · QLoRA adapter · Hugging Face",
    Icon: LayersIcon,
  },
  {
    href: "https://huggingface.co/datasets/khoilamalphaai/chess-coach-move-review",
    label: "Dataset",
    hint: "chess-coach-move-review · Hugging Face",
    Icon: DatabaseIcon,
  },
  {
    href: "https://huggingface.co/spaces/khoilamalphaai/chess-coach-studio",
    label: "Space",
    hint: "Live v4 coach demo · Hugging Face",
    Icon: RocketIcon,
  },
  {
    href: "https://github.com/Alpha-AI-Engineering-Khoi/chess-instructor-llm",
    label: "GitHub",
    hint: "Source repository",
    Icon: GitHubIcon,
  },
  {
    href: "https://github.com/Alpha-AI-Engineering-Khoi/chess-instructor-llm/blob/main/BRAINLIFT.md",
    label: "BrainLift",
    hint: "The written thesis behind the model",
    Icon: BookIcon,
  },
];

export default function EvidenceBar() {
  return (
    <footer
      aria-labelledby="evidence-heading"
      className="relative z-[1] mt-8 border-t border-[color:var(--separator)] bg-[color:var(--surface)]"
    >
      <div className="mx-auto flex w-full max-w-[1320px] flex-col gap-5 px-4 py-5 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:gap-8 lg:py-4">
        {/* The deterministic proof: the honest headline number, stated once. */}
        <div className="flex min-w-0 flex-col gap-1.5">
          <h2
            id="evidence-heading"
            className="flex items-center gap-2 text-sm font-semibold text-ink"
          >
            <ShieldCheckIcon width={15} height={15} className="text-signal" />
            Deterministic proof
          </h2>
          <p className="max-w-2xl text-pretty text-[13px] leading-relaxed text-muted">
            Tier-appropriate move selection lifts{" "}
            <span className="font-mono text-muted tnum">34.7%</span>{" "}
            <span className="text-muted">base</span> →{" "}
            <span className="font-mono font-semibold text-signal tnum">76.7%</span>{" "}
            <span className="text-ink">tuned</span> on the strict held-out eval &mdash; the top
            tier-fit of every model measured, past the best frontier at{" "}
            <span className="font-mono text-ink tnum">55.3%</span>. On the{" "}
            <span className="font-mono text-ink tnum">62</span> held-out positions where OURS diverges
            from the best frontier, it wins the tier-appropriate move{" "}
            <span className="font-mono font-semibold text-signal tnum">51&ndash;5</span> (6 ties).
          </p>
        </div>

        {/* The artifacts anyone can check: each opens in a new tab. */}
        <nav
          aria-label="Model, dataset, and project links"
          className="flex shrink-0 flex-wrap items-center gap-2"
        >
          {LINKS.map(({ href, label, hint, Icon }) => (
            <a
              key={href}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              title={hint}
              aria-label={`${label}: ${hint} (opens in a new tab)`}
              className="group inline-flex items-center gap-1.5 rounded-full border border-[color:var(--border)] bg-[color:var(--surface-secondary)]/60 px-3 py-1.5 text-[13px] font-medium text-muted transition-colors hover:border-[color:var(--field-border)] hover:bg-[color:var(--surface-tertiary)] hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60"
            >
              <Icon width={14} height={14} className="text-faint transition-colors group-hover:text-signal" />
              {label}
              <ExternalLinkIcon
                width={11}
                height={11}
                className="text-faint/70 transition-colors group-hover:text-muted"
              />
            </a>
          ))}
        </nav>
      </div>
    </footer>
  );
}
