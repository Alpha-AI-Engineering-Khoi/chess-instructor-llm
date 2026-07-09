// Minimal inline icons (stroke = currentColor). Decorative unless given a title.

type IconProps = React.SVGProps<SVGSVGElement>;

const base = {
  width: 20,
  height: 20,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export function LampIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M8 3h8l3 7H5l3-7Z" />
      <path d="M12 10v7" />
      <path d="M9 20h6" />
      <path d="M10 17h4" />
    </svg>
  );
}

export function ArrowRightIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M5 12h14" />
      <path d="m13 6 6 6-6 6" />
    </svg>
  );
}

export function FlipIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M4 8a8 8 0 0 1 13.7-5.6L20 5" />
      <path d="M20 3v3h-3" />
      <path d="M20 16a8 8 0 0 1-13.7 5.6L4 19" />
      <path d="M4 21v-3h3" />
    </svg>
  );
}

export function FlipVerticalIcon({ style, ...props }: IconProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={18}
      height={18}
      viewBox="0 0 12 12"
      fill="currentColor"
      aria-hidden
      {...props}
      style={{ transform: "rotate(90deg)", ...style }}
    >
      <path d="M6.9248 0.074198C7.18465 -0.050692 7.49362 -0.0160603 7.71875 0.164042L8.96875 1.16404C9.14665 1.30637 9.25 1.52215 9.25 1.74998C9.24999 1.9778 9.14665 2.1936 8.96875 2.33592L7.71875 3.33592C7.49363 3.516 7.18464 3.55064 6.9248 3.42576C6.66515 3.30081 6.50002 3.03815 6.5 2.74998V2.49998H3.25C2.55964 2.49998 2 3.05962 2 3.74998V8.24998C1.99999 8.66418 1.66421 8.99998 1.25 8.99998C0.835793 8.99998 0.500011 8.66418 0.5 8.24998V3.74998C0.5 2.2312 1.73122 0.99998 3.25 0.999979H6.5V0.749979C6.5 0.461801 6.66515 0.199154 6.9248 0.074198Z" />
      <path d="M10.75 3C11.1642 3 11.5 3.33579 11.5 3.75V8.25C11.5 9.76878 10.2688 11 8.75 11H5.75V11.25C5.75 11.5382 5.58485 11.8008 5.3252 11.9258C5.06535 12.0507 4.75638 12.016 4.53125 11.8359L3.28125 10.8359C3.10334 10.6936 3 10.4778 3 10.25C3.00001 10.0222 3.10334 9.80639 3.28125 9.66406L4.53125 8.66406C4.75637 8.48397 5.06535 8.44934 5.3252 8.57422C5.58485 8.69917 5.74999 8.96182 5.75 9.25V9.5H8.75C9.44036 9.5 10 8.94036 10 8.25V3.75C10 3.33579 10.3358 3 10.75 3Z" />
    </svg>
  );
}

// A shield with a check: marks an engine-verified, truth-gated explanation.
export function ShieldCheckIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 3 5 6v5c0 4 3 6.6 7 8 4-1.4 7-4 7-8V6l-7-3Z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}

export function SparkIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 3v4" />
      <path d="M12 17v4" />
      <path d="M3 12h4" />
      <path d="M17 12h4" />
      <path d="m6 6 2.5 2.5" />
      <path d="m15.5 15.5 2.5 2.5" />
      <path d="m18 6-2.5 2.5" />
      <path d="m8.5 15.5-2.5 2.5" />
    </svg>
  );
}

export function QuoteIcon(props: IconProps) {
  return (
    <svg {...base} fill="currentColor" stroke="none" {...props}>
      <path d="M9.5 6C6.5 6.8 5 9 5 12.2V18h5.5v-5.5H8c0-2 .8-3.2 2.6-3.8L9.5 6Zm9 0c-3 .8-4.5 3-4.5 6.2V18H19.5v-5.5H17c0-2 .8-3.2 2.6-3.8L18.5 6Z" />
    </svg>
  );
}

export function BoardIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <rect x="3" y="3" width="18" height="18" rx="1.5" />
      <path d="M3 9h18M3 15h18M9 3v18M15 3v18" />
    </svg>
  );
}

export function ResetIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
      <path d="M3 3v5h5" />
    </svg>
  );
}

// Take-back: an arrow curving back (distinct from the circular reset/clear glyph).
export function UndoIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M9 14 4 9l5-5" />
      <path d="M4 9h10a6 6 0 0 1 0 12h-3" />
    </svg>
  );
}

// Two stacked sheets: the copy-to-clipboard affordance.
export function CopyIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

// A check: confirms the copy landed.
export function CheckIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="m5 12 5 5L20 7" />
    </svg>
  );
}

// Stacked planes: the released model weights (Evidence bar link).
export function LayersIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="m12 3 9 5-9 5-9-5 9-5Z" />
      <path d="m3 13 9 5 9-5" />
    </svg>
  );
}

// A database cylinder: the benchmark dataset (Evidence bar link).
export function DatabaseIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5" />
      <path d="M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6" />
    </svg>
  );
}

// A rocket: the live interactive Space (Evidence bar link).
export function RocketIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09Z" />
      <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2Z" />
      <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0" />
      <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" />
    </svg>
  );
}

// A book: the BrainLift write-up (Evidence bar link).
export function BookIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z" />
    </svg>
  );
}

// The GitHub mark: the source repository (Evidence bar link). Filled glyph.
export function GitHubIcon({ "aria-hidden": ariaHidden = true, ...props }: IconProps) {
  return (
    <svg
      width={20}
      height={20}
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden={ariaHidden}
      {...props}
    >
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.604-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .269.18.579.688.481A10.02 10.02 0 0 0 22 12.017C22 6.484 17.523 2 12 2Z"
      />
    </svg>
  );
}

// A small "opens in a new tab" arrow: pairs with external links.
export function ExternalLinkIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M15 3h6v6" />
      <path d="M10 14 21 3" />
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
    </svg>
  );
}
