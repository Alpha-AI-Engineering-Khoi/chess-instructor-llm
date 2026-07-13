"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

// The real in-app routes (static export serves them as .html). Benchmark is a
// separate HF Space, linked externally.
const LINKS: { href: string; label: string; match: string }[] = [
  { href: "/", label: "Studio", match: "/" },
  { href: "/showcase.html", label: "Showcase", match: "/showcase" },
  { href: "/showdown.html", label: "Showdown", match: "/showdown" },
];
const BENCHMARK_URL = "https://khoilamalphaai-chess-coach-benchmark.static.hf.space";

/** One uniform top bar shared across every page (rendered once in the root
 *  layout). Tournament-Hall identity: felt surface, brass wordmark + active
 *  state. Keyboard-focusable, responsive (wraps on narrow screens), and it uses
 *  only the existing design tokens. */
export default function NavBar() {
  // Static export can serve pages at "/route.html"; normalize so the active
  // state matches whether the URL carries ".html" or a trailing slash.
  const raw = usePathname() ?? "/";
  const path = raw.replace(/\.html$/, "").replace(/\/+$/, "") || "/";

  return (
    <header
      className="sticky top-0 z-40 border-b border-[color:var(--border)]"
      style={{ backgroundColor: "color-mix(in oklab, var(--background) 90%, transparent)" }}
    >
      <nav
        aria-label="Primary"
        className="mx-auto flex w-full max-w-[1240px] flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-2.5 backdrop-blur-sm sm:px-6 lg:px-8"
      >
        <Link
          href="/"
          aria-label="Chess Coach — Studio home"
          className="group inline-flex items-center gap-2 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60"
        >
          <span
            aria-hidden
            className="grid size-6 place-items-center rounded-[6px] bg-signal/15 font-serif text-sm leading-none text-signal ring-1 ring-signal/30"
          >
            ♞
          </span>
          <span className="font-semibold tracking-tight text-ink">Chess&nbsp;Coach</span>
        </Link>

        <div className="flex flex-wrap items-center gap-1">
          {LINKS.map((l) => {
            const active = path === l.match;
            return (
              <Link
                key={l.href}
                href={l.href}
                aria-current={active ? "page" : undefined}
                className={`inline-flex min-h-9 cursor-pointer items-center rounded-full px-3 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60 motion-reduce:transition-none ${
                  active
                    ? "bg-signal/15 text-signal ring-1 ring-signal/40"
                    : "text-muted hover:bg-[color:var(--surface-tertiary)] hover:text-ink"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
          <a
            href={BENCHMARK_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex min-h-9 cursor-pointer items-center gap-1 rounded-full px-3 text-sm font-medium text-muted transition-colors hover:bg-[color:var(--surface-tertiary)] hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60 motion-reduce:transition-none"
          >
            Benchmark
            <span aria-hidden className="text-faint">↗</span>
          </a>
        </div>
      </nav>
    </header>
  );
}
