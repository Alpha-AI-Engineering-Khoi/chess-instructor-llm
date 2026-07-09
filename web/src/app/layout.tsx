import type { Metadata } from "next";
import { Archivo, Spline_Sans_Mono, Merriweather } from "next/font/google";
import "./globals.css";
import EvidenceBar from "@/components/EvidenceBar";
// Chessground (Lichess board) styles loaded GLOBALLY in the root layout so the
// board is always styled on first paint — not dependent on the client-component
// bundle loading first (which could briefly render an unstyled board).
// base = board layout, brown = square theme, cburnett = embedded piece sprites.
import "chessground/assets/chessground.base.css";
import "chessground/assets/chessground.brown.css";
import "chessground/assets/chessground.cburnett.css";

// UI + display + coaching prose. One technical grotesque, hierarchy via weight +
// tracking + size.
const ui = Archivo({
  subsets: ["latin"],
  variable: "--font-ui",
  display: "swap",
});

// The notation face: recommended move (hero), evals, FEN, engine lines. Mono is
// strictly earned here — it renders real tabular chess data, never plain prose.
const mono = Spline_Sans_Mono({
  subsets: ["latin"],
  variable: "--font-data",
  display: "swap",
});

// The signature chess-publishing SERIF for SAN move notation (Informant feel).
const notation = Merriweather({
  weight: ["300", "400", "700", "900"],
  subsets: ["latin"],
  variable: "--font-notation",
  display: "swap",
});

export const metadata: Metadata = {
  title: "AI Chess Instructor",
  description:
    "An engine-grounded chess coach that reliably picks the level-appropriate move for your rating — a move plus a short principle tag, with a full written explanation as an optional layer. Set a position, mark the move you are unsure about, and pick your level.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`dark ${ui.variable} ${mono.variable} ${notation.variable} antialiased`}
      data-theme="dark"
      suppressHydrationWarning
    >
      <body className="flex min-h-dvh flex-col bg-background font-sans text-foreground">
        {/* Page content grows to fill; the global Evidence + Trust bar sits below it
            (and pins to the bottom of short pages via the flex column). */}
        <div className="flex flex-1 flex-col">{children}</div>
        <EvidenceBar />
      </body>
    </html>
  );
}
