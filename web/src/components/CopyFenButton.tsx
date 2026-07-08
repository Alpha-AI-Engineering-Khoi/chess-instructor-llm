"use client";

// A subtle "Copy FEN" control: writes the current position's FEN to the
// clipboard and shows a brief transient "Copied" confirmation. Shared by the
// Showcase detail view and the Coach Studio board so both read identically.

import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@heroui/react";
import { CheckIcon, CopyIcon } from "./icons";

interface CopyFenButtonProps {
  fen: string;
  /** Extra classes for placement/height tuning at the call site. */
  className?: string;
}

const COPIED_MS = 1500;

export default function CopyFenButton({ fen, className }: CopyFenButtonProps) {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Reset the confirmation if the position changes out from under us, and clear
  // any pending timer on unmount so we never setState after teardown.
  useEffect(() => {
    setCopied(false);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [fen]);

  const copy = useCallback(() => {
    // Clipboard is unavailable in insecure contexts / older browsers — bail
    // quietly rather than throw. (localhost + https are secure contexts.)
    if (!navigator.clipboard) return;
    // Handle the promise so a denied/failed write never becomes an unhandled
    // rejection; on success flip to "Copied" and revert after a beat.
    void navigator.clipboard
      .writeText(fen)
      .then(() => {
        setCopied(true);
        if (timerRef.current) clearTimeout(timerRef.current);
        timerRef.current = setTimeout(() => setCopied(false), COPIED_MS);
      })
      .catch(() => {
        // Permission denied or clipboard unavailable — leave the label as-is.
      });
  }, [fen]);

  return (
    <Button
      variant="tertiary"
      size="sm"
      className={`gap-1.5 ${className ?? ""}`}
      aria-label={copied ? "FEN copied to clipboard" : "Copy position FEN to clipboard"}
      onPress={copy}
    >
      {copied ? (
        <CheckIcon width={14} height={14} className="text-[color:var(--good)]" />
      ) : (
        <CopyIcon width={14} height={14} className="text-faint" />
      )}
      {copied ? "Copied" : "Copy FEN"}
    </Button>
  );
}
