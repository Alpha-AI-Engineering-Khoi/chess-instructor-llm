"use client";

import { ToggleButton, ToggleButtonGroup } from "@heroui/react";
import type { Tier } from "@/lib/api";

const TIERS: { id: Tier; label: string; band: string }[] = [
  { id: "beginner", label: "Beginner", band: "1000-1200" },
  { id: "intermediate", label: "Intermediate", band: "1300-1600" },
  { id: "advanced", label: "Advanced", band: "1700-2000" },
];

export default function TierControl({
  tier,
  onChange,
  disabled,
}: {
  tier: Tier;
  onChange: (t: Tier) => void;
  disabled?: boolean;
}) {
  const band = TIERS.find((t) => t.id === tier)?.band ?? "";
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        <span id="tier-control-label" className="text-sm font-medium text-ink">
          Coach at my level
        </span>
        <span className="text-xs text-muted tnum">{band.trim()}</span>
      </div>
      <ToggleButtonGroup
        aria-labelledby="tier-control-label"
        selectionMode="single"
        disallowEmptySelection
        fullWidth
        size="lg"
        isDisabled={disabled}
        selectedKeys={new Set([tier])}
        onSelectionChange={(keys) => {
          const next = [...keys][0];
          if (next) onChange(next as Tier);
        }}
      >
        {TIERS.map((t, i) => (
          <ToggleButton
            key={t.id}
            id={t.id}
            className="min-h-11"
            // Announce the rating band with the tier so the choice is unambiguous
            // to a screen reader (the band is only shown visually otherwise).
            aria-label={`${t.label}, rated ${t.band}`}
          >
            {i > 0 && <ToggleButtonGroup.Separator />}
            {t.label}
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
    </div>
  );
}
