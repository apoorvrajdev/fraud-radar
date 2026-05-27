import type { ReactNode } from "react";
import { Card } from "./Card";
import { cn } from "../../lib/cn";

interface StatProps {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  className?: string;
}

/**
 * KPI tile: bold value on top, muted label above, optional hint
 * (delta, sub-metric, or unit) below.
 *
 * Loading and error states are rendered by the parent passing a
 * skeleton or fallback node into `value` — this component stays
 * presentational so it has no opinion on data fetching.
 */
export function Stat({ label, value, hint, className }: StatProps) {
  return (
    <Card className={cn("p-5", className)}>
      <div className="text-xs uppercase tracking-wide text-neutral-500">
        {label}
      </div>
      <div className="mt-2 text-3xl font-semibold tabular-nums text-neutral-100">
        {value}
      </div>
      {hint !== undefined && (
        <div className="mt-1 text-xs text-neutral-400">{hint}</div>
      )}
    </Card>
  );
}
