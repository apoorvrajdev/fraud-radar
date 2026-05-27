/**
 * Color-coded pill for a transaction's fraud decision.
 *
 * Colors map directly to the operational meaning analysts already
 * have in their heads: green = clean, amber = needs eyes, red =
 * blocked, neutral = not scored yet.
 */
import type { Decision } from "../../types/api";
import { cn } from "../../lib/cn";

interface Props {
  decision: Decision | null;
}

const STYLES: Record<Decision, string> = {
  APPROVE: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
  REVIEW: "bg-amber-500/10 text-amber-300 border-amber-500/30",
  DECLINE: "bg-rose-500/10 text-rose-300 border-rose-500/30",
  PENDING: "bg-neutral-500/10 text-neutral-400 border-neutral-500/30",
};

export function DecisionBadge({ decision }: Props) {
  const value: Decision = decision ?? "PENDING";
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5",
        "text-[10px] font-medium uppercase tracking-wider",
        STYLES[value],
      )}
    >
      {value}
    </span>
  );
}
