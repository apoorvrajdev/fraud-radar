/**
 * Score chip aligned to the AlertsSummary bucket boundaries
 * (Phase 3H). Keeping the per-row chip and the header-strip stat
 * card in lockstep means the visual vocabulary stays consistent
 * across the page: rose = high, amber = mid, neutral = low.
 *
 * See `docs/adr/PHASE_3H_DESIGN.md` for the empirical justification
 * of the 0.20 / 0.50 boundaries.
 */
import { cn } from "../../lib/cn";

type Bucket = "low" | "mid" | "high";

function scoreBucket(score: number): Bucket {
  if (score >= 0.5) return "high";
  if (score >= 0.2) return "mid";
  return "low";
}

const TONE: Record<Bucket, string> = {
  high: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  mid: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  low: "bg-neutral-700/30 text-neutral-400 border-neutral-700/60",
};

export function ScoreChip({ score }: { score: string }) {
  const n = Number(score);
  if (!Number.isFinite(n)) {
    return <span className="text-neutral-500">—</span>;
  }
  const bucket = scoreBucket(n);
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5",
        "text-xs font-medium tabular-nums",
        TONE[bucket],
      )}
      title={`Bucket: ${bucket}`}
    >
      {n.toFixed(4)}
    </span>
  );
}
