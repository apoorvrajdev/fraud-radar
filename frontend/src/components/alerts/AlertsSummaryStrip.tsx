/**
 * Header strip for the alerts queue (Phase 3H).
 *
 * Four KPI tiles, all reading from the queue-wide `AlertsSummary`
 * block on page 0 of the infinite query. The strip stays *whole-
 * queue*: it deliberately ignores the analyst's filters so the
 * triage operator always sees the full pending picture, not the
 * filtered subset.
 */
import { Stat } from "../ui/Stat";
import { formatAge, formatInt } from "../../lib/format";
import type { AlertsSummary } from "../../types/api";

interface Props {
  summary: AlertsSummary | undefined;
  isLoading: boolean;
  isError: boolean;
}

export function AlertsSummaryStrip({ summary, isLoading, isError }: Props) {
  const value = (render: (s: AlertsSummary) => string) => {
    if (isLoading) return <Skeleton />;
    if (isError || !summary)
      return <span className="text-neutral-500">—</span>;
    return render(summary);
  };

  const hint = (render: (s: AlertsSummary) => string) => {
    if (isLoading) return undefined;
    if (isError || !summary) return "data unavailable";
    return render(summary);
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <Stat
        label="Pending review"
        value={value((s) => formatInt(s.pending_count))}
        hint={hint((s) =>
          s.pending_count === 0
            ? "queue clear"
            : `${formatInt(s.score_buckets.low)} low · ${formatInt(
                s.score_buckets.mid,
              )} mid · ${formatInt(s.score_buckets.high)} high`,
        )}
      />
      <Stat
        label="Oldest pending"
        value={value((s) => formatAge(s.oldest_pending_seconds))}
        hint={hint((s) =>
          s.oldest_pending_seconds == null
            ? "nothing waiting"
            : "time since the row entered the queue",
        )}
      />
      <Stat
        label="Strong ML signal"
        value={value((s) => formatInt(s.score_buckets.high))}
        hint={hint(() => "score ≥ 0.50")}
      />
      <Stat
        label="Mid-band"
        value={value((s) => formatInt(s.score_buckets.mid))}
        hint={hint(() => "score 0.20 – 0.50")}
      />
    </div>
  );
}

function Skeleton() {
  return (
    <span className="inline-block h-7 w-24 rounded-md bg-neutral-800 animate-pulse align-middle" />
  );
}
