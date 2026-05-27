import { Stat } from "../ui/Stat";
import { useStatsOverview } from "../../hooks/useStatsOverview";
import {
  formatCompactInt,
  formatInt,
  formatMoney,
  formatPercent,
} from "../../lib/format";

/**
 * Five-tile KPI row driven by `/stats/overview`. Each tile renders
 * its own loading skeleton; a query error swaps the value for an
 * em-dash plus a muted "unavailable" hint so a single broken endpoint
 * never blanks the whole row (ADR decision 6).
 */
export function KpiTiles() {
  const { data, isLoading, isError } = useStatsOverview();

  const value = (render: (d: NonNullable<typeof data>) => string) => {
    if (isLoading) return <Skeleton width="w-24" />;
    if (isError || !data) return <span className="text-neutral-500">—</span>;
    return render(data);
  };

  const hint = (render: (d: NonNullable<typeof data>) => string) => {
    if (isLoading) return undefined;
    if (isError || !data) return "data unavailable";
    return render(data);
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
      <Stat
        label="Transactions · 24h"
        value={value((d) => formatCompactInt(d.total_transactions_24h))}
        hint={hint((d) => `${formatInt(d.total_transactions_24h)} exact`)}
      />
      <Stat
        label="Approved rate"
        value={value((d) => formatPercent(d.approved_rate))}
        hint={hint((d) =>
          `${formatInt(d.approved_count_24h)} approved`,
        )}
      />
      <Stat
        label="Declined · 24h"
        value={value((d) => formatInt(d.declined_count_24h))}
        hint={hint((d) =>
          d.avg_fraud_score === null
            ? "no scored rows in window"
            : `avg score ${d.avg_fraud_score.toFixed(3)}`,
        )}
      />
      <Stat
        label="Pending review"
        value={value((d) => formatInt(d.pending_review_count))}
        hint="analyst queue depth"
      />
      <Stat
        label="Fraud caught · 24h"
        value={value((d) => formatMoney(d.fraud_caught_amount))}
        hint="declined + review amounts"
      />
    </div>
  );
}

function Skeleton({ width }: { width: string }) {
  return (
    <span
      className={`inline-block h-7 ${width} rounded-md bg-neutral-800 animate-pulse align-middle`}
    />
  );
}
