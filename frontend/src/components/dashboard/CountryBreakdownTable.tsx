import { Card } from "../ui/Card";
import { useStatsBreakdown } from "../../hooks/useStatsBreakdown";
import { formatInt, formatMoney, formatPercent } from "../../lib/format";

/**
 * Severity bands for the per-country decline rate. The thresholds are
 * intentionally generous — at a healthy 1.5% baseline fraud rate, a
 * country sitting above 10% declines is a real operational outlier
 * worth surfacing in red without crying wolf at every small market.
 */
function declineRateClass(rate: number): string {
  if (rate >= 0.1) return "text-rose-400";
  if (rate >= 0.02) return "text-amber-400";
  return "text-emerald-400";
}

/**
 * Top-10 countries by transaction count over the trailing 24h.
 * Rendered as a plain table — sparkline columns belong on the
 * transactions page, not the overview.
 */
export function CountryBreakdownTable() {
  const { data, isLoading, isError } = useStatsBreakdown();

  return (
    <Card className="p-5">
      <div className="mb-4">
        <h2 className="text-sm font-semibold text-neutral-200">
          Top countries · trailing 24h
        </h2>
        <p className="text-xs text-neutral-500 mt-0.5">
          Ranked by transaction count
        </p>
      </div>

      {isLoading ? (
        <RowSkeleton />
      ) : isError || !data ? (
        <div className="py-8 text-center text-sm text-neutral-500">
          Country breakdown unavailable.
        </div>
      ) : data.items.length === 0 ? (
        <div className="py-8 text-center text-sm text-neutral-500">
          No transactions in the last 24 hours.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[10px] uppercase tracking-wider text-neutral-500">
                <th className="font-medium pb-2 pr-3">Country</th>
                <th className="font-medium pb-2 pr-3 text-right">Transactions</th>
                <th className="font-medium pb-2 pr-3 text-right">Declined</th>
                <th className="font-medium pb-2 pr-3 text-right">Decline rate</th>
                <th className="font-medium pb-2 text-right">Total amount</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800/80">
              {data.items.map((row) => {
                const declineRate =
                  row.transaction_count > 0
                    ? row.declined_count / row.transaction_count
                    : 0;
                return (
                  <tr key={row.category} className="text-neutral-200">
                    <td className="py-2 pr-3 font-medium tabular-nums">
                      {row.category}
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums">
                      {formatInt(row.transaction_count)}
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums text-neutral-400">
                      {formatInt(row.declined_count)}
                    </td>
                    <td
                      className={`py-2 pr-3 text-right tabular-nums ${declineRateClass(
                        declineRate,
                      )}`}
                    >
                      {formatPercent(declineRate)}
                    </td>
                    <td className="py-2 text-right tabular-nums">
                      {formatMoney(row.total_amount)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function RowSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 5 }).map((_, i) => (
        <div
          key={i}
          className="h-6 w-full rounded-md bg-neutral-800/40 animate-pulse"
        />
      ))}
    </div>
  );
}
