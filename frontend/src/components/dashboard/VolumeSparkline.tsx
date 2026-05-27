import { ResponsiveContainer, BarChart, Bar, Tooltip } from "recharts";
import { Card } from "../ui/Card";
import { useStatsTimeseries } from "../../hooks/useStatsTimeseries";
import { formatInt } from "../../lib/format";

const HOUR_FMT = new Intl.DateTimeFormat("en-US", {
  hour: "2-digit",
  hour12: false,
  timeZone: "UTC",
});

/**
 * Compact 24h transaction-volume sparkline, paired visually with
 * the fraud-rate line chart. Shares the same `/stats/timeseries`
 * query so only one network round-trip per polling tick.
 */
export function VolumeSparkline() {
  const { data, isLoading, isError } = useStatsTimeseries();

  const points =
    data?.points.map((p) => ({
      hour: `${HOUR_FMT.format(new Date(p.timestamp))}:00`,
      transaction_count: p.transaction_count,
    })) ?? [];

  const total = points.reduce((acc, p) => acc + p.transaction_count, 0);

  return (
    <Card className="p-5">
      <div className="flex items-baseline justify-between mb-2">
        <div>
          <h2 className="text-sm font-semibold text-neutral-200">
            Volume · trailing 24h
          </h2>
          <p className="text-xs text-neutral-500 mt-0.5">
            Transactions per hour
          </p>
        </div>
        <div className="text-right">
          <div className="text-lg font-semibold tabular-nums text-neutral-100">
            {isLoading || isError || !data ? "—" : formatInt(total)}
          </div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">
            sum
          </div>
        </div>
      </div>

      <div className="h-24">
        {isLoading ? (
          <div className="h-full w-full rounded-md bg-neutral-800/40 animate-pulse" />
        ) : isError || !data ? (
          <div className="h-full flex items-center justify-center text-xs text-neutral-500">
            Volume unavailable.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={points} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
              <Tooltip
                cursor={{ fill: "#262626" }}
                contentStyle={{
                  background: "#0a0a0a",
                  border: "1px solid #262626",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelStyle={{ color: "#a3a3a3" }}
                itemStyle={{ color: "#e5e5e5" }}
                formatter={(value) => [formatInt(Number(value)), "Transactions"]}
              />
              <Bar
                dataKey="transaction_count"
                fill="#3b82f6"
                radius={[2, 2, 0, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </Card>
  );
}
