import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { Card } from "../ui/Card";
import { useStatsTimeseries } from "../../hooks/useStatsTimeseries";

const HOUR_FMT = new Intl.DateTimeFormat("en-US", {
  hour: "2-digit",
  hour12: false,
  timeZone: "UTC",
});

/**
 * 24h hourly fraud-rate line chart. Each point is the per-hour
 * `fraud_count / transaction_count` produced by the service layer;
 * empty hours land at 0 so the x-axis stays continuous.
 */
export function FraudRateChart() {
  const { data, isLoading, isError } = useStatsTimeseries();

  const points =
    data?.points.map((p) => ({
      hour: `${HOUR_FMT.format(new Date(p.timestamp))}:00`,
      // Empty hours have no fraud rate to speak of — a 0% point would
      // be a lie that flattens the line. Emit null so Recharts breaks
      // the line and the reader sees a genuine gap.
      fraud_rate_pct:
        p.transaction_count > 0 ? +(p.fraud_rate * 100).toFixed(2) : null,
      transaction_count: p.transaction_count,
    })) ?? [];

  return (
    <Card className="p-5">
      <div className="flex items-baseline justify-between mb-4">
        <div>
          <h2 className="text-sm font-semibold text-neutral-200">
            Fraud rate · trailing 24h
          </h2>
          <p className="text-xs text-neutral-500 mt-0.5">
            Hourly buckets, UTC
          </p>
        </div>
      </div>

      <div className="h-64">
        {isLoading ? (
          <div className="h-full w-full rounded-md bg-neutral-800/40 animate-pulse" />
        ) : isError || !data ? (
          <div className="h-full flex items-center justify-center text-sm text-neutral-500">
            Timeseries unavailable.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={points}
              margin={{ top: 4, right: 8, bottom: 4, left: -16 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#262626"
                vertical={false}
              />
              <XAxis
                dataKey="hour"
                stroke="#525252"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: "#262626" }}
                interval={2}
              />
              <YAxis
                stroke="#525252"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: "#262626" }}
                tickFormatter={(v: number) => `${v}%`}
                width={48}
              />
              <Tooltip
                contentStyle={{
                  background: "#0a0a0a",
                  border: "1px solid #262626",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelStyle={{ color: "#a3a3a3" }}
                itemStyle={{ color: "#e5e5e5" }}
                formatter={(value, key) =>
                  key === "fraud_rate_pct"
                    ? [`${value}%`, "Fraud rate"]
                    : [String(value), "Transactions"]
                }
              />
              <Line
                type="monotone"
                dataKey="fraud_rate_pct"
                stroke="#a78bfa"
                strokeWidth={2}
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </Card>
  );
}
