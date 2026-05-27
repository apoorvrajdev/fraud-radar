import { KpiTiles } from "../components/dashboard/KpiTiles";
import { FraudRateChart } from "../components/dashboard/FraudRateChart";
import { VolumeSparkline } from "../components/dashboard/VolumeSparkline";
import { CountryBreakdownTable } from "../components/dashboard/CountryBreakdownTable";

/**
 * Live dashboard overview.
 *
 * Each tile / chart owns its own query so partial backend failures
 * never blank the whole page (ADR decision 6). Polling cadences live
 * on the hooks: KPIs + breakdown at 30s, timeseries at 60s.
 */
export function DashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
        <p className="text-sm text-neutral-500 mt-1">
          Trailing 24-hour fraud activity. Updates automatically.
        </p>
      </div>

      <KpiTiles />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <FraudRateChart />
        </div>
        <VolumeSparkline />
      </div>

      <CountryBreakdownTable />
    </div>
  );
}
