import { Card } from "../components/ui/Card";

/**
 * Dashboard overview — Phase 3E-2 placeholder.
 *
 * Slice 3E-3 will replace the empty state with KPI tiles, the 24h
 * fraud-rate chart, and the country breakdown table. For now this
 * page exists so the router has a route to mount and `npm run build`
 * has a real page entry to type-check.
 */
export function DashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
        <p className="text-sm text-neutral-500 mt-1">
          Trailing 24-hour fraud activity, updated on a polling interval.
        </p>
      </div>

      <Card className="p-8 text-center text-neutral-500 text-sm">
        Dashboard tiles ship in Phase 3E-3.
      </Card>
    </div>
  );
}
