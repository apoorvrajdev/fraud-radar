import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  ListChecks,
  ShieldAlert,
} from "lucide-react";
import { cn } from "../../lib/cn";
import { demoSnapshotDate, isDemoMode } from "../../lib/demoMode";
import { useModelInfo } from "../../hooks/useModelInfo";

interface NavItem {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
  disabled?: boolean;
  comingIn?: string;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/transactions", label: "Transactions", icon: ListChecks },
  { to: "/alerts", label: "Alerts", icon: ShieldAlert },
];

/**
 * Left rail. Overview, Transactions, and Alerts are all live
 * (transaction detail is reached by clicking a row in either the
 * Transactions table or the Alerts queue). Disabled entries would
 * render as muted placeholders with their target phase noted, so the
 * nav shape stays visible without being misleading.
 */
export function Sidebar() {
  return (
    <aside className="hidden md:flex w-60 shrink-0 flex-col border-r border-neutral-800 bg-neutral-950">
      <div className="px-5 py-5 border-b border-neutral-800">
        <div className="text-sm font-semibold tracking-tight bg-gradient-to-r from-blue-400 to-violet-400 bg-clip-text text-transparent">
          Fraud Radar
        </div>
        <div className="text-[10px] uppercase tracking-wider text-neutral-500 mt-0.5">
          Dashboard
        </div>
      </div>

      <nav className="flex-1 px-2 py-4 space-y-1">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          if (item.disabled) {
            return (
              <div
                key={item.to}
                className={cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm",
                  "text-neutral-600 cursor-not-allowed select-none",
                )}
                title={`Coming in Phase ${item.comingIn}`}
              >
                <Icon size={16} aria-hidden />
                <span className="flex-1">{item.label}</span>
                <span className="text-[10px] uppercase tracking-wider text-neutral-700">
                  {item.comingIn}
                </span>
              </div>
            );
          }
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                  isActive
                    ? "bg-neutral-900 text-neutral-100"
                    : "text-neutral-400 hover:bg-neutral-900/60 hover:text-neutral-100",
                )
              }
            >
              <Icon size={16} aria-hidden />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      <div className="px-5 py-3 border-t border-neutral-800 space-y-1">
        <ModelBadge />
        <div className="text-[10px] text-neutral-600">
          {isDemoMode()
            ? `Demo · snapshot from ${demoSnapshotDate()}`
            : "Live backend"}
        </div>
      </div>
    </aside>
  );
}

/**
 * Compact serving-model line. Names the dataset and featureset the API
 * is actually running, not the phase of the build — a phase number
 * goes stale on the next commit and tells a reviewer nothing about
 * what produced the numbers on screen. The full provenance, including
 * which tracks are benchmark-only, is on the dashboard panel.
 */
function ModelBadge() {
  const { data } = useModelInfo();
  if (!data) return null;

  const trained = data.serving.trained_at_utc?.slice(0, 10);
  return (
    <div
      className="truncate font-mono text-[10px] text-neutral-500"
      title={`Serving ${data.serving.dataset_name} (${data.serving.dataset_kind}) on featureset ${data.serving.featureset_version}. ${data.serving.metrics_caveat}`}
    >
      model: {data.serving.dataset_name} · {data.serving.featureset_version}
      {trained ? ` · ${trained}` : ""}
    </div>
  );
}
