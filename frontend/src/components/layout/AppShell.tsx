import type { PropsWithChildren } from "react";
import { Sidebar } from "./Sidebar";
import { DemoBanner } from "./DemoBanner";
import { isDemoMode } from "../../lib/demoMode";

/**
 * Top-level chrome — sidebar + topbar + scrollable main content
 * region. Pages render inside `<main>`. In demo mode a banner is
 * rendered above the topbar (see PHASE_4A_DEMO_SCOPE.md).
 */
export function AppShell({ children }: PropsWithChildren) {
  const demo = isDemoMode();
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        {demo && <DemoBanner />}
        <header className="h-14 shrink-0 border-b border-neutral-800 bg-neutral-950/80 backdrop-blur flex items-center justify-between px-6">
          <div className="flex items-center gap-2 text-sm text-neutral-400">
            <span
              className={
                demo
                  ? "w-2 h-2 rounded-full bg-amber-500"
                  : "w-2 h-2 rounded-full bg-emerald-500 animate-pulse"
              }
            />
            <span>{demo ? "Snapshot" : "Live"}</span>
          </div>
          <div className="text-xs text-neutral-500 tabular-nums">
            apoorvrajdev · Fraud Radar
          </div>
        </header>
        <main className="flex-1 overflow-y-auto px-6 py-6">{children}</main>
      </div>
    </div>
  );
}
