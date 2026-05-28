/**
 * Demo-mode runtime flags.
 *
 * The flags are baked in at Vite build time via `VITE_DEMO_MODE` and
 * `VITE_DEMO_SNAPSHOT_DATE`. The Vercel project sets both; local
 * `npm run dev` leaves them unset and the app runs against the live
 * backend exactly as before.
 *
 * Contract locked in `docs/adr/PHASE_4A_DEMO_SCOPE.md`.
 */

export const isDemoMode = (): boolean =>
  import.meta.env.VITE_DEMO_MODE === "true";

export const demoSnapshotDate = (): string =>
  import.meta.env.VITE_DEMO_SNAPSHOT_DATE ?? "unknown";

/**
 * Convenience: collapse a live polling interval to `false` in demo
 * mode so TanStack Query stops re-fetching static JSON every 30s.
 * The ADR explicitly disables polling in demo mode.
 */
export const demoRefetchInterval = (live: number): number | false =>
  isDemoMode() ? false : live;

/**
 * Public links surfaced in the demo banner and sidebar footer.
 */
export const DEMO_REPO_URL = "https://github.com/apoorvrajdev/fraud-radar";
export const DEMO_LOOM_URL = ""; // filled in once Phase 4E records it
