/**
 * Top-of-page banner shown only in public-demo mode.
 *
 * The ADR (PHASE_4A_DEMO_SCOPE.md) locks the copy:
 *   "Demo mode — snapshot from {DATE}. [GitHub] · [Loom]"
 *
 * One line, factual, links out. No apologies, no "limited demo"
 * language — a snapshot is a normal thing.
 */
import { useState } from "react";
import { Info, X } from "lucide-react";
import {
  DEMO_LOOM_URL,
  DEMO_REPO_URL,
  demoSnapshotDate,
} from "../../lib/demoMode";

export function DemoBanner() {
  const [dismissed, setDismissed] = useState(false);
  if (dismissed) return null;

  return (
    <div
      role="status"
      className="flex items-center gap-3 border-b border-amber-500/20 bg-amber-500/10 px-6 py-2 text-xs text-amber-200"
    >
      <Info size={14} aria-hidden className="shrink-0" />
      <span className="flex-1">
        <span className="font-medium">Demo mode</span> — snapshot from{" "}
        <span className="tabular-nums">{demoSnapshotDate()}</span>. Polling
        and analyst write actions are disabled here; run locally for the
        full live experience.{" "}
        <a
          href={DEMO_REPO_URL}
          target="_blank"
          rel="noreferrer"
          className="underline decoration-amber-500/40 underline-offset-2 hover:text-amber-100"
        >
          Source on GitHub
        </a>
        {DEMO_LOOM_URL && (
          <>
            {" · "}
            <a
              href={DEMO_LOOM_URL}
              target="_blank"
              rel="noreferrer"
              className="underline decoration-amber-500/40 underline-offset-2 hover:text-amber-100"
            >
              Watch the 2-minute walkthrough
            </a>
          </>
        )}
      </span>
      <button
        type="button"
        onClick={() => setDismissed(true)}
        className="shrink-0 rounded p-0.5 text-amber-300/70 hover:bg-amber-500/10 hover:text-amber-100"
        aria-label="Dismiss demo banner"
      >
        <X size={14} />
      </button>
    </div>
  );
}
