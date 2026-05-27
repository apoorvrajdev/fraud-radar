/**
 * Audit-log panel.
 *
 * Renders the trailing 20 audit entries newest-first. Action codes
 * map to friendly labels so the log reads as a narrative rather than
 * a dump of constants. The raw payload is exposed in a collapsible
 * `<details>` for operators who need it without crowding the page.
 */
import { Card } from "../../ui/Card";
import { Clock } from "lucide-react";
import { formatDateTime } from "../../../lib/format";
import type { AuditEntry } from "../../../types/api";

interface Props {
  entries: AuditEntry[];
}

const ACTION_LABELS: Record<string, string> = {
  ANALYST_DECISION: "Analyst decision recorded",
  ANALYST_DECISION_REVISED: "Analyst decision revised",
  "scored.approve": "Scored — approved",
  "scored.decline": "Scored — declined",
  "scored.review": "Scored — sent to review",
  "scored.hard_block": "Scored — hard-blocked by rule",
};

export function AuditPanel({ entries }: Props) {
  if (entries.length === 0) {
    return (
      <Card className="px-5 py-4">
        <Header />
        <div className="mt-3 text-sm text-neutral-400">
          No audit entries for this transaction yet.
        </div>
      </Card>
    );
  }

  return (
    <Card className="px-5 py-4">
      <Header count={entries.length} />
      <ol className="mt-3 relative border-l border-neutral-800 pl-4 space-y-3">
        {entries.map((entry) => (
          <AuditRow key={entry.id} entry={entry} />
        ))}
      </ol>
    </Card>
  );
}

function Header({ count }: { count?: number }) {
  return (
    <div className="flex items-center justify-between">
      <h2 className="text-sm font-medium text-neutral-200">Audit log</h2>
      {count !== undefined && (
        <span className="text-[10px] uppercase tracking-wider text-neutral-500 tabular-nums">
          {count} entr{count === 1 ? "y" : "ies"}
        </span>
      )}
    </div>
  );
}

function AuditRow({ entry }: { entry: AuditEntry }) {
  const label = ACTION_LABELS[entry.action] ?? entry.action;
  const hasPayload =
    entry.payload !== null && Object.keys(entry.payload).length > 0;

  return (
    <li className="relative">
      <span className="absolute -left-[21px] top-1.5 inline-block h-2 w-2 rounded-full bg-neutral-600" />
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-sm text-neutral-200">{label}</div>
        <div className="flex items-center gap-1 text-[11px] text-neutral-500 tabular-nums">
          <Clock size={11} aria-hidden />
          {formatDateTime(entry.created_at)}
        </div>
      </div>
      <div className="mt-0.5 text-[11px] text-neutral-500">
        by <span className="font-mono text-neutral-400">{entry.actor}</span>
      </div>
      {hasPayload && (
        <details className="mt-1 group">
          <summary className="cursor-pointer text-[11px] text-neutral-500 hover:text-neutral-300 select-none">
            Payload
          </summary>
          <pre className="mt-1 overflow-x-auto rounded-md border border-neutral-800 bg-neutral-950 px-2 py-1.5 text-[11px] text-neutral-300 font-mono">
            {JSON.stringify(entry.payload, null, 2)}
          </pre>
        </details>
      )}
    </li>
  );
}
