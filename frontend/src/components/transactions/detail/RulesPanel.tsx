/**
 * Rules-triggered panel.
 *
 * Surfaces the deterministic-rule names that fired at scoring time.
 * Empty list is rendered as a positive ("no rules fired") rather than
 * a blank pane so analysts know they're not looking at missing data.
 */
import { Card } from "../../ui/Card";
import { AlertTriangle, ShieldCheck } from "lucide-react";

interface Props {
  rules: string[];
}

export function RulesPanel({ rules }: Props) {
  if (rules.length === 0) {
    return (
      <Card className="px-5 py-4">
        <PanelHeader title="Rules triggered" />
        <div className="mt-3 flex items-center gap-2 text-sm text-neutral-400">
          <ShieldCheck size={14} className="text-emerald-400" aria-hidden />
          No rules fired at scoring time.
        </div>
      </Card>
    );
  }

  return (
    <Card className="px-5 py-4">
      <PanelHeader title="Rules triggered" count={rules.length} />
      <ul className="mt-3 flex flex-wrap gap-2">
        {rules.map((rule) => (
          <li
            key={rule}
            className="inline-flex items-center gap-1.5 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-xs text-amber-200"
          >
            <AlertTriangle size={12} aria-hidden />
            <span className="font-mono">{rule}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function PanelHeader({ title, count }: { title: string; count?: number }) {
  return (
    <div className="flex items-center justify-between">
      <h2 className="text-sm font-medium text-neutral-200">{title}</h2>
      {count !== undefined && (
        <span className="text-[10px] uppercase tracking-wider text-neutral-500 tabular-nums">
          {count}
        </span>
      )}
    </div>
  );
}
