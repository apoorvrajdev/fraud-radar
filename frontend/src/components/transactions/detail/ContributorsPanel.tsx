/**
 * SHAP contributors panel.
 *
 * Hand-rolled CSS bars rather than Recharts — there are only ~5 rows,
 * the visual is purely directional, and CSS gives us pixel control
 * over centred zero-origin layout without a chart library round-trip.
 *
 * Bar direction:
 *   - shap_value > 0  → red bar growing right (pushes toward fraud)
 *   - shap_value <= 0 → green bar growing left (pulls toward legit)
 *
 * Bars are scaled to the largest absolute SHAP value in the set so a
 * weak contributor never dominates a strong one visually.
 */
import { Card } from "../../ui/Card";
import { Info } from "lucide-react";
import type { ContributorEntry } from "../../../types/api";

interface Props {
  contributors: ContributorEntry[];
  threshold: number | null;
}

export function ContributorsPanel({ contributors, threshold }: Props) {
  if (contributors.length === 0) {
    return (
      <Card className="px-5 py-4">
        <Header threshold={threshold} />
        <div className="mt-3 flex items-center gap-2 text-sm text-neutral-400">
          <Info size={14} aria-hidden />
          No SHAP attribution was persisted for this transaction.
        </div>
      </Card>
    );
  }

  const maxAbs = Math.max(
    ...contributors.map((c) => Math.abs(c.shap_value)),
    1e-9,
  );

  return (
    <Card className="px-5 py-4">
      <Header threshold={threshold} />
      <ul className="mt-4 space-y-2">
        {contributors.map((c) => (
          <ContributorRow key={c.feature} entry={c} maxAbs={maxAbs} />
        ))}
      </ul>
      <div className="mt-4 flex items-center gap-4 text-[10px] uppercase tracking-wider text-neutral-500">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-sm bg-rose-500/70" />
          Pushes toward fraud
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-sm bg-emerald-500/70" />
          Pulls toward legit
        </span>
      </div>
    </Card>
  );
}

function Header({ threshold }: { threshold: number | null }) {
  return (
    <div className="flex items-center justify-between">
      <h2 className="text-sm font-medium text-neutral-200">
        Top contributors
      </h2>
      {threshold !== null && (
        <span className="text-[10px] uppercase tracking-wider text-neutral-500 tabular-nums">
          Threshold {threshold.toFixed(2)}
        </span>
      )}
    </div>
  );
}

function ContributorRow({
  entry,
  maxAbs,
}: {
  entry: ContributorEntry;
  maxAbs: number;
}) {
  const widthPct = Math.min(100, (Math.abs(entry.shap_value) / maxAbs) * 100);
  const isFraud = entry.direction === "fraud";

  return (
    <li className="grid grid-cols-[1fr_auto_1fr_auto] items-center gap-2">
      {/* Left half — legit bars grow leftward from centre */}
      <div className="flex justify-end h-2">
        {!isFraud && (
          <div
            className="h-2 rounded-sm bg-emerald-500/70"
            style={{ width: `${widthPct}%` }}
            aria-hidden
          />
        )}
      </div>

      {/* Feature name + value, centred on the zero axis */}
      <div className="px-2 min-w-[140px] text-right">
        <div className="font-mono text-xs text-neutral-200 truncate">
          {entry.feature}
        </div>
        <div className="text-[10px] text-neutral-500 tabular-nums">
          = {formatFeatureValue(entry.feature_value)}
        </div>
      </div>

      {/* Right half — fraud bars grow rightward from centre */}
      <div className="h-2">
        {isFraud && (
          <div
            className="h-2 rounded-sm bg-rose-500/70"
            style={{ width: `${widthPct}%` }}
            aria-hidden
          />
        )}
      </div>

      {/* Signed SHAP value, right-aligned */}
      <div
        className={`pl-2 text-xs font-mono tabular-nums ${
          isFraud ? "text-rose-300" : "text-emerald-300"
        }`}
      >
        {entry.shap_value >= 0 ? "+" : ""}
        {entry.shap_value.toFixed(3)}
      </div>
    </li>
  );
}

function formatFeatureValue(v: number): string {
  if (Number.isInteger(v)) return v.toString();
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toFixed(4);
}
