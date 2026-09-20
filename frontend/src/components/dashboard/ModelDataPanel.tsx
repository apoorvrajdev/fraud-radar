/**
 * Model and dataset identity panel (Phase 5G).
 *
 * The dashboard's headline numbers come from one specific model
 * trained on one specific dataset, and this repository keeps three
 * data layers deliberately apart. A KPI row with no provenance quietly
 * invites the reader to assume the most flattering interpretation —
 * that these are real-world fraud-detection numbers. They are not, and
 * this panel is where that is said plainly.
 *
 * Restraint is the design goal. It is a provenance note, not a
 * marketing tile: no large metrics, no colour coding of scores, no
 * claim the backend did not send.
 */
import { CircleCheck, Minus } from "lucide-react";
import { Card } from "../ui/Card";
import { useModelInfo } from "../../hooks/useModelInfo";
import { cn } from "../../lib/cn";
import type { BenchmarkTrack } from "../../types/api";

const KIND_LABEL: Record<BenchmarkTrack["kind"], string> = {
  synthetic: "Synthetic · in-house",
  "synthetic-external": "Synthetic · external generator",
  "real-anonymised": "Real · anonymised",
};

const REPO_BLOB =
  "https://github.com/apoorvrajdev/fraud-radar/blob/main/";

export function ModelDataPanel() {
  const { data, isLoading, isError } = useModelInfo();

  return (
    <Card className="p-5">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-medium text-neutral-200">Model &amp; data</h2>
        {data && (
          <span className="font-mono text-[10px] text-neutral-500">
            {data.serving.dataset_name} · featureset{" "}
            {data.serving.featureset_version}
          </span>
        )}
      </div>

      {isLoading && (
        <div className="mt-4 space-y-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <div
              key={i}
              className="h-4 w-full rounded bg-neutral-800/70 animate-pulse"
            />
          ))}
        </div>
      )}

      {isError && (
        <p className="mt-3 text-xs text-neutral-500">
          Model information is unavailable.
        </p>
      )}

      {data && (
        <>
          <p className="mt-2 text-xs leading-relaxed text-neutral-400">
            Scoring runs a {data.serving.feature_count}-feature XGBoost model
            {data.serving.trained_at_utc
              ? ` trained on ${data.serving.trained_at_utc.slice(0, 10)}`
              : ""}
            {data.serving.threshold !== null
              ? `, at operating threshold ${data.serving.threshold.toFixed(4)}`
              : ""}
            . {data.serving.metrics_caveat}
          </p>

          <ul className="mt-4 space-y-2">
            {data.benchmarks.map((track) => (
              <TrackRow key={track.name} track={track} />
            ))}
          </ul>

          <p className="mt-4 border-t border-neutral-800 pt-3 text-[11px] leading-relaxed text-neutral-500">
            Benchmark results are reported in their own cards and are never
            merged or averaged with each other. Amounts in the tiles above are
            summed in {data.reporting_currency}.
          </p>
        </>
      )}
    </Card>
  );
}

function TrackRow({ track }: { track: BenchmarkTrack }) {
  return (
    <li className="flex items-start gap-3">
      <span
        className={cn(
          "mt-0.5 shrink-0",
          track.served ? "text-emerald-400" : "text-neutral-600",
        )}
        title={track.served ? "Served by this API" : "Benchmark only — not served"}
      >
        {track.served ? (
          <CircleCheck size={14} aria-hidden />
        ) : (
          <Minus size={14} aria-hidden />
        )}
      </span>
      <div className="min-w-0">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="font-mono text-xs text-neutral-200">
            {track.name}
          </span>
          <span className="text-[10px] uppercase tracking-wider text-neutral-500">
            {KIND_LABEL[track.kind]}
          </span>
          <span
            className={cn(
              "rounded px-1.5 py-px text-[10px] font-medium",
              track.served
                ? "bg-emerald-500/10 text-emerald-300"
                : "bg-neutral-800 text-neutral-400",
            )}
          >
            {track.served ? "in production" : "benchmark only"}
          </span>
        </div>
        <p className="mt-0.5 text-[11px] leading-relaxed text-neutral-500">
          {track.description}
          {track.card_path && (
            <>
              {" "}
              <a
                href={`${REPO_BLOB}${track.card_path}`}
                target="_blank"
                rel="noreferrer"
                className="text-neutral-400 underline decoration-neutral-700 underline-offset-2 hover:text-neutral-200"
              >
                Results
              </a>
            </>
          )}
        </p>
      </div>
    </li>
  );
}
