/**
 * Transaction detail page (Phase 3G).
 *
 * Read-only composite view of one scored transaction: header,
 * triggered rules, top SHAP contributors, identity panel, and audit
 * trail. Analyst decision controls land in Slice 3G-4.
 */
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, AlertCircle } from "lucide-react";
import { Card } from "../components/ui/Card";
import { DetailHeader } from "../components/transactions/detail/DetailHeader";
import { RulesPanel } from "../components/transactions/detail/RulesPanel";
import { ContributorsPanel } from "../components/transactions/detail/ContributorsPanel";
import { FeaturesPanel } from "../components/transactions/detail/FeaturesPanel";
import { AuditPanel } from "../components/transactions/detail/AuditPanel";
import { useTransactionDetail } from "../hooks/useTransactionDetail";

export function TransactionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const query = useTransactionDetail(id);

  return (
    <div className="space-y-5">
      <Link
        to="/transactions"
        className="inline-flex items-center gap-1.5 text-sm text-neutral-400 hover:text-neutral-100"
      >
        <ArrowLeft size={14} aria-hidden />
        Back to transactions
      </Link>

      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Transaction detail
        </h1>
        {!query.data && (
          <p className="text-sm text-neutral-500 mt-1 font-mono">{id}</p>
        )}
      </div>

      {query.isPending ? (
        <DetailSkeleton />
      ) : query.isError ? (
        <ErrorState message={query.error?.message} />
      ) : query.data ? (
        <>
          <DetailHeader detail={query.data} />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <div className="lg:col-span-2 space-y-5">
              <ContributorsPanel
                contributors={query.data.top_contributors}
                threshold={query.data.threshold}
              />
              <RulesPanel rules={query.data.rules_triggered} />
            </div>
            <div className="space-y-5">
              <FeaturesPanel detail={query.data} />
            </div>
          </div>

          <AuditPanel entries={query.data.audit} />
        </>
      ) : null}
    </div>
  );
}

function DetailSkeleton() {
  return (
    <div className="space-y-5">
      <Card className="px-5 py-4">
        <div className="h-3 w-24 rounded bg-neutral-800 animate-pulse" />
        <div className="mt-2 h-4 w-72 rounded bg-neutral-800 animate-pulse" />
      </Card>
      <Card className="px-5 py-4">
        <div className="h-3 w-32 rounded bg-neutral-800 animate-pulse" />
        <div className="mt-4 space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <div
              key={i}
              className="h-2 w-full rounded bg-neutral-800 animate-pulse"
            />
          ))}
        </div>
      </Card>
    </div>
  );
}

function ErrorState({ message }: { message?: string }) {
  const notFound = message?.toLowerCase().includes("404") ?? false;
  return (
    <Card className="px-6 py-8">
      <div className="flex flex-col items-center gap-2 text-center">
        <div className="rounded-full bg-rose-500/10 border border-rose-500/30 p-3">
          <AlertCircle size={20} className="text-rose-300" aria-hidden />
        </div>
        <div className="text-sm text-neutral-200">
          {notFound ? "Transaction not found" : "Failed to load transaction"}
        </div>
        {!notFound && message && (
          <div className="text-xs text-neutral-500 max-w-md">{message}</div>
        )}
      </div>
    </Card>
  );
}
