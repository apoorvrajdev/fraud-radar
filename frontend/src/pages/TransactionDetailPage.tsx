/**
 * Stub for the transaction detail page. The full view — including the
 * fraud-score breakdown, SHAP contributors, and analyst feedback
 * controls — ships in Phase 3G. This stub exists so row links from
 * the transactions list don't 404 in the meantime.
 */
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Construction } from "lucide-react";
import { Card } from "../components/ui/Card";

export function TransactionDetailPage() {
  const { id } = useParams<{ id: string }>();

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
        <p className="text-sm text-neutral-500 mt-1 font-mono">{id}</p>
      </div>

      <Card className="px-6 py-8">
        <div className="flex flex-col items-center gap-3 text-center">
          <div className="rounded-full bg-neutral-800 p-3">
            <Construction size={20} className="text-neutral-400" />
          </div>
          <div className="text-sm text-neutral-300">
            Detail view ships in Phase 3G
          </div>
          <div className="text-xs text-neutral-500 max-w-md">
            Will include the per-rule trigger list, top-5 SHAP contributors
            with directional bars, raw feature values, and the analyst
            decision controls for transactions in <code>REVIEW</code>.
          </div>
        </div>
      </Card>
    </div>
  );
}
