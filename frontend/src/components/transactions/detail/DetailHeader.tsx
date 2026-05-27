/**
 * Header strip for the transaction detail page.
 *
 * Shows the primary identifying facts (id, time, amount, country) and
 * the model's call alongside the effective decision. When the analyst
 * has overridden the verdict, both badges render so the audit trail
 * is visible at a glance.
 */
import { DecisionBadge } from "../DecisionBadge";
import { Card } from "../../ui/Card";
import {
  formatDateTime,
  formatFraudScore,
  formatMoneyPrecise,
} from "../../../lib/format";
import type { TransactionDetail } from "../../../types/api";

interface Props {
  detail: TransactionDetail;
}

export function DetailHeader({ detail }: Props) {
  const overridden =
    detail.fraud_decision !== null &&
    detail.fraud_decision !== detail.effective_decision;

  return (
    <Card className="px-5 py-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">
            Transaction
          </div>
          <div className="mt-0.5 font-mono text-sm text-neutral-200 break-all">
            {detail.id}
          </div>
          <div className="mt-2 text-xs text-neutral-500">
            {formatDateTime(detail.created_at)} · {detail.country} ·{" "}
            {detail.payment_method}
          </div>
        </div>

        <div className="flex items-center gap-6">
          <Field
            label="Amount"
            value={`${formatMoneyPrecise(detail.amount)} ${detail.currency}`}
          />
          <Field
            label="Fraud score"
            value={formatFraudScore(detail.fraud_score)}
            mono
          />
          <div>
            <div className="text-[10px] uppercase tracking-wider text-neutral-500">
              {overridden ? "Model" : "Decision"}
            </div>
            <div className="mt-1.5">
              <DecisionBadge decision={detail.fraud_decision} />
            </div>
          </div>
          {overridden && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-amber-300/80">
                After review
              </div>
              <div className="mt-1.5">
                <DecisionBadge decision={detail.effective_decision} />
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

function Field({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-neutral-500">
        {label}
      </div>
      <div
        className={`mt-0.5 text-sm text-neutral-100 ${
          mono ? "font-mono tabular-nums" : ""
        }`}
      >
        {value}
      </div>
    </div>
  );
}
