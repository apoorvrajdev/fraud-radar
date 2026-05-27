/**
 * Raw feature/identity block for the transaction detail page.
 *
 * Mostly identity and channel signals — anything that helps an analyst
 * sanity-check the row before acting. Empty values render as "—" so
 * gaps read as data rather than rendering bugs.
 */
import { Card } from "../../ui/Card";
import type { TransactionDetail } from "../../../types/api";

interface Props {
  detail: TransactionDetail;
}

export function FeaturesPanel({ detail }: Props) {
  const rows: Array<[string, string | null]> = [
    ["Customer", detail.customer_id],
    ["Merchant", detail.merchant_id],
    ["Card last 4", detail.card_last4],
    ["Card present", detail.is_card_present ? "Yes" : "No"],
    ["IP address", detail.ip_address],
    ["Device", detail.device_id],
    ["Status", detail.status],
  ];

  return (
    <Card className="px-5 py-4">
      <h2 className="text-sm font-medium text-neutral-200">Identity & channel</h2>
      <dl className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2.5">
        {rows.map(([label, value]) => (
          <div
            key={label}
            className="flex items-baseline justify-between gap-2 border-b border-neutral-800/60 pb-1"
          >
            <dt className="text-[11px] uppercase tracking-wider text-neutral-500">
              {label}
            </dt>
            <dd
              className={`text-xs text-neutral-200 truncate ${
                isMonoField(label) ? "font-mono" : ""
              }`}
              title={value ?? undefined}
            >
              {value ?? "—"}
            </dd>
          </div>
        ))}
      </dl>
    </Card>
  );
}

function isMonoField(label: string): boolean {
  return (
    label === "Customer" ||
    label === "Merchant" ||
    label === "IP address" ||
    label === "Device" ||
    label === "Card last 4"
  );
}
