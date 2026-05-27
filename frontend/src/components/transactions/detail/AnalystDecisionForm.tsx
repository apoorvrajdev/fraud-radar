/**
 * Analyst decision form (Phase 3G-4).
 *
 * Rendered only for rows that are still in REVIEW (the backend
 * returns 409 on anything else, so the form should never offer the
 * action where it would fail). When the row already carries an
 * `analyst_label`, this is a *revision* — the previous label is
 * pre-selected and the submit button reads "Update decision".
 *
 * No optimistic update: the response is the authoritative envelope
 * and the mutation hook writes it straight into the detail cache, so
 * the UI flips state in one round-trip without an inconsistent
 * intermediate frame.
 */
import { useId, useMemo, useState } from "react";
import axios from "axios";
import { AlertCircle, CheckCircle2, ShieldX, UserCog } from "lucide-react";
import { Card } from "../../ui/Card";
import { cn } from "../../../lib/cn";
import { useAnalystId } from "../../../hooks/useAnalystId";
import { useSubmitAnalystDecision } from "../../../hooks/useSubmitAnalystDecision";
import { AnalystIdModal } from "./AnalystIdModal";
import type { AnalystLabel, TransactionDetail } from "../../../types/api";

const NOTES_MAX = 2000;

interface Props {
  detail: TransactionDetail;
}

export function AnalystDecisionForm({ detail }: Props) {
  // Only REVIEW rows are valid override targets server-side.
  if (detail.fraud_decision !== "REVIEW") return null;

  return <AnalystDecisionFormInner detail={detail} />;
}

function AnalystDecisionFormInner({ detail }: Props) {
  const { analystId, setAnalystId } = useAnalystId();
  const mutation = useSubmitAnalystDecision();
  const headingId = useId();
  const notesId = useId();

  const [label, setLabel] = useState<AnalystLabel | null>(detail.analyst_label);
  const [notes, setNotes] = useState<string>(detail.analyst_notes ?? "");
  const [modalOpen, setModalOpen] = useState(false);

  // If the upstream detail changes (e.g. a refetch picked up a
  // revision from another tab), reseed the form so it doesn't show
  // stale local edits. React 19 idiom: adjust state during render
  // rather than from inside an effect.
  const upstreamSig = `${detail.id}|${detail.analyst_label ?? ""}|${
    detail.analyst_notes ?? ""
  }`;
  const [prevSig, setPrevSig] = useState(upstreamSig);
  if (prevSig !== upstreamSig) {
    setPrevSig(upstreamSig);
    setLabel(detail.analyst_label);
    setNotes(detail.analyst_notes ?? "");
  }

  const isRevision = detail.analyst_label !== null;
  const noChange =
    label === detail.analyst_label &&
    (notes.trim() || null) === (detail.analyst_notes ?? null);
  const canSubmit =
    label !== null && notes.length <= NOTES_MAX && !mutation.isPending;

  const errorMessage = useMemo(
    () => extractErrorMessage(mutation.error),
    [mutation.error],
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit || label === null) return;
    if (!analystId) {
      setModalOpen(true);
      return;
    }
    mutation.mutate({
      id: detail.id,
      analystId,
      label,
      notes: notes.trim(),
    });
  };

  return (
    <>
      <Card className="px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2
              id={headingId}
              className="text-sm font-medium text-neutral-200"
            >
              {isRevision ? "Revise decision" : "Record decision"}
            </h2>
            <p className="mt-0.5 text-xs text-neutral-500">
              Your call overrides the model's verdict for this transaction.
              The model's original score is preserved on record.
            </p>
          </div>
          <AnalystChip
            analystId={analystId}
            onChange={() => setModalOpen(true)}
          />
        </div>

        <form
          className="mt-4 space-y-4"
          onSubmit={handleSubmit}
          aria-labelledby={headingId}
        >
          <fieldset>
            <legend className="text-[10px] uppercase tracking-wider text-neutral-500">
              Verdict
            </legend>
            <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2">
              <LabelChoice
                value="CONFIRMED_FRAUD"
                current={label}
                onSelect={setLabel}
                tone="fraud"
                title="Confirm fraud"
                subtitle="Final decision becomes DECLINE."
                icon={<ShieldX size={14} aria-hidden />}
              />
              <LabelChoice
                value="CONFIRMED_LEGIT"
                current={label}
                onSelect={setLabel}
                tone="legit"
                title="Confirm legit"
                subtitle="Final decision becomes APPROVE."
                icon={<CheckCircle2 size={14} aria-hidden />}
              />
            </div>
          </fieldset>

          <div>
            <label
              htmlFor={notesId}
              className="text-[10px] uppercase tracking-wider text-neutral-500"
            >
              Notes (optional)
            </label>
            <textarea
              id={notesId}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={3}
              maxLength={NOTES_MAX}
              placeholder="Add any context — chargeback ID, customer call ref, etc."
              className={cn(
                "mt-1 w-full rounded-md border border-neutral-700 bg-neutral-950",
                "px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-600",
                "focus:border-neutral-500 focus:outline-none",
                "resize-y",
              )}
            />
            <div className="mt-1 text-[10px] text-neutral-600 tabular-nums">
              {notes.length} / {NOTES_MAX}
            </div>
          </div>

          {errorMessage && (
            <div
              role="alert"
              className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-200"
            >
              <AlertCircle size={14} aria-hidden className="mt-0.5 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}

          <div className="flex items-center justify-end gap-3">
            {isRevision && noChange && !mutation.isPending && (
              <span className="text-[11px] text-neutral-500">
                No changes to submit
              </span>
            )}
            <button
              type="submit"
              disabled={!canSubmit || (isRevision && noChange)}
              className={cn(
                "rounded-md px-4 py-1.5 text-xs font-medium",
                "border border-emerald-500/30 bg-emerald-500/10 text-emerald-200",
                "hover:bg-emerald-500/20",
                "disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-emerald-500/10",
              )}
            >
              {mutation.isPending
                ? "Submitting…"
                : isRevision
                  ? "Update decision"
                  : "Record decision"}
            </button>
          </div>
        </form>
      </Card>

      <AnalystIdModal
        open={modalOpen}
        initialValue={analystId}
        onClose={() => setModalOpen(false)}
        onSubmit={(value) => {
          setAnalystId(value);
          setModalOpen(false);
          // If the user opened the modal mid-submit, auto-fire once
          // the id is captured so they don't have to click twice.
          if (label !== null) {
            mutation.mutate({
              id: detail.id,
              analystId: value,
              label,
              notes: notes.trim(),
            });
          }
        }}
      />
    </>
  );
}

function AnalystChip({
  analystId,
  onChange,
}: {
  analystId: string;
  onChange: () => void;
}) {
  if (!analystId) {
    return (
      <button
        type="button"
        onClick={onChange}
        className={cn(
          "inline-flex items-center gap-1 rounded-md border border-dashed",
          "border-neutral-700 px-2 py-1 text-[11px] text-neutral-400",
          "hover:border-neutral-500 hover:text-neutral-200",
        )}
      >
        <UserCog size={12} aria-hidden />
        Set your analyst ID
      </button>
    );
  }
  return (
    <div className="text-right">
      <div className="text-[10px] uppercase tracking-wider text-neutral-500">
        Acting as
      </div>
      <div className="mt-0.5 flex items-center justify-end gap-2">
        <span className="font-mono text-xs text-neutral-200">{analystId}</span>
        <button
          type="button"
          onClick={onChange}
          className="text-[10px] uppercase tracking-wider text-neutral-500 hover:text-neutral-200"
        >
          Change
        </button>
      </div>
    </div>
  );
}

function LabelChoice({
  value,
  current,
  onSelect,
  tone,
  title,
  subtitle,
  icon,
}: {
  value: AnalystLabel;
  current: AnalystLabel | null;
  onSelect: (v: AnalystLabel) => void;
  tone: "fraud" | "legit";
  title: string;
  subtitle: string;
  icon: React.ReactNode;
}) {
  const selected = current === value;
  const toneStyles = selected
    ? tone === "fraud"
      ? "border-rose-500/50 bg-rose-500/15 text-rose-100"
      : "border-emerald-500/50 bg-emerald-500/15 text-emerald-100"
    : "border-neutral-700 bg-neutral-900/40 text-neutral-300 hover:border-neutral-500";

  return (
    <button
      type="button"
      onClick={() => onSelect(value)}
      aria-pressed={selected}
      className={cn(
        "flex items-start gap-2 rounded-md border px-3 py-2 text-left transition-colors",
        toneStyles,
      )}
    >
      <span className="mt-0.5 shrink-0">{icon}</span>
      <span>
        <span className="block text-sm font-medium">{title}</span>
        <span
          className={cn(
            "block text-[11px]",
            selected ? "text-current/80" : "text-neutral-500",
          )}
        >
          {subtitle}
        </span>
      </span>
    </button>
  );
}

function extractErrorMessage(err: unknown): string | null {
  if (!err) return null;
  if (axios.isAxiosError(err)) {
    const status = err.response?.status;
    const detail = (err.response?.data as { detail?: unknown } | undefined)
      ?.detail;
    if (status === 409) {
      return typeof detail === "string"
        ? detail
        : "This transaction can no longer be overridden.";
    }
    if (status === 422) {
      return "Invalid input — check the label and notes length.";
    }
    if (status === 404) {
      return "Transaction not found.";
    }
    if (typeof detail === "string") return detail;
    return err.message;
  }
  if (err instanceof Error) return err.message;
  return "Failed to submit decision.";
}
