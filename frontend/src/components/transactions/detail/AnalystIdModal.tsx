/**
 * One-shot capture modal for the analyst identifier (Phase 3G-4).
 *
 * Rendered when the analyst tries to submit a decision without an
 * identifier set, and also reachable explicitly via the "change"
 * affordance in the decision form. Trusts whatever is typed — there
 * is no auth in this project — but the value is persisted to
 * localStorage so the audit log narrative stays consistent across
 * sessions.
 */
import { useEffect, useId, useRef, useState } from "react";
import { X } from "lucide-react";
import { cn } from "../../../lib/cn";
import { ANALYST_ID_MAX_LENGTH } from "../../../hooks/useAnalystId";

interface Props {
  open: boolean;
  initialValue?: string;
  onSubmit: (value: string) => void;
  onClose: () => void;
}

export function AnalystIdModal({
  open,
  initialValue = "",
  onSubmit,
  onClose,
}: Props) {
  const [value, setValue] = useState(initialValue);
  const [prevOpen, setPrevOpen] = useState(open);
  const inputRef = useRef<HTMLInputElement>(null);
  const inputId = useId();

  // React 19 idiom: reset derived state in response to a prop
  // transition during render rather than inside an effect.
  if (open !== prevOpen) {
    setPrevOpen(open);
    if (open) setValue(initialValue);
  }

  useEffect(() => {
    if (open) {
      // Defer focus a tick so the input is in the DOM tree.
      const t = setTimeout(() => inputRef.current?.focus(), 0);
      return () => clearTimeout(t);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const trimmed = value.trim();
  const isValid = trimmed.length > 0 && trimmed.length <= ANALYST_ID_MAX_LENGTH;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="analyst-id-modal-title"
      onClick={onClose}
    >
      <div
        className={cn(
          "w-full max-w-md rounded-xl border border-neutral-800 bg-neutral-900",
          "shadow-2xl shadow-black/40",
        )}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between px-5 pt-4 pb-2">
          <div>
            <h3
              id="analyst-id-modal-title"
              className="text-sm font-medium text-neutral-100"
            >
              Identify yourself
            </h3>
            <p className="mt-0.5 text-xs text-neutral-500">
              Stamped on the audit log for every decision you record.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-md p-1 text-neutral-500 hover:bg-neutral-800 hover:text-neutral-200"
          >
            <X size={14} aria-hidden />
          </button>
        </div>

        <form
          className="px-5 pb-5 pt-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (isValid) onSubmit(trimmed);
          }}
        >
          <label
            htmlFor={inputId}
            className="block text-[10px] uppercase tracking-wider text-neutral-500"
          >
            Analyst ID
          </label>
          <input
            id={inputId}
            ref={inputRef}
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            maxLength={ANALYST_ID_MAX_LENGTH}
            placeholder="e.g. analyst-1, jdoe, ops-3"
            className={cn(
              "mt-1 w-full rounded-md border border-neutral-700 bg-neutral-950",
              "px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-600",
              "focus:border-neutral-500 focus:outline-none",
            )}
          />
          <div className="mt-1 text-[10px] text-neutral-600 tabular-nums">
            {trimmed.length} / {ANALYST_ID_MAX_LENGTH}
          </div>

          <div className="mt-4 flex items-center justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-md px-3 py-1.5 text-xs text-neutral-400 hover:text-neutral-100"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!isValid}
              className={cn(
                "rounded-md px-3 py-1.5 text-xs font-medium",
                "border border-emerald-500/30 bg-emerald-500/10 text-emerald-200",
                "hover:bg-emerald-500/20",
                "disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-emerald-500/10",
              )}
            >
              Save
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
