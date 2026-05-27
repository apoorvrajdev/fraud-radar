/**
 * Tiny localStorage-backed identity hook for the analyst capture flow
 * (Phase 3G-4).
 *
 * We trust whatever the user types — there is no auth layer in this
 * project — but persisting the value across reloads keeps the audit
 * log narrative readable ("by analyst-1" rather than "by anon" every
 * row). The first decision submission triggers a capture modal if
 * the value is empty.
 */
import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "fraud-radar.analyst-id";
const MAX_LEN = 64;

function read(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function useAnalystId() {
  const [analystId, setAnalystIdState] = useState<string>(read);

  // Cross-tab sync: if another tab changes the value we pick it up.
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) {
        setAnalystIdState(e.newValue ?? "");
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const setAnalystId = useCallback((value: string) => {
    const trimmed = value.trim().slice(0, MAX_LEN);
    try {
      if (trimmed) {
        localStorage.setItem(STORAGE_KEY, trimmed);
      } else {
        localStorage.removeItem(STORAGE_KEY);
      }
    } catch {
      // Persistence is best-effort; in-memory state still works.
    }
    setAnalystIdState(trimmed);
  }, []);

  return { analystId, setAnalystId };
}

export const ANALYST_ID_MAX_LENGTH = MAX_LEN;
