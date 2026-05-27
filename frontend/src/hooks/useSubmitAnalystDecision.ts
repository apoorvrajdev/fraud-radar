/**
 * Mutation wrapper for the analyst-override endpoint (Phase 3G-4).
 *
 * On success we both write the response into the detail cache
 * directly (cheaper than a refetch, the server already returned the
 * full envelope) AND invalidate the transactions list so any row
 * showing this transaction picks up the new effective decision on its
 * next refetch tick.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  submitAnalystDecision,
  type AnalystDecisionInput,
} from "../lib/transactions";
import type { TransactionDetail } from "../types/api";

export function useSubmitAnalystDecision() {
  const queryClient = useQueryClient();
  return useMutation<TransactionDetail, Error, AnalystDecisionInput>({
    mutationFn: submitAnalystDecision,
    onSuccess: (updated) => {
      queryClient.setQueryData<TransactionDetail>(
        ["transaction-detail", updated.id],
        updated,
      );
      queryClient.invalidateQueries({ queryKey: ["transactions"] });
      // Phase 3H: a submitted verdict moves the row off the alerts
      // queue, so refresh that surface too. TanStack matches the key
      // prefix so every filtered alerts query updates in one go.
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
    },
  });
}
