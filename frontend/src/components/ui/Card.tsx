import type { HTMLAttributes, PropsWithChildren } from "react";
import { cn } from "../../lib/cn";

/**
 * Dark-mode card container used as the base for every tile and panel
 * on the dashboard. Children compose freely inside.
 */
export function Card({
  className,
  children,
  ...rest
}: PropsWithChildren<HTMLAttributes<HTMLDivElement>>) {
  return (
    <div
      className={cn(
        "rounded-xl border border-neutral-800 bg-neutral-900/50",
        "shadow-sm shadow-black/20",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}
