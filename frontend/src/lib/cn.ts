/**
 * Tailwind class merger.
 *
 * `clsx` handles conditional class composition; `twMerge` resolves
 * conflicts between Tailwind utility classes that target the same
 * CSS property (e.g. `p-2` and `p-4`) so the last one wins
 * deterministically.
 */
import { clsx } from "clsx";
import type { ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
