/**
 * Centralised axios instance for the Fraud Radar backend.
 *
 * The base URL comes from `VITE_API_URL` (see `.env.example`). All
 * dashboard hooks go through this client so retries, timeouts, and
 * future auth headers have one place to live.
 *
 * In demo mode (`VITE_DEMO_MODE === "true"`) the adapter is swapped
 * for `demoAdapter`, which resolves every request from the bundled
 * snapshot under `public/demo-data/`. No hook needs to know.
 */
import axios from "axios";
import { demoAdapter } from "./demoApi";
import { isDemoMode } from "./demoMode";

const baseURL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";

export const api = axios.create({
  baseURL,
  timeout: 10_000,
  headers: {
    Accept: "application/json",
  },
  ...(isDemoMode() ? { adapter: demoAdapter } : {}),
});
