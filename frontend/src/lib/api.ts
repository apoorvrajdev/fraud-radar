/**
 * Centralised axios instance for the Fraud Radar backend.
 *
 * The base URL comes from `VITE_API_URL` (see `.env.example`). All
 * dashboard hooks go through this client so retries, timeouts, and
 * future auth headers have one place to live.
 */
import axios from "axios";

const baseURL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";

export const api = axios.create({
  baseURL,
  timeout: 10_000,
  headers: {
    Accept: "application/json",
  },
});
