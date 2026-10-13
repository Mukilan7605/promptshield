/** Axios API client — all backend calls go through this module. */

import axios from "axios";

// In production, always point to the Render backend.
// In development, requests go through the Vite proxy (vite.config.ts).
const isProd = import.meta.env.PROD;

export const BACKEND_BASE = isProd
  ? (import.meta.env.VITE_API_BASE_URL ?? "https://promptshield-cy8k.onrender.com")
  : "";

const BASE_URL = `${BACKEND_BASE}/api/v1`;

const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: 30_000,
});

// ── Response interceptor for global error handling ──────────
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error("[API Error]", error?.response?.data ?? error.message);
    return Promise.reject(error);
  }
);

/**
 * Silently wake the Render backend on app load.
 * Free-tier Render services sleep after 15 min of inactivity.
 * This ping starts the wake-up before the user hits a real endpoint,
 * so cold-start delay is hidden. Retries up to 5 times with 3 s gap.
 */
export function wakeBackend(): void {
  if (!isProd) return; // only needed in production

  let attempts = 0;
  const maxAttempts = 5;

  const ping = async () => {
    attempts++;
    try {
      await axios.get(`${BACKEND_BASE}/api/v1/health`, { timeout: 10_000 });
      console.info("[PromptShield] Backend is awake ✅");
    } catch {
      if (attempts < maxAttempts) {
        console.info(`[PromptShield] Backend waking up… attempt ${attempts}/${maxAttempts}`);
        setTimeout(ping, 3_000);
      } else {
        console.warn("[PromptShield] Backend did not respond after 5 attempts.");
      }
    }
  };

  ping();
}

export default apiClient;

