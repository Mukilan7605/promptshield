/** Axios API client — all backend calls go through this module. */

import axios from "axios";

// In production on Vercel, requests go through the Vercel proxy rewrite
// (vercel.json routes /api/* → Render backend server-side).
// This means ZERO CORS issues — browser only talks to vercel.app.
// In development, requests go through the Vite proxy (vite.config.ts).
const BASE_URL = "/api/v1";

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
  let attempts = 0;
  const maxAttempts = 5;

  const ping = async () => {
    attempts++;
    try {
      await axios.get("/api/v1/health", { timeout: 10_000 });
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

export const BACKEND_BASE = "";
export default apiClient;
