/** Axios API client — all backend calls go through this module. */

import axios from "axios";

// In production, always point to the Render backend.
// In development, requests go through the Vite proxy (vite.config.ts).
const isProd = import.meta.env.PROD;
const BASE_URL = isProd
  ? `${import.meta.env.VITE_API_BASE_URL ?? "https://promptshield-cy8k.onrender.com"}/api/v1`
  : "/api/v1";

const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: 30_000,
});

// ── Response interceptor for global error handling ──────────
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // TODO: Add toast/notification in Phase 2
    console.error("[API Error]", error?.response?.data ?? error.message);
    return Promise.reject(error);
  }
);

export default apiClient;
