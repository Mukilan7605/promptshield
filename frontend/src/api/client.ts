/** Axios API client — all backend calls go through this module. */

import axios from "axios";

// In production (Vercel), VITE_API_BASE_URL points to the Render backend.
// In development, requests go through the Vite proxy defined in vite.config.ts.
const BASE_URL = import.meta.env.VITE_API_BASE_URL
  ? `${import.meta.env.VITE_API_BASE_URL}/api/v1`
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
