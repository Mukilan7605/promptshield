/** Axios API client — all backend calls go through this module. */

import axios from "axios";

const apiClient = axios.create({
  baseURL: "/api/v1",
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
