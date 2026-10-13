import apiClient from "./client";
import type { DetectionResult, AnalyticsSummary, ChatGatewayResponse } from "@/types";

export interface DetectionRequest {
  prompt: string;
  context?: string;
  model_version?: string;
}

export interface RiskBreakdown { low: number; medium: number; high: number; critical: number; total: number; }
export interface RecentEvent { timestamp: number; is_injection: boolean; confidence: number; risk_level: string; processing_ms: number; prompt_preview: string; }
export interface AnalyticsExtended extends AnalyticsSummary { benign_count: number; uptime_seconds: number; }
export interface DetectorStatus { rule_engine: boolean; ml_classifier: boolean; ml_model_path: string | null; }

export const detect = (req: DetectionRequest) =>
  apiClient.post<DetectionResult>("/detect", req).then((r) => r.data);

export const processChatGateway = (prompt: string) =>
  apiClient.post<ChatGatewayResponse>("/chat", { prompt }).then((r) => r.data);

export const getAnalytics = () =>
  apiClient.get<AnalyticsExtended>("/analytics").then((r) => r.data);


export const getRiskBreakdown = () =>
  apiClient.get<RiskBreakdown>("/analytics/risk-levels").then((r) => r.data);

export const getRecentEvents = (n = 20) =>
  apiClient.get<{ events: RecentEvent[]; count: number }>(`/analytics/recent?n=${n}`).then((r) => r.data);

export const getDetectorStatus = () =>
  apiClient.get<DetectorStatus>("/detect/status").then((r) => r.data);

export const getHealth = () =>
  apiClient.get("/health/ready").then((r) => r.data);
