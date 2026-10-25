/** Shared TypeScript types for the PromptShield frontend. */

export interface LayerResult {
  layer: string;
  triggered: boolean;
  confidence: number;
  details?: string;
}

export type RiskLevel = "low" | "medium" | "high" | "critical";

export interface DetectionResult {
  is_injection: boolean;
  final_score: number;
  overall_confidence: number;
  confidence: number;
  risk_level: RiskLevel;
  decision: "ALLOW" | "WARN" | "SANITIZE" | "BLOCK";
  attack_category: string;
  reasons: string[];
  layers: LayerResult[];
  sanitized_prompt?: string;
  processing_time_ms: number;
}

export interface AnalyticsSummary {
  total_scans: number;
  injections_detected: number;
  benign_count: number;
  detection_rate: number;
  avg_confidence: number;
  avg_processing_time_ms: number;
  uptime_seconds: number;
}

export interface ScanRecord {
  id: number;
  prompt_preview: string;
  is_injection: boolean;
  overall_confidence: number;
  risk_level: RiskLevel;
  processing_time_ms: number;
  model_version: string;
  created_at: string;
}

export interface ChatGatewayResponse {
  request_id: string;
  original_prompt: string;
  final_prompt: string;
  decision: "ALLOW" | "WARN" | "SANITIZE" | "BLOCK";
  risk_score: number;
  was_sanitized: boolean;
  sanitization_reason?: string | null;
  blocked_parts?: string[];
  llm_called: boolean;
  llm_provider: string;
  response?: string | Record<string, any> | null;
  processing_time_ms: number;
  can_sanitize?: boolean;
  sanitized_prompt?: string | null;
}


