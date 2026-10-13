import { useEffect, useState } from "react";
import { getAnalytics, getRiskBreakdown, getRecentEvents, getDetectorStatus } from "@/api/detection";
import type { AnalyticsExtended, RiskBreakdown, RecentEvent, DetectorStatus } from "@/api/detection";

function StatCard({ label, value, sub, icon, colorClass = "" }: {
  label: string; value: string | number; sub?: string; icon: React.ReactNode; colorClass?: string;
}) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${colorClass}`}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
      <div className="stat-icon">{icon}</div>
    </div>
  );
}

function RiskBar({ label, count, total, color }: { label: string; count: number; total: number; color: string }) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <div className="risk-row">
      <span className="risk-label-text">{label}</span>
      <div className="risk-bar-track">
        <div className="risk-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="risk-count">{count}</span>
    </div>
  );
}

function timeAgo(ts?: number) {
  if (typeof ts !== "number" || isNaN(ts)) return "—";
  const diff = Math.floor(Date.now() / 1000 - ts);
  if (diff < 0) return "Just now";
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

export default function DashboardPage() {
  const [analytics, setAnalytics] = useState<AnalyticsExtended | null>(null);
  const [risk, setRisk] = useState<RiskBreakdown | null>(null);
  const [events, setEvents] = useState<RecentEvent[]>([]);
  const [status, setStatus] = useState<DetectorStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      const [aRes, rRes, eRes, sRes] = await Promise.allSettled([
        getAnalytics(),
        getRiskBreakdown(),
        getRecentEvents(10),
        getDetectorStatus(),
      ]);

      let hasError = false;
      if (aRes.status === "fulfilled") setAnalytics(aRes.value);
      else hasError = true;

      if (rRes.status === "fulfilled") setRisk(rRes.value);
      else hasError = true;

      if (eRes.status === "fulfilled") {
        setEvents(Array.isArray(eRes.value?.events) ? eRes.value.events : []);
      } else hasError = true;

      if (sRes.status === "fulfilled") setStatus(sRes.value);

      if (hasError && aRes.status === "rejected" && rRes.status === "rejected") {
        setError("Failed to fetch dashboard data. Ensure the backend server is running.");
      } else {
        setError(null);
      }
    } catch (e: any) {
      setError(e?.message ?? "An error occurred while connecting to the server.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    const id = setInterval(loadData, 5000);
    return () => clearInterval(id);
  }, []);

  if (loading && !analytics) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", paddingTop: "3rem" }}>
        <div className="spinner" /> <span className="text-secondary">Loading dashboard data…</span>
      </div>
    );
  }

  const detRate = typeof analytics?.detection_rate === "number"
    ? `${(analytics.detection_rate * 100).toFixed(1)}%`
    : "—";

  return (
    <div>
      <div className="page-title">Dashboard</div>
      <div className="page-sub">Real-time prompt injection monitoring &amp; analytics</div>

      {error && (
        <div className="card mb-3" style={{ borderColor: "var(--danger)", background: "var(--danger-dim)" }}>
          <span className="text-danger">{error}</span>
          <button className="btn btn-ghost" style={{ marginLeft: "1rem", padding: "0.2rem 0.6rem", fontSize: "0.75rem" }} onClick={loadData}>
            Retry
          </button>
        </div>
      )}

      {/* Stats row */}
      <div className="stats-grid mb-3">
        <StatCard
          label="Total Scans"
          value={analytics?.total_scans ?? 0}
          icon={
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
          }
        />
        <StatCard
          label="Injections"
          value={analytics?.injections_detected ?? 0}
          colorClass="danger"
          sub={`${detRate} detection rate`}
          icon={
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          }
        />
        <StatCard
          label="Avg Confidence"
          value={typeof analytics?.avg_confidence === "number" ? `${(analytics.avg_confidence * 100).toFixed(0)}%` : "—"}
          colorClass="accent"
          icon={
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <circle cx="12" cy="12" r="6" />
              <circle cx="12" cy="12" r="2" />
            </svg>
          }
        />
        <StatCard
          label="Avg Latency"
          value={typeof analytics?.avg_processing_time_ms === "number" ? `${analytics.avg_processing_time_ms.toFixed(1)}ms` : "—"}
          icon={
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
          }
        />
      </div>

      <div className="grid-2">
        {/* Risk breakdown */}
        <div className="card">
          <div className="section-title">Risk Level Breakdown</div>
          {risk ? (
            <div className="risk-breakdown">
              <RiskBar label="Critical" count={risk.critical ?? 0} total={risk.total ?? 0} color="var(--danger)" />
              <RiskBar label="High"     count={risk.high ?? 0}     total={risk.total ?? 0} color="#f97316" />
              <RiskBar label="Medium"   count={risk.medium ?? 0}   total={risk.total ?? 0} color="var(--warning)" />
              <RiskBar label="Low"      count={risk.low ?? 0}      total={risk.total ?? 0} color="var(--success)" />
            </div>
          ) : <div className="text-muted text-sm">No risk breakdown data available</div>}
        </div>

        {/* Detector status */}
        <div className="card">
          <div className="section-title">Detector Status</div>
          <div className="flex-col gap-1">
            {[
              { label: "Rule Engine", active: status?.rule_engine ?? false, detail: "38 rules active" },
              { label: "ML Classifier", active: status?.ml_classifier ?? false, detail: "distilbert-base-uncased" },
              { label: "Semantic Analyzer", active: true, detail: "all-MiniLM-L6-v2" },
            ].map(({ label, active, detail }) => (
              <div key={label} className="layer-row">
                <span className={`layer-triggered ${active ? "yes" : "no"}`}>
                  {active ? "ACTIVE" : "INACTIVE"}
                </span>
                <span className="layer-name">{label}</span>
                <span className="layer-details">{detail}</span>
              </div>
            ))}
          </div>
          <hr className="divider" />
          <div className="text-sm text-muted">
            Uptime: {typeof analytics?.uptime_seconds === "number"
              ? `${Math.floor(analytics.uptime_seconds / 60)}m ${Math.floor(analytics.uptime_seconds % 60)}s`
              : "—"}
          </div>
        </div>
      </div>

      {/* Recent events */}
      <div className="card mt-2">
        <div className="section-title">Recent Detections</div>
        {!events || events.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 12h-6l-2 3h-4l-2-3H2v7a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-7z" />
                <path d="M5.45 5.11 2 12v0h20v0l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
              </svg>
            </div>
            <div>No scans recorded yet — submit a prompt on the Analyze page</div>
          </div>
        ) : (
          <table className="events-table">
            <thead>
              <tr>
                <th>Time</th><th>Prompt Preview</th><th>Verdict</th><th>Risk Level</th><th>Confidence</th><th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => (
                <tr key={i}>
                  <td className="mono" style={{ whiteSpace: "nowrap" }}>{timeAgo(e.timestamp)}</td>
                  <td><div className="prompt-preview" title={e.prompt_preview}>{e.prompt_preview || "—"}</div></td>
                  <td>
                    <span className={e.is_injection ? "text-danger" : "text-success"} style={{ fontWeight: 600 }}>
                      {e.is_injection ? "Injection" : "Safe"}
                    </span>
                  </td>
                  <td><span className={`risk-badge risk-${e.risk_level || "low"}`}>{e.risk_level || "low"}</span></td>
                  <td className="mono">{typeof e.confidence === "number" ? `${(e.confidence * 100).toFixed(0)}%` : "0%"}</td>
                  <td className="mono">{typeof e.processing_ms === "number" ? `${e.processing_ms.toFixed(1)}ms` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

