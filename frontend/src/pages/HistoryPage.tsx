import { useEffect, useState } from "react";
import { getRecentEvents } from "@/api/detection";
import type { RecentEvent } from "@/api/detection";

function timeAgo(ts?: number) {
  if (typeof ts !== "number" || isNaN(ts)) return "—";
  const d = Math.floor(Date.now() / 1000 - ts);
  if (d < 0) return "Just now";
  if (d < 60) return `${d}s ago`;
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  return new Date(ts * 1000).toLocaleTimeString();
}

export default function HistoryPage() {
  const [events, setEvents] = useState<RecentEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "injection" | "safe">("all");

  const loadData = () => {
    setLoading(true);
    getRecentEvents(100)
      .then((r) => {
        setEvents(Array.isArray(r?.events) ? r.events : []);
        setError(null);
      })
      .catch((e: any) => {
        console.error("Failed to load history events", e);
        setError(e?.message ?? "Failed to load scan history from backend.");
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
    const id = setInterval(loadData, 10000);
    return () => clearInterval(id);
  }, []);

  const safeEvents = Array.isArray(events) ? events : [];

  const filtered = safeEvents.filter((e) =>
    filter === "all" ? true : filter === "injection" ? e.is_injection : !e.is_injection
  );

  const injCount = safeEvents.filter((e) => e.is_injection).length;

  return (
    <div>
      <div className="page-title">Scan History</div>
      <div className="page-sub">Historical log of all prompt security scans from the application database</div>

      {error && (
        <div className="card mb-3" style={{ borderColor: "var(--danger)", background: "var(--danger-dim)" }}>
          <span className="text-danger">{error}</span>
          <button className="btn btn-ghost" style={{ marginLeft: "1rem", padding: "0.2rem 0.6rem", fontSize: "0.75rem" }} onClick={loadData}>
            Retry
          </button>
        </div>
      )}

      {/* Summary chips */}
      <div style={{ display: "flex", gap: "1rem", marginBottom: "1.25rem" }}>
        <div className="card" style={{ padding: "0.85rem 1.4rem", display: "flex", gap: "0.5rem", alignItems: "center", flex: 1 }}>
          <div>
            <div className="stat-label">Total Scans</div>
            <div style={{ fontWeight: 700, fontSize: "1.2rem", color: "#fff" }}>{safeEvents.length}</div>
          </div>
        </div>
        <div className="card" style={{ padding: "0.85rem 1.4rem", display: "flex", gap: "0.5rem", alignItems: "center", flex: 1 }}>
          <div>
            <div className="stat-label">Injections</div>
            <div style={{ fontWeight: 700, fontSize: "1.2rem", color: "var(--danger)" }}>{injCount}</div>
          </div>
        </div>
        <div className="card" style={{ padding: "0.85rem 1.4rem", display: "flex", gap: "0.5rem", alignItems: "center", flex: 1 }}>
          <div>
            <div className="stat-label">Safe</div>
            <div style={{ fontWeight: 700, fontSize: "1.2rem", color: "var(--success)" }}>{safeEvents.length - injCount}</div>
          </div>
        </div>
      </div>

      {/* Filter buttons */}
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
        {(["all", "injection", "safe"] as const).map((f) => (
          <button
            key={f}
            className={`btn ${filter === f ? "btn-primary" : "btn-ghost"}`}
            style={{ fontSize: "0.8rem", padding: "0.4rem 1.1rem" }}
            onClick={() => setFilter(f)}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
        <button className="btn btn-ghost" style={{ marginLeft: "auto", fontSize: "0.8rem" }} onClick={loadData}>
          Refresh
        </button>
      </div>

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        {loading && safeEvents.length === 0 ? (
          <div style={{ padding: "2rem", display: "flex", gap: "0.75rem", alignItems: "center" }}>
            <div className="spinner" /><span className="text-secondary">Loading history records…</span>
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
            </div>
            <div>{safeEvents.length === 0 ? "No scan records found in database — use the Analyze page to get started" : "No results for this filter"}</div>
          </div>
        ) : (
          <table className="events-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Prompt Preview</th>
                <th>Verdict</th>
                <th>Risk Level</th>
                <th>Confidence</th>
                <th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((e, i) => (
                <tr key={i}>
                  <td className="mono" style={{ whiteSpace: "nowrap", color: "var(--text-muted)" }}>
                    {timeAgo(e.timestamp)}
                  </td>
                  <td>
                    <div className="prompt-preview" title={e.prompt_preview}>
                      {e.prompt_preview || <span className="text-muted">—</span>}
                    </div>
                  </td>
                  <td>
                    <span style={{ fontWeight: 600, color: e.is_injection ? "var(--danger)" : "var(--success)" }}>
                      {e.is_injection ? "Injection" : "Safe"}
                    </span>
                  </td>
                  <td><span className={`risk-badge risk-${e.risk_level || "low"}`}>{e.risk_level || "low"}</span></td>
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <div style={{ width: 40, height: 4, background: "rgba(255,255,255,0.1)", borderRadius: 99, overflow: "hidden" }}>
                        <div style={{
                          height: "100%", borderRadius: 99,
                          width: `${(e.confidence ?? 0) * 100}%`,
                          background: e.is_injection ? "var(--danger)" : "var(--success)"
                        }} />
                      </div>
                      <span className="mono" style={{ fontSize: "0.75rem" }}>{((e.confidence ?? 0) * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                  <td className="mono" style={{ fontSize: "0.75rem" }}>
                    {typeof e.processing_ms === "number" ? `${e.processing_ms.toFixed(1)}ms` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

