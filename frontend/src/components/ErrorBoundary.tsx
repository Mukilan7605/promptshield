import { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught React error:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: "2rem", color: "#ef4444", background: "#1a2235", borderRadius: "10px", margin: "2rem", border: "1px solid #ef4444" }}>
          <h2>UI Component Rendering Error</h2>
          <p style={{ marginTop: "0.5rem", color: "#94a3b8" }}>
            A rendering error occurred in this view:
          </p>
          <pre style={{ whiteSpace: "pre-wrap", marginTop: "1rem", padding: "1rem", background: "#080b12", borderRadius: "6px", color: "#f1f5f9", fontSize: "0.85rem" }}>
            {this.state.error?.toString()}
          </pre>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{ marginTop: "1rem", padding: "0.5rem 1rem", background: "#6366f1", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer", fontWeight: 600 }}
          >
            Reload View
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
