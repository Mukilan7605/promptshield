/** 404 Not Found page. */
import { Link } from "react-router-dom";

export default function NotFoundPage() {
  return (
    <div style={{ textAlign: "center", padding: "4rem" }}>
      <h1 style={{ fontSize: "3rem", fontWeight: 800, color: "var(--color-accent-primary)" }}>404</h1>
      <p style={{ color: "var(--color-text-secondary)", marginBottom: "1.5rem" }}>
        Page not found.
      </p>
      <Link to="/" style={{ color: "var(--color-accent-primary)", fontWeight: 600 }}>
        ← Back to Dashboard
      </Link>
    </div>
  );
}
