import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import { ErrorBoundary } from "../ErrorBoundary";

export default function Layout() {
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="app-main">
        <ErrorBoundary>
          <Outlet />
        </ErrorBoundary>
      </main>
    </div>
  );
}
