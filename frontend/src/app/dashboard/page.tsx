"use client";

import { useEffect, useState } from "react";
import DashboardNavbar from "@/components/layout/DashboardNavbar";
import Sidebar from "@/components/layout/Sidebar";
import DashboardMetrics from "@/components/dashboard/DashboardMetrics";
import AsrBarChart from "@/components/dashboard/AsrBarChart";
import RobustnessTrend from "@/components/dashboard/RobustnessTrend";
import RecentRuns from "@/components/dashboard/RecentRuns";
import { getStats, getRuns, type StatsData, type RunInfo } from "@/lib/api";

export default function DashboardPage() {
  const [stats, setStats] = useState<StatsData | null>(null);
  const [runs, setRuns] = useState<RunInfo[]>([]);

  useEffect(() => {
    getStats()
      .then(setStats)
      .catch(() => console.info("ARTP backend not reachable; using demo data."));

    getRuns()
      .then((data) => setRuns(data.runs))
      .catch(() => {});
  }, []);

  const lastRun = runs.length > 0 ? runs[0] : null;

  return (
    <>
      <DashboardNavbar />
      <Sidebar />

      <main className="lg:ml-[260px] pt-24 px-6 pb-12 relative min-h-screen">
        {/* Radial glow */}
        <div
          className="absolute top-0 left-1/2 -translate-x-1/2 w-full pointer-events-none"
          style={{ height: "600px", background: "radial-gradient(ellipse at top, rgba(255,255,255,0.03) 0%, transparent 70%)", zIndex: 0 }}
        />

        {/* Header */}
        <section className="relative z-10 flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8">
          <div>
            <h1 className="text-2xl font-medium mb-1" style={{ color: "var(--text-primary)" }}>Welcome back</h1>
            <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
              {stats?.has_data
                ? `${stats.total_runs || runs.length} test runs completed. Review robustness status.`
                : "Run your first test to see metrics here."}
            </p>
          </div>
          {lastRun && (
            <div className="text-right">
              <p className="font-bold uppercase" style={{ fontSize: "11px", letterSpacing: "0.1em", color: "var(--text-muted)" }}>Last Run</p>
              <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                {lastRun.model_name} — Score: {lastRun.robustness_score || "—"}
              </p>
            </div>
          )}
        </section>

        {/* Metrics */}
        <DashboardMetrics stats={stats} />

        {/* Charts */}
        <section className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-10 relative z-10">
          <AsrBarChart />
          <RobustnessTrend />
        </section>

        {/* Recent Runs — fed with real data */}
        <RecentRuns
          runs={runs.map((r) => ({
            id: r.id,
            model_name: r.model_name,
            attacks: r.attacks,
            robustness_score: r.robustness_score,
            status: r.status,
          }))}
        />
      </main>
    </>
  );
}
