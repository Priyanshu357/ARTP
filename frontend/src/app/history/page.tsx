"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import DashboardNavbar from "@/components/layout/DashboardNavbar";
import Sidebar from "@/components/layout/Sidebar";
import Link from "next/link";
import { Eye, Filter } from "lucide-react";
import { getRuns, type RunInfo } from "@/lib/api";

function StatusBadge({ status }: { status: string }) {
  const color = status === "completed" ? "var(--success)" : status === "failed" ? "#ef4444" : "var(--text-secondary)";
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      <span className="text-xs capitalize" style={{ color }}>{status}</span>
    </div>
  );
}

export default function HistoryPage() {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState("all");

  useEffect(() => {
    getRuns()
      .then((data) => setRuns(data.runs))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const filtered = statusFilter === "all" ? runs : runs.filter(r => r.status === statusFilter);

  return (
    <>
      <DashboardNavbar />
      <Sidebar />
      <main className="lg:ml-[260px] pt-24 px-6 pb-12 relative min-h-screen">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full pointer-events-none" style={{ height: "600px", background: "radial-gradient(ellipse at top, rgba(255,255,255,0.03) 0%, transparent 70%)", zIndex: 0 }} />

        <div className="relative z-10 max-w-5xl mx-auto space-y-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-medium" style={{ color: "var(--text-primary)" }}>Run History</h1>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                {loading ? "Loading..." : `${runs.length} test runs across all models.`}
              </p>
            </div>
          </div>

          {/* Filters */}
          <div className="flex items-center gap-3">
            <Filter size={14} style={{ color: "var(--text-muted)" }} />
            {["all", "completed", "failed"].map(f => (
              <button
                key={f}
                className="px-3 py-1.5 rounded-full text-xs font-medium transition-all capitalize"
                style={{
                  background: statusFilter === f ? "rgba(255,255,255,0.08)" : "transparent",
                  color: statusFilter === f ? "var(--text-primary)" : "var(--text-muted)",
                  border: statusFilter === f ? "1px solid rgba(255,255,255,0.15)" : "1px solid transparent",
                }}
                onClick={() => setStatusFilter(f)}
              >
                {f}
              </button>
            ))}
          </div>

          {/* Table */}
          <div className="glass-card rounded-2xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr style={{ background: "rgba(255,255,255,0.02)", borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
                    {["Run ID", "Model", "Attacks", "Score", "Health", "Status", "Date", ""].map((h, i) => (
                      <th key={h || i} className="px-6 py-4 font-bold uppercase" style={{ fontSize: "11px", letterSpacing: "0.08em", color: "var(--text-muted)" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((run, i) => (
                    <motion.tr
                      key={run.id}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.04, duration: 0.3 }}
                      className="transition-colors"
                      style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.02)")}
                      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                    >
                      <td className="px-6 py-4 text-sm font-medium" style={{ color: "var(--text-primary)", fontFamily: "var(--font-jetbrains), monospace" }}>{run.id}</td>
                      <td className="px-6 py-4 text-sm" style={{ color: "var(--text-primary)" }}>{run.model_name}</td>
                      <td className="px-6 py-4">
                        <div className="flex flex-wrap gap-1.5">
                          {run.attacks.map(a => (
                            <span key={a} className="px-2 py-0.5 rounded text-[10px] font-medium" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)", color: "var(--text-secondary)" }}>{a}</span>
                          ))}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm" style={{ fontFamily: "var(--font-jetbrains), monospace", color: run.robustness_score ? "var(--text-primary)" : "var(--text-muted)" }}>
                        {run.robustness_score || "—"}
                      </td>
                      <td className="px-6 py-4">
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase" style={{
                          background: run.health === "CRITICAL" ? "rgba(163,68,61,0.15)" : "rgba(255,255,255,0.06)",
                          color: run.health === "CRITICAL" ? "var(--red-team)" : "var(--text-secondary)",
                        }}>
                          {run.health}
                        </span>
                      </td>
                      <td className="px-6 py-4"><StatusBadge status={run.status} /></td>
                      <td className="px-6 py-4 text-xs" style={{ color: "var(--text-muted)" }}>
                        {run.timestamp ? new Date(run.timestamp).toLocaleDateString() : "—"}
                      </td>
                      <td className="px-6 py-4">
                        <Link href={`/results?run=${run.id}`} className="p-2 rounded-lg inline-flex transition-all" style={{ color: "var(--text-muted)" }}>
                          <Eye size={16} />
                        </Link>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
