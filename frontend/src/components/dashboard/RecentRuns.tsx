"use client";

import { motion } from "framer-motion";
import { Eye, Square, ChevronRight } from "lucide-react";
import Link from "next/link";

interface Run {
  id: string;
  model_name: string;
  attacks: string[];
  robustness_score: number | null;
  status: string;
}

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { color: string; dot: string; animate: boolean }> = {
    completed: { color: "var(--success)", dot: "var(--success)", animate: false },
    running: { color: "var(--text-secondary)", dot: "#fff", animate: true },
    failed: { color: "var(--error)", dot: "var(--error)", animate: false },
  };
  const c = config[status] || config.completed;

  return (
    <div className="flex items-center gap-2">
      <div className={`w-2 h-2 rounded-full ${c.animate ? "animate-breathe" : ""}`} style={{ background: c.dot, boxShadow: c.animate ? "0 0 8px rgba(255,255,255,0.4)" : undefined }} />
      <span className="text-xs capitalize" style={{ color: c.color }}>{status}</span>
    </div>
  );
}

interface RecentRunsProps {
  runs?: Run[];
}

export default function RecentRuns({ runs }: RecentRunsProps) {
  const data = runs && runs.length > 0 ? runs : [];

  return (
    <section className="relative z-10">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-medium" style={{ color: "var(--text-primary)" }}>Recent Runs</h2>
        <Link href="/history" className="text-sm flex items-center gap-1 transition-all" style={{ color: "var(--text-muted)" }}>View All<ChevronRight size={16} /></Link>
      </div>

      <div className="glass-card rounded-2xl overflow-hidden">
        {data.length === 0 ? (
          <div className="p-8 text-center">
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>No test runs yet. Start one with "+ New Test".</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr style={{ background: "rgba(255,255,255,0.02)", borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
                  {["Model Name", "Attacks", "Score", "Status", "Action"].map((h, i) => (
                    <th key={h} className="px-6 py-4 font-bold uppercase" style={{ fontSize: "11px", letterSpacing: "0.08em", color: "var(--text-muted)", textAlign: i === 2 ? "right" : i === 4 ? "center" : "left" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.map((run, index) => (
                  <motion.tr
                    key={run.id}
                    className="transition-all group"
                    style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}
                    initial={{ opacity: 0, x: -12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.02)"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                  >
                    <td className="px-6 py-5">
                      <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>{run.model_name}</span>
                    </td>
                    <td className="px-6 py-5">
                      <div className="flex flex-wrap gap-2">
                        {run.attacks.map(a => (
                          <span key={a} className="px-2 py-0.5 rounded-md" style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", fontSize: "10px", color: "var(--text-secondary)" }}>{a}</span>
                        ))}
                      </div>
                    </td>
                    <td className="px-6 py-5 text-right text-sm" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--text-primary)" }}>
                      {run.robustness_score ?? "—"}
                    </td>
                    <td className="px-6 py-5"><StatusBadge status={run.status} /></td>
                    <td className="px-6 py-5 text-center">
                      <Link href={`/results?run=${run.id}`} className="transition-colors" style={{ color: "var(--text-muted)" }}>
                        {run.status === "running" ? <Square size={16} /> : <Eye size={16} />}
                      </Link>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
