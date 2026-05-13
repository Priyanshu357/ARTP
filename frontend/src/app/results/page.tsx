"use client";

import { useState, useEffect, useRef, Suspense } from "react";
import { motion, useInView } from "framer-motion";
import { useSearchParams } from "next/navigation";
import DashboardNavbar from "@/components/layout/DashboardNavbar";
import Sidebar from "@/components/layout/Sidebar";
import { Download, Code, RefreshCw } from "lucide-react";
import { getRuns, getRunSummary, getRunDiagnostics, getRunAttacks, type RunInfo, type RunSummary, type DiagnosticsData, type AttackStat } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const TABS = ["Summary", "Attacks", "Detection", "Mitigation"];

function RobustnessRing({ score }: { score: number }) {
  const ref = useRef<SVGSVGElement>(null);
  const isInView = useInView(ref as React.RefObject<Element>, { once: true });
  const r = 70, cx = 80, cy = 80, C = 2 * Math.PI * r;
  const offset = C - (score / 100) * C;

  return (
    <div className="flex flex-col items-center">
      <svg ref={ref} width="160" height="160" viewBox="0 0 160 160">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="8" />
        <motion.circle
          cx={cx} cy={cy} r={r} fill="none" stroke="#E5E5E5" strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={C}
          initial={{ strokeDashoffset: C }}
          animate={isInView ? { strokeDashoffset: offset } : {}}
          transition={{ duration: 1.2, ease: "easeInOut" }}
          transform={`rotate(-90 ${cx} ${cy})`}
          style={{ filter: "drop-shadow(0 0 8px rgba(255,255,255,0.15))" }}
        />
        <text x={cx} y={cy - 5} textAnchor="middle" fill="white" fontSize="32" fontWeight="300">{score}</text>
        <text x={cx} y={cy + 16} textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize="12">/ 100</text>
      </svg>
      <p className="text-xs mt-2 text-center" style={{ color: "var(--text-muted)", fontFamily: "var(--font-jetbrains), monospace", fontSize: "10px" }}>
        (1 - ASR) × DA × (1 - FPR) × 100
      </p>
    </div>
  );
}

function ResultsContent() {
  const searchParams = useSearchParams();
  const runId = searchParams.get("run");

  const [activeTab, setActiveTab] = useState("Summary");
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>("");
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [diagnostics, setDiagnostics] = useState<DiagnosticsData | null>(null);
  const [attacks, setAttacks] = useState<AttackStat[]>([]);
  const [loading, setLoading] = useState(true);

  // Load runs list
  useEffect(() => {
    getRuns().then((data) => {
      setRuns(data.runs);
      if (runId && data.runs.find(r => r.id === runId)) {
        setSelectedRun(runId);
      } else if (data.runs.length > 0) {
        setSelectedRun(data.runs[0].id);
      }
    }).catch(() => {}).finally(() => setLoading(false));
  }, [runId]);

  // Load data for selected run
  useEffect(() => {
    if (!selectedRun) return;
    setLoading(true);

    Promise.all([
      getRunSummary(selectedRun).catch(() => null),
      getRunDiagnostics(selectedRun).catch(() => null),
      getRunAttacks(selectedRun).catch(() => ({ attacks: [], total_entries: 0 })),
    ]).then(([s, d, a]) => {
      setSummary(s);
      setDiagnostics(d);
      setAttacks(a?.attacks || []);
    }).finally(() => setLoading(false));
  }, [selectedRun]);

  const currentRunInfo = runs.find(r => r.id === selectedRun);

  return (
    <div className="relative z-10 max-w-5xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <h1 className="text-2xl font-medium" style={{ color: "var(--text-primary)" }}>
              {currentRunInfo?.model_name || "Results"}
            </h1>
            {currentRunInfo && (
              <>
                <span className="text-sm" style={{ color: "var(--text-muted)" }}>·</span>
                <span className="text-sm" style={{ color: "var(--text-secondary)" }}>{currentRunInfo.id}</span>
                <span className="text-sm" style={{ color: "var(--text-muted)" }}>·</span>
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full" style={{ background: "var(--success)" }} />
                  <span className="text-sm" style={{ color: "var(--success)" }}>Completed</span>
                </div>
              </>
            )}
          </div>
        </div>
        {/* Run selector */}
        {runs.length > 1 && (
          <select
            value={selectedRun}
            onChange={(e) => setSelectedRun(e.target.value)}
            className="px-3 py-2 rounded-lg text-sm outline-none"
            style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", color: "var(--text-primary)" }}
          >
            {runs.map(r => <option key={r.id} value={r.id}>{r.model_name}</option>)}
          </select>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-2">
        {TABS.map((tab) => (
          <button
            key={tab}
            className="px-4 py-2 rounded-lg text-sm font-medium transition-all"
            style={{
              background: activeTab === tab ? "rgba(255,255,255,0.06)" : "transparent",
              color: activeTab === tab ? "var(--text-primary)" : "var(--text-muted)",
            }}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="glass-card rounded-2xl p-12 text-center">
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>Loading results...</p>
        </div>
      ) : !summary ? (
        <div className="glass-card rounded-2xl p-12 text-center">
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>No results found. Run a test first.</p>
        </div>
      ) : (
        <>
          {/* Robustness Ring */}
          <div className="glass-card rounded-2xl p-8 flex flex-col items-center">
            <RobustnessRing score={Math.round(summary.robustness_score * 10) / 10} />
          </div>

          {/* Metric pills */}
          <div className="grid grid-cols-3 gap-6">
            {[
              { label: "Attack Success Rate", value: summary.attack_success_rate.toFixed(2) },
              { label: "Detection Accuracy", value: summary.detection_accuracy.toFixed(2) },
              { label: "False Positive Rate", value: summary.false_positive_rate.toFixed(2) },
            ].map((m) => (
              <div key={m.label} className="glass-card rounded-2xl p-5 text-center">
                <p className="text-xs font-bold uppercase tracking-widest mb-2" style={{ color: "var(--text-muted)", letterSpacing: "0.1em" }}>{m.label}</p>
                <p className="text-3xl" style={{ fontWeight: 300, color: "var(--text-primary)" }}>{m.value}</p>
              </div>
            ))}
          </div>

          {/* Per-Attack Table */}
          {attacks.length > 0 && (
            <div className="glass-card rounded-2xl overflow-hidden">
              <div className="p-6 pb-0">
                <h3 className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>Per-Attack Breakdown</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr style={{ background: "rgba(255,255,255,0.02)", borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
                      {["Attack", "ASR", "Samples", "Avg Perturbation"].map((h, i) => (
                        <th key={h} className="px-6 py-4 font-bold uppercase" style={{ fontSize: "11px", letterSpacing: "0.08em", color: "var(--text-muted)", textAlign: i > 0 ? "right" : "left" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {attacks.map((a) => (
                      <tr key={a.name} style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
                        <td className="px-6 py-4 text-sm font-medium" style={{ color: "var(--text-primary)" }}>{a.name}</td>
                        <td className="px-6 py-4 text-sm text-right" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--text-primary)" }}>{a.asr}</td>
                        <td className="px-6 py-4 text-sm text-right" style={{ color: "var(--text-secondary)" }}>{a.samples}</td>
                        <td className="px-6 py-4 text-sm text-right" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--text-secondary)" }}>{a.avg_perturbation}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Diagnostics */}
          {diagnostics && diagnostics.diagnostics.length > 0 && (
            <div className="glass-card rounded-2xl p-6">
              <h3 className="text-sm font-medium mb-4" style={{ color: "var(--text-primary)" }}>
                Diagnostics — Health: <span style={{ color: diagnostics.overall_health === "CRITICAL" ? "var(--red-team)" : "var(--success)" }}>{diagnostics.overall_health}</span>
              </h3>
              <div className="space-y-4">
                {diagnostics.diagnostics.map((d, i) => (
                  <div key={i} className="rounded-xl p-4" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase" style={{
                        background: d.severity === "CRITICAL" ? "rgba(163,68,61,0.15)" : "rgba(255,255,255,0.06)",
                        color: d.severity === "CRITICAL" ? "var(--red-team)" : "var(--text-secondary)",
                      }}>
                        {d.severity}
                      </span>
                      <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>{d.diagnostic.replace(/_/g, " ")}</span>
                    </div>
                    <p className="text-xs mb-2" style={{ color: "var(--text-secondary)" }}>{d.description}</p>
                    {d.interpretation && (
                      <p className="text-xs italic" style={{ color: "var(--text-muted)" }}>{d.interpretation}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex items-center gap-4">
            <button
              className="btn-primary flex items-center gap-2 px-6 py-3 text-sm"
              onClick={() => {
                if (!selectedRun) return;
                window.open(`${API_BASE}/api/reports/${selectedRun}/pdf`, "_blank");
              }}
            >
              <Download size={16} /> Download PDF
            </button>
            <button
              className="btn-secondary flex items-center gap-2 px-6 py-3 text-sm"
              onClick={async () => {
                if (!selectedRun) return;
                try {
                  const res = await fetch(`${API_BASE}/api/reports/${selectedRun}/json`);
                  if (!res.ok) throw new Error("Report not found");
                  const blob = await res.blob();
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `${selectedRun}_attack_results.json`;
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                  URL.revokeObjectURL(url);
                } catch (err) {
                  console.error("Download failed:", err);
                }
              }}
            >
              <Code size={16} /> Export JSON
            </button>
            <button className="btn-secondary flex items-center gap-2 px-6 py-3 text-sm"><RefreshCw size={16} /> Re-run</button>
          </div>
        </>
      )}
    </div>
  );
}

export default function ResultsPage() {
  return (
    <>
      <DashboardNavbar />
      <Sidebar />
      <main className="lg:ml-[260px] pt-24 px-6 pb-12 relative min-h-screen">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full pointer-events-none" style={{ height: "600px", background: "radial-gradient(ellipse at top, rgba(255,255,255,0.03) 0%, transparent 70%)", zIndex: 0 }} />
        <Suspense fallback={<div className="relative z-10 max-w-5xl mx-auto p-12 text-center text-sm" style={{ color: "var(--text-muted)" }}>Loading...</div>}>
          <ResultsContent />
        </Suspense>
      </main>
    </>
  );
}
