"use client";

import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import DashboardNavbar from "@/components/layout/DashboardNavbar";
import Sidebar from "@/components/layout/Sidebar";
import { CheckCircle, Loader, Clock, Square } from "lucide-react";
import { getActiveRun, type ActiveRun } from "@/lib/api";

function StatusIcon({ status }: { status: string }) {
  if (status === "done") return <CheckCircle size={14} style={{ color: "var(--success)" }} />;
  if (status === "active")
    return <Loader size={14} className="animate-spin" style={{ color: "var(--text-secondary)" }} />;
  return <Clock size={14} style={{ color: "var(--text-muted)", opacity: 0.4 }} />;
}

export default function LiveMonitorPage() {
  const router = useRouter();
  const [run, setRun] = useState<ActiveRun>({ status: "idle" });
  const logRef = useRef<HTMLDivElement>(null);

  // Poll active run every 2 seconds
  useEffect(() => {
    let mounted = true;

    const poll = async () => {
      try {
        const data = await getActiveRun();
        if (mounted) setRun(data);

        // If completed, redirect to results after a brief delay
        if (data.status === "completed" && data.run_id) {
          setTimeout(() => {
            if (mounted) router.push(`/results?run=${data.run_id}`);
          }, 3000);
        }
      } catch {
        // Backend not running — stay on page
      }
    };

    poll();
    const interval = setInterval(poll, 2000);
    return () => { mounted = false; clearInterval(interval); };
  }, [router]);

  // Auto-scroll log
  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [run.logs]);

  const progress = run.progress || 0;
  const logs = run.logs || [];

  if (run.status === "idle") {
    return (
      <>
        <DashboardNavbar />
        <Sidebar />
        <main className="lg:ml-[260px] pt-24 px-6 pb-12 relative min-h-screen">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full pointer-events-none" style={{ height: "600px", background: "radial-gradient(ellipse at top, rgba(255,255,255,0.03) 0%, transparent 70%)", zIndex: 0 }} />
          <div className="relative z-10 max-w-5xl mx-auto">
            <div className="glass-card rounded-2xl p-12 text-center">
              <Clock size={32} style={{ color: "var(--text-muted)", margin: "0 auto 16px" }} />
              <h2 className="text-lg font-medium mb-2" style={{ color: "var(--text-primary)" }}>No Active Run</h2>
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                Start a new test from the wizard to see live progress here.
              </p>
            </div>
          </div>
        </main>
      </>
    );
  }

  return (
    <>
      <DashboardNavbar />
      <Sidebar />
      <main className="lg:ml-[260px] pt-24 px-6 pb-12 relative min-h-screen">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full pointer-events-none" style={{ height: "600px", background: "radial-gradient(ellipse at top, rgba(255,255,255,0.03) 0%, transparent 70%)", zIndex: 0 }} />

        <div className="relative z-10 max-w-5xl mx-auto space-y-8">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-3 mb-1">
                <h1 className="text-2xl font-medium" style={{ color: "var(--text-primary)" }}>
                  {run.model_name || "Test Run"}
                </h1>
                <span className="text-sm" style={{ color: "var(--text-muted)" }}>·</span>
                <span className="text-sm" style={{ color: "var(--text-secondary)" }}>{run.model_type}</span>
                <span className="text-sm" style={{ color: "var(--text-muted)" }}>·</span>
                <div className="flex items-center gap-1.5">
                  {run.status === "running" ? (
                    <>
                      <div className="w-2 h-2 rounded-full bg-white animate-breathe" style={{ boxShadow: "0 0 8px rgba(255,255,255,0.4)" }} />
                      <span className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>Running</span>
                    </>
                  ) : run.status === "completed" ? (
                    <>
                      <div className="w-2 h-2 rounded-full" style={{ background: "var(--success)" }} />
                      <span className="text-sm" style={{ color: "var(--success)" }}>Completed</span>
                    </>
                  ) : (
                    <>
                      <div className="w-2 h-2 rounded-full" style={{ background: "#ef4444" }} />
                      <span className="text-sm" style={{ color: "#ef4444" }}>Failed</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Progress Bar */}
          <div>
            <div className="flex justify-between text-xs mb-2">
              <span style={{ color: "var(--text-muted)" }}>{run.stage || "Initializing"}</span>
              <span style={{ color: "var(--text-primary)" }}>{progress}%</span>
            </div>
            <div className="h-2 rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
              <motion.div className="h-full rounded-full" style={{ background: "#fff" }} animate={{ width: `${progress}%` }} transition={{ duration: 0.5 }} />
            </div>
          </div>

          {/* Current attack status */}
          {run.current_attack && (
            <div className="glass-card rounded-2xl p-6" style={{ borderLeft: "3px solid var(--red-team)" }}>
              <h3 className="text-xs font-bold uppercase tracking-widest mb-3" style={{ color: "var(--red-team)" }}>Current Attack</h3>
              <div className="flex items-center gap-3">
                <Loader size={16} className="animate-spin" style={{ color: "var(--text-secondary)" }} />
                <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>{run.current_attack}</span>
              </div>
            </div>
          )}

          {/* Attacks list */}
          {run.attacks && run.attacks.length > 0 && (
            <div className="glass-card rounded-2xl p-6">
              <h3 className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: "var(--text-muted)" }}>Attacks In Queue</h3>
              <div className="flex flex-wrap gap-2">
                {run.attacks.map(a => (
                  <span key={a} className="px-3 py-1.5 rounded-full text-xs font-medium" style={{
                    background: a === run.current_attack ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.03)",
                    border: a === run.current_attack ? "1px solid rgba(255,255,255,0.2)" : "1px solid rgba(255,255,255,0.06)",
                    color: a === run.current_attack ? "var(--text-primary)" : "var(--text-muted)",
                  }}>
                    {a}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Live Output */}
          <div className="glass-card rounded-2xl p-6">
            <h3 className="text-sm font-medium mb-4" style={{ color: "var(--text-primary)" }}>Live Output</h3>
            <div ref={logRef} className="h-64 overflow-y-auto rounded-lg p-4 space-y-1" style={{ background: "rgba(0,0,0,0.3)", fontFamily: "var(--font-jetbrains), monospace", fontSize: "12px" }}>
              {logs.map((log, i) => (
                <div key={i} className="flex gap-3">
                  <span style={{ color: "var(--text-muted)" }}>{log.time}</span>
                  <span style={{ color: log.msg.startsWith("✓") ? "var(--success)" : log.msg.startsWith("✗") ? "#ef4444" : "var(--text-secondary)" }}>{log.msg}</span>
                </div>
              ))}
              {run.status === "running" && (
                <div className="flex items-center gap-1 mt-1">
                  <div className="w-1.5 h-4 bg-white animate-pulse" />
                </div>
              )}
            </div>
          </div>

          {/* Completed → link to results */}
          {run.status === "completed" && run.run_id && (
            <div className="glass-card rounded-2xl p-6 text-center">
              <CheckCircle size={24} style={{ color: "var(--success)", margin: "0 auto 12px" }} />
              <p className="text-sm mb-3" style={{ color: "var(--text-primary)" }}>Test completed! Redirecting to results...</p>
            </div>
          )}
        </div>
      </main>
    </>
  );
}
