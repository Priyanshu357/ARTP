"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import DashboardNavbar from "@/components/layout/DashboardNavbar";
import Sidebar from "@/components/layout/Sidebar";
import Link from "next/link";
import { Box, Image, Brain, Volume2, Cpu } from "lucide-react";
import { getModels, type ModelInfo } from "@/lib/api";

function getIcon(modality: string) {
  const size = 20;
  const style = { color: "var(--text-muted)" };
  switch (modality) {
    case "vision": return <Image size={size} style={style} />;
    case "nlp": return <Brain size={size} style={style} />;
    case "audio": return <Volume2 size={size} style={style} />;
    default: return <Cpu size={size} style={style} />;
  }
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModels()
      .then((data) => setModels(data.models))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <>
      <DashboardNavbar />
      <Sidebar />
      <main className="lg:ml-[260px] pt-24 px-6 pb-12 relative min-h-screen">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full pointer-events-none" style={{ height: "600px", background: "radial-gradient(ellipse at top, rgba(255,255,255,0.03) 0%, transparent 70%)", zIndex: 0 }} />

        <div className="relative z-10 max-w-5xl mx-auto space-y-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-medium" style={{ color: "var(--text-primary)" }}>Models</h1>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                {loading ? "Loading..." : `${models.length} models registered in your library.`}
              </p>
            </div>
            <Link href="/runs/new" className="btn-primary flex items-center gap-2 px-5 py-2.5 text-sm font-semibold">
              + Upload Model
            </Link>
          </div>

          {!loading && models.length === 0 && (
            <div className="glass-card rounded-2xl p-12 text-center">
              <Box size={32} style={{ color: "var(--text-muted)", margin: "0 auto 16px" }} />
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>No models found. Upload a model to get started.</p>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {models.map((model, i) => (
              <motion.div
                key={model.filename}
                className="glass-card rounded-2xl p-6 group cursor-default"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: i * 0.06, ease: [0.16, 1, 0.3, 1] }}
              >
                <div className="flex items-start gap-4 mb-4">
                  <div className="p-2.5 rounded-xl" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
                    {getIcon(model.modality)}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>{model.name}</h3>
                    <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                      {model.format.toUpperCase()} · {formatSize(model.size_bytes)} · {model.modality}
                    </p>
                  </div>
                </div>

                <div className="flex items-center justify-between mb-5">
                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>Best Robustness</p>
                    <p className="text-xl" style={{ fontWeight: 300, color: "var(--text-primary)" }}>
                      {model.best_score ?? "—"}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>Runs</p>
                    <p className="text-xl" style={{ fontWeight: 300, color: "var(--text-primary)" }}>{model.runs}</p>
                  </div>
                </div>

                <div className="flex gap-2">
                  <Link href="/runs/new" className="flex-1 text-center px-3 py-2 rounded-lg text-xs font-semibold transition-all" style={{ background: "#fff", color: "#0A0A0A" }}>
                    Test
                  </Link>
                  <Link href={`/results?run=${model.name.replace(/-/g, "_").toLowerCase()}`} className="flex-1 text-center px-3 py-2 rounded-lg text-xs font-medium transition-all" style={{ border: "1px solid rgba(255,255,255,0.1)", color: "var(--text-secondary)" }}>
                    Details →
                  </Link>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </main>
    </>
  );
}
