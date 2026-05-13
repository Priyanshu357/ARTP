"use client";

import { useState, useRef, useEffect } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Metrics {
  snr_db?: number | string;
  l2_norm?: number;
  linf_norm?: number;
  spectral_centroid_shift_hz?: number;
  original_centroid_hz?: number;
  adversarial_centroid_hz?: number;
  duration_s?: number;
  sample_rate?: number;
  rms?: number;
  peak_amplitude?: number;
  spectral_centroid_hz?: number;
  db_min?: number;
  db_max?: number;
  band_energies?: { low_0_500hz: number; mid_500_4000hz: number; high_4000hz_plus: number };
}

interface Spectrograms {
  original?: string | null;
  adversarial?: string | null;
  difference?: string | null;
}

interface SavedPair {
  id: string;
  original_url: string | null;
  adversarial_url: string | null;
  has_pair: boolean;
}

interface SavedAttack {
  attack_name: string;
  n_pairs: number;
  pairs: SavedPair[];
}

type Mode = "single" | "compare";

export default function AudioAnalysisPage() {
  const [mode, setMode] = useState<Mode>("compare");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Single file state
  const [singleFile, setSingleFile] = useState<File | null>(null);
  const [singleMetrics, setSingleMetrics] = useState<Metrics | null>(null);
  const [singleSpec, setSingleSpec] = useState<string | null>(null);

  // Compare state
  const [origFile, setOrigFile] = useState<File | null>(null);
  const [advFile, setAdvFile] = useState<File | null>(null);
  const [compareMetrics, setCompareMetrics] = useState<Metrics | null>(null);
  const [compareSpecs, setCompareSpecs] = useState<Spectrograms | null>(null);

  // Saved WAV pairs from backend
  const [savedAttacks, setSavedAttacks] = useState<SavedAttack[]>([]);
  const [savedLoading, setSavedLoading] = useState(false);
  const [selectedPairLoading, setSelectedPairLoading] = useState(false);

  const origRef = useRef<HTMLInputElement>(null);
  const advRef = useRef<HTMLInputElement>(null);
  const singleRef = useRef<HTMLInputElement>(null);

  // Fetch saved WAV samples on mount
  useEffect(() => {
    setSavedLoading(true);
    fetch(`${API}/api/audio/samples`)
      .then((r) => r.json())
      .then((d) => setSavedAttacks(d.attacks || []))
      .catch(() => setSavedAttacks([]))
      .finally(() => setSavedLoading(false));
  }, []);

  // Load a saved pair directly into the compare tool
  async function loadSavedPair(pair: SavedPair) {
    if (!pair.original_url || !pair.adversarial_url) return;
    setSelectedPairLoading(true);
    setError(null);
    try {
      const [origRes, advRes] = await Promise.all([
        fetch(`${API}${pair.original_url}`),
        fetch(`${API}${pair.adversarial_url}`),
      ]);
      if (!origRes.ok || !advRes.ok) throw new Error("Failed to download WAV files");
      const [origBlob, advBlob] = await Promise.all([origRes.blob(), advRes.blob()]);
      setOrigFile(new File([origBlob], `${pair.id}_original.wav`, { type: "audio/wav" }));
      setAdvFile(new File([advBlob], `${pair.id}_adversarial.wav`, { type: "audio/wav" }));
      setCompareMetrics(null);
      setCompareSpecs(null);
      setMode("compare");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setSelectedPairLoading(false);
    }
  }


  async function analyzeSingle() {
    if (!singleFile) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", singleFile);
      const res = await fetch(`${API}/api/audio/spectrogram`, { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      setSingleMetrics(data.metrics);
      setSingleSpec(data.spectrogram_png_b64);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function analyzeCompare() {
    if (!origFile || !advFile) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("original", origFile);
      form.append("adversarial", advFile);
      const res = await fetch(`${API}/api/audio/compare`, { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      setCompareMetrics(data.metrics);
      setCompareSpecs(data.spectrograms);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const snrColor = (v: number | string | undefined) => {
    if (v === undefined || v === null) return "#94a3b8";
    const n = typeof v === "string" ? Infinity : v;
    if (n === Infinity || n > 30) return "#22c55e";
    if (n > 15) return "#f59e0b";
    return "#ef4444";
  };

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <h1 style={styles.title}>🎵 Audio Spectrogram Analysis</h1>
        <p style={styles.subtitle}>
          Upload audio files to visualize spectrograms and compare adversarial perturbations
        </p>
        <div style={styles.modeToggle}>
          {(["single", "compare"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              style={{ ...styles.modeBtn, ...(mode === m ? styles.modeBtnActive : {}) }}
            >
              {m === "single" ? "📊 Single File" : "🔀 Compare Original vs Adversarial"}
            </button>
          ))}
        </div>
      </div>

      {/* ── Saved Samples Panel ────────────────────────────────────────────── */}
      {(savedAttacks.length > 0 || savedLoading) && (
        <div style={{ ...styles.card, borderColor: "#1e3a5f" }}>
          <h2 style={styles.cardTitle}>
            📁 Load from Last Run
            <span style={{ fontSize: 12, fontWeight: 400, color: "#64748b", marginLeft: 10 }}>
              WAV files saved by <code>--save-audio</code>
            </span>
          </h2>
          {savedLoading ? (
            <p style={{ color: "#64748b", fontSize: 14 }}>Loading saved samples…</p>
          ) : (
            savedAttacks.map((attack) => (
              <div key={attack.attack_name} style={{ marginBottom: 20 }}>
                <p style={{ fontSize: 13, fontWeight: 700, color: "#818cf8", marginBottom: 8 }}>
                  ⚡ {attack.attack_name} — {attack.n_pairs} pair{attack.n_pairs !== 1 ? "s" : ""}
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {attack.pairs.filter((p) => p.has_pair).map((pair) => (
                    <button
                      key={pair.id}
                      onClick={() => loadSavedPair(pair)}
                      disabled={selectedPairLoading}
                      style={styles.pairBtn}
                      title={`Load ${pair.id} into compare tool`}
                    >
                      {selectedPairLoading ? "⏳" : "🎵"} {pair.id}
                    </button>
                  ))}
                </div>
              </div>
            ))
          )}
          <p style={{ fontSize: 12, color: "#475569", marginTop: 4 }}>
            💡 Click any sample to auto-load it into the Compare tool below
          </p>
        </div>
      )}

      {/* No saved samples — show instructions */}
      {!savedLoading && savedAttacks.length === 0 && (
        <div style={{ ...styles.card, borderColor: "#334155", textAlign: "center", padding: "24px 32px" }}>
          <p style={{ color: "#64748b", fontSize: 14, margin: 0 }}>
            📭 <b>No saved WAV files yet.</b> Run the pipeline with <code style={{ background: "#1e293b", padding: "2px 6px", borderRadius: 4, color: "#818cf8" }}>--save-audio</code> to generate them:
          </p>
          <pre style={{ background: "#0a0f1e", border: "1px solid #1e293b", borderRadius: 8, padding: "12px 16px", fontSize: 12, color: "#94a3b8", marginTop: 12, textAlign: "left", overflowX: "auto" }}>
{`python main_audio.py \\
  --attacks noise psychoacoustic reverb \\
  --save-audio \\
  --output reports/`}
          </pre>
        </div>
      )}

      {mode === "single" && (
        <div style={styles.card}>
          <h2 style={styles.cardTitle}>Analyze Single Audio File</h2>
          <DropZone
            label="Drop WAV file here or click to browse"
            file={singleFile}
            inputRef={singleRef}
            onChange={(f) => { setSingleFile(f); setSingleMetrics(null); setSingleSpec(null); }}
            accept=".wav,.mp3,.flac"
          />
          <button
            style={{ ...styles.btn, opacity: (!singleFile || loading) ? 0.5 : 1 }}
            disabled={!singleFile || loading}
            onClick={analyzeSingle}
          >
            {loading ? "Analyzing…" : "Generate Spectrogram"}
          </button>

          {singleSpec && (
            <div style={styles.specBox}>
              <h3 style={styles.sectionLabel}>Spectrogram</h3>
              <img src={`data:image/png;base64,${singleSpec}`} style={styles.specImg} alt="spectrogram" />
            </div>
          )}
          {singleMetrics && <MetricGrid metrics={singleMetrics} snrColor={snrColor} />}
        </div>
      )}

      {/* Compare Mode */}
      {mode === "compare" && (
        <div style={styles.card}>
          <h2 style={styles.cardTitle}>Compare Original vs Adversarial Audio</h2>
          <div style={styles.row}>
            <div style={styles.half}>
              <p style={styles.dropLabel}>🟢 Original Audio</p>
              <DropZone
                label="Drop original WAV"
                file={origFile}
                inputRef={origRef}
                onChange={(f) => { setOrigFile(f); setCompareMetrics(null); setCompareSpecs(null); }}
                accept=".wav,.mp3,.flac"
                color="#22c55e"
              />
            </div>
            <div style={styles.half}>
              <p style={styles.dropLabel}>🔴 Adversarial Audio</p>
              <DropZone
                label="Drop adversarial WAV"
                file={advFile}
                inputRef={advRef}
                onChange={(f) => { setAdvFile(f); setCompareMetrics(null); setCompareSpecs(null); }}
                accept=".wav,.mp3,.flac"
                color="#ef4444"
              />
            </div>
          </div>

          <button
            style={{ ...styles.btn, opacity: (!origFile || !advFile || loading) ? 0.5 : 1 }}
            disabled={!origFile || !advFile || loading}
            onClick={analyzeCompare}
          >
            {loading ? "Computing…" : "Compare Spectrograms"}
          </button>

          {compareSpecs && (
            <div>
              <div style={styles.specGrid}>
                {(["original", "adversarial", "difference"] as const).map((key) => (
                  compareSpecs[key] ? (
                    <div key={key} style={styles.specCell}>
                      <p style={styles.specLabel}>
                        {key === "original" ? "🟢 Original" : key === "adversarial" ? "🔴 Adversarial" : "📊 Difference (Adv − Orig)"}
                      </p>
                      <img src={`data:image/png;base64,${compareSpecs[key]}`} style={styles.specImg} alt={key} />
                    </div>
                  ) : null
                ))}
              </div>
              {compareMetrics && (
                <div style={styles.metricsPanel}>
                  <h3 style={styles.sectionLabel}>Attack Quality Metrics</h3>
                  <div style={styles.metricRow}>
                    <MetricChip label="SNR" value={compareMetrics.snr_db !== undefined ? `${compareMetrics.snr_db} dB` : "—"} color={snrColor(compareMetrics.snr_db)} />
                    <MetricChip label="L2 Norm" value={compareMetrics.l2_norm?.toFixed(4) ?? "—"} color="#818cf8" />
                    <MetricChip label="L∞ Norm" value={compareMetrics.linf_norm?.toFixed(6) ?? "—"} color="#c084fc" />
                    <MetricChip label="Centroid Shift" value={`${compareMetrics.spectral_centroid_shift_hz ?? "—"} Hz`} color="#38bdf8" />
                    <MetricChip label="Duration" value={`${compareMetrics.duration_s ?? "—"} s`} color="#94a3b8" />
                    <MetricChip label="Sample Rate" value={`${compareMetrics.sample_rate ?? "—"} Hz`} color="#94a3b8" />
                  </div>
                  <div style={styles.centroidDetail}>
                    <span>Orig centroid: <b>{compareMetrics.original_centroid_hz} Hz</b></span>
                    <span style={{ margin: "0 12px", color: "#64748b" }}>→</span>
                    <span>Adv centroid: <b style={{ color: "#ef4444" }}>{compareMetrics.adversarial_centroid_hz} Hz</b></span>
                  </div>
                  <SnrInterpretation snr={compareMetrics.snr_db} />
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {error && (
        <div style={styles.errorBox}>
          <span>⚠️ {error}</span>
        </div>
      )}
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function DropZone({ label, file, inputRef, onChange, accept, color = "#6366f1" }: {
  label: string; file: File | null; inputRef: React.RefObject<HTMLInputElement>;
  onChange: (f: File) => void; accept: string; color?: string;
}) {
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) onChange(f);
  };
  return (
    <div
      style={{ ...styles.dropZone, borderColor: file ? color : "#334155" }}
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={() => inputRef.current?.click()}
    >
      <input ref={inputRef} type="file" accept={accept} style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) onChange(f); }} />
      {file ? (
        <div>
          <div style={{ fontSize: 28 }}>🎵</div>
          <p style={{ color, fontWeight: 600, margin: "4px 0" }}>{file.name}</p>
          <p style={{ color: "#64748b", fontSize: 12 }}>{(file.size / 1024).toFixed(1)} KB</p>
        </div>
      ) : (
        <div>
          <div style={{ fontSize: 28 }}>📂</div>
          <p style={{ color: "#64748b", margin: "4px 0" }}>{label}</p>
          <p style={{ color: "#475569", fontSize: 12 }}>WAV, MP3, FLAC</p>
        </div>
      )}
    </div>
  );
}

function MetricGrid({ metrics, snrColor }: { metrics: Metrics; snrColor: (v: any) => string }) {
  return (
    <div style={styles.metricsPanel}>
      <h3 style={styles.sectionLabel}>Audio Metrics</h3>
      <div style={styles.metricRow}>
        <MetricChip label="RMS Energy" value={metrics.rms?.toFixed(4) ?? "—"} color="#22c55e" />
        <MetricChip label="Peak Amp" value={metrics.peak_amplitude?.toFixed(4) ?? "—"} color="#f59e0b" />
        <MetricChip label="Centroid" value={`${metrics.spectral_centroid_hz?.toFixed(0) ?? "—"} Hz`} color="#38bdf8" />
        <MetricChip label="dB Range" value={`${metrics.db_min?.toFixed(0)}–${metrics.db_max?.toFixed(0)} dB`} color="#c084fc" />
      </div>
      {metrics.band_energies && (
        <div style={styles.bandBar}>
          <BandBar label="Low (0–500 Hz)" value={metrics.band_energies.low_0_500hz} color="#22c55e" />
          <BandBar label="Mid (500–4k Hz)" value={metrics.band_energies.mid_500_4000hz} color="#f59e0b" />
          <BandBar label="High (4k+ Hz)" value={metrics.band_energies.high_4000hz_plus} color="#ef4444" />
        </div>
      )}
    </div>
  );
}

function MetricChip({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ ...styles.chip, borderColor: color }}>
      <span style={{ color: "#94a3b8", fontSize: 11 }}>{label}</span>
      <span style={{ color, fontWeight: 700, fontSize: 16 }}>{value}</span>
    </div>
  );
}

function BandBar({ label, value, color }: { label: string; value: number; color: string }) {
  const max = 0.01; // rough max for normalization
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#94a3b8", marginBottom: 2 }}>
        <span>{label}</span><span>{value.toExponential(2)}</span>
      </div>
      <div style={{ height: 6, background: "#1e293b", borderRadius: 3 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 3, transition: "width 0.6s" }} />
      </div>
    </div>
  );
}

function SnrInterpretation({ snr }: { snr?: number | string }) {
  const n = typeof snr === "string" ? Infinity : (snr ?? Infinity);
  let msg = "", color = "#94a3b8";
  if (n === Infinity || n > 40) { msg = "✅ Perturbation is nearly imperceptible (SNR > 40 dB). Excellent attack stealth."; color = "#22c55e"; }
  else if (n > 25) { msg = "🟡 Perturbation is subtle (SNR 25–40 dB). May be barely noticeable to careful listening."; color = "#f59e0b"; }
  else if (n > 10) { msg = "🟠 Perturbation is noticeable (SNR 10–25 dB). Environmental-class attack."; color = "#f97316"; }
  else { msg = "🔴 High perturbation (SNR < 10 dB). Audio quality significantly degraded."; color = "#ef4444"; }
  return <div style={{ ...styles.interpretation, borderColor: color, color }}>{msg}</div>;
}

// ── Styles ─────────────────────────────────────────────────────────────────────

const styles: Record<string, React.CSSProperties> = {
  page: { minHeight: "100vh", background: "#0a0f1e", padding: "32px 24px", fontFamily: "'Inter', sans-serif", color: "#e2e8f0" },
  header: { maxWidth: 1100, margin: "0 auto 32px", textAlign: "center" },
  title: { fontSize: 32, fontWeight: 800, background: "linear-gradient(135deg,#6366f1,#a855f7,#06b6d4)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", margin: "0 0 8px" },
  subtitle: { color: "#64748b", fontSize: 15, margin: "0 0 20px" },
  modeToggle: { display: "flex", justifyContent: "center", gap: 12 },
  modeBtn: { padding: "8px 20px", borderRadius: 8, border: "1px solid #334155", background: "#0f172a", color: "#94a3b8", cursor: "pointer", fontSize: 14, transition: "all 0.2s" },
  modeBtnActive: { borderColor: "#6366f1", background: "rgba(99,102,241,0.15)", color: "#818cf8" },
  card: { maxWidth: 1100, margin: "0 auto 24px", background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, padding: 32 },
  cardTitle: { fontSize: 20, fontWeight: 700, marginBottom: 24, color: "#e2e8f0" },
  dropZone: { border: "2px dashed #334155", borderRadius: 12, padding: 32, textAlign: "center", cursor: "pointer", transition: "border-color 0.2s", marginBottom: 20 },
  dropLabel: { fontWeight: 600, color: "#94a3b8", marginBottom: 8, fontSize: 14 },
  row: { display: "flex", gap: 20, marginBottom: 4 },
  half: { flex: 1 },
  btn: { display: "block", width: "100%", padding: "14px", borderRadius: 10, border: "none", background: "linear-gradient(135deg,#6366f1,#8b5cf6)", color: "#fff", fontSize: 15, fontWeight: 700, cursor: "pointer", marginBottom: 28, transition: "opacity 0.2s" },
  specBox: { marginTop: 8 },
  specGrid: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginTop: 8 },
  specCell: { background: "#0a0f1e", borderRadius: 10, padding: 12, border: "1px solid #1e293b" },
  specLabel: { fontSize: 13, fontWeight: 600, color: "#94a3b8", margin: "0 0 8px" },
  specImg: { width: "100%", borderRadius: 6, display: "block" },
  sectionLabel: { fontSize: 15, fontWeight: 700, color: "#94a3b8", marginBottom: 14 },
  metricsPanel: { marginTop: 24, background: "#0a0f1e", borderRadius: 12, padding: 20, border: "1px solid #1e293b" },
  metricRow: { display: "flex", flexWrap: "wrap" as const, gap: 12, marginBottom: 16 },
  chip: { flex: "1 1 140px", background: "#0f172a", border: "1px solid #334155", borderRadius: 10, padding: "12px 14px", display: "flex", flexDirection: "column" as const, gap: 4 },
  bandBar: { marginTop: 8 },
  centroidDetail: { fontSize: 13, color: "#94a3b8", marginTop: 8, marginBottom: 12 },
  interpretation: { fontSize: 13, padding: "10px 14px", borderRadius: 8, border: "1px solid", marginTop: 12 },
  errorBox: { maxWidth: 1100, margin: "0 auto", background: "rgba(239,68,68,0.1)", border: "1px solid #ef4444", borderRadius: 10, padding: "12px 20px", color: "#ef4444", fontSize: 14 },
  pairBtn: { padding: "6px 14px", borderRadius: 8, border: "1px solid #334155", background: "#0f172a", color: "#818cf8", cursor: "pointer", fontSize: 13, fontWeight: 600, transition: "all 0.15s" },
};

