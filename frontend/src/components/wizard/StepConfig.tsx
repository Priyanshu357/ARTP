"use client";

import { useState, useRef } from "react";
import { Sliders, Shield, Cpu, FileText, Upload, CheckCircle, Loader, Volume2 } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ConfigStepData {
  epsilon: string;
  batchSize: string;
  maxBatches: string;
  enableDetection: boolean;
  gpuAcceleration: boolean;
  datasetPath: string;
  datasetFileName: string;
  targetSnr: string;
}

interface StepConfigProps {
  data: ConfigStepData;
  onChange: (data: ConfigStepData) => void;
  modelType?: string;
}

export default function StepConfig({ data, onChange, modelType = "auto" }: StepConfigProps) {
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const dataRef = useRef<HTMLInputElement>(null);

  const showDataset = modelType === "nlp" || modelType === "audio" || modelType === "auto";
  const showSnr = modelType === "audio";
  const showEpsilon = modelType !== "nlp";

  const handleDatasetUpload = async (file: File) => {
    setUploading(true);
    setUploadError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      // Upload to a datasets directory on the server
      const res = await fetch(`${API_BASE}/api/datasets/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        // If the endpoint doesn't exist yet, allow manual path entry
        throw new Error("Dataset upload not available — enter the path manually below");
      }
      const result = await res.json();
      onChange({ ...data, datasetPath: result.path, datasetFileName: file.name });
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
      // Fall back — just set filename for manual path
      onChange({ ...data, datasetFileName: file.name });
    } finally {
      setUploading(false);
    }
  };

  const inputStyle = {
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.1)",
    color: "var(--text-primary)",
    fontFamily: "var(--font-jetbrains), monospace",
  };

  return (
    <div className="space-y-8">
      {/* Run Configuration */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
          Run Configuration
        </h3>
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
          Fine-tune parameters for the adversarial robustness test.
        </p>
      </div>

      {/* Parameter Inputs */}
      <div className="grid grid-cols-2 gap-6">
        {showEpsilon && (
          <div className="space-y-2">
            <label
              className="font-bold uppercase text-xs tracking-widest flex items-center gap-2"
              style={{ color: "var(--text-muted)", letterSpacing: "0.1em" }}
            >
              <Sliders size={14} />
              Epsilon (ε)
            </label>
            <input
              type="text"
              placeholder="0.03"
              value={data.epsilon}
              onChange={(e) => onChange({ ...data, epsilon: e.target.value })}
              className="w-full px-4 py-3 rounded-lg text-sm outline-none transition-all"
              style={inputStyle}
              onFocus={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.3)")}
              onBlur={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.1)")}
            />
            <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
              Perturbation budget. Lower = more constrained.
            </p>
          </div>
        )}

        <div className="space-y-2">
          <label
            className="font-bold uppercase text-xs tracking-widest flex items-center gap-2"
            style={{ color: "var(--text-muted)", letterSpacing: "0.1em" }}
          >
            <Cpu size={14} />
            Batch Size
          </label>
          <input
            type="text"
            placeholder={modelType === "nlp" || modelType === "audio" ? "4" : "16"}
            value={data.batchSize}
            onChange={(e) => onChange({ ...data, batchSize: e.target.value })}
            className="w-full px-4 py-3 rounded-lg text-sm outline-none transition-all"
            style={inputStyle}
            onFocus={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.3)")}
            onBlur={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.1)")}
          />
          <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
            Samples per batch during evaluation.
          </p>
        </div>

        <div className="space-y-2">
          <label
            className="font-bold uppercase text-xs tracking-widest flex items-center gap-2"
            style={{ color: "var(--text-muted)", letterSpacing: "0.1em" }}
          >
            <Sliders size={14} />
            Max Batches
          </label>
          <input
            type="text"
            placeholder="3"
            value={data.maxBatches}
            onChange={(e) => onChange({ ...data, maxBatches: e.target.value })}
            className="w-full px-4 py-3 rounded-lg text-sm outline-none transition-all"
            style={inputStyle}
            onFocus={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.3)")}
            onBlur={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.1)")}
          />
          <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
            Limits scope of test. Use -1 for all batches.
          </p>
        </div>

        {showSnr && (
          <div className="space-y-2">
            <label
              className="font-bold uppercase text-xs tracking-widest flex items-center gap-2"
              style={{ color: "var(--text-muted)", letterSpacing: "0.1em" }}
            >
              <Volume2 size={14} />
              Target SNR (dB)
            </label>
            <input
              type="text"
              placeholder="20.0"
              value={data.targetSnr}
              onChange={(e) => onChange({ ...data, targetSnr: e.target.value })}
              className="w-full px-4 py-3 rounded-lg text-sm outline-none transition-all"
              style={inputStyle}
              onFocus={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.3)")}
              onBlur={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.1)")}
            />
            <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
              Signal-to-Noise Ratio for audio noise injection.
            </p>
          </div>
        )}
      </div>

      {/* ── Dataset Upload (NLP / Audio) ─────────────── */}
      {showDataset && (
        <div
          className="space-y-4 p-5 rounded-xl"
          style={{
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.08)",
          }}
        >
          <label
            className="font-bold uppercase text-xs tracking-widest flex items-center gap-2"
            style={{ color: "var(--text-muted)", letterSpacing: "0.1em" }}
          >
            <FileText size={14} />
            Test Dataset {modelType === "nlp" ? "(CSV / JSON)" : modelType === "audio" ? "(Audio Directory)" : "(Optional)"}
          </label>

          <div
            className="rounded-xl cursor-pointer transition-all flex flex-col items-center justify-center py-8 gap-3"
            style={{
              border: data.datasetPath
                ? "2px solid rgba(74,222,128,0.4)"
                : "2px dashed rgba(255,255,255,0.12)",
              background: data.datasetPath ? "rgba(74,222,128,0.04)" : "transparent",
            }}
            onClick={() => !uploading && dataRef.current?.click()}
          >
            <input
              ref={dataRef}
              type="file"
              accept={modelType === "audio" ? ".wav,.zip" : ".csv,.json,.txt"}
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleDatasetUpload(file);
              }}
            />
            {uploading ? (
              <Loader size={20} className="animate-spin" style={{ color: "var(--text-secondary)" }} />
            ) : data.datasetPath ? (
              <>
                <CheckCircle size={20} style={{ color: "#4ade80" }} />
                <span className="text-sm font-medium text-white">{data.datasetFileName}</span>
              </>
            ) : (
              <>
                <Upload size={20} style={{ color: "var(--text-muted)" }} />
                <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                  Click to upload or enter path below
                </span>
              </>
            )}
          </div>
          {uploadError && (
            <p className="text-xs" style={{ color: "#facc15" }}>⚠ {uploadError}</p>
          )}

          {/* Manual path fallback */}
          <input
            type="text"
            placeholder={modelType === "audio" ? "Or enter path: datasets/audio/" : "Or enter path: datasets/validation_20.csv"}
            value={data.datasetPath}
            onChange={(e) => onChange({ ...data, datasetPath: e.target.value })}
            className="w-full px-4 py-3 rounded-lg text-sm outline-none"
            style={{
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: "var(--text-primary)",
            }}
          />
          <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
            {modelType === "nlp"
              ? "CSV or JSON file with text and label columns. Leave blank to use default sample data."
              : "Directory path containing .wav files. Leave blank for synthetic test audio."}
          </p>
        </div>
      )}

      {/* Toggle Options */}
      <div className="space-y-4">
        <label
          className="font-bold uppercase text-xs tracking-widest"
          style={{ color: "var(--text-muted)", letterSpacing: "0.1em" }}
        >
          Options
        </label>

        {/* Detection Toggle */}
        <div
          className="flex items-center justify-between p-4 rounded-xl transition-all"
          style={{
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          <div className="flex items-center gap-3">
            <Shield size={18} style={{ color: "var(--text-muted)" }} />
            <div>
              <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                Enable Detection
              </p>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                Run adversarial example detectors alongside attacks
              </p>
            </div>
          </div>
          <button
            className="w-11 h-6 rounded-full transition-all relative"
            style={{
              background: data.enableDetection ? "#fff" : "rgba(255,255,255,0.15)",
            }}
            onClick={() => onChange({ ...data, enableDetection: !data.enableDetection })}
          >
            <div
              className="w-4 h-4 rounded-full absolute top-1 transition-all"
              style={{
                background: data.enableDetection ? "#0A0A0A" : "rgba(255,255,255,0.5)",
                left: data.enableDetection ? "24px" : "4px",
              }}
            />
          </button>
        </div>

        {/* GPU Toggle */}
        <div
          className="flex items-center justify-between p-4 rounded-xl transition-all"
          style={{
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          <div className="flex items-center gap-3">
            <Cpu size={18} style={{ color: "var(--text-muted)" }} />
            <div>
              <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                GPU Acceleration
              </p>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                Use CUDA-enabled GPU for faster inference
              </p>
            </div>
          </div>
          <button
            className="w-11 h-6 rounded-full transition-all relative"
            style={{
              background: data.gpuAcceleration ? "#fff" : "rgba(255,255,255,0.15)",
            }}
            onClick={() => onChange({ ...data, gpuAcceleration: !data.gpuAcceleration })}
          >
            <div
              className="w-4 h-4 rounded-full absolute top-1 transition-all"
              style={{
                background: data.gpuAcceleration ? "#0A0A0A" : "rgba(255,255,255,0.5)",
                left: data.gpuAcceleration ? "24px" : "4px",
              }}
            />
          </button>
        </div>
      </div>

      {/* Footer badges */}
      <div className="flex gap-8 pt-4">
        {["Encrypted Pipeline", "GPU Accelerated", "Auto-Versioning"].map((badge) => (
          <div key={badge} className="flex items-center gap-2">
            <Shield size={12} style={{ color: "var(--text-muted)", opacity: 0.5 }} />
            <span
              className="font-bold uppercase"
              style={{ fontSize: "10px", letterSpacing: "0.15em", color: "var(--text-muted)", opacity: 0.5 }}
            >
              {badge}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
