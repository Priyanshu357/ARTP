"use client";

import { useState, useRef } from "react";
import { Upload, CheckCircle, Loader, Globe, HardDrive } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const MODEL_TYPES = [
  { id: "auto", label: "Automated" },
  { id: "image", label: "Computer Vision" },
  { id: "nlp", label: "NLP / LLM" },
  { id: "audio", label: "Audio" },
];

export interface ModelStepData {
  modelName: string;
  version: string;
  modelType: string;
  fileName: string;
  modelPath: string;
  sourceMode: "upload" | "huggingface";
  huggingfaceId: string;
  tokenizerName: string;
  labelMapping: string;
}

interface StepModelProps {
  data: ModelStepData;
  onChange: (data: ModelStepData) => void;
}

export default function StepModel({ data, onChange }: StepModelProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const isNlpOnnx =
    data.sourceMode === "upload" &&
    (data.modelType === "nlp" || data.modelType === "auto") &&
    data.fileName.endsWith(".onnx");

  const uploadFile = async (file: File) => {
    setUploading(true);
    setUploadError("");
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/api/models/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Upload failed");
      }

      const result = await res.json();
      const detectedType = result.detected_type || "auto";
      const isTfidf = file.name.toLowerCase().includes("tfidf");
      onChange({
        ...data,
        fileName: file.name,
        modelPath: result.path,
        modelName: data.modelName || file.name.replace(/\.[^/.]+$/, ""),
        modelType: detectedType,
        // Auto-fill NLP defaults when NLP model is detected
        tokenizerName: detectedType === "nlp" ? (isTfidf ? "" : "distilbert-base-uncased") : "",
        labelMapping: detectedType === "nlp" ? '{"0":"NEGATIVE","1":"POSITIVE"}' : "",
      });
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
      onChange({ ...data, fileName: "", modelPath: "" });
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
  };

  const inputStyle = {
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.1)",
    color: "var(--text-primary)",
  };

  return (
    <div className="space-y-8">
      {/* ── Model Source Toggle ─────────────────────── */}
      <div className="space-y-4">
        <label
          className="font-bold uppercase text-xs tracking-widest"
          style={{ color: "var(--text-muted)", letterSpacing: "0.1em" }}
        >
          Model Source
        </label>
        <div className="flex gap-3">
          {[
            { id: "upload" as const, label: "Upload File", icon: <HardDrive size={16} /> },
            { id: "huggingface" as const, label: "HuggingFace ID", icon: <Globe size={16} /> },
          ].map((opt) => {
            const active = data.sourceMode === opt.id;
            return (
              <button
                key={opt.id}
                className="flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-medium transition-all"
                style={{
                  background: active ? "#fff" : "rgba(255,255,255,0.05)",
                  color: active ? "#000" : "var(--text-muted)",
                  border: active
                    ? "1px solid #fff"
                    : "1px solid rgba(255,255,255,0.1)",
                }}
                onClick={() =>
                  onChange({
                    ...data,
                    sourceMode: opt.id,
                    fileName: "",
                    modelPath: "",
                    huggingfaceId: "",
                  })
                }
              >
                {opt.icon} {opt.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Upload Zone (only when sourceMode=upload) ─── */}
      {data.sourceMode === "upload" && (
        <div className="space-y-4">
          <div className="flex items-baseline justify-between">
            <label
              className="font-bold uppercase text-xs tracking-widest"
              style={{ color: "var(--text-muted)", letterSpacing: "0.1em" }}
            >
              Model File
            </label>
            <span className="text-xs text-green-400">
              Tip: Upload a .zip if your ONNX model has external .data weights
            </span>
          </div>
          <div
            className="rounded-xl cursor-pointer transition-all flex flex-col items-center justify-center py-12 gap-4"
            style={{
              border: isDragOver
                ? "2px dashed rgba(255,255,255,0.4)"
                : data.modelPath
                ? "2px solid rgba(74,222,128,0.4)"
                : "2px dashed rgba(255,255,255,0.12)",
              background: isDragOver
                ? "rgba(255,255,255,0.04)"
                : data.modelPath
                ? "rgba(74,222,128,0.04)"
                : "transparent",
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => !uploading && fileRef.current?.click()}
          >
            <input
              ref={fileRef}
              type="file"
              accept=".onnx,.pth,.bin,.h5,.pt,.zip"
              className="hidden"
              onChange={handleFileSelect}
            />
            {uploading ? (
              <>
                <Loader
                  size={28}
                  className="animate-spin"
                  style={{ color: "var(--text-secondary)" }}
                />
                <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                  Uploading…
                </span>
              </>
            ) : data.modelPath ? (
              <>
                <CheckCircle size={28} style={{ color: "#4ade80" }} />
                <span className="text-sm font-medium text-white">{data.fileName}</span>
                <span className="text-xs" style={{ color: "#4ade80" }}>
                  Saved to {data.modelPath}
                </span>
              </>
            ) : (
              <>
                <Upload size={28} style={{ color: "var(--text-muted)" }} />
                <span className="text-sm font-medium" style={{ color: "#d4d4d8" }}>
                  Click to upload or drag and drop
                </span>
                <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                  Supported: .onnx, .pth, .bin, .h5, .pt (Max 2 GB)
                </span>
              </>
            )}
          </div>
          {uploadError && (
            <p className="text-xs" style={{ color: "#ef4444" }}>
              ✗ {uploadError}
            </p>
          )}
        </div>
      )}

      {/* ── HuggingFace ID (only when sourceMode=huggingface) ─── */}
      {data.sourceMode === "huggingface" && (
        <div className="space-y-4">
          <label
            className="font-bold uppercase text-xs tracking-widest"
            style={{ color: "var(--text-muted)", letterSpacing: "0.1em" }}
          >
            HuggingFace Model ID
          </label>
          <input
            type="text"
            placeholder="e.g.  distilbert-base-uncased-finetuned-sst-2-english"
            value={data.huggingfaceId}
            onChange={(e) =>
              onChange({
                ...data,
                huggingfaceId: e.target.value,
                modelName: data.modelName || e.target.value.split("/").pop() || "",
                tokenizerName: e.target.value, // Auto-fill tokenizer with HF model ID
              })
            }
            className="w-full px-4 py-3 rounded-lg text-sm outline-none transition-all"
            style={inputStyle}
            onFocus={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.3)")}
            onBlur={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.1)")}
          />
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>
            Any public HuggingFace model name. The platform will download it automatically.
          </p>
        </div>
      )}

      {/* ── Architecture Type ─────────────────────── */}
      <div className="space-y-4">
        <label
          className="font-bold uppercase text-xs tracking-widest"
          style={{ color: "var(--text-muted)", letterSpacing: "0.1em" }}
        >
          Architecture Type
        </label>
        <div className="flex flex-wrap gap-3">
          {MODEL_TYPES.map((t) => {
            const isSelected = data.modelType === t.id;
            return (
              <button
                key={t.id}
                className="px-6 py-2.5 rounded-full text-sm font-medium transition-all"
                style={{
                  background: isSelected ? "#fff" : "rgba(255,255,255,0.05)",
                  color: isSelected ? "#000" : "var(--text-muted)",
                  border: isSelected
                    ? "1px solid #fff"
                    : "1px solid rgba(255,255,255,0.1)",
                  boxShadow: isSelected ? "0 0 12px rgba(255,255,255,0.1)" : "none",
                }}
                onClick={() => {
                  const updates: Partial<ModelStepData> = { modelType: t.id };
                  // Auto-fill tokenizer defaults when switching to NLP
                  if (t.id === "nlp") {
                    if (data.sourceMode === "huggingface" && data.huggingfaceId) {
                      updates.tokenizerName = data.huggingfaceId;
                    } else if (!data.tokenizerName) {
                      updates.tokenizerName = data.fileName.toLowerCase().includes("tfidf") ? "" : "distilbert-base-uncased";
                    }
                    if (!data.labelMapping) {
                      updates.labelMapping = '{"0":"NEGATIVE","1":"POSITIVE"}';
                    }
                  } else {
                    // Clear NLP-specific fields when switching away
                    updates.tokenizerName = "";
                    updates.labelMapping = "";
                  }
                  onChange({ ...data, ...updates });
                }}
              >
                {t.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* ── NLP-specific: Tokenizer & Labels ─── */}
      {(data.modelType === "nlp" || isNlpOnnx) && (
        <div
          className="space-y-4 p-5 rounded-xl"
          style={{
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.08)",
          }}
        >
          <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "#facc15" }}>
            ⚠ NLP Model — Review Preprocessing Settings
          </p>

          {/* Disclaimer */}
          <div className="rounded-lg p-3" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}>
            <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
              <strong style={{ color: "var(--text-primary)" }}>What is a Tokenizer?</strong> A tokenizer converts raw text into numerical tokens that a model can process.
              Each model is trained with a specific tokenizer, and using the wrong one will produce incorrect results.
              {data.sourceMode === "huggingface" ? (
                <> For HuggingFace models, the tokenizer is automatically set to match the model ID.</>
              ) : data.fileName.toLowerCase().includes("tfidf") ? (
                <> TF-IDF models use their own vectorizer and do not require a transformer tokenizer.</>
              ) : (
                <> For uploaded ONNX models, specify the tokenizer used during training (e.g., <code style={{ color: "#facc15" }}>distilbert-base-uncased</code>).</>
              )}
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-xs" style={{ color: "var(--text-muted)" }}>
              Tokenizer Name
            </label>
            <input
              type="text"
              placeholder={
                data.sourceMode === "huggingface"
                  ? "Auto-filled from HuggingFace model ID"
                  : data.fileName.toLowerCase().includes("tfidf")
                  ? "Not required for TF-IDF models"
                  : "e.g.  distilbert-base-uncased"
              }
              value={data.tokenizerName}
              onChange={(e) => onChange({ ...data, tokenizerName: e.target.value })}
              className="w-full px-4 py-3 rounded-lg text-sm outline-none"
              style={inputStyle}
              disabled={data.sourceMode === "huggingface"}
            />
            {data.sourceMode === "huggingface" && (
              <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                ✓ Auto-detected from HuggingFace model ID
              </p>
            )}
            {data.fileName.toLowerCase().includes("tfidf") && data.sourceMode === "upload" && (
              <p className="text-[10px]" style={{ color: "#4ade80" }}>
                ✓ TF-IDF model detected — tokenizer is handled by the built-in vectorizer
              </p>
            )}
          </div>
          <div className="space-y-2">
            <label className="text-xs" style={{ color: "var(--text-muted)" }}>
              Label Mapping (JSON)
            </label>
            <input
              type="text"
              placeholder={'e.g.  {"0": "NEGATIVE", "1": "POSITIVE"}'}
              value={data.labelMapping}
              onChange={(e) => onChange({ ...data, labelMapping: e.target.value })}
              className="w-full px-4 py-3 rounded-lg text-sm outline-none"
              style={inputStyle}
            />
            <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
              Maps numeric output indices to human-readable labels. Leave blank for auto-detection.
            </p>
          </div>
        </div>
      )}

      {/* ── Model Name + Version ─────────────────── */}
      <div className="grid grid-cols-2 gap-6">
        <div className="space-y-2">
          <label
            className="font-bold uppercase text-xs tracking-widest"
            style={{ color: "var(--text-muted)", letterSpacing: "0.1em" }}
          >
            Model Name
          </label>
          <input
            type="text"
            placeholder="e.g. BERT-Red-01"
            value={data.modelName}
            onChange={(e) => onChange({ ...data, modelName: e.target.value })}
            className="w-full px-4 py-3 rounded-lg text-sm outline-none transition-all"
            style={inputStyle}
            onFocus={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.3)")}
            onBlur={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.1)")}
          />
        </div>
        <div className="space-y-2">
          <label
            className="font-bold uppercase text-xs tracking-widest"
            style={{ color: "var(--text-muted)", letterSpacing: "0.1em" }}
          >
            Version
          </label>
          <input
            type="text"
            placeholder="v1.0.4-stable"
            value={data.version}
            onChange={(e) => onChange({ ...data, version: e.target.value })}
            className="w-full px-4 py-3 rounded-lg text-sm outline-none transition-all"
            style={inputStyle}
            onFocus={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.3)")}
            onBlur={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.1)")}
          />
        </div>
      </div>
    </div>
  );
}
