/**
 * API client for communicating with the ARTP FastAPI backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ── Types ────────────────────────────────────────────────────────────────── */

export interface StatsData {
  robustness_score: number;
  robustness_change: number;
  attack_success_rate: number;
  asr_change: number;
  detection_accuracy: number;
  detection_change: number;
  false_positive_rate: number;
  fpr_change: number;
  total_attacks_tested: number;
  total_runs?: number;
  has_data: boolean;
}

export interface ModelInfo {
  name: string;
  filename: string;
  path: string;
  format: string;
  size_bytes: number;
  modality: string;
  best_score: number | null;
  runs: number;
}

export interface RunInfo {
  id: string;
  model_name: string;
  robustness_score: number;
  overall_asr: number;
  detection_accuracy: number;
  false_positive_rate: number;
  status: string;
  has_report: boolean;
  attacks: string[];
  health: string;
  timestamp: string;
}

export interface RunSummary {
  attack_success_rate: number;
  detection_accuracy: number;
  false_positive_rate: number;
  robustness_score: number;
  score_penalty_applied?: boolean;
}

export interface DiagnosticItem {
  diagnostic: string;
  severity: string;
  description: string;
  evidence: Record<string, unknown>;
  root_causes: string[];
  recommendations: string[];
  interpretation: string;
}

export interface DiagnosticsData {
  timestamp: string;
  model_info: Record<string, string>;
  diagnostics: DiagnosticItem[];
  severity_counts: Record<string, number>;
  overall_health: string;
  llm_enhanced: boolean;
}

export interface AttackStat {
  name: string;
  asr: number;
  samples: number;
  avg_perturbation: number;
}

export interface ActiveRun {
  status: string;
  run_id?: string;
  model_name?: string;
  model_type?: string;
  attacks?: string[];
  progress?: number;
  stage?: string;
  current_attack?: string | null;
  started_at?: string;
  logs?: { time: string; msg: string }[];
}

export interface LaunchRequest {
  model_path: string;
  model_name?: string;
  model_type?: string;
  attacks?: string[];
  epsilon?: number;
  batch_size?: number;
  max_batches?: number;
  enable_detection?: boolean;
  gpu?: boolean;
  // NLP-specific
  huggingface_id?: string;
  tokenizer_name?: string;
  label_mapping?: string;
  dataset_path?: string;
  // Audio-specific
  target_snr?: number;
}

/* ── Fetch helpers ────────────────────────────────────────────────────────── */

async function apiFetch<T>(endpoint: string): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

async function apiPost<T>(endpoint: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

/* ── API functions ────────────────────────────────────────────────────────── */

export async function healthCheck(): Promise<{ status: string }> {
  return apiFetch<{ status: string }>("/api/health");
}

export async function getStats(): Promise<StatsData> {
  return apiFetch<StatsData>("/api/stats");
}

export async function getModels(): Promise<{ models: ModelInfo[] }> {
  return apiFetch<{ models: ModelInfo[] }>("/api/models");
}

export async function getRuns(): Promise<{ runs: RunInfo[] }> {
  return apiFetch<{ runs: RunInfo[] }>("/api/runs");
}

export async function getRunSummary(runId: string): Promise<RunSummary> {
  return apiFetch<RunSummary>(`/api/runs/${runId}/summary`);
}

export async function getRunDiagnostics(runId: string): Promise<DiagnosticsData> {
  return apiFetch<DiagnosticsData>(`/api/runs/${runId}/diagnostics`);
}

export async function getRunAttacks(runId: string): Promise<{ attacks: AttackStat[]; total_entries: number }> {
  return apiFetch<{ attacks: AttackStat[]; total_entries: number }>(`/api/runs/${runId}/attacks`);
}

export async function getActiveRun(): Promise<ActiveRun> {
  return apiFetch<ActiveRun>("/api/runs/active");
}

export async function launchRun(req: LaunchRequest): Promise<{ status: string; run_id: string }> {
  return apiPost<{ status: string; run_id: string }>("/api/runs/launch", req);
}
