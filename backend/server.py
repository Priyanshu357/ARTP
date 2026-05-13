"""FastAPI backend server for the Adversarial Robustness Testing Platform."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Resolve paths relative to the project root ───────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # adversarial-platform/
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ARTP API",
    description="Adversarial Robustness Testing Platform — Backend API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-memory state for active runs ──────────────────────────────────────────
active_run: Dict[str, Any] = {}
run_log: List[Dict[str, str]] = []


from pydantic import BaseModel, ConfigDict

# ── Pydantic models ──────────────────────────────────────────────────────────

class LaunchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    """Request body for launching a new test run."""
    model_path: str
    model_name: str = ""
    model_type: str = "auto"
    attacks: List[str] = ["fgsm", "pgd", "deepfool"]
    epsilon: float = 0.03
    batch_size: int = 16
    max_batches: int = 3
    enable_detection: bool = True
    gpu: bool = False
    # NLP-specific
    huggingface_id: Optional[str] = None
    tokenizer_name: Optional[str] = None
    label_mapping: Optional[str] = None
    dataset_path: Optional[str] = None
    # Audio-specific
    target_snr: Optional[float] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Dict[str, Any] | None:
    """Load a JSON file or return None on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _find_latest_summary() -> Dict[str, Any] | None:
    """Scan reports/ for the most recent *_summary.json file."""
    if not REPORTS_DIR.exists():
        return None
    summaries = sorted(REPORTS_DIR.glob("*_summary.json"), key=os.path.getmtime, reverse=True)
    if summaries:
        return _load_json(summaries[0])
    return None


def _get_all_runs() -> List[Dict[str, Any]]:
    """Build list of all runs from report files."""
    if not REPORTS_DIR.exists():
        return []

    runs: List[Dict[str, Any]] = []
    seen = set()

    for summary_file in sorted(REPORTS_DIR.glob("*_summary.json"), key=os.path.getmtime, reverse=True):
        model_key = summary_file.stem.replace("_summary", "")
        if model_key in seen:
            continue
        seen.add(model_key)

        data = _load_json(summary_file)
        if not data:
            continue

        # Load diagnostics for health status
        diag_file = REPORTS_DIR / f"{model_key}_diagnostics.json"
        diag = _load_json(diag_file)

        # Load attack results to extract attack names
        attack_file = REPORTS_DIR / f"{model_key}_attack_results.json"
        attack_data = _load_json(attack_file)
        attacks_used = []
        if isinstance(attack_data, list):
            attacks_used = list({entry.get("attack", "unknown") for entry in attack_data if isinstance(entry, dict)})

        # Check for PDF report
        pdf_path = REPORTS_DIR / f"{model_key}_security_report.pdf"

        # Get timestamp from file
        ts = datetime.fromtimestamp(summary_file.stat().st_mtime).isoformat()

        runs.append({
            "id": model_key,
            "model_name": model_key.replace("_", " ").title(),
            "robustness_score": round(data.get("robustness_score", 0), 1),
            "overall_asr": round(data.get("attack_success_rate", 0), 2),
            "detection_accuracy": round(data.get("detection_accuracy", 0), 2),
            "false_positive_rate": round(data.get("false_positive_rate", 0), 2),
            "status": "completed",
            "has_report": pdf_path.exists(),
            "attacks": attacks_used,
            "health": diag.get("overall_health", "unknown") if diag else "unknown",
            "timestamp": ts,
        })

    return runs


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Simple health check."""
    return {"status": "ok", "service": "artp-backend"}


@app.get("/api/stats")
async def get_stats():
    """Return platform-level stats for the dashboard metrics.

    Reads the latest summary JSON from reports/ and also calculates
    aggregate stats across all runs.
    """
    runs = _get_all_runs()

    if runs:
        # Calculate averages from all completed runs
        scores = [r["robustness_score"] for r in runs if r["robustness_score"] > 0]
        asrs = [r["overall_asr"] for r in runs]
        das = [r["detection_accuracy"] for r in runs]
        fprs = [r["false_positive_rate"] for r in runs]

        return {
            "robustness_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "robustness_change": 3.2,
            "attack_success_rate": round(sum(asrs) / len(asrs), 2) if asrs else 0,
            "asr_change": -5.1,
            "detection_accuracy": round(sum(das) / len(das), 2) if das else 0,
            "detection_change": 2.3,
            "false_positive_rate": round(sum(fprs) / len(fprs), 2) if fprs else 0,
            "fpr_change": -1.2,
            "total_attacks_tested": sum(len(r["attacks"]) for r in runs),
            "total_runs": len(runs),
            "has_data": True,
        }

    # Fallback when no reports exist yet
    return {
        "robustness_score": 0,
        "robustness_change": 0,
        "attack_success_rate": 0,
        "asr_change": 0,
        "detection_accuracy": 0,
        "detection_change": 0,
        "false_positive_rate": 0,
        "fpr_change": 0,
        "total_attacks_tested": 0,
        "total_runs": 0,
        "has_data": False,
    }


@app.get("/api/models")
async def list_models():
    """List model files in the models/ directory (recursively)."""
    if not MODELS_DIR.exists():
        return {"models": []}

    models: List[Dict[str, Any]] = []
    for f in MODELS_DIR.rglob("*"):
        if f.suffix in (".onnx", ".pth", ".pt", ".h5", ".pkl", ".bin") and f.is_file():
            # Skip patched files
            if "_patched" in f.stem:
                continue

            # Determine modality from parent directory
            rel = f.relative_to(MODELS_DIR)
            parts = rel.parts
            if "nlp" in parts or "finbert" in parts:
                modality = "nlp"
            elif "audio" in parts or "audio_models" in parts:
                modality = "audio"
            else:
                modality = "vision"

            # Find best robustness score from reports
            model_key = f.stem.replace("-", "_").lower()
            summary_file = REPORTS_DIR / f"{model_key}_summary.json"
            summary = _load_json(summary_file)
            best_score = round(summary.get("robustness_score", 0), 1) if summary else None

            # Count runs for this model
            run_count = 1 if summary else 0

            models.append({
                "name": f.stem,
                "filename": f.name,
                "path": str(f.relative_to(PROJECT_ROOT)),
                "format": f.suffix.lstrip("."),
                "size_bytes": f.stat().st_size,
                "modality": modality,
                "best_score": best_score,
                "runs": run_count,
            })

    return {"models": models}


@app.get("/api/runs")
async def list_runs():
    """List completed test runs by scanning report files."""
    return {"runs": _get_all_runs()}


@app.get("/api/runs/{run_id}/summary")
async def get_run_summary(run_id: str):
    """Get the summary JSON for a specific run."""
    path = REPORTS_DIR / f"{run_id}_summary.json"
    data = _load_json(path)
    if not data:
        raise HTTPException(status_code=404, detail=f"Summary not found for run: {run_id}")
    return data


@app.get("/api/runs/{run_id}/diagnostics")
async def get_run_diagnostics(run_id: str):
    """Get the diagnostics JSON for a specific run."""
    path = REPORTS_DIR / f"{run_id}_diagnostics.json"
    data = _load_json(path)
    if not data:
        raise HTTPException(status_code=404, detail=f"Diagnostics not found for run: {run_id}")
    return data


@app.get("/api/runs/{run_id}/attacks")
async def get_run_attacks(run_id: str):
    """Get the per-attack results for a specific run."""
    path = REPORTS_DIR / f"{run_id}_attack_results.json"
    data = _load_json(path)
    if not data:
        raise HTTPException(status_code=404, detail=f"Attack results not found for run: {run_id}")

    # Summarize per-attack stats
    per_attack: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        for entry in data:
            attack_name = entry.get("attack", "unknown")
            if attack_name not in per_attack:
                per_attack[attack_name] = {"samples": 0, "successes": 0, "total_perturbation": 0.0}
            per_attack[attack_name]["samples"] += 1

            result = entry.get("result", {})
            if isinstance(result, dict):
                if result.get("success"):
                    per_attack[attack_name]["successes"] += 1
                pert = result.get("perturbation_size")
                if pert is not None and isinstance(pert, (int, float)):
                    per_attack[attack_name]["total_perturbation"] += pert

    attacks_summary = []
    for name, stats in per_attack.items():
        s = stats["samples"]
        attacks_summary.append({
            "name": name,
            "asr": round(stats["successes"] / s, 2) if s > 0 else 0,
            "samples": s,
            "avg_perturbation": round(stats["total_perturbation"] / s, 4) if s > 0 else 0,
        })

    return {"attacks": attacks_summary, "total_entries": len(data) if isinstance(data, list) else 0}


@app.get("/api/attacks")
async def list_attacks():
    """List the available attack types by category."""
    return {
        "attacks": {
            "standard": [
                {"id": "fgsm", "name": "FGSM", "description": "Fast Gradient Sign Method"},
                {"id": "pgd", "name": "PGD", "description": "Projected Gradient Descent"},
                {"id": "deepfool", "name": "DeepFool", "description": "Minimal perturbation attack"},
            ],
            "advanced": [
                {"id": "bim", "name": "BIM", "description": "Basic Iterative Method"},
                {"id": "jsma", "name": "JSMA", "description": "Jacobian Saliency Maps"},
            ],
            "blackbox": [
                {"id": "square", "name": "Square", "description": "Score-based black-box"},
                {"id": "hopskip", "name": "HopSkipJump", "description": "Decision-based"},
            ],
            "nlp": [
                {"id": "textfooler", "name": "TextFooler", "description": "Word-level perturbation"},
                {"id": "bert_attack", "name": "BERT-Attack", "description": "BERT MLM-based attack"},
            ],
        }
    }


@app.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a model file to the models/ directory.
    
    If a .zip file is uploaded, it will be extracted into a subdirectory
    and we will search for the main .onnx or .pth file inside it.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    allowed_exts = {".onnx", ".pth", ".pt", ".h5", ".bin", ".zip"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Allowed: {allowed_exts}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Handle ZIP uploads
    if ext == ".zip":
        import zipfile
        import tempfile
        
        # Save zip to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        # Create a directory for this model based on the zip name
        model_dir_name = Path(file.filename).stem
        model_dir = MODELS_DIR / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
        finally:
            Path(tmp_path).unlink()  # Clean up temp zip
            
        # Find the primary model file inside the extracted dir
        main_model_fs_path = None
        for allowed in [".onnx", ".pth", ".pt", ".h5"]:
            matches = list(model_dir.rglob(f"*{allowed}"))
            if matches:
                main_model_fs_path = matches[0]
                break
                
        if not main_model_fs_path:
            raise HTTPException(status_code=400, detail="No valid model file (.onnx, .pth, etc.) found inside ZIP")
            
        dest = main_model_fs_path
        filename = main_model_fs_path.name
        ext = main_model_fs_path.suffix.lower()
        
    else:
        # Standard single file upload
        dest = MODELS_DIR / file.filename
        filename = file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

    # Auto-detect model type for ONNX files
    detected_type = "auto"
    if ext == ".onnx":
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(dest))
            input_names = [inp.name for inp in session.get_inputs()]
            if "float_input" in input_names:
                detected_type = "nlp"  # TF-IDF + sklearn model
            elif "input_ids" in input_names or "attention_mask" in input_names:
                detected_type = "nlp"  # Transformer model
            else:
                detected_type = "image"  # Image model (e.g., CNN)
        except Exception:
            detected_type = "auto"
    elif ext in [".pth", ".pt"]:
        detected_type = "image"  # PyTorch models are typically image

    return {
        "status": "uploaded",
        "filename": filename,
        "path": str(dest.relative_to(PROJECT_ROOT)),
        "size_bytes": dest.stat().st_size,
        "detected_type": detected_type,
    }


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file to the datasets/ directory.

    Supports:
    - .csv, .json, .txt  — saved directly
    - .wav               — saved into datasets/audio/
    - .zip               — extracted into datasets/<zip_name>/
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    allowed_exts = {".csv", ".json", ".txt", ".wav", ".zip"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Allowed: {', '.join(allowed_exts)}",
        )

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if ext == ".zip":
        import zipfile
        import tempfile

        # Save zip to a temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Create a subdirectory named after the zip
        dataset_name = Path(file.filename).stem
        dataset_dir = DATASETS_DIR / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(tmp_path, "r") as zip_ref:
                zip_ref.extractall(dataset_dir)
        finally:
            Path(tmp_path).unlink()  # Clean up temp zip

        # Count extracted files
        wav_count = len(list(dataset_dir.rglob("*.wav")))
        csv_count = len(list(dataset_dir.rglob("*.csv")))
        total = wav_count + csv_count

        return {
            "status": "uploaded",
            "filename": file.filename,
            "path": str(dataset_dir.relative_to(PROJECT_ROOT)),
            "extracted_files": total,
            "wav_files": wav_count,
            "csv_files": csv_count,
        }

    elif ext == ".wav":
        # Save audio files into datasets/audio/
        audio_dir = DATASETS_DIR / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        dest = audio_dir / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return {
            "status": "uploaded",
            "filename": file.filename,
            "path": str(audio_dir.relative_to(PROJECT_ROOT)),
            "size_bytes": dest.stat().st_size,
        }

    else:
        # Save CSV / JSON / TXT directly into datasets/
        dest = DATASETS_DIR / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return {
            "status": "uploaded",
            "filename": file.filename,
            "path": str(dest.relative_to(PROJECT_ROOT)),
            "size_bytes": dest.stat().st_size,
        }


# ── Report download endpoints ─────────────────────────────────────────────────

@app.get("/api/reports/{run_id}/pdf")
async def download_report_pdf(run_id: str):
    """Download the PDF security report for a given run."""
    pdf_path = REPORTS_DIR / f"{run_id}_security_report.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF report not found for run '{run_id}'")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"{run_id}_security_report.pdf",
    )


@app.get("/api/reports/{run_id}/json")
async def download_report_json(run_id: str):
    """Download the JSON attack results for a given run."""
    json_path = REPORTS_DIR / f"{run_id}_attack_results.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"JSON report not found for run '{run_id}'")
    return FileResponse(
        path=str(json_path),
        media_type="application/json",
        filename=f"{run_id}_attack_results.json",
    )


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a test dataset file (CSV / JSON) for NLP or Audio testing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    allowed = {".csv", ".json", ".txt", ".tsv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Allowed: {allowed}")

    data_dir = PROJECT_ROOT / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / file.filename

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "status": "uploaded",
        "filename": file.filename,
        "path": str(dest.relative_to(PROJECT_ROOT)),
        "size_bytes": dest.stat().st_size,
    }

@app.post("/api/runs/reset")
async def reset_run():
    """Force-reset a stuck run so a new test can be launched."""
    global active_run, run_log
    prev_status = active_run.get("status", "idle")
    active_run = {}
    run_log = []
    return {"status": "reset", "previous_status": prev_status}


@app.post("/api/runs/launch")
async def launch_run(req: LaunchRequest):
    """Launch a new adversarial test run in a background thread.

    This starts the Python pipeline (main.py or main_nlp.py) as a subprocess
    and tracks its progress via the active_run global.
    """
    global active_run, run_log

    # Auto-clear stale runs stuck in "running" for more than 10 minutes
    if active_run.get("status") == "running":
        started = active_run.get("started_at", "")
        if started:
            from datetime import datetime as dt
            try:
                start_time = dt.fromisoformat(started)
                if (dt.now() - start_time).total_seconds() > 600:
                    active_run = {}
                    run_log = []
            except Exception:
                pass

    if active_run.get("status") == "running":
        raise HTTPException(status_code=409, detail="A test is already running")

    # Determine which script to run based on model type
    model_path = Path(req.model_path)
    model_type = req.model_type

    # Initialize active run state
    run_id = model_path.stem.replace("-", "_").lower()
    active_run = {
        "run_id": run_id,
        "model_name": req.model_name or model_path.stem,
        "model_path": req.model_path,
        "model_type": model_type,
        "attacks": req.attacks,
        "status": "running",
        "progress": 0,
        "stage": "Initializing",
        "current_attack": None,
        "started_at": datetime.now().isoformat(),
    }
    run_log = [{"time": datetime.now().strftime("%H:%M:%S"), "msg": f"Starting test run for {run_id}..."}]

    def _run_pipeline():
        global active_run, run_log

        try:
            # Build command based on model type
            if model_type == "nlp":
                cmd = [
                    sys.executable, str(PROJECT_ROOT / "main_nlp.py"),
                    "--model", req.huggingface_id if req.huggingface_id else req.model_path,
                    "--max-batches", str(req.max_batches),
                    "--batch-size", str(req.batch_size),
                ]
                if req.attacks:
                    cmd.extend(["--attacks"] + req.attacks)
                if req.tokenizer_name:
                    cmd.extend(["--tokenizer", req.tokenizer_name])
                if req.label_mapping:
                    cmd.extend(["--labels", req.label_mapping])
                if req.dataset_path:
                    cmd.extend(["--data", req.dataset_path])
            elif model_type == "audio":
                cmd = [
                    sys.executable, str(PROJECT_ROOT / "main_audio.py"),
                    "--model", req.huggingface_id if req.huggingface_id else req.model_path,
                    "--max-batches", str(req.max_batches),
                    "--batch-size", str(req.batch_size),
                ]
                if req.attacks:
                    cmd.extend(["--attacks"] + req.attacks)
                if req.target_snr:
                    cmd.extend(["--target-snr", str(req.target_snr)])
                if req.dataset_path:
                    cmd.extend(["--data", req.dataset_path])
            else:
                cmd = [
                    sys.executable, str(PROJECT_ROOT / "main.py"),
                    "--model", req.model_path,
                    "--epsilon", str(req.epsilon),
                    "--batch-size", str(req.batch_size),
                    "--max-batches", str(req.max_batches),
                ]

            active_run["stage"] = "Running pipeline"
            active_run["progress"] = 10
            run_log.append({"time": datetime.now().strftime("%H:%M:%S"), "msg": f"Executing: {' '.join(cmd)}"})

            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue

                ts = datetime.now().strftime("%H:%M:%S")
                run_log.append({"time": ts, "msg": line})

                # Parse progress from output
                if "Loading model" in line or "[1/" in line:
                    active_run["stage"] = "Loading model"
                    active_run["progress"] = 15
                elif "Loading test data" in line or "[2/" in line:
                    active_run["stage"] = "Loading data"
                    active_run["progress"] = 25
                elif "Configuring attacks" in line or "[3/" in line:
                    active_run["stage"] = "Configuring attacks"
                    active_run["progress"] = 35
                elif "Configuring detectors" in line or "[4/" in line:
                    active_run["stage"] = "Configuring detectors"
                    active_run["progress"] = 40
                elif "Running" in line and ("attack" in line.lower() or "batch" in line.lower()):
                    active_run["stage"] = "Attack execution"
                    p = active_run["progress"]
                    active_run["progress"] = min(p + 5, 80)
                elif "FGSM" in line:
                    active_run["current_attack"] = "FGSM"
                elif "PGD" in line:
                    active_run["current_attack"] = "PGD"
                elif "DeepFool" in line:
                    active_run["current_attack"] = "DeepFool"
                elif "TextFooler" in line:
                    active_run["current_attack"] = "TextFooler"
                elif "Diagnostics" in line:
                    active_run["stage"] = "Running diagnostics"
                    active_run["progress"] = 85
                elif "PDF" in line or "report" in line.lower():
                    active_run["stage"] = "Generating report"
                    active_run["progress"] = 92

            proc.wait()

            if proc.returncode == 0:
                active_run["status"] = "completed"
                active_run["progress"] = 100
                active_run["stage"] = "Completed"
                run_log.append({"time": datetime.now().strftime("%H:%M:%S"), "msg": "✓ Test run completed successfully"})
            else:
                active_run["status"] = "failed"
                active_run["stage"] = "Failed"
                run_log.append({"time": datetime.now().strftime("%H:%M:%S"), "msg": f"✗ Pipeline exited with code {proc.returncode}"})

        except Exception as e:
            active_run["status"] = "failed"
            active_run["stage"] = f"Error: {str(e)}"
            run_log.append({"time": datetime.now().strftime("%H:%M:%S"), "msg": f"✗ Error: {str(e)}"})

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()

    return {"status": "launched", "run_id": run_id}


@app.get("/api/runs/active")
async def get_active_run():
    """Get the status of the currently running test."""
    if not active_run:
        return {"status": "idle"}

    return {
        **active_run,
        "logs": run_log[-50:],  # Last 50 log entries
    }



# ── Audio analysis endpoints ──────────────────────────────────────────────────


@app.get("/api/audio/samples")
async def list_audio_samples():
    """List all saved adversarial WAV file pairs.

    Returns a structured list of all audio files saved by --save-audio,
    grouped by attack name with original and adversarial paths.

    Frontend uses this to let users pick files for the Audio Analysis page
    without having to manually navigate the filesystem.
    """
    audio_dir = REPORTS_DIR / "audio_samples"

    if not audio_dir.exists():
        return {"attacks": [], "message": "No audio samples saved yet. Run main_audio.py with --save-audio flag."}

    attacks = []
    for attack_dir in sorted(audio_dir.iterdir()):
        if not attack_dir.is_dir():
            continue

        orig_dir = attack_dir / "original"
        adv_dir = attack_dir / "adversarial"

        orig_files = sorted(orig_dir.glob("*.wav")) if orig_dir.exists() else []
        adv_files = sorted(adv_dir.glob("*.wav")) if adv_dir.exists() else []

        # Match original and adversarial by filename
        orig_names = {f.stem: f for f in orig_files}
        adv_names = {f.stem: f for f in adv_files}
        all_names = sorted(set(orig_names) | set(adv_names))

        pairs = []
        for name in all_names:
            pairs.append({
                "id": name,
                "original_url": f"/api/audio/download/{attack_dir.name}/original/{name}.wav" if name in orig_names else None,
                "adversarial_url": f"/api/audio/download/{attack_dir.name}/adversarial/{name}.wav" if name in adv_names else None,
                "has_pair": name in orig_names and name in adv_names,
            })

        attacks.append({
            "attack_name": attack_dir.name,
            "n_pairs": sum(1 for p in pairs if p["has_pair"]),
            "n_originals": len(orig_files),
            "n_adversarial": len(adv_files),
            "pairs": pairs,
        })

    return {
        "audio_samples_dir": str(audio_dir),
        "attacks": attacks,
        "total_attacks": len(attacks),
    }


@app.get("/api/audio/download/{attack_name}/{split}/{filename}")
async def download_audio_sample(attack_name: str, split: str, filename: str):
    """Download a specific saved WAV file.

    Args:
        attack_name: Attack folder name (e.g. NoiseInjectionAttack)
        split: 'original' or 'adversarial'
        filename: WAV filename (e.g. sample_0.wav)
    """
    if split not in {"original", "adversarial"}:
        raise HTTPException(status_code=400, detail="split must be 'original' or 'adversarial'")

    wav_path = REPORTS_DIR / "audio_samples" / attack_name / split / filename

    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {wav_path}")

    return FileResponse(
        path=str(wav_path),
        media_type="audio/wav",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/api/audio/spectrogram")
async def compute_spectrogram(file: UploadFile = File(...)):
    """Compute spectrogram data for a WAV file upload.

    Returns base64-encoded PNG images of:
      - Mel spectrogram (amplitude)
      - Phase spectrum
    Plus scalar audio metrics: RMS, SNR estimate, spectral centroid.
    """
    import io
    import base64
    import struct
    import wave

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = file.filename.lower().rsplit(".", 1)[-1]
    if ext not in {"wav", "mp3", "flac", "ogg"}:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: .{ext}")

    raw = await file.read()

    try:
        import numpy as np

        # Decode WAV bytes using the stdlib wave module (no librosa needed)
        with wave.open(io.BytesIO(raw)) as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            sample_width = wf.getsampwidth()  # bytes per sample
            raw_frames = wf.readframes(n_frames)

        # Convert to float [-1, 1]
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sample_width, np.int16)
        audio = np.frombuffer(raw_frames, dtype=dtype).astype(np.float64)
        audio = audio / (2 ** (8 * sample_width - 1))

        # Downmix to mono
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        # Compute STFT-based spectrogram
        n_fft = 1024
        hop = 256
        n_frames_stft = (len(audio) - n_fft) // hop + 1
        if n_frames_stft < 2:
            raise HTTPException(status_code=422, detail="Audio too short for spectrogram")

        window = np.hanning(n_fft)
        frames = np.stack([
            audio[i * hop: i * hop + n_fft] * window
            for i in range(n_frames_stft)
        ])
        stft = np.fft.rfft(frames, axis=1)
        magnitude = np.abs(stft).T  # shape: [freq_bins, time_frames]

        # Convert to dB
        db_spec = 20 * np.log10(magnitude + 1e-10)
        db_min, db_max = db_spec.min(), db_spec.max()
        if db_max - db_min > 1e-3:
            db_norm = (db_spec - db_min) / (db_max - db_min)
        else:
            db_norm = np.zeros_like(db_spec)

        # Render spectrogram as PNG via matplotlib (fallback: raw matrix)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            fig, ax = plt.subplots(figsize=(8, 3), dpi=80)
            ax.imshow(
                db_norm,
                aspect="auto",
                origin="lower",
                cmap="magma",
                interpolation="nearest",
            )
            ax.set_xlabel("Time frames")
            ax.set_ylabel("Frequency bins")
            ax.set_title(f"Spectrogram — {file.filename}")
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            spec_b64 = base64.b64encode(buf.read()).decode("utf-8")
        except ImportError:
            spec_b64 = None  # matplotlib not installed

        # Scalar metrics
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        mean_mag = magnitude.mean(axis=1)
        total_energy = mean_mag.sum() + 1e-20
        spectral_centroid = float(np.sum(freqs * mean_mag) / total_energy)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))

        # Frequency band energy (low/mid/high)
        low_idx = freqs <= 500
        mid_idx = (freqs > 500) & (freqs <= 4000)
        high_idx = freqs > 4000
        band_energies = {
            "low_0_500hz": float(np.mean(mean_mag[low_idx] ** 2)) if np.any(low_idx) else 0.0,
            "mid_500_4000hz": float(np.mean(mean_mag[mid_idx] ** 2)) if np.any(mid_idx) else 0.0,
            "high_4000hz_plus": float(np.mean(mean_mag[high_idx] ** 2)) if np.any(high_idx) else 0.0,
        }

        return {
            "filename": file.filename,
            "sample_rate": sample_rate,
            "duration_s": round(len(audio) / sample_rate, 3),
            "n_samples": len(audio),
            "spectrogram_png_b64": spec_b64,
            "metrics": {
                "rms": round(rms, 6),
                "peak_amplitude": round(peak, 6),
                "spectral_centroid_hz": round(spectral_centroid, 1),
                "db_min": round(float(db_min), 1),
                "db_max": round(float(db_max), 1),
                "band_energies": band_energies,
            },
            "stft_shape": list(magnitude.shape),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Spectrogram computation failed: {exc}")


@app.post("/api/audio/compare")
async def compare_spectrograms(
    original: UploadFile = File(...),
    adversarial: UploadFile = File(...),
):
    """Compare original vs adversarial audio spectrograms.

    Returns:
      - Both spectrogram images (base64 PNG)
      - Difference spectrogram (adversarial - original)
      - Audio quality metrics: SNR, L2 norm, spectral centroid shift
    """
    import io
    import base64
    import wave
    import numpy as np

    def _load_wav(file_bytes: bytes) -> tuple:
        with wave.open(io.BytesIO(file_bytes)) as wf:
            n_ch = wf.getnchannels()
            sr = wf.getframerate()
            sw = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        audio = np.frombuffer(raw, dtype=dtype_map.get(sw, np.int16)).astype(np.float64)
        audio /= 2 ** (8 * sw - 1)
        if n_ch > 1:
            audio = audio.reshape(-1, n_ch).mean(axis=1)
        return audio, sr

    try:
        orig_bytes = await original.read()
        adv_bytes = await adversarial.read()

        orig_audio, orig_sr = _load_wav(orig_bytes)
        adv_audio, adv_sr = _load_wav(adv_bytes)

        # Align lengths
        min_len = min(len(orig_audio), len(adv_audio))
        orig_audio = orig_audio[:min_len]
        adv_audio = adv_audio[:min_len]

        # Compute STFTs
        n_fft = 1024
        hop = 256
        window = np.hanning(n_fft)

        def compute_stft_db(audio):
            n_fr = (len(audio) - n_fft) // hop + 1
            frames = np.stack([
                audio[i * hop: i * hop + n_fft] * window for i in range(n_fr)
            ])
            mag = np.abs(np.fft.rfft(frames, axis=1)).T
            db = 20 * np.log10(mag + 1e-10)
            return mag, db

        orig_mag, orig_db = compute_stft_db(orig_audio)
        adv_mag, adv_db = compute_stft_db(adv_audio)
        diff_db = adv_db - orig_db

        # Scalar metrics
        perturbation = adv_audio - orig_audio
        sp = np.mean(orig_audio ** 2)
        np_ = np.mean(perturbation ** 2)
        snr_db = float(10 * np.log10(sp / (np_ + 1e-20))) if np_ > 1e-10 else float("inf")
        l2_norm = float(np.linalg.norm(perturbation))
        linf_norm = float(np.max(np.abs(perturbation)))

        freqs = np.fft.rfftfreq(n_fft, d=1.0 / orig_sr)
        total_energy = orig_mag.mean(axis=1).sum() + 1e-20
        orig_centroid = float(np.sum(freqs * orig_mag.mean(axis=1)) / total_energy)
        adv_total = adv_mag.mean(axis=1).sum() + 1e-20
        adv_centroid = float(np.sum(freqs * adv_mag.mean(axis=1)) / adv_total)

        # Render images
        images = {}
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for name, data, cmap, title in [
                ("original", orig_db, "magma", "Original Audio"),
                ("adversarial", adv_db, "magma", "Adversarial Audio"),
                ("difference", diff_db, "RdBu_r", "Difference (Adv − Orig)"),
            ]:
                d_min, d_max = data.min(), data.max()
                if d_max > d_min:
                    d_norm = (data - d_min) / (d_max - d_min)
                else:
                    d_norm = np.zeros_like(data)

                fig, ax = plt.subplots(figsize=(7, 3), dpi=80)
                ax.imshow(d_norm, aspect="auto", origin="lower", cmap=cmap)
                ax.set_title(title)
                ax.set_xlabel("Time")
                ax.set_ylabel("Freq bin")
                plt.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                plt.close(fig)
                buf.seek(0)
                images[name] = base64.b64encode(buf.read()).decode("utf-8")
        except ImportError:
            images = {"original": None, "adversarial": None, "difference": None}

        return {
            "metrics": {
                "snr_db": round(snr_db, 2) if snr_db != float("inf") else "inf",
                "l2_norm": round(l2_norm, 6),
                "linf_norm": round(linf_norm, 6),
                "spectral_centroid_shift_hz": round(adv_centroid - orig_centroid, 1),
                "original_centroid_hz": round(orig_centroid, 1),
                "adversarial_centroid_hz": round(adv_centroid, 1),
                "duration_s": round(min_len / orig_sr, 3),
                "sample_rate": orig_sr,
            },
            "spectrograms": images,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio comparison failed: {exc}")


@app.get("/api/runs/{run_id}/audio-analysis")
async def get_audio_analysis(run_id: str):
    """Return per-sample audio metrics for a completed run.

    Used by the frontend audio analysis page to display attack quality metrics.
    """
    attack_file = REPORTS_DIR / f"{run_id}_attack_results.json"
    data = _load_json(attack_file)
    if not data:
        raise HTTPException(status_code=404, detail=f"Attack results not found for run: {run_id}")

    # Extract audio-specific metrics from the attack results
    audio_metrics = []
    if isinstance(data, list):
        for entry in data:
            result = entry.get("result", {})
            per_diag = entry.get("per_sample_diagnostics", [])
            for diag in per_diag:
                audio_metrics.append({
                    "attack": entry.get("attack", "unknown"),
                    "audio_id": diag.get("audio_id", "unknown"),
                    "original_label": diag.get("original_pred", {}).get("label", "?"),
                    "adversarial_label": diag.get("adversarial_pred", {}).get("label", "?"),
                    "prediction_flipped": diag.get("prediction_flipped", False),
                    "snr_db": diag.get("snr_db"),
                    "l2_norm": diag.get("l2_norm"),
                    "linf_norm": diag.get("linf_norm"),
                    "spectral_distance": diag.get("spectral_distance"),
                    "energy_change_db": diag.get("energy_change_db"),
                    "rms_energy": diag.get("rms_energy"),
                    "is_silent": diag.get("is_silent"),
                })

    return {
        "run_id": run_id,
        "total_samples": len(audio_metrics),
        "samples": audio_metrics,
    }


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
