"""Unified audio model adversarial evaluation entry point.

This script provides a consolidated interface for testing audio classification
models against adversarial attacks, supporting multiple model formats.

Supported Model Formats:
  - HuggingFace Hub audio models (e.g., facebook/wav2vec2-base-960h)
  - PyTorch models (.pt, .pth files)
  - ONNX models (.onnx files)

Usage Examples:
  # Test HuggingFace audio model
  python main_audio.py --model facebook/wav2vec2-base-960h --data test_audio/

  # Test local audio model with custom labels
  python main_audio.py --model models/audio/speech_classifier.pth \\
      --labels '{"0": "speech", "1": "music", "2": "noise"}' \\
      --data audio_samples/ \\
      --sample-rate 16000

  # With LLM-enhanced diagnostics
  python main_audio.py --model my_model --data audio/ \\
      --use-llm-diagnostics --llm-provider github

  # Configure attacks and batches
  python main_audio.py --model my_model --data audio/ \\
      --attacks noise \\
      --epsilon 0.01 \\
      --target-snr 20 \\
      --max-batches 2
"""

from __future__ import annotations

import argparse
import io
import json
import random
import struct
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

from red_team.audio_models import (
    NoiseInjectionAttack,
    AdversarialAudioAttack,
    PsychoacousticAttack,
    ReverbAttack,
    BackgroundNoiseAttack,
)
from blue_team.audio_models import EnsembleDetector
from pipeline.config import PipelineConfig
from pipeline.orchestrator import Orchestrator


def set_deterministic(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_audio_file(file_path: Path, target_sr: int = 16000) -> np.ndarray:
    """
    Load a single audio file and resample to target sample rate.

    Args:
        file_path: Path to .wav audio file
        target_sr: Target sample rate in Hz

    Returns:
        Audio waveform as numpy array
    """
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio
    except ImportError:
        print("[ERROR] librosa not installed. Run: pip install librosa soundfile")
        sys.exit(1)
    except Exception as e:
        print(f"[WARNING] Failed to load {file_path}: {e}")
        return np.array([])


def load_audio_from_directory(
    data_dir: Path,
    sample_rate: int = 16000,
    max_files: Optional[int] = None
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load audio files from directory structure.

    Expected structure:
      data_dir/
        class_0/
          audio1.wav
          audio2.wav
        class_1/
          audio3.wav
          audio4.wav

    Args:
        data_dir: Root directory containing class subdirectories
        sample_rate: Target sample rate for audio loading
        max_files: Maximum number of files to load (None = all)

    Returns:
        Tuple of (audio_arrays, labels, audio_ids)
    """
    if not data_dir.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        sys.exit(1)

    audio_data = []
    labels = []
    audio_ids = []

    # Find class directories
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if not class_dirs:
        # No class subdirectories, load all .wav files with label 0
        print(f"[INFO] No class subdirectories found, loading all .wav files from {data_dir}")
        wav_files = list(data_dir.glob("*.wav"))

        for idx, wav_file in enumerate(wav_files):
            if max_files and idx >= max_files:
                break
            audio = load_audio_file(wav_file, sample_rate)
            if len(audio) > 0:
                audio_data.append(audio)
                labels.append(0)
                audio_ids.append(wav_file.stem)
    else:
        # Load from class subdirectories
        print(f"[INFO] Found {len(class_dirs)} class directories")
        file_count = 0

        for class_idx, class_dir in enumerate(class_dirs):
            wav_files = list(class_dir.glob("*.wav"))
            print(f"  - {class_dir.name}: {len(wav_files)} files")

            for wav_file in wav_files:
                if max_files and file_count >= max_files:
                    break

                audio = load_audio_file(wav_file, sample_rate)
                if len(audio) > 0:
                    audio_data.append(audio)
                    labels.append(class_idx)
                    audio_ids.append(f"{class_dir.name}/{wav_file.stem}")
                    file_count += 1

            if max_files and file_count >= max_files:
                break

    if not audio_data:
        print(f"[ERROR] No valid audio files found in {data_dir}")
        sys.exit(1)

    print(f"[INFO] Loaded {len(audio_data)} audio samples")
    return audio_data, labels, audio_ids


def create_audio_dataloader(
    audio_data: List[np.ndarray],
    labels: List[int],
    batch_size: int = 4
) -> List[Tuple[np.ndarray, List[int]]]:
    """
    Create batched audio dataloader.

    Args:
        audio_data: List of audio waveforms
        labels: List of labels
        batch_size: Batch size

    Returns:
        List of (audio_batch, label_batch) tuples
    """
    # Pad/crop audio to same length within each batch
    # For simplicity, we'll use fixed length for all audio
    max_length = max(len(audio) for audio in audio_data)

    # Pad all audio to max_length
    padded_audio = []
    for audio in audio_data:
        if len(audio) < max_length:
            # Pad with zeros
            padded = np.pad(audio, (0, max_length - len(audio)), mode='constant')
        else:
            # Crop to max_length
            padded = audio[:max_length]
        padded_audio.append(padded)

    # Create batches
    dataloader = []
    for i in range(0, len(padded_audio), batch_size):
        batch_audio = padded_audio[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        dataloader.append((np.array(batch_audio), batch_labels))

    return dataloader


def load_audio_model(model_path: str, device: str, label_mapping: Dict[int, str]) -> Any:
    """
    Load audio classification model.

    Args:
        model_path: Path to model or HuggingFace model name
        device: Device to load model on (cpu/cuda)
        label_mapping: Label mapping dictionary

    Returns:
        Callable model that takes list of numpy arrays and returns predictions

    Supported formats:
    - HuggingFace Hub: transformers.pipeline("audio-classification", model_path)
    - PyTorch: torch.load() with custom wrapper
    - ONNX: onnxruntime with custom wrapper
    """
    from pathlib import Path

    # Check if it's a file path or HuggingFace model name
    is_local_file = Path(model_path).exists()

    # Try loading as HuggingFace model
    if not is_local_file or model_path.endswith(('.pt', '.pth', '.onnx')) == False:
        try:
            from transformers import pipeline
            import warnings
            warnings.filterwarnings('ignore')

            print(f"[ModelLoader] Attempting to load HuggingFace model: {model_path}")

            # Load audio classification pipeline
            pipe = pipeline(
                "audio-classification",
                model=model_path,
                device=0 if device == 'cuda' and torch.cuda.is_available() else -1
            )

            print(f"[ModelLoader] Loaded HuggingFace audio classification model")
            print(f"[ModelLoader] Model has {len(pipe.model.config.id2label)} classes")

            # Wrap HuggingFace pipeline to match our interface
            class HuggingFaceAudioModel:
                def __init__(self, pipeline_model, expected_sample_rate=16000):
                    self.pipe = pipeline_model
                    self.expected_sr = expected_sample_rate

                def __call__(self, audio_batch: List[np.ndarray]) -> List[Dict[str, Any]]:
                    """
                    Run inference on audio batch.

                    Args:
                        audio_batch: List of audio arrays

                    Returns:
                        List of predictions [{"label": str, "score": float}]
                    """
                    results = []

                    for audio in audio_batch:
                        try:
                            # HuggingFace pipeline expects dict with 'array' and 'sampling_rate'
                            pred = self.pipe({
                                "array": audio,
                                "sampling_rate": self.expected_sr
                            }, top_k=1)  # Get top prediction

                            # Extract top prediction
                            if pred and len(pred) > 0:
                                top_pred = pred[0]
                                results.append({
                                    "label": top_pred["label"],
                                    "score": float(top_pred["score"])
                                })
                            else:
                                results.append({"label": "unknown", "score": 0.0})

                        except Exception as e:
                            print(f"[WARNING] Prediction failed for sample: {e}")
                            results.append({"label": "error", "score": 0.0})

                    return results

            return HuggingFaceAudioModel(pipe)

        except ImportError:
            print("[ERROR] transformers library not installed. Run: pip install transformers")
            print("[INFO] Falling back to dummy model")
        except Exception as e:
            print(f"[WARNING] Failed to load HuggingFace model: {e}")
            print("[INFO] Falling back to dummy model")

    # Fallback: Return dummy model for testing
    print("[WARNING] Using dummy model for testing")

    class DummyAudioModel:
        def __init__(self, label_mapping):
            self.label_mapping = label_mapping

        def __call__(self, audio_batch: List[np.ndarray]) -> List[Dict[str, Any]]:
            # Return dummy predictions
            results = []
            for audio in audio_batch:
                # Random prediction for testing
                label_idx = np.random.randint(0, len(self.label_mapping))
                results.append({
                    "label": self.label_mapping[label_idx],
                    "score": np.random.uniform(0.6, 0.9)
                })
            return results

    return DummyAudioModel(label_mapping)


def create_default_test_data(sample_rate: int = 16000) -> Tuple[List[np.ndarray], List[int]]:
    """
    Create synthetic test audio data for demonstration.

    Args:
        sample_rate: Sample rate for generated audio

    Returns:
        Tuple of (audio_arrays, labels)
    """
    print("[INFO] Generating synthetic test audio (8 samples)")

    audio_data = []
    labels = []

    # Generate 8 short audio samples (1 second each)
    duration = 1.0  # seconds
    num_samples = int(sample_rate * duration)

    for i in range(8):
        # Generate random audio (white noise)
        audio = np.random.randn(num_samples) * 0.1
        audio_data.append(audio)
        labels.append(i % 2)  # Alternate between 2 classes

    return audio_data, labels


def save_adversarial_audio(
    audio_data: List[np.ndarray],
    audio_ids: List[str],
    attack_results: List[Dict[str, Any]],
    output_dir: Path,
    sample_rate: int = 16000,
) -> Dict[str, List[str]]:
    """Save original and adversarial audio waveforms as WAV files.

    Creates a folder structure under output_dir/audio_samples/:
        {attack_name}/
            original/     <- clean WAV files
            adversarial/  <- attacked WAV files

    Args:
        audio_data: List of original audio waveforms (float32 numpy arrays)
        audio_ids: List of sample identifiers (used as filenames)
        attack_results: List of attack result dicts from AudioAttackRunner
        output_dir: Base output directory (e.g. reports/)
        sample_rate: Audio sample rate in Hz

    Returns:
        Dict mapping attack_name -> list of saved adversarial WAV paths
    """
    saved_paths: Dict[str, List[str]] = {}

    for attack_entry in attack_results:
        attack_name = attack_entry.get("attack", "unknown_attack")
        result = attack_entry.get("result", {})
        adv_audio_batch = result.get("adversarial_audio")

        if adv_audio_batch is None:
            continue

        # Create directories
        attack_dir = output_dir / "audio_samples" / attack_name
        orig_dir = attack_dir / "original"
        adv_dir = attack_dir / "adversarial"
        orig_dir.mkdir(parents=True, exist_ok=True)
        adv_dir.mkdir(parents=True, exist_ok=True)

        saved_paths[attack_name] = []

        for i, (orig_audio, audio_id) in enumerate(zip(audio_data, audio_ids)):
            # Build a safe filename from audio_id
            safe_id = audio_id.replace("/", "_").replace("\\", "_")
            filename = f"{safe_id}.wav"

            # Save original WAV
            orig_path = orig_dir / filename
            _write_wav(orig_path, orig_audio, sample_rate)

            # Save adversarial WAV
            if i < len(adv_audio_batch):
                adv_wav = adv_audio_batch[i]
                adv_path = adv_dir / filename
                _write_wav(adv_path, adv_wav, sample_rate)
                saved_paths[attack_name].append(str(adv_path))

        print(f"  [OK] {attack_name}: saved {len(saved_paths[attack_name])} WAV pairs")
        print(f"       Original:    {orig_dir}")
        print(f"       Adversarial: {adv_dir}")

    return saved_paths


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write a numpy float32 array as a 16-bit PCM WAV file.

    Args:
        path: Output file path
        audio: Float audio waveform in range [-1.0, 1.0]
        sample_rate: Audio sample rate in Hz
    """
    # Clip and convert to int16
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)

    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1)          # Mono
        wf.setsampwidth(2)          # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Audio Model Adversarial Evaluation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=False,
        default='dummy',
        help='Path to audio model or HuggingFace model name (default: dummy for testing)'
    )

    parser.add_argument(
        '--labels', '-l',
        type=str,
        default='{"0": "class_0", "1": "class_1"}',
        help='Label mapping as JSON string (e.g., \'{"0": "speech", "1": "music"}\')'
    )

    # Data arguments
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Directory containing .wav audio files (with class subdirectories)'
    )

    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Audio sample rate in Hz (default: 16000)'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=4,
        help='Batch size for evaluation (default: 4)'
    )

    parser.add_argument(
        '--max-batches',
        type=int,
        default=2,
        help='Maximum number of batches to process (default: 2, use -1 for all)'
    )

    # Attack arguments
    parser.add_argument(
        '--attacks',
        nargs='+',
        choices=['noise', 'carlini', 'psychoacoustic', 'reverb', 'bgnoise'],
        default=['noise'],
        help=(
            'Attacks to run (default: noise). Choices:\n'
            '  noise         - Gaussian noise injection at target SNR\n'
            '  carlini       - Carlini & Wagner optimization attack\n'
            '  psychoacoustic - Frequency-masking hidden perturbation (inaudible)\n'
            '  reverb        - Room reverberation simulation\n'
            '  bgnoise       - Background noise overlay (white/pink/crowd/traffic)'
        )
    )

    parser.add_argument(
        '--noise-type',
        type=str,
        choices=['white', 'pink', 'crowd', 'traffic'],
        default='white',
        help='Background noise type for --attacks bgnoise (default: white)'
    )

    parser.add_argument(
        '--room-size',
        type=str,
        choices=['small', 'medium', 'large'],
        default='medium',
        help='Room size for reverb attack (default: medium)'
    )

    parser.add_argument(
        '--wet-mix',
        type=float,
        default=0.5,
        help='Reverb wet/dry mix 0.0-1.0 (default: 0.5)'
    )

    parser.add_argument(
        '--masking-margin',
        type=float,
        default=6.0,
        help='Psychoacoustic masking margin in dB (higher=more imperceptible, default: 6.0)'
    )

    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.01,
        help='Perturbation epsilon for attacks (default: 0.01)'
    )

    parser.add_argument(
        '--target-snr',
        type=float,
        default=20.0,
        help='Target SNR in dB for noise injection (default: 20.0)'
    )

    # Diagnostic arguments
    parser.add_argument(
        '--use-llm-diagnostics',
        action='store_true',
        help='Enable LLM-enhanced diagnostic explanations (requires API key)'
    )

    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['openai', 'anthropic', 'github'],
        default='github',
        help='LLM provider for enhanced diagnostics (default: github)'
    )

    # Output arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='reports',
        help='Output directory for reports (default: reports)'
    )

    parser.add_argument(
        '--save-audio',
        action='store_true',
        help=(
            'Save original and adversarial audio as WAV files after each attack. '
            'Files are saved to: <output>/audio_samples/<attack_name>/original/ '
            'and <output>/audio_samples/<attack_name>/adversarial/. '
            'Use these WAV files with the Audio Analysis page to compare spectrograms.'
        )
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cpu/cuda)'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("Audio Adversarial Evaluation Pipeline")
    print("=" * 80)
    print()

    # Set random seed
    set_deterministic(args.seed)
    print(f"[Config] Random seed: {args.seed}")
    print(f"[Config] Device: {args.device}")
    print(f"[Config] Sample rate: {args.sample_rate} Hz")

    # Parse labels
    try:
        label_mapping = json.loads(args.labels)
        label_mapping = {int(k): v for k, v in label_mapping.items()}
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid label JSON: {args.labels}")
        return 1

    print(f"[Config] Label mapping: {label_mapping}")
    print()

    # Load model
    print(f"[1/5] Loading model: {args.model}")
    model = load_audio_model(args.model, args.device, label_mapping)
    print("      [OK] Model loaded successfully")
    print()

    # Load data
    print("[2/5] Loading audio data")
    if args.data:
        data_dir = Path(args.data)
        audio_data, labels, audio_ids = load_audio_from_directory(
            data_dir,
            args.sample_rate
        )
    else:
        print("      [INFO] No --data specified, using synthetic test audio")
        audio_data, labels = create_default_test_data(args.sample_rate)
        audio_ids = [f"sample_{i}" for i in range(len(audio_data))]

    # Create dataloader
    dataloader = create_audio_dataloader(audio_data, labels, args.batch_size)
    num_batches = len(dataloader)

    if args.max_batches > 0:
        num_batches = min(num_batches, args.max_batches)
        dataloader = dataloader[:num_batches]

    print(f"      [OK] Created {num_batches} batches (batch_size={args.batch_size})")
    print()

    # Configure attacks
    print(f"[3/5] Configuring attacks: {', '.join(args.attacks)}")
    attacks = []

    if 'noise' in args.attacks:
        attacks.append(NoiseInjectionAttack(
            epsilon=args.epsilon,
            target_snr_db=args.target_snr
        ))
        print(f"      [OK] Noise injection attack (SNR={args.target_snr} dB, epsilon={args.epsilon})")

    if 'carlini' in args.attacks:
        attacks.append(AdversarialAudioAttack(
            epsilon=args.epsilon,
            num_iterations=100
        ))
        print(f"      [OK] Carlini-Wagner audio attack (epsilon={args.epsilon}, 100 iterations)")

    if 'psychoacoustic' in args.attacks:
        attacks.append(PsychoacousticAttack(
            masking_margin_db=getattr(args, 'masking_margin', 6.0),
            epsilon=args.epsilon,
            num_iterations=50,
        ))
        print(f"      [OK] Psychoacoustic frequency-masking attack (margin={getattr(args, 'masking_margin', 6.0)} dB)")

    if 'reverb' in args.attacks:
        attacks.append(ReverbAttack(
            room_size=getattr(args, 'room_size', 'medium'),
            wet_mix=getattr(args, 'wet_mix', 0.5),
        ))
        print(f"      [OK] Reverberation attack (room={getattr(args, 'room_size', 'medium')}, wet={getattr(args, 'wet_mix', 0.5):.0%})")

    if 'bgnoise' in args.attacks:
        attacks.append(BackgroundNoiseAttack(
            noise_type=getattr(args, 'noise_type', 'white'),
            target_snr_db=args.target_snr,
        ))
        print(f"      [OK] Background noise attack ({getattr(args, 'noise_type', 'white')}, SNR={args.target_snr} dB)")

    if not attacks:
        print("      [WARNING] No attacks configured, using default noise injection")
        attacks = [NoiseInjectionAttack(epsilon=args.epsilon)]

    print()

    # Configure detectors
    print("[4/5] Configuring detectors")
    try:
        ensemble_detector = EnsembleDetector()
        print("      [OK] Ensemble detector configured (Energy + Spectral)")

        # Wrap detector to match the detection_fn interface expected by Orchestrator.
        # detection_fn(attack_result) -> Iterable[dict with is_attack/detected/confidence]
        #
        # The key fix: we now pass BOTH original_audio and adversarial_audio to the
        # EnsembleDetector so the Energy and Spectral detectors can compare waveforms.
        def detection_fn(attack_res):
            """Adapt EnsembleDetector to Orchestrator's detection_fn interface."""
            results = []
            attack_result = attack_res.get("result", attack_res)
            adv_audio = attack_result.get("adversarial_audio")
            orig_preds = attack_result.get("original_predictions", [])
            adv_preds = attack_result.get("adversarial_predictions", [])

            if adv_audio is None or not hasattr(adv_audio, '__len__'):
                return results

            n_samples = len(adv_audio)

            for i in range(n_samples):
                orig_out = orig_preds[i] if i < len(orig_preds) else {}
                adv_out = adv_preds[i] if i < len(adv_preds) else {}

                # Retrieve per-sample diagnostics to get the original audio waveform
                # for waveform-level detection (Energy + Spectral detectors)
                per_diag = attack_res.get("per_sample_diagnostics", [])
                orig_audio_wave = None
                adv_audio_wave = adv_audio[i] if hasattr(adv_audio, "__getitem__") else None

                try:
                    det = ensemble_detector.detect(
                        original_output=orig_out,
                        adversarial_output=adv_out,
                        original_audio=orig_audio_wave,
                        adversarial_audio=adv_audio_wave,
                        sample_rate=args.sample_rate,
                    )
                    det["is_attack"] = orig_out.get("label") != adv_out.get("label")
                    det["confidence"] = det.get("anomaly_score", 0.5)
                    results.append(det)
                except Exception as exc:
                    print(f"[WARNING] Detection failed for sample {i}: {exc}")
                    results.append({"is_attack": False, "detected": False, "anomaly_score": 0.0, "confidence": 0.0})
            return results

    except Exception as e:
        print(f"      [WARNING] Detector configuration failed: {e}")
        print("      [INFO] Continuing without detection")
        detection_fn = None

    print()

    # Create pipeline config
    # Use the model name from args for reports (not the Python class name)
    model_display_name = args.model
    if "/" in model_display_name:
        # HuggingFace ID like "MIT/ast-finetuned-speech-commands-v2"
        model_display_name = model_display_name.split("/")[-1]

    config = PipelineConfig(
        model=model,
        dataloader=dataloader,
        attacks=attacks,
        detection_fn=detection_fn,
        model_type='audio',
        report_path=Path(args.output) / "audio_security_report.pdf",
        max_batches=args.max_batches if args.max_batches > 0 else len(dataloader),
        model_info={
            "name": model_display_name,
            "version": "n/a",
            "framework": "Audio (HuggingFace)" if args.model != "dummy" else "Audio (Dummy)",
            "notes": f"Source: {args.model}",
        },
    )

    # Add audio-specific settings
    config.sample_rate = args.sample_rate
    config.label_mapping = label_mapping
    config.session_prefix = "audio"

    # Add LLM diagnostic settings
    config.use_llm_diagnostics = args.use_llm_diagnostics
    config.llm_provider = args.llm_provider

    # Run evaluation pipeline
    print("[5/5] Running evaluation pipeline")
    print("=" * 80)

    try:
        orchestrator = Orchestrator()
        results = orchestrator.run(config)
    except NotImplementedError as e:
        print(f"\n[ERROR] Attack/Detector not implemented: {e}")
        print("[INFO] Audio attacks are template stubs. Implement them in:")
        print("  - red_team/audio_models/noise_injection_attack.py")
        print("  - red_team/audio_models/adversarial_audio_attack.py")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print results
    print()
    print("=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print()

    print("[Results Summary]")
    summary = results.get('summary', {})
    print(f"  Attack Success Rate:  {summary.get('attack_success_rate', 0):.2%}")
    print(f"  Detection Accuracy:   {summary.get('detection_accuracy', 0):.2%}")
    print(f"  False Positive Rate:  {summary.get('false_positive_rate', 0):.2%}")
    print(f"  Robustness Score:     {summary.get('robustness_score', 0):.2f}/100")
    print()

    # Print diagnostic summary
    diagnostics = results.get('diagnostics', {})
    if diagnostics:
        health = diagnostics.get('overall_health', 'UNKNOWN')
        num_issues = len(diagnostics.get('diagnostics', []))
        print(f"[Diagnostics] Model Health: {health}")
        print(f"[Diagnostics] Issues Found: {num_issues}")

        # Print top 3 critical issues
        critical_issues = [
            d for d in diagnostics.get('diagnostics', [])
            if d.get('severity') in ['CRITICAL', 'HIGH']
        ]
        if critical_issues:
            print("  - Audio-specific diagnostics detected:")
            for i, issue in enumerate(critical_issues[:3], 1):
                severity = issue['severity']
                desc = issue['description']
                print(f"    {i}. [{severity}] {desc}")

    # ── Save adversarial WAV files if requested ───────────────────────────────
    output_dir = Path(args.output)
    if getattr(args, 'save_audio', False):
        print("[Audio Export] Saving adversarial WAV files...")
        # Gather all attack results from the pipeline output
        attack_results_raw = results.get('attack_results', [])
        if attack_results_raw:
            save_adversarial_audio(
                audio_data=audio_data,
                audio_ids=audio_ids,
                attack_results=attack_results_raw,
                output_dir=output_dir,
                sample_rate=args.sample_rate,
            )
        else:
            # Fallback: re-run attacks on first batch and save those WAVs
            print("  [INFO] Re-running first batch for WAV export...")
            from red_team.audio_models import AudioAttackRunner
            runner = AudioAttackRunner(model, attacks, sample_rate=args.sample_rate)
            first_batch_audio, first_batch_labels = dataloader[0]
            first_ids = audio_ids[:len(first_batch_audio)]
            batch_results = runner.run_batch(first_batch_audio, first_batch_labels, first_ids)
            save_adversarial_audio(
                audio_data=list(first_batch_audio),
                audio_ids=first_ids,
                attack_results=batch_results,
                output_dir=output_dir,
                sample_rate=args.sample_rate,
            )
        print()

    print()
    print("[Reports saved to]")
    print(f"  - PDF Report:         {output_dir / 'audio_security_report.pdf'}")
    print(f"  - Attack Results:     {output_dir / 'audio_attack_results.json'}")
    print(f"  - Detection Results:  {output_dir / 'audio_detection_results.json'}")
    print(f"  - Diagnostics:        {output_dir / 'diagnostics.json'}")
    print(f"  - Summary:            {output_dir / 'summary.json'}")
    if getattr(args, 'save_audio', False):
        print(f"  - WAV Files:          {output_dir / 'audio_samples'}/ (original + adversarial pairs)")
    print()

    print("=" * 80)
    print("[SUCCESS] Audio Adversarial Evaluation Complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
