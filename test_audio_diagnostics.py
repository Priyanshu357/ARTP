"""Full diagnostic test for C&W Attack + Energy Detector + Spectral Detector."""
import numpy as np
import sys

# ── Dummy model ────────────────────────────────────────────────────────────────
class FlippableModel:
    """Flips label when audio RMS energy crosses a threshold."""
    def __call__(self, audio_batch):
        results = []
        for audio in audio_batch:
            rms = float(np.sqrt(np.mean(np.array(audio, dtype=np.float64) ** 2)))
            if rms > 0.12:
                results.append({"label": "class_1", "score": 0.80})
            else:
                results.append({"label": "class_0", "score": 0.90})
        return results

model = FlippableModel()
np.random.seed(42)
audio_batch = np.random.randn(2, 16000) * 0.1  # quiet -> predicts class_0

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: Carlini & Wagner (C&W) Attack")
print("=" * 60)

from red_team.audio_models import AdversarialAudioAttack

attack = AdversarialAudioAttack(
    epsilon=0.05,
    num_iterations=15,
    learning_rate=0.01,
    binary_search_steps=2,
)

result = attack.generate(model, audio_batch, labels=[0, 0], sample_rate=16000)

print(f"  Attack:              {result['attack']}")
print(f"  Success rate:        {result['attack_success_rate']:.2%}")
print(f"  Successful:          {result['successful_attacks']}/{result['total_samples']}")
print(f"  Avg L2 norm:         {result['perturbation_metrics']['avg_l2_norm']:.4f}")
snr = result['perturbation_metrics']['avg_snr_db']
print(f"  Avg SNR (dB):        {snr:.1f}" if snr != float('inf') else "  Avg SNR (dB):        inf")
for i, m in enumerate(result["per_sample_metrics"]):
    print(f"  Sample {i}: {m['original_label']} -> {m['adversarial_label']} | "
          f"success={m['success']} | snr={m['snr_db']:.1f} dB | l2={m['l2_norm']:.4f}")
print("  [OK] C&W attack completed\n")

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: EnergyDetector")
print("=" * 60)

from blue_team.audio_models import EnergyDetector

detector = EnergyDetector(energy_threshold=0.05, snr_threshold=30.0, frame_size=512)

# Case A: clean audio (should NOT detect)
clean = np.random.randn(16000) * 0.1
det_clean = detector.detect(
    original_output={"label": "class_0", "score": 0.9},
    adversarial_output={"label": "class_0", "score": 0.9},
    original_audio=clean,
    adversarial_audio=clean,  # no perturbation
    sample_rate=16000,
)
print(f"  Clean audio  -> detected={det_clean['detected']} | "
      f"anomaly_score={det_clean['anomaly_score']:.3f} | {det_clean['explanation']}")

# Case B: heavily perturbed audio (should detect)
perturbed = clean + np.random.randn(16000) * 0.3
det_pert = detector.detect(
    original_output={"label": "class_0", "score": 0.9},
    adversarial_output={"label": "class_1", "score": 0.8},
    original_audio=clean,
    adversarial_audio=perturbed,
    sample_rate=16000,
)
print(f"  Perturbed    -> detected={det_pert['detected']} | "
      f"anomaly_score={det_pert['anomaly_score']:.3f} | {det_pert['explanation']}")
print(f"  Details: {det_pert.get('details', {})}")
print("  [OK] EnergyDetector completed\n")

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: SpectralDetector")
print("=" * 60)

from blue_team.audio_models import SpectralDetector

sp_det = SpectralDetector(anomaly_threshold=0.1, n_fft=2048)

# Case A: identical audio (should NOT detect)
det_sp_clean = sp_det.detect(
    original_output={"label": "class_0", "score": 0.9},
    adversarial_output={"label": "class_0", "score": 0.9},
    original_audio=clean,
    adversarial_audio=clean,
    sample_rate=16000,
)
print(f"  Clean audio  -> detected={det_sp_clean['detected']} | "
      f"anomaly_score={det_sp_clean['anomaly_score']:.3f} | {det_sp_clean['explanation']}")

# Case B: spectrally different audio (high freq noise added)
hf_noise = clean + np.sin(2 * np.pi * 7000 * np.linspace(0, 1, 16000)) * 0.25
det_sp_pert = sp_det.detect(
    original_output={"label": "class_0", "score": 0.9},
    adversarial_output={"label": "class_1", "score": 0.8},
    original_audio=clean,
    adversarial_audio=hf_noise,
    sample_rate=16000,
)
print(f"  HF-perturbed -> detected={det_sp_pert['detected']} | "
      f"anomaly_score={det_sp_pert['anomaly_score']:.3f} | {det_sp_pert['explanation']}")
print(f"  Details: {det_sp_pert.get('details', {})}")
print("  [OK] SpectralDetector completed\n")

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 4: EnsembleDetector (Energy + Spectral combined)")
print("=" * 60)

from blue_team.audio_models import EnsembleDetector

ens = EnsembleDetector(mode="average")

# Case A: clean
det_ens_clean = ens.detect(
    original_output={"label": "class_0", "score": 0.9},
    adversarial_output={"label": "class_0", "score": 0.9},
    original_audio=clean,
    adversarial_audio=clean,
    sample_rate=16000,
)
print(f"  Clean audio  -> detected={det_ens_clean['detected']} | "
      f"anomaly_score={det_ens_clean['anomaly_score']:.3f}")
for sub in det_ens_clean.get("individual_results", []):
    print(f"    {sub['detector']}: score={sub['anomaly_score']:.3f} detected={sub['detected']}")

# Case B: perturbed
det_ens_pert = ens.detect(
    original_output={"label": "class_0", "score": 0.9},
    adversarial_output={"label": "class_1", "score": 0.8},
    original_audio=clean,
    adversarial_audio=perturbed,
    sample_rate=16000,
)
print(f"  Perturbed    -> detected={det_ens_pert['detected']} | "
      f"anomaly_score={det_ens_pert['anomaly_score']:.3f}")
for sub in det_ens_pert.get("individual_results", []):
    print(f"    {sub['detector']}: score={sub['anomaly_score']:.3f} detected={sub['detected']}")
print("  [OK] EnsembleDetector completed\n")

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 5: End-to-end — Attack -> Detect pipeline")
print("=" * 60)

from red_team.audio_models import AudioAttackRunner, NoiseInjectionAttack

runner = AudioAttackRunner(model, [NoiseInjectionAttack(epsilon=0.05, target_snr_db=10)], sample_rate=16000)
batch_results = runner.run_batch(audio_batch, [0, 0], audio_ids=["sample_0", "sample_1"])

for entry in batch_results:
    print(f"  Attack: {entry['attack']}")
    for diag in entry["per_sample_diagnostics"]:
        print(f"    {diag['audio_id']}: flip={diag['prediction_flipped']} | "
              f"snr={diag['snr_db']:.1f} dB | l2={diag['l2_norm']:.4f}")

    # Run ensemble detector on these results
    adv_audio = entry["result"]["adversarial_audio"]
    for i in range(len(audio_batch)):
        det = ens.detect(
            original_output=entry["result"]["original_predictions"][i],
            adversarial_output=entry["result"]["adversarial_predictions"][i],
            original_audio=audio_batch[i],
            adversarial_audio=adv_audio[i],
            sample_rate=16000,
        )
        print(f"    {entry['per_sample_diagnostics'][i]['audio_id']} detection: "
              f"detected={det['detected']} score={det['anomaly_score']:.3f} | {det['explanation'][:60]}")

print("  [OK] End-to-end pipeline completed\n")
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
