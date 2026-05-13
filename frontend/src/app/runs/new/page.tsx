"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { X, ArrowRight, ArrowLeft, Loader } from "lucide-react";
import DashboardNavbar from "@/components/layout/DashboardNavbar";
import Sidebar from "@/components/layout/Sidebar";
import StepIndicator from "@/components/wizard/StepIndicator";
import StepModel from "@/components/wizard/StepModel";
import type { ModelStepData } from "@/components/wizard/StepModel";
import StepAttacks from "@/components/wizard/StepAttacks";
import StepConfig from "@/components/wizard/StepConfig";
import type { ConfigStepData } from "@/components/wizard/StepConfig";
import { launchRun } from "@/lib/api";

const STEPS = ["Model", "Attacks", "Config"];

export default function NewTestPage() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(1);

  // Step 1 — Model source
  const [modelData, setModelData] = useState<ModelStepData>({
    modelName: "",
    version: "",
    modelType: "auto",
    fileName: "",
    modelPath: "",
    sourceMode: "upload",
    huggingfaceId: "",
    tokenizerName: "",
    labelMapping: "",
  });

  // Step 2 — Attack selection
  const [attackData, setAttackData] = useState({
    selectedCategory: "standard",
    selectedAttacks: [] as string[],
  });

  // Step 3 — Configuration
  const [configData, setConfigData] = useState<ConfigStepData>({
    epsilon: "",
    batchSize: "",
    maxBatches: "3",
    enableDetection: true,
    gpuAcceleration: false,
    datasetPath: "",
    datasetFileName: "",
    targetSnr: "",
  });

  const nextStep = () => {
    // When moving from step 1 (Model) to step 2 (Attacks), auto-select attacks
    if (currentStep === 1) {
      const defaultAttacks: Record<string, string[]> = {
        image: ["fgsm", "pgd", "deepfool"],
        nlp: ["textfooler", "bertattack"],
        audio: ["noise", "carlini"],
        auto: ["fgsm", "pgd", "deepfool"],
      };
      const autoAttacks = defaultAttacks[modelData.modelType] || defaultAttacks.auto;
      const category = modelData.modelType === "nlp" ? "nlp" : modelData.modelType === "audio" ? "audio" : "standard";
      setAttackData({
        selectedCategory: category,
        selectedAttacks: autoAttacks,
      });
    }
    setCurrentStep((s) => Math.min(s + 1, 3));
  };
  const prevStep = () => setCurrentStep((s) => Math.max(s - 1, 1));

  const [launching, setLaunching] = useState(false);

  /* ── Build and send the launch request ──────────────────────────────────── */
  const handleLaunch = async () => {
    setLaunching(true);
    try {
      // Determine model path — for huggingface, the ID goes via huggingface_id field
      const isHF = modelData.sourceMode === "huggingface";
      const modelPath = isHF
        ? modelData.huggingfaceId
        : modelData.modelPath || modelData.fileName || "models/cifar_net.onnx";

      // Default attacks per modality if none selected
      const defaultAttacks: Record<string, string[]> = {
        image: ["fgsm", "pgd", "deepfool"],
        nlp: ["textfooler", "bertattack"],
        audio: ["noise"],
        auto: ["fgsm", "pgd", "deepfool"],
      };
      const attacks =
        attackData.selectedAttacks.length > 0
          ? attackData.selectedAttacks
          : defaultAttacks[modelData.modelType] || defaultAttacks.auto;

      await launchRun({
        model_path: modelPath,
        model_name: modelData.modelName || modelData.fileName || modelData.huggingfaceId,
        model_type: modelData.modelType,
        attacks,
        epsilon: parseFloat(configData.epsilon) || 0.03,
        batch_size: parseInt(configData.batchSize) || (modelData.modelType === "nlp" ? 4 : 16),
        max_batches: parseInt(configData.maxBatches) || 3,
        enable_detection: configData.enableDetection,
        gpu: configData.gpuAcceleration,
        // NLP-specific
        huggingface_id: isHF ? modelData.huggingfaceId : undefined,
        tokenizer_name: modelData.tokenizerName || undefined,
        label_mapping: modelData.labelMapping || undefined,
        dataset_path: configData.datasetPath || undefined,
        // Audio-specific
        target_snr: configData.targetSnr ? parseFloat(configData.targetSnr) : undefined,
      });
      router.push("/runs/monitor");
    } catch (err) {
      console.error("Launch failed:", err);
      setLaunching(false);
    }
  };

  return (
    <>
      <DashboardNavbar />
      <Sidebar />

      <main className="lg:ml-[260px] pt-24 px-6 pb-12 relative min-h-screen">
        {/* Radial glow */}
        <div
          className="absolute top-0 left-1/2 -translate-x-1/2 w-full pointer-events-none"
          style={{
            height: "600px",
            background:
              "radial-gradient(ellipse at top, rgba(255,255,255,0.03) 0%, transparent 70%)",
            zIndex: 0,
          }}
        />

        <div className="relative z-10 max-w-3xl mx-auto">
          {/* Header */}
          <div className="flex items-center justify-between mb-2">
            <div>
              <h1
                className="text-2xl font-bold tracking-tight"
                style={{ color: "var(--text-primary)" }}
              >
                New Test Run
              </h1>
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                Step {currentStep}: {STEPS[currentStep - 1]}
                {currentStep === 1 && " — Select the Model"}
                {currentStep === 2 && " — Select Attacks"}
                {currentStep === 3 && " — Configure & Launch"}
              </p>
            </div>
            <div className="flex gap-2">
              <button
                className="px-4 py-2 rounded-lg text-sm transition-all"
                style={{
                  border: "1px solid rgba(255,255,255,0.1)",
                  color: "#94a3b8",
                }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.borderColor = "rgba(255,255,255,0.25)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)")
                }
              >
                Save Draft
              </button>
              {currentStep < 3 ? (
                <button
                  className="px-4 py-2 rounded-lg text-sm font-bold transition-all"
                  style={{
                    background: "#fff",
                    color: "#000",
                  }}
                  onClick={nextStep}
                >
                  Next Step
                </button>
              ) : (
                <button
                  className="px-4 py-2 rounded-lg text-sm font-bold transition-all flex items-center gap-2"
                  style={{
                    background: launching ? "#a3a3a3" : "#fff",
                    color: "#000",
                  }}
                  onClick={handleLaunch}
                  disabled={launching}
                >
                  {launching ? (
                    <>
                      <Loader size={14} className="animate-spin" />
                      Launching…
                    </>
                  ) : (
                    <>
                      Launch Test
                      <ArrowRight size={14} />
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Step Indicator */}
          <div className="my-8">
            <StepIndicator currentStep={currentStep} steps={STEPS} />
          </div>

          {/* Progress bar */}
          <div
            className="h-1.5 rounded-full mb-8"
            style={{ background: "rgba(255,255,255,0.05)" }}
          >
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${(currentStep / 3) * 100}%`,
                background: "#fff",
              }}
            />
          </div>

          {/* Wizard Card */}
          <div
            className="rounded-xl p-8"
            style={{
              background: "rgba(255,255,255,0.03)",
              backdropFilter: "blur(6px)",
              WebkitBackdropFilter: "blur(6px)",
              border: "1px solid rgba(255,255,255,0.1)",
              boxShadow: "0 25px 50px -12px rgba(0,0,0,0.25)",
            }}
          >
            <AnimatePresence mode="wait">
              <motion.div
                key={currentStep}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
              >
                {currentStep === 1 && (
                  <StepModel data={modelData} onChange={setModelData} />
                )}
                {currentStep === 2 && (
                  <StepAttacks
                    data={attackData}
                    onChange={setAttackData}
                    modelType={modelData.modelType}
                  />
                )}
                {currentStep === 3 && (
                  <StepConfig
                    data={configData}
                    onChange={setConfigData}
                    modelType={modelData.modelType}
                  />
                )}
              </motion.div>
            </AnimatePresence>

            {/* Bottom Actions */}
            <div
              className="flex items-center justify-between mt-8 pt-6"
              style={{ borderTop: "1px solid rgba(255,255,255,0.1)" }}
            >
              <button
                className="flex items-center gap-2 px-6 py-3 text-sm font-bold transition-colors"
                style={{ color: "var(--text-muted)" }}
                onClick={() => {
                  if (currentStep === 1) router.push("/dashboard");
                  else prevStep();
                }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.color = "var(--text-secondary)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.color = "var(--text-muted)")
                }
              >
                {currentStep === 1 ? (
                  <>
                    <X size={14} />
                    Cancel
                  </>
                ) : (
                  <>
                    <ArrowLeft size={14} />
                    Back
                  </>
                )}
              </button>

              <button
                className="flex items-center gap-2 px-10 py-3 rounded-lg text-sm font-bold transition-all active:scale-95"
                style={{
                  background: launching && currentStep === 3 ? "#a3a3a3" : "#fff",
                  color: "#000",
                }}
                onClick={currentStep === 3 ? handleLaunch : nextStep}
                disabled={launching && currentStep === 3}
              >
                {currentStep === 3 ? (
                  launching ? (
                    <>
                      <Loader size={14} className="animate-spin" />
                      Launching…
                    </>
                  ) : (
                    "Launch Test"
                  )
                ) : (
                  "Next Step"
                )}
                {!(launching && currentStep === 3) && <ArrowRight size={14} />}
              </button>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
