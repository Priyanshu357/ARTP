"use client";

import { motion } from "framer-motion";

interface StepIndicatorProps {
  currentStep: number;
  steps: string[];
}

export default function StepIndicator({ currentStep, steps }: StepIndicatorProps) {
  return (
    <div className="flex items-center gap-4 w-full">
      {steps.map((label, i) => {
        const stepNum = i + 1;
        const isDone = currentStep > stepNum;
        const isActive = currentStep === stepNum;
        const isFuture = currentStep < stepNum;

        return (
          <div key={label} className="flex items-center gap-4 flex-1 last:flex-none">
            {/* Step circle + label */}
            <div className="flex items-center gap-2">
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all"
                style={{
                  border: isActive
                    ? "2px solid #fff"
                    : isDone
                    ? "2px solid rgba(255,255,255,0.4)"
                    : "1px solid rgba(255,255,255,0.1)",
                  background: isDone
                    ? "rgba(255,255,255,0.1)"
                    : isActive
                    ? "rgba(255,255,255,0.1)"
                    : "transparent",
                  color: isActive ? "#fff" : isDone ? "rgba(255,255,255,0.6)" : "rgba(148,163,184,0.5)",
                  opacity: isFuture ? 0.3 : 1,
                }}
              >
                {isDone ? "✓" : String(stepNum).padStart(2, "0")}
              </div>
              <span
                className="text-xs font-bold uppercase tracking-wider"
                style={{
                  color: isActive ? "#94a3b8" : isDone ? "rgba(255,255,255,0.6)" : "#94a3b8",
                  opacity: isFuture ? 0.3 : 1,
                }}
              >
                {label}
              </span>
            </div>

            {/* Connector line */}
            {i < steps.length - 1 && (
              <div className="flex-1 h-px" style={{
                background: isDone
                  ? "rgba(255,255,255,0.3)"
                  : "rgba(255,255,255,0.1)",
              }} />
            )}
          </div>
        );
      })}
    </div>
  );
}
