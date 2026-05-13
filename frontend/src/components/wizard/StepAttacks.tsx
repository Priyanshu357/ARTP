"use client";

import { Zap, Shield, Eye, Cpu, Type, Mic } from "lucide-react";
import { type ReactNode } from "react";

/* ── Attack definitions per modality ─────────────────────────────────────── */

interface Attack {
  id: string;
  name: string;
  desc: string;
  category: string;
  icon: ReactNode;
}

const IMAGE_ATTACKS: Attack[] = [
  { id: "fgsm", name: "FGSM", desc: "Fast Gradient Sign Method — single-step L∞ attack", category: "standard", icon: <Zap size={16} /> },
  { id: "pgd", name: "PGD", desc: "Projected Gradient Descent — multi-step iterative", category: "standard", icon: <Shield size={16} /> },
  { id: "deepfool", name: "DeepFool", desc: "Minimal perturbation to cross decision boundary", category: "standard", icon: <Eye size={16} /> },
];

const NLP_ATTACKS: Attack[] = [
  { id: "textfooler", name: "TextFooler", desc: "Word-level substitution via synonyms", category: "nlp", icon: <Type size={16} /> },
  { id: "bertattack", name: "BERTAttack", desc: "Context-aware substitution using BERT MLM", category: "nlp", icon: <Cpu size={16} /> },
];

const AUDIO_ATTACKS: Attack[] = [
  { id: "noise", name: "Noise Injection", desc: "Adds random noise at target SNR", category: "audio", icon: <Mic size={16} /> },
  { id: "carlini", name: "Carlini & Wagner", desc: "Optimization-based adversarial audio", category: "audio", icon: <Shield size={16} /> },
];

const ALL_ATTACKS = [...IMAGE_ATTACKS, ...NLP_ATTACKS, ...AUDIO_ATTACKS];

function getAttacksForType(modelType: string): Attack[] {
  switch (modelType) {
    case "image": return IMAGE_ATTACKS;
    case "nlp": return NLP_ATTACKS;
    case "audio": return AUDIO_ATTACKS;
    default: return ALL_ATTACKS;  // "auto" shows everything
  }
}

function getCategoriesForType(modelType: string) {
  switch (modelType) {
    case "image": return [{ id: "standard", label: "Image Attacks" }];
    case "nlp": return [{ id: "nlp", label: "NLP Attacks" }];
    case "audio": return [{ id: "audio", label: "Audio Attacks" }];
    default: return [
      { id: "standard", label: "Image" },
      { id: "nlp", label: "NLP" },
      { id: "audio", label: "Audio" },
    ];
  }
}

/* ── Component ───────────────────────────────────────────────────────────── */

interface StepAttacksProps {
  data: {
    selectedCategory: string;
    selectedAttacks: string[];
  };
  onChange: (data: StepAttacksProps["data"]) => void;
  modelType?: string;
}

export default function StepAttacks({ data, onChange, modelType = "auto" }: StepAttacksProps) {
  const categories = getCategoriesForType(modelType);
  const attacks = getAttacksForType(modelType);

  // If model type changed and current category is no longer valid, auto-switch
  const validCat = categories.find(c => c.id === data.selectedCategory);
  const activeCat = validCat ? data.selectedCategory : categories[0]?.id || "standard";
  if (!validCat && categories.length > 0) {
    onChange({ ...data, selectedCategory: categories[0].id, selectedAttacks: [] });
  }

  const filteredAttacks = attacks.filter(a => a.category === activeCat);

  const toggleAttack = (id: string) => {
    const selected = data.selectedAttacks.includes(id)
      ? data.selectedAttacks.filter(a => a !== id)
      : [...data.selectedAttacks, id];
    onChange({ ...data, selectedAttacks: selected });
  };

  return (
    <div className="space-y-8">
      {/* Category Tabs */}
      <div
        className="flex gap-6 pb-px"
        style={{ borderBottom: "1px solid rgba(255,255,255,0.1)" }}
      >
        {categories.map(cat => {
          const isActive = activeCat === cat.id;
          return (
            <button
              key={cat.id}
              className="pb-3.5 text-sm font-bold transition-colors relative"
              style={{
                color: isActive ? "#fff" : "var(--text-muted)",
                borderBottom: isActive ? "2px solid #fff" : "2px solid transparent",
              }}
              onClick={() => onChange({ ...data, selectedCategory: cat.id })}
            >
              {cat.label}
            </button>
          );
        })}
      </div>

      {/* Attack Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {filteredAttacks.map(attack => {
          const isSelected = data.selectedAttacks.includes(attack.id);
          return (
            <button
              key={attack.id}
              className="rounded-xl p-5 text-left transition-all"
              style={{
                background: isSelected
                  ? "rgba(255,255,255,0.08)"
                  : "rgba(255,255,255,0.02)",
                border: isSelected
                  ? "1px solid rgba(255,255,255,0.25)"
                  : "1px solid rgba(255,255,255,0.06)",
                boxShadow: isSelected
                  ? "0 0 16px rgba(255,255,255,0.04)"
                  : "none",
              }}
              onClick={() => toggleAttack(attack.id)}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <div
                    className="p-1.5 rounded-lg"
                    style={{
                      background: "rgba(255,255,255,0.06)",
                      color: isSelected ? "#fff" : "var(--text-muted)",
                    }}
                  >
                    {attack.icon}
                  </div>
                  <span
                    className="text-sm font-medium"
                    style={{ color: isSelected ? "#fff" : "var(--text-secondary)" }}
                  >
                    {attack.name}
                  </span>
                </div>
                {/* Checkbox */}
                <div
                  className="w-5 h-5 rounded flex items-center justify-center text-xs transition-all"
                  style={{
                    border: isSelected
                      ? "1px solid #fff"
                      : "1px solid rgba(255,255,255,0.15)",
                    background: isSelected ? "#fff" : "transparent",
                    color: isSelected ? "#000" : "transparent",
                  }}
                >
                  ✓
                </div>
              </div>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                {attack.desc}
              </p>
            </button>
          );
        })}
      </div>

      {/* Selected count */}
      {data.selectedAttacks.length > 0 && (
        <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
          {data.selectedAttacks.length} attack{data.selectedAttacks.length > 1 ? "s" : ""} selected:{" "}
          {data.selectedAttacks.map(id => ALL_ATTACKS.find(a => a.id === id)?.name).filter(Boolean).join(", ")}
        </div>
      )}
    </div>
  );
}
