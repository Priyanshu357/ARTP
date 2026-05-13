"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import { MoreVertical } from "lucide-react";

const BARS = [
  { label: "FGSM", value: 0.31, height: "31%" },
  { label: "PGD", value: 0.45, height: "45%" },
  { label: "DF", value: 0.22, height: "22%" },
  { label: "C&W", value: 0.15, height: "15%" },
];

const BAR_GRADIENTS = [
  "linear-gradient(180deg, #555555, #333333)",
  "linear-gradient(180deg, #444444, #222222)",
  "linear-gradient(180deg, #666666, #3A3A3A)",
  "linear-gradient(180deg, #505050, #2A2A2A)",
];

export default function AsrBarChart() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, amount: 0.3 });

  return (
    <div ref={ref} className="glass-card rounded-2xl p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h3 className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
            ASR by Attack Type
          </h3>
          <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
            Comparison of success rates per method
          </p>
        </div>
        <button style={{ color: "var(--text-muted)" }} className="hover:opacity-80 transition-opacity">
          <MoreVertical size={20} />
        </button>
      </div>

      {/* Grid lines */}
      <div className="h-64 relative flex items-end justify-between gap-4 px-2">
        {/* Horizontal grid lines */}
        {[0.25, 0.5, 0.75, 1.0].map((v) => (
          <div
            key={v}
            className="absolute w-full left-0"
            style={{
              bottom: `${v * 100}%`,
              borderTop: "1px dashed rgba(255,255,255,0.03)",
            }}
          >
            <span
              className="absolute -left-1 -top-3 text-[9px]"
              style={{ color: "var(--text-muted)", fontFamily: "var(--font-jetbrains), monospace" }}
            >
              {v.toFixed(2)}
            </span>
          </div>
        ))}

        {/* Bars */}
        {BARS.map((bar, i) => (
          <div key={bar.label} className="flex flex-col items-center flex-1">
            <motion.div
              className="w-full max-w-[40px] rounded-t-lg relative"
              style={{
                background: BAR_GRADIENTS[i],
                transformOrigin: "bottom center",
                boxShadow: "inset 0 1px 0 rgba(255,255,255,0.1)",
              }}
              initial={{ scaleY: 0 }}
              animate={isInView ? { scaleY: 1 } : { scaleY: 0 }}
              transition={{
                duration: 0.8,
                delay: i * 0.08,
                ease: [0.16, 1, 0.3, 1],
              }}
            >
              <div style={{ height: bar.height, minHeight: "100%" }} />
              {/* Value tooltip on hover */}
              <div
                className="absolute -top-7 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 text-xs font-mono transition-opacity"
                style={{ color: "var(--text-primary)" }}
              >
                {bar.value}
              </div>
            </motion.div>
            <span
              className="mt-4 font-bold uppercase"
              style={{ fontSize: "10px", color: "var(--text-muted)" }}
            >
              {bar.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
