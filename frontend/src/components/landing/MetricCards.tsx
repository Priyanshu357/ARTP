"use client";

import { useEffect, useState, useRef } from "react";
import { motion, useInView } from "framer-motion";
import { TrendingUp, Shield, Eye } from "lucide-react";

interface MetricItem {
  label: string;
  value: number;
  format: "decimal" | "percent";
  change: number;
  changeLabel: string;
  icon: React.ReactNode;
}

function useCountUp(target: number, duration: number, inView: boolean, decimals: number = 1) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (!inView) return;
    const start = performance.now();
    const step = (now: number) => {
      const progress = Math.min((now - start) / duration, 1);
      // Spring-like easing
      const ease = 1 - Math.pow(1 - progress, 3);
      setValue(Number((ease * target).toFixed(decimals)));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [target, duration, inView, decimals]);

  return value;
}

interface MetricCardProps {
  metric: MetricItem;
  index: number;
}

function MetricCard({ metric, index }: MetricCardProps) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, amount: 0.3 });
  const decimals = metric.format === "decimal" ? 2 : 1;
  const count = useCountUp(metric.value, 1000, isInView, decimals);

  const isPositive = metric.change >= 0;
  const changeColor = isPositive ? "var(--success)" : "var(--error)";
  const arrow = isPositive ? "▲" : "▼";

  return (
    <motion.div
      ref={ref}
      className="glass-card rounded-2xl p-6 group cursor-default"
      initial={{ opacity: 0, y: 20 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{
        duration: 0.5,
        delay: index * 0.1,
        ease: [0.16, 1, 0.3, 1],
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span
          className="text-xs tracking-widest uppercase font-semibold"
          style={{ color: "var(--text-muted)", fontSize: "10px", letterSpacing: "0.1em" }}
        >
          {metric.label}
        </span>
        <div style={{ color: "var(--text-muted)" }}>{metric.icon}</div>
      </div>

      {/* Value */}
      <div
        className="text-4xl text-white mb-2"
        style={{ fontWeight: 300 }}
      >
        {count}
      </div>

      {/* Change */}
      <div className="text-xs flex items-center gap-1" style={{ color: changeColor }}>
        <span>
          {arrow} {isPositive ? "+" : ""}
          {metric.change}%
        </span>
        <span style={{ color: "var(--text-muted)", opacity: 0.5 }}>
          {metric.changeLabel}
        </span>
      </div>

      {/* Bottom accent line */}
      <div
        className="mt-6 h-0.5 mx-auto transition-all duration-500"
        style={{
          width: "60%",
          background: "var(--gradient-accent-line)",
          opacity: 0.2,
        }}
        onMouseEnter={(e) => {
          const el = e.currentTarget;
          el.style.width = "100%";
          el.style.opacity = "0.4";
        }}
        onMouseLeave={(e) => {
          const el = e.currentTarget;
          el.style.width = "60%";
          el.style.opacity = "0.2";
        }}
      />
    </motion.div>
  );
}

interface MetricCardsProps {
  stats?: {
    robustness_score: number;
    robustness_change: number;
    attack_success_rate: number;
    asr_change: number;
    detection_accuracy: number;
    detection_change: number;
  } | null;
}

export default function MetricCards({ stats }: MetricCardsProps) {
  const metrics: MetricItem[] = [
    {
      label: "Attack Success Rate",
      value: stats?.attack_success_rate ?? 0.31,
      format: "decimal",
      change: stats?.asr_change ?? -5.1,
      changeLabel: "vs baseline",
      icon: <TrendingUp size={16} />,
    },
    {
      label: "Robustness Score",
      value: stats?.robustness_score ?? 72.4,
      format: "percent",
      change: stats?.robustness_change ?? 3.2,
      changeLabel: "last 7 days",
      icon: <Shield size={16} />,
    },
    {
      label: "Detection Accuracy",
      value: stats?.detection_accuracy ?? 0.87,
      format: "decimal",
      change: stats?.detection_change ?? 2.3,
      changeLabel: "stable",
      icon: <Eye size={16} />,
    },
  ];

  return (
    <section className="relative z-10 max-w-7xl mx-auto px-6 pb-24">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {metrics.map((m, i) => (
          <MetricCard key={m.label} metric={m} index={i} />
        ))}
      </div>
    </section>
  );
}
