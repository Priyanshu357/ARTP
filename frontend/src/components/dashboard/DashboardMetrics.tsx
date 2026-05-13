"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import { Shield, Crosshair, Search, AlertTriangle } from "lucide-react";

interface MetricDef {
  label: string;
  value: number;
  decimals: number;
  change: number;
  changeLabel: string;
  icon: React.ReactNode;
}

function useCountUp(target: number, duration: number, inView: boolean, decimals: number) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    if (!inView) return;
    const start = performance.now();
    const step = (now: number) => {
      const progress = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      setValue(Number((ease * target).toFixed(decimals)));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [target, duration, inView, decimals]);
  return value;
}

function DashMetricCard({ metric, index }: { metric: MetricDef; index: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, amount: 0.3 });
  const count = useCountUp(metric.value, 1000, isInView, metric.decimals);

  const isPositive = metric.change >= 0;
  const changeColor = isPositive ? "var(--success)" : "var(--error)";
  const arrow = isPositive ? "▲" : "▼";

  return (
    <motion.div
      ref={ref}
      className="glass-card rounded-2xl p-6 group cursor-default"
      initial={{ opacity: 0, y: 16 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay: index * 0.08, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="flex items-center gap-3 mb-4">
        <div
          className="p-2 rounded-lg"
          style={{ background: "rgba(255,255,255,0.05)", color: "var(--text-muted)" }}
        >
          {metric.icon}
        </div>
        <span
          className="font-bold uppercase"
          style={{ color: "var(--text-muted)", fontSize: "11px", letterSpacing: "0.1em" }}
        >
          {metric.label}
        </span>
      </div>

      <div className="mb-4">
        <span className="text-4xl leading-none" style={{ fontWeight: 300, color: "var(--text-primary)" }}>
          {count}
        </span>
      </div>

      <div className="flex items-center gap-2 text-xs mb-6" style={{ color: changeColor }}>
        <span className="font-bold">
          {arrow} {isPositive ? "+" : ""}{metric.change}%
        </span>
        <span style={{ color: "var(--text-muted)" }}>{metric.changeLabel}</span>
      </div>

      <div
        className="h-0.5 mx-auto transition-all duration-500 accent-line-dash"
        style={{
          width: "60%",
          background: "var(--gradient-accent-line)",
          opacity: 0.3,
        }}
      />
    </motion.div>
  );
}

interface DashboardMetricsProps {
  stats?: {
    robustness_score: number;
    robustness_change: number;
    attack_success_rate: number;
    asr_change: number;
    detection_accuracy: number;
    detection_change: number;
    false_positive_rate: number;
    fpr_change: number;
  } | null;
}

export default function DashboardMetrics({ stats }: DashboardMetricsProps) {
  const metrics: MetricDef[] = [
    {
      label: "Robustness Score",
      value: stats?.robustness_score ?? 72.4,
      decimals: 1,
      change: stats?.robustness_change ?? 3.2,
      changeLabel: "from last run",
      icon: <Shield size={18} />,
    },
    {
      label: "Attack Success Rate",
      value: stats?.attack_success_rate ?? 0.31,
      decimals: 2,
      change: stats?.asr_change ?? -5.1,
      changeLabel: "improvement",
      icon: <Crosshair size={18} />,
    },
    {
      label: "Det. Accuracy",
      value: stats?.detection_accuracy ?? 0.87,
      decimals: 2,
      change: stats?.detection_change ?? 2.3,
      changeLabel: "detection rate",
      icon: <Search size={18} />,
    },
    {
      label: "False Pos. Rate",
      value: stats?.false_positive_rate ?? 0.08,
      decimals: 2,
      change: stats?.fpr_change ?? -1.2,
      changeLabel: "fewer errors",
      icon: <AlertTriangle size={18} />,
    },
  ];

  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-6 mb-10 relative z-10">
      {metrics.map((m, i) => (
        <DashMetricCard key={m.label} metric={m} index={i} />
      ))}
    </section>
  );
}
