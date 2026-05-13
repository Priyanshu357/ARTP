"use client";

import { MoreVertical } from "lucide-react";

export default function RobustnessTrend() {
  return (
    <div className="glass-card rounded-2xl p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h3 className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
            Robustness Trend
          </h3>
          <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
            Score progression over historical runs
          </p>
        </div>
        <button style={{ color: "var(--text-muted)" }} className="hover:opacity-80 transition-opacity">
          <MoreVertical size={20} />
        </button>
      </div>

      <div className="h-64 relative">
        <svg
          className="w-full h-full overflow-visible"
          viewBox="0 0 400 100"
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id="trend-area-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="white" />
              <stop offset="100%" stopColor="transparent" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[25, 50, 75].map((y) => (
            <line
              key={y}
              x1="0"
              y1={y}
              x2="400"
              y2={y}
              stroke="rgba(255,255,255,0.03)"
              strokeDasharray="4 4"
            />
          ))}

          {/* Area fill */}
          <path
            d="M0 80 Q 50 70, 100 85 T 200 60 T 300 40 T 400 30 V 100 H 0 Z"
            fill="url(#trend-area-grad)"
            opacity="0.06"
          />

          {/* Line */}
          <path
            d="M0 80 Q 50 70, 100 85 T 200 60 T 300 40 T 400 30"
            fill="none"
            stroke="rgba(255,255,255,0.4)"
            strokeWidth="2"
            strokeLinecap="round"
            className="trend-line"
          />

          {/* Data points */}
          {[
            { cx: 0, cy: 80 },
            { cx: 100, cy: 85 },
            { cx: 200, cy: 60 },
            { cx: 300, cy: 40 },
            { cx: 400, cy: 30 },
          ].map((pt, i) => (
            <circle
              key={i}
              cx={pt.cx}
              cy={pt.cy}
              r="3"
              fill="white"
              style={{
                filter: i === 4 ? "drop-shadow(0 0 4px rgba(255,255,255,0.5))" : undefined,
              }}
            />
          ))}
        </svg>

        {/* X-axis labels */}
        <div className="absolute bottom-0 w-full flex justify-between px-2 pt-4">
          {["Jan 01", "Jan 15", "Feb 01", "Today"].map((label) => (
            <span key={label} className="text-[10px]" style={{ color: "var(--text-muted)" }}>
              {label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
