"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import {
  LayoutDashboard,
  Box,
  Play,
  Clock,
  Plus,
  BarChart3,
  Headphones,
} from "lucide-react";

const NAV_ITEMS = [
  { label: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { label: "Models", href: "/models", icon: Box },
  { label: "Runs", href: "/runs/monitor", icon: Play },
  { label: "Results", href: "/results", icon: BarChart3 },
  { label: "Audio Analysis", href: "/audio-analysis", icon: Headphones },
  { label: "History", href: "/history", icon: Clock },
];


export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside
      className="fixed left-0 top-16 h-[calc(100vh-64px)] w-[260px] hidden lg:flex flex-col z-40 transition-all duration-300"
      style={{
        background: "rgba(5, 5, 5, 0.5)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderRight: "1px solid rgba(255, 255, 255, 0.04)",
      }}
    >
      {/* Nav links */}
      <div className="p-6 space-y-1 flex-grow">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className="flex items-center gap-3 px-4 py-3 rounded-xl transition-all"
              style={{
                color: isActive ? "#fff" : "var(--text-muted)",
                background: isActive ? "rgba(255,255,255,0.05)" : "transparent",
                borderLeft: isActive
                  ? "2px solid rgba(255,255,255,0.3)"
                  : "2px solid transparent",
              }}
              onMouseEnter={(e) => {
                if (!isActive) {
                  e.currentTarget.style.color = "var(--text-secondary)";
                  e.currentTarget.style.background = "rgba(255,255,255,0.02)";
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  e.currentTarget.style.color = "var(--text-muted)";
                  e.currentTarget.style.background = "transparent";
                }
              }}
            >
              <Icon size={20} />
              <span className="text-sm font-medium">{item.label}</span>
            </Link>
          );
        })}
      </div>

      {/* New Test button */}
      <div className="p-6">
        <Link
          href="/runs/new"
          className="w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-sm transition-all active:scale-95"
          style={{
            background: "#fff",
            color: "var(--bg-void)",
            boxShadow: "0 0 20px rgba(255,255,255,0.1)",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "rgba(255,255,255,0.9)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "#fff";
          }}
        >
          <Plus size={18} strokeWidth={2.5} />
          New Test
        </Link>
      </div>
    </aside>
  );
}
