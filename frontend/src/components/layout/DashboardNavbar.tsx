"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { Bell } from "lucide-react";

const NAV_LINKS = [
  { label: "Dashboard", href: "/dashboard" },
  { label: "Models", href: "/models" },
  { label: "Reports", href: "/reports" },
  { label: "History", href: "/history" },
];

export default function DashboardNavbar() {
  const [scrolled, setScrolled] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className="fixed top-0 w-full z-50 flex items-center justify-between px-6 h-16 transition-all duration-300"
      style={{
        background: scrolled ? "rgba(5, 5, 5, 0.95)" : "rgba(5, 5, 5, 0.8)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
      }}
    >
      {/* Left: Logo */}
      <div className="flex items-center gap-3">
        <Link href="/" className="flex items-center gap-2 group">
          <div
            className="w-2 h-2 rounded-full bg-white group-hover:scale-125 transition-transform"
            style={{ boxShadow: "0 0 6px rgba(255,255,255,0.6)" }}
          />
          <span className="font-semibold text-lg tracking-tight text-white">
            ARTP
          </span>
        </Link>
      </div>

      {/* Center: Nav links */}
      <nav className="hidden md:flex items-center gap-8">
        {NAV_LINKS.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
          return (
            <Link
              key={item.href}
              href={item.href}
              className="text-sm transition-colors"
              style={{
                color: isActive ? "var(--text-primary)" : "var(--text-muted)",
                fontWeight: isActive ? 500 : 400,
                background: isActive ? "rgba(255,255,255,0.05)" : "transparent",
                padding: "6px 16px",
                borderRadius: "8px",
              }}
              onMouseEnter={(e) => {
                if (!isActive) e.currentTarget.style.color = "var(--text-secondary)";
              }}
              onMouseLeave={(e) => {
                if (!isActive) e.currentTarget.style.color = "var(--text-muted)";
              }}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Right: Actions */}
      <div className="flex items-center gap-4">
        <button
          className="p-2 transition-colors"
          style={{ color: "var(--text-muted)" }}
          onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-primary)")}
          onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
        >
          <Bell size={20} />
        </button>
        <div
          className="h-8 w-8 rounded-full flex items-center justify-center text-xs font-medium"
          style={{
            border: "1px solid rgba(255,255,255,0.1)",
            background: "rgba(255,255,255,0.05)",
          }}
        >
          JD
        </div>
      </div>
    </header>
  );
}
