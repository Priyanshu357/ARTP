"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const navRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      ref={navRef}
      className="fixed top-0 w-full z-50 transition-all duration-300"
      style={{
        background: scrolled
          ? "rgba(5, 5, 5, 0.95)"
          : "rgba(5, 5, 5, 0.8)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
      }}
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Left: Logo + Nav */}
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-2 group">
            <div
              className="w-2 h-2 rounded-full bg-white group-hover:scale-125 transition-transform"
              style={{ boxShadow: "0 0 6px rgba(255,255,255,0.6)" }}
            />
            <span className="text-xl font-semibold tracking-tight text-white">
              ARTP
            </span>
          </Link>

          <nav className="hidden md:flex items-center gap-1">
            {["Dashboard", "Models", "Reports", "History"].map((item) => (
              <Link
                key={item}
                href={`/${item.toLowerCase()}`}
                className="px-4 py-1.5 text-sm transition-colors rounded-lg"
                style={{ color: "var(--text-muted)" }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.color = "var(--text-secondary)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.color = "var(--text-muted)")
                }
              >
                {item}
              </Link>
            ))}
          </nav>
        </div>

        {/* Right: Auth & CTA */}
        <div className="flex items-center gap-4">
          <button
            className="text-sm transition-colors"
            style={{ color: "var(--text-secondary)" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "#fff")}
            onMouseLeave={(e) =>
              (e.currentTarget.style.color = "var(--text-secondary)")
            }
          >
            Sign In
          </button>
          <button className="btn-primary">Get Started</button>
        </div>
      </div>
    </header>
  );
}
