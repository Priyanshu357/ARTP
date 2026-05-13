"use client";

import { useEffect, useState } from "react";
import Navbar from "@/components/layout/Navbar";
import ParticleField from "@/components/landing/ParticleField";
import WordSphere from "@/components/landing/WordSphere";
import HeroSection from "@/components/landing/HeroSection";
import MetricCards from "@/components/landing/MetricCards";
import type { StatsData } from "@/lib/api";

export default function LandingPage() {
  const [stats, setStats] = useState<StatsData | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/stats`
        );
        if (res.ok) {
          const data: StatsData = await res.json();
          setStats(data);
        }
      } catch {
        // Backend not reachable — metric cards will use fallback data
        console.info("ARTP backend not reachable; using demo data.");
      }
    };
    fetchStats();
  }, []);

  return (
    <>
      <ParticleField />
      <Navbar />

      <main className="relative">
        {/* Hero area: sphere + text + CTA */}
        <section className="relative z-10 min-h-screen flex flex-col items-center justify-center pt-16 overflow-hidden">
          {/* Radial glow behind sphere */}
          <div
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none"
            style={{
              width: "800px",
              height: "600px",
              background:
                "radial-gradient(ellipse at center, rgba(255,255,255,0.03) 0%, transparent 70%)",
            }}
          />

          <WordSphere />
          <HeroSection />

          {/* Decorative bottom accent */}
          <div className="absolute bottom-12 left-1/2 -translate-x-1/2 w-full max-w-2xl px-6">
            <div
              className="h-px w-full"
              style={{
                background: "var(--gradient-accent-line)",
                opacity: 0.3,
              }}
            />
            <div
              className="mt-4 flex justify-between uppercase"
              style={{
                color: "var(--text-muted)",
                fontFamily: "var(--font-jetbrains), monospace",
                fontSize: "10px",
                letterSpacing: "0.2em",
              }}
            >
              <span>Scale 0.012</span>
              <span>Vector 0x4F2A</span>
              <span>Secure Layer v2</span>
            </div>
          </div>
        </section>

        {/* Metric cards */}
        <MetricCards stats={stats} />
      </main>
    </>
  );
}
