"use client";

import { motion } from "framer-motion";
import Link from "next/link";

export default function HeroSection() {
  return (
    <div className="text-center px-6 max-w-4xl mx-auto space-y-8">
      <motion.h1
        className="text-5xl md:text-7xl tracking-tight leading-tight text-white"
        style={{ fontWeight: 200, letterSpacing: "-0.03em" }}
        initial={{ opacity: 0, y: 20, filter: "blur(8px)" }}
        animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      >
        Adversarial Robustness
        <br />
        <span style={{ opacity: 0.8 }}>Testing Platform</span>
      </motion.h1>

      <motion.p
        className="text-lg md:text-xl max-w-2xl mx-auto leading-relaxed"
        style={{ color: "var(--text-secondary)", fontWeight: 300 }}
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
      >
        Test. Attack. Defend. Harden your AI models against sophisticated
        adversarial perturbations with industry-standard methodology.
      </motion.p>

      <motion.div
        className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-4"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
      >
        <Link
          href="/dashboard"
          className="btn-primary btn-primary-lg group flex items-center gap-3"
        >
          <div className="w-2 h-2 rounded-full bg-[#0A0A0A] group-hover:scale-125 transition-transform" />
          Launch Platform
        </Link>
        <button className="btn-secondary btn-secondary-lg">
          Documentation
        </button>
      </motion.div>
    </div>
  );
}
