"use client";

import { useEffect, useRef, useCallback } from "react";

interface Particle {
  x: number;
  y: number;
  size: number;
  speedX: number;
  speedY: number;
  opacity: number;
  wobble: number;
  wobbleSpeed: number;
  hasTrail: boolean;
  trail: { x: number; y: number }[];
}

interface LightStreak {
  x: number;
  y: number;
  angle: number;
  length: number;
  life: number;
  maxLife: number;
  opacity: number;
}

export default function ParticleField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const streaksRef = useRef<LightStreak[]>([]);
  const mouseRef = useRef({ x: 0, y: 0 });
  const animRef = useRef<number>(0);

  const createParticle = useCallback((w: number, h: number): Particle => {
    return {
      x: Math.random() * w,
      y: Math.random() * h,
      size: Math.random() * 2 + 1,
      speedX: (Math.random() - 0.5) * 0.4,
      speedY: (Math.random() - 0.5) * 0.4,
      opacity: Math.random() * 0.32 + 0.08,
      wobble: Math.random() * Math.PI * 2,
      wobbleSpeed: Math.random() * 0.02 + 0.005,
      hasTrail: Math.random() < 0.2,
      trail: [],
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;

      const count = Math.min(120, Math.max(60, Math.floor(canvas.width / 15)));
      particlesRef.current = Array.from({ length: count }, () =>
        createParticle(canvas.width, canvas.height)
      );
    };

    const spawnStreak = () => {
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      streaksRef.current.push({
        x: cx,
        y: cy,
        angle: Math.random() * Math.PI * 2,
        length: Math.random() * 120 + 80,
        life: 0,
        maxLife: 600,
        opacity: Math.random() * 0.1 + 0.15,
      });
    };

    let streakTimer: ReturnType<typeof setInterval>;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Particles
      for (const p of particlesRef.current) {
        // Mouse parallax (subtle)
        const dx = (mouseRef.current.x - canvas.width / 2) / canvas.width;
        const dy = (mouseRef.current.y - canvas.height / 2) / canvas.height;

        p.x += p.speedX + Math.sin(p.wobble) * 0.1 + dx * 0.3;
        p.y += p.speedY + dy * 0.3;
        p.wobble += p.wobbleSpeed;

        // Wrap
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        // Trail
        if (p.hasTrail) {
          p.trail.push({ x: p.x, y: p.y });
          if (p.trail.length > 8) p.trail.shift();
          for (let i = 0; i < p.trail.length; i++) {
            const t = p.trail[i];
            const alpha = (i / p.trail.length) * p.opacity * 0.4;
            ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
            ctx.beginPath();
            ctx.arc(t.x, t.y, p.size * 0.6, 0, Math.PI * 2);
            ctx.fill();
          }
        }

        // Dot
        ctx.fillStyle = `rgba(255, 255, 255, ${p.opacity})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      }

      // Light streaks
      for (let i = streaksRef.current.length - 1; i >= 0; i--) {
        const s = streaksRef.current[i];
        s.life += 16;
        const progress = s.life / s.maxLife;
        if (progress >= 1) {
          streaksRef.current.splice(i, 1);
          continue;
        }
        const fade = 1 - progress;
        const ex = s.x + Math.cos(s.angle) * s.length * progress;
        const ey = s.y + Math.sin(s.angle) * s.length * progress;

        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(ex, ey);
        ctx.strokeStyle = `rgba(255, 255, 255, ${s.opacity * fade})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      animRef.current = requestAnimationFrame(animate);
    };

    const onMouseMove = (e: MouseEvent) => {
      mouseRef.current.x = e.clientX;
      mouseRef.current.y = e.clientY;
    };

    resize();
    animate();
    streakTimer = setInterval(spawnStreak, 3000 + Math.random() * 2000);

    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", onMouseMove);

    return () => {
      cancelAnimationFrame(animRef.current);
      clearInterval(streakTimer);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMouseMove);
    };
  }, [createParticle]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-full h-full pointer-events-none"
      style={{ zIndex: 0 }}
    />
  );
}
