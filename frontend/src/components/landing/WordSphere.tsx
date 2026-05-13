"use client";

import { useEffect, useRef, useCallback } from "react";

const WORDS = [
  "FGSM", "PGD", "DeepFool", "Robust", "Detection",
  "Entropy", "Gradient", "Perturbation", "C&W", "ARTP",
  "Defense", "Model", "Security", "Attack", "AI",
  "Neutral", "Sparse", "Elastic", "JSMA", "BIM",
  "BlackBox", "WhiteBox", "Adversarial", "Feature", "Logit",
  "Transfer", "Universal", "Boundary", "AutoAttack", "Harden",
];

const RADIUS = 180;

interface Word3D {
  x: number;
  y: number;
  z: number;
  text: string;
}

export default function WordSphere() {
  const containerRef = useRef<HTMLDivElement>(null);
  const wordsRef = useRef<Word3D[]>([]);
  const angleRef = useRef({ x: 0.002, y: 0.003 });
  const animRef = useRef<number>(0);
  const elementsRef = useRef<HTMLDivElement[]>([]);

  const init = useCallback(() => {
    wordsRef.current = WORDS.map((text, i) => {
      const phi = Math.acos(-1 + (2 * i) / WORDS.length);
      const theta = Math.sqrt(WORDS.length * Math.PI) * phi;
      return {
        text,
        x: RADIUS * Math.cos(theta) * Math.sin(phi),
        y: RADIUS * Math.sin(theta) * Math.sin(phi),
        z: RADIUS * Math.cos(phi),
      };
    });
  }, []);

  useEffect(() => {
    init();
    const container = containerRef.current;
    if (!container) return;

    // Create DOM elements
    container.innerHTML = "";
    elementsRef.current = wordsRef.current.map((w) => {
      const el = document.createElement("div");
      el.textContent = w.text;
      el.style.position = "absolute";
      el.style.top = "50%";
      el.style.left = "50%";
      el.style.transformStyle = "preserve-3d";
      el.style.userSelect = "none";
      el.style.pointerEvents = "none";
      el.style.whiteSpace = "nowrap";
      el.style.fontSize = "12px";
      el.style.fontFamily = "var(--font-jetbrains), monospace";
      el.style.letterSpacing = "0.08em";
      el.style.fontWeight = "400";
      container.appendChild(el);
      return el;
    });

    const rotate = () => {
      const ax = angleRef.current.x;
      const ay = angleRef.current.y;

      wordsRef.current.forEach((w, i) => {
        // Y-axis rotation
        const x1 = w.x * Math.cos(ay) - w.z * Math.sin(ay);
        const z1 = w.z * Math.cos(ay) + w.x * Math.sin(ay);
        // X-axis rotation
        const y2 = w.y * Math.cos(ax) - z1 * Math.sin(ax);
        const z2 = z1 * Math.cos(ax) + w.y * Math.sin(ax);

        w.x = x1;
        w.y = y2;
        w.z = z2;

        const scale = (z2 + RADIUS * 2) / (RADIUS * 3);
        const opacity = Math.max(0.08, (z2 + RADIUS) / (RADIUS * 2));

        const el = elementsRef.current[i];
        if (el) {
          el.style.transform = `translate3d(${w.x}px, ${w.y}px, ${w.z}px) scale(${scale})`;
          el.style.opacity = `${opacity}`;
          el.style.color = opacity > 0.5 ? "rgba(255,255,255,0.6)" : "rgba(255,255,255,0.15)";
        }
      });

      animRef.current = requestAnimationFrame(rotate);
    };

    const onMouseMove = (e: MouseEvent) => {
      const dx = (e.clientX - window.innerWidth / 2) / window.innerWidth;
      const dy = (e.clientY - window.innerHeight / 2) / window.innerHeight;
      angleRef.current.y = dx * 0.02;
      angleRef.current.x = dy * 0.02;
    };

    rotate();
    document.addEventListener("mousemove", onMouseMove);

    return () => {
      cancelAnimationFrame(animRef.current);
      document.removeEventListener("mousemove", onMouseMove);
    };
  }, [init]);

  return (
    <div className="relative mb-12 flex items-center justify-center animate-float">
      <div
        ref={containerRef}
        style={{
          perspective: "1000px",
          width: "400px",
          height: "400px",
          position: "relative",
        }}
      />
    </div>
  );
}
