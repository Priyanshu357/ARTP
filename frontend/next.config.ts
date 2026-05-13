import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  turbopack: {
    // Ensure Turbopack resolves tailwindcss from frontend/node_modules
    // This fixes "Can't resolve 'tailwindcss'" when the frontend
    // is nested inside a parent project without its own node_modules.
    resolveAlias: {
      tailwindcss: path.resolve(__dirname, "node_modules/tailwindcss"),
    },
  },
};

export default nextConfig;
