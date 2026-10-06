import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    resolveAlias: {
      // pdfjs-dist optionally requires 'canvas' for Node.js environments.
      // We don't need it — PDF rendering is client-only via react-pdf.
      canvas: "",
    },
  },
};

export default nextConfig;
