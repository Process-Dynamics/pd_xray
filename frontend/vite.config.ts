import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run build` writes into the server package, which serves it at "/".
// `npm run dev` serves the UI with hot reload and forwards API calls to `pd-xray serve` on port 8000.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../packages/pd-xray-server/src/pd_xray/server/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1000, // one bundle (~560 kB, mostly OpenSeadragon), loaded once and cached
  },
  server: {
    proxy: { "/api": { target: "http://127.0.0.1:8000", ws: true } },
  },
});
