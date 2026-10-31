import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy: talk to the local JagX backend on :10000 without CORS issues.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://localhost:10000",
      "/chat": "http://localhost:10000",
      "/jobs": "http://localhost:10000",
      "/agents": "http://localhost:10000",
      "/tool": "http://localhost:10000",
      "/billing": "http://localhost:10000",
      "/keys": "http://localhost:10000",
      "/create-key": "http://localhost:10000",
    },
  },
});
