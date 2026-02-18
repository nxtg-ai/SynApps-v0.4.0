import path from "node:path";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const clientEnv = Object.fromEntries(
    Object.entries(env).filter(([key]) => key.startsWith("REACT_APP_") || key.startsWith("VITE_"))
  );

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    envPrefix: ["VITE_", "REACT_APP_"],
    define: {
      "process.env": JSON.stringify({
        ...clientEnv,
        NODE_ENV: mode,
      }),
    },
    server: {
      host: "0.0.0.0",
      port: 3000,
    },
  };
});
