import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const isGitHubPagesBuild = process.env.GITHUB_ACTIONS === "true";
const base = process.env.VITE_BASE_PATH ?? (isGitHubPagesBuild ? "/plant.ai/" : "/");

export default defineConfig({
  base,
  plugins: [react()]
});
