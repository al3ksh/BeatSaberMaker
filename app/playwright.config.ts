import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60000,
  // One retry on CI only: the hosted Windows runner has no GPU and is slow, so GUI timing can flake.
  retries: process.env.CI ? 1 : 0,
  use: {
    trace: "retain-on-failure",
  },
});
