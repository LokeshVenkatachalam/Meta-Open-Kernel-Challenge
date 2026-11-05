import { defineConfig } from "@playwright/test";
import { createServer } from "node:net";

async function unusedPort() {
  const server = createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    server.close();
    throw new Error("Could not allocate a local preview port");
  }
  await new Promise((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
  return address.port;
}

const externalURL = process.env.PLAYWRIGHT_BASE_URL;
const port = process.env.PLAYWRIGHT_TEST_PORT ?? String(await unusedPort());
if (!/^\d+$/.test(port) || Number(port) < 1 || Number(port) > 65535) {
  throw new Error("PLAYWRIGHT_TEST_PORT must be an integer between 1 and 65535");
}
process.env.PLAYWRIGHT_TEST_PORT = port;
const prefix = "/Meta-Open-Kernel-Challenge/";
const baseURL = externalURL ?? `http://127.0.0.1:${port}${prefix}`;

export default defineConfig({
  testDir: "./tests",
  testMatch: "**/*.spec.js",
  workers: 1,
  fullyParallel: false,
  retries: 0,
  timeout: 30_000,
  expect: { timeout: 5_000 },
  reporter: [["list"]],
  use: {
    baseURL: baseURL.endsWith("/") ? baseURL : `${baseURL}/`,
    browserName: "chromium",
    headless: true,
    reducedMotion: "reduce",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    launchOptions: {
      args: [
        "--disable-gpu",
        "--disable-gpu-compositing",
        "--disable-webgl",
        "--disable-vulkan",
        "--disable-dev-shm-usage"
      ]
    }
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1440, height: 1000 } } },
    {
      name: "mobile",
      use: {
        viewport: { width: 390, height: 844 },
        isMobile: true,
        hasTouch: true,
        deviceScaleFactor: 1
      }
    }
  ],
  webServer: externalURL
    ? undefined
    : {
        command: `python3 scripts/serve_preview.py --port ${port} --prefix ${prefix}`,
        url: baseURL,
        reuseExistingServer: false,
        timeout: 30_000,
        env: {
          ...process.env,
          CUDA_VISIBLE_DEVICES: "",
          LIBGL_ALWAYS_SOFTWARE: "1",
          PYTHONDONTWRITEBYTECODE: "1"
        }
      }
});
