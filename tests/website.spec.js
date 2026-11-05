import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";

const results = JSON.parse(readFileSync(new URL("../docs/data/results.json", import.meta.url), "utf8"));
const runs = JSON.parse(readFileSync(new URL("../docs/data/runs.json", import.meta.url), "utf8")).runs;
const routes = [
  "index.html",
  "methodology.html",
  "experiments.html",
  ...results.kernels.map((kernel) => `kernels/${kernel.slug}.html`)
];

function watchPageErrors(page, baseURL) {
  const errors = [];
  const origin = new URL(baseURL).origin;
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("requestfailed", (request) => {
    if (request.url().startsWith(origin)) {
      errors.push(`${request.url()}: ${request.failure()?.errorText}`);
    }
  });
  page.on("response", (response) => {
    if (
      response.url().startsWith(origin) &&
      response.request().resourceType() !== "document" &&
      response.status() >= 400
    ) {
      errors.push(`${response.status()} ${response.url()}`);
    }
  });
  return errors;
}

for (const route of routes) {
  test(`${route} is readable and stays within the project path`, async ({ page, baseURL }, testInfo) => {
    const errors = watchPageErrors(page, baseURL);
    const response = await page.goto(route);
    expect(response.status()).toBe(200);
    await expect(page.getByRole("heading", { level: 1 })).toHaveCount(1);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    await expect(page.getByRole("main")).toBeVisible();
    await expect(page).not.toHaveTitle("");
    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    const dimensions = await page.evaluate(() => ({
      width: document.documentElement.clientWidth,
      scroll: document.documentElement.scrollWidth
    }));
    expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.width + 2);
    const localLinks = await page.locator("a[href]").evaluateAll((anchors) =>
      anchors.map((anchor) => anchor.href)
    );
    const base = new URL(baseURL);
    for (const link of localLinks) {
      const target = new URL(link);
      if (target.origin === base.origin) {
        expect(target.pathname).toMatch(new RegExp(`^${base.pathname}`));
      }
    }
    expect(errors).toEqual([]);
    if (route === "index.html") {
      await page.screenshot({ path: testInfo.outputPath("homepage.png"), fullPage: true, animations: "disabled" });
    }
  });
}

test("experiment filters preserve exact counts, failures, and empty states", async ({ page, baseURL }) => {
  const errors = watchPageErrors(page, baseURL);
  await page.goto("experiments.html");
  const visibleRows = page.locator("[data-run-row]:visible");
  const summary = page.locator("[data-filter-summary]");
  await expect(visibleRows).toHaveCount(runs.length);
  await expect(summary).toHaveText("Showing all 79 attempts.");

  await page.getByLabel("Mode", { exact: true }).selectOption("run");
  await expect(visibleRows).toHaveCount(results.campaign.scoredCount);
  await expect(summary).toHaveText("Showing 18 of 79 attempts.");

  await page.getByLabel("Problem", { exact: true }).selectOption("muse_prefill_layer");
  await expect(visibleRows).toHaveCount(2);
  await page.getByLabel("Pass / fail", { exact: true }).selectOption("fail");
  await expect(visibleRows).toHaveCount(0);
  await expect(page.locator("[data-no-match]")).toBeVisible();
  await expect(summary).toHaveText("Showing 0 of 79 attempts.");

  await page.getByRole("button", { name: "Reset filters" }).click();
  await expect(visibleRows).toHaveCount(79);
  await expect(page.locator("[data-no-match]")).toBeHidden();
  await page.getByLabel("Pass / fail", { exact: true }).selectOption("fail");
  const failed = runs.filter((run) => run.passed === false).length;
  await expect(visibleRows).toHaveCount(failed);
  for (const row of await visibleRows.all()) {
    await expect(row.locator("td.numeric").first()).toHaveText("\u2014");
  }
  expect(errors).toEqual([]);
});

test("valid filter links reproduce the requested subset", async ({ page }) => {
  await page.goto("experiments.html?problem=muse_moe_w4a16&mode=check&outcome=pass");
  const count = runs.filter((run) =>
    run.problem === "muse_moe_w4a16" && run.mode === "check" && run.passed === true
  ).length;
  await expect(page.locator("[data-run-row]:visible")).toHaveCount(count);
  await expect(page.getByLabel("Problem", { exact: true })).toHaveValue("muse_moe_w4a16");
});

test("invalid URL filters are surfaced without hiding the history", async ({ page }) => {
  await page.goto("experiments.html?problem=unknown&mode=invalid&outcome=invalid");
  await expect(page.locator("[data-run-row]:visible")).toHaveCount(79);
  await expect(page.getByText(/invalid.*filter|unsupported.*filter/i)).toBeVisible();
});

test("a broken filter component shows an explicit error", async ({ page }) => {
  await page.route("**/experiments.html", async (route) => {
    const response = await route.fetch();
    const html = (await response.text()).replace('data-filter="mode"', 'data-broken-mode="mode"');
    await route.fulfill({ response, body: html });
  });
  await page.goto("experiments.html");
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(page.locator("[data-run-row]:visible")).toHaveCount(79);
});

test("articles link to their exact repository sources and show the measured score", async ({ page }) => {
  for (const kernel of results.kernels) {
    await page.goto(`kernels/${kernel.slug}.html`);
    const sourceLinks = page.locator(`a[href$="/${kernel.source.path}"]`);
    expect(await sourceLinks.count()).toBeGreaterThan(0);
    const text = await page.getByRole("main").innerText();
    expect(text).toContain(kernel.scored.speedup.toFixed(6));
    expect(text).toContain(kernel.source.sha256);
  }
});

test("the full ledger and articles remain readable without JavaScript", async ({ browser, baseURL }) => {
  const context = await browser.newContext({ baseURL, javaScriptEnabled: false });
  try {
    const page = await context.newPage();
    await page.goto("experiments.html");
    await expect(page.locator("[data-run-row]:visible")).toHaveCount(79);
    await expect(page.getByText(/JavaScript is off/)).toBeVisible();
    await page.goto("kernels/prefill-layer.html");
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    await expect(page.getByRole("main")).toContainText("1.919190");
  } finally {
    await context.close();
  }
});

test("keyboard users can skip navigation", async ({ page }) => {
  await page.goto("index.html");
  await page.keyboard.press("Tab");
  const skip = page.getByRole("link", { name: "Skip to content" });
  await expect(skip).toBeFocused();
  await expect(skip).toBeVisible();
  await page.keyboard.press("Enter");
  expect(new URL(page.url()).hash).toBe("#main");
});

for (const missing of ["does-not-exist.html", "missing/deep/page.html"]) {
  test(`404 at ${missing} keeps working assets and a local home link`, async ({ page, baseURL }) => {
    const errors = watchPageErrors(page, baseURL);
    const response = await page.goto(missing);
    expect(response.status()).toBe(404);
    await expect(page.getByRole("heading", { level: 1 })).toContainText(/does not exist|not found/i);
    const home = page.getByRole("link", { name: "Return to field notes" });
    await home.click();
    const url = new URL(page.url());
    const base = new URL(baseURL);
    expect(url.origin).toBe(base.origin);
    expect(url.pathname).toBe(`${base.pathname}index.html`);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    expect(errors).toEqual([]);
  });
}
