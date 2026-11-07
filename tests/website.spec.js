import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";

const results = JSON.parse(readFileSync(new URL("../docs/data/results.json", import.meta.url), "utf8"));
const runs = JSON.parse(readFileSync(new URL("../docs/data/runs.json", import.meta.url), "utf8")).runs;
const routes = [
  "index.html",
  "disclosure.html",
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
    const notice = page.locator("[data-agentic-disclosure]").first();
    await expect(notice).toBeVisible();
    await expect(notice).toBeInViewport({ ratio: 0.2 });
    await expect(notice).toContainText(/agentic|agent-driven|AI agents/i);
    await expect(notice).toContainText(/copy|copied|verbatim/i);
    await expect(notice).toContainText(/audit|originality/i);
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
  await expect(page.locator('[data-run-row][data-result-disposition="withdrawn"]')).toHaveCount(2);
  await expect(page.locator('[data-run-row][data-result-disposition="excluded-history"]')).toHaveCount(8);
  await expect(page.locator('[data-run-row][data-result-disposition="retained-history"]')).toHaveCount(69);

  await page.getByLabel("Mode", { exact: true }).selectOption("run");
  await expect(visibleRows).toHaveCount(results.campaign.scoredCount);
  await expect(summary).toHaveText("Showing 18 of 79 attempts.");

  await page.getByLabel("Problem", { exact: true }).selectOption("muse_prefill_layer");
  await expect(visibleRows).toHaveCount(2);
  for (const row of await visibleRows.all()) {
    await expect(row).toHaveAttribute("data-result-disposition", "withdrawn");
    await expect(row).toContainText(/withdrawn/i);
  }
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

test("articles link to evidence without crediting the copied result", async ({ page }) => {
  for (const kernel of results.kernels) {
    await page.goto(`kernels/${kernel.slug}.html`);
    const sourceLinks = page.locator(`a[href$="/${kernel.source.path}"]`);
    expect(await sourceLinks.count()).toBeGreaterThan(0);
    const text = await page.getByRole("main").innerText();
    if (kernel.includedInSummary) {
      expect(text).toContain(kernel.scored.speedup.toFixed(6));
      expect(text).toMatch(/audit|originality/i);
    } else {
      expect(text).toMatch(/copied|verbatim|byte-for-byte/i);
      expect(text).toMatch(/withdrawn|excluded/i);
      await expect(page.locator(".result-ticket")).toHaveCount(0);
      await expect(page.locator("[data-claimed-result]")).toHaveCount(0);
    }
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
    await expect(page.getByRole("main")).toContainText(/copied|verbatim|byte-for-byte/i);
    await expect(page.getByRole("main")).toContainText(/withdrawn|excluded/i);
  } finally {
    await context.close();
  }
});

test("homepage retains only four unreviewed results and flags the excluded problem", async ({ page }) => {
  await page.goto("index.html");
  const cards = page.locator("[data-claimed-result]");
  await expect(cards).toHaveCount(4);
  const ids = await cards.evaluateAll((elements) =>
    elements.map((element) => element.dataset.claimedResult).sort()
  );
  expect(ids).toEqual(results.kernels.filter((kernel) => kernel.includedInSummary).map((kernel) => kernel.id).sort());
  expect(ids).not.toContain("muse_prefill_layer");
  await expect(page.locator('[data-excluded-result="muse_prefill_layer"]')).toBeVisible();
  await expect(page.getByRole("heading", { name: /five baseline wins/i })).toHaveCount(0);
});

test("public JSON separates removed claims from immutable historical measurements", async ({ request }) => {
  const response = await request.get("data/results.json");
  expect(response.ok()).toBe(true);
  const data = await response.json();
  expect(data.schemaVersion).toBe(2);
  expect(data.publication.workflow).toBe("fully-agentic");
  expect(data.publication.originalityAudit).toBe("incomplete");
  expect(data.publication.withdrawnScoredJobIds.slice().sort()).toEqual(["23b1cc7e2a6040af", "c43f7aba5ae84778"]);
  const prefill = data.kernels.find((kernel) => kernel.id === "muse_prefill_layer");
  expect(prefill.includedInSummary).toBe(false);
  for (const field of ["scored", "check", "ranking", "firstCampaignScoredSpeedup"]) {
    expect(prefill[field]).toBeNull();
  }
  expect(prefill.historical.scored.jobId).toBe("23b1cc7e2a6040af");
  expect(prefill.historical.scored.speedup).toBeCloseTo(1.9191900674402353, 10);
  for (const kernel of data.kernels.filter((item) => item.includedInSummary)) {
    expect(kernel.source.originalityStatus).toBe("not-comprehensively-audited");
    expect(kernel.source.reusedPublicSource).toBeNull();
  }
});

test("public disclosure names both exact-copy sources without claiming an organizer ruling", async ({ page }) => {
  await page.goto("disclosure.html");
  const text = await page.getByRole("main").innerText();
  expect(text).toContain("c43f7aba5ae84778");
  expect(text).toContain("23b1cc7e2a6040af");
  expect(text).toMatch(/byte-for-byte|verbatim|exact cop/i);
  expect(text).toMatch(/not.*audit|audit.*incomplete|not.*originality/i);
  expect(text).toMatch(/voluntar/i);
  expect(text).toContain("Madhavan113");
  expect(text).toContain("keell0renz");
  expect(text).toContain("vjeux");
  expect(text).toContain("alygg77");
  expect(text).toContain("ZhimaoL");
  expect(text).toMatch(/not necessarily sole authors/i);
  expect(await page.locator('a[href*="keell0renz"]').count()).toBeGreaterThan(0);
  expect(await page.locator('a[href*="Madhavan113"]').count()).toBeGreaterThan(0);
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
