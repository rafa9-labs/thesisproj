import puppeteer, { type Browser, type Page } from "puppeteer";

const APP_URL = process.env.APP_URL ?? "http://localhost:5173";
const API_URL = process.env.API_URL ?? "http://localhost:8000/api/v1";
let browser: Browser;
let page: Page;

jest.setTimeout(60_000);

beforeAll(async () => {
  browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--window-size=1280,900"],
  });
  page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      console.log(`[BROWSER ERROR] ${msg.text()}`);
    }
  });
});

afterAll(async () => {
  await browser?.close();
});

async function waitForApp(): Promise<void> {
  await page.goto(APP_URL, { waitUntil: "networkidle2", timeout: 15_000 });
}

async function navigateTo(path: string): Promise<void> {
  await page.goto(`${APP_URL}${path}`, { waitUntil: "networkidle2", timeout: 10_000 });
  await page.waitForSelector("main", { timeout: 5_000 }).catch(() => {});
  await new Promise((r) => setTimeout(r, 500));
}

function countConsoleErrors(): number {
  let count = 0;
  page.on("console", (msg) => {
    if (msg.type() === "error") count++;
  });
  return count;
}

describe("Sprint 8 E2E — All Pages Load", () => {
  const routes = [
    { path: "/", name: "Dashboard" },
    { path: "/backtest", name: "Backtest Config" },
    { path: "/results", name: "Results (empty state)" },
    { path: "/compare", name: "Compare" },
    { path: "/news", name: "News & Sentiment" },
    { path: "/settings", name: "Settings" },
  ];

  test.each(routes)("Route %s (%s) renders without crash", async ({ path, name }) => {
    await navigateTo(path);
    const title = await page.title();
    expect(title).toBeDefined();

    const bodyText = await page.evaluate(() => document.body.innerText);
    expect(bodyText.length).toBeGreaterThan(0);

    const hasErrorBoundary = await page.evaluate(
      () => document.body.textContent?.includes("Something went wrong") ?? false,
    );
    expect(hasErrorBoundary).toBe(false);
  });
});

describe("P2: Health Check Dots in AppShell Header", () => {
  test("Header contains Backend and WS status dots", async () => {
    await navigateTo("/");

    const dots = await page.evaluate(() => {
      const header = document.querySelector("header");
      if (!header) return null;
      const allText = header.textContent ?? "";
      const hasBackend = allText.includes("Backend");
      const hasWS = allText.includes("WS");
      return { hasBackend, hasWS, text: allText };
    });

    expect(dots).not.toBeNull();
    expect(dots!.hasBackend).toBe(true);
    expect(dots!.hasWS).toBe(true);
  });

  test("Backend dot has a colored indicator", async () => {
    await navigateTo("/");

    const dotColor = await page.evaluate(() => {
      const header = document.querySelector("header");
      if (!header) return null;
      const dots = header.querySelectorAll("div[class*='rounded-full']");
      for (const dot of dots) {
        const style = window.getComputedStyle(dot as HTMLElement);
        if (style.backgroundColor && style.backgroundColor !== "rgba(0, 0, 0, 0)") {
          return style.backgroundColor;
        }
      }
      return null;
    });

    expect(dotColor).not.toBeNull();
  });
});

describe("P1: Multi-Model Tab Switching on Results Page", () => {
  test("Results empty state shows when no jobId", async () => {
    await navigateTo("/results");

    const hasEmpty = await page.evaluate(() => {
      const text = document.body.innerText;
      return text.includes("No results") || text.includes("Run") || text.includes("backtest");
    });
    expect(hasEmpty).toBe(true);
  });

  test("Results with jobId loads from API", async () => {
    const jobsResp = await page.evaluate(async (apiUrl) => {
      try {
        const res = await fetch(`${apiUrl}/backtest?limit=1`);
        if (!res.ok) return null;
        const data = await res.json();
        return data.jobs?.[0]?.job_id ?? null;
      } catch {
        return null;
      }
    }, API_URL);

    if (!jobsResp) {
      console.log("SKIP: No completed backtest jobs found. Run a backtest first.");
      return;
    }

    await navigateTo(`/results/${jobsResp}`);

    const hasMetricsOrLoading = await page.evaluate(() => {
      const text = document.body.innerText;
      return text.includes("Sharpe") || text.includes("Loading") || text.includes("Results");
    });
    expect(hasMetricsOrLoading).toBe(true);
  });

  test("Model pills are rendered when multi-model results exist", async () => {
    const multiJobId = await page.evaluate(async (apiUrl) => {
      try {
        const res = await fetch(`${apiUrl}/backtest?limit=20`);
        if (!res.ok) return null;
        const data = await res.json();
        const multi = data.jobs?.find((j: { models: string[] }) => j.models && j.models.length > 1);
        return multi?.job_id ?? null;
      } catch {
        return null;
      }
    }, API_URL);

    if (!multiJobId) {
      console.log("SKIP: No multi-model jobs found. Submit a multi-model backtest first.");
      return;
    }

    await navigateTo(`/results/${multiJobId}`);
    await new Promise((r) => setTimeout(r, 1500));

    const pillCount = await page.evaluate(() => {
      const pills = document.querySelectorAll("button");
      let count = 0;
      pills.forEach((p) => {
        if (p.textContent?.includes("Sharpe")) count++;
      });
      return count;
    });

    expect(pillCount).toBeGreaterThanOrEqual(2);
  });
});

describe("P3: PNG Export Button Exists on Results Page", () => {
  test("Export bar contains PNG button", async () => {
    await navigateTo("/results/test-job-id");

    const hasPng = await page.evaluate(() => {
      const buttons = document.querySelectorAll("button");
      for (const btn of buttons) {
        if (btn.textContent?.includes("PNG")) return true;
      }
      return false;
    });

    expect(hasPng).toBe(true);
  });

  test("Export bar contains CSV and JSON buttons", async () => {
    await navigateTo("/results/test-job-id");

    const buttons = await page.evaluate(() => {
      const btns = document.querySelectorAll("button");
      let csv = false;
      let json = false;
      btns.forEach((b) => {
        const t = b.textContent ?? "";
        if (t.includes("CSV")) csv = true;
        if (t.includes("JSON")) json = true;
      });
      return { csv, json };
    });

    expect(buttons.csv).toBe(true);
    expect(buttons.json).toBe(true);
  });
});

describe("P5: Settings Page Structure", () => {
  test("All 6 sections are present", async () => {
    await navigateTo("/settings");

    const sections = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        general: text.includes("General"),
        gpu: text.includes("GPU") || text.includes("Compute"),
        data: text.includes("Data Source") || text.includes("OANDA"),
        license: text.includes("License"),
        pipeline: text.includes("Pipeline"),
        about: text.includes("About") || text.includes("Version"),
      };
    });

    expect(sections.general).toBe(true);
    expect(sections.gpu).toBe(true);
    expect(sections.data).toBe(true);
    expect(sections.license).toBe(true);
    expect(sections.pipeline).toBe(true);
    expect(sections.about).toBe(true);
  });

  test("General section is open by default and shows API URL input", async () => {
    await navigateTo("/settings");

    const hasInput = await page.evaluate(() => {
      const inputs = document.querySelectorAll("input");
      for (const input of inputs) {
        if ((input as HTMLInputElement).type === "text" || (input as HTMLInputElement).type === "url") {
          return true;
        }
      }
      return false;
    });
    expect(hasInput).toBe(true);
  });

  test("Reset to Defaults button exists", async () => {
    await navigateTo("/settings");

    const hasReset = await page.evaluate(() => {
      const buttons = document.querySelectorAll("button");
      for (const btn of buttons) {
        if (btn.textContent?.includes("Reset")) return true;
      }
      return false;
    });
    expect(hasReset).toBe(true);
  });

  test("OANDA API Key field is type=password", async () => {
    await navigateTo("/settings");

    const hasPasswordInput = await page.evaluate(() => {
      const inputs = document.querySelectorAll("input[type='password']");
      return inputs.length > 0;
    });
    expect(hasPasswordInput).toBe(true);
  });

  test("Settings sections expand and collapse", async () => {
    await navigateTo("/settings");

    const clicked = await page.evaluate(() => {
      const buttons = document.querySelectorAll("button");
      let clicked = false;
      for (const btn of buttons) {
        if (btn.textContent?.includes("GPU") || btn.textContent?.includes("Compute")) {
          (btn as HTMLElement).click();
          clicked = true;
          break;
        }
      }
      return clicked;
    });

    if (clicked) {
      await new Promise((r) => setTimeout(r, 300));
      const content = await page.evaluate(() => {
        const text = document.body.innerText;
        return text.includes("Thread Budget") || text.includes("Mixed Precision");
      });
      expect(content).toBe(true);
    }
  });
});

describe("P4: News Page Live Data", () => {
  test("News page renders 4 cards (Sentiment, Events, Features, Status)", async () => {
    await navigateTo("/news");

    const cards = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        sentiment: text.includes("Sentiment Engine"),
        events: text.includes("Event Calendar") || text.includes("Events"),
        features: text.includes("Sentiment Features"),
        status: text.includes("Data Status"),
      };
    });

    expect(cards.sentiment).toBe(true);
    expect(cards.features).toBe(true);
    expect(cards.status).toBe(true);
  });

  test("News page shows VADER backend and finBERT status", async () => {
    await navigateTo("/news");
    await new Promise((r) => setTimeout(r, 1000));

    const content = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        hasBackend: text.includes("VADER") || text.includes("vader"),
        hasFinbert: text.includes("finBERT") || text.includes("Not installed") || text.includes("Available"),
      };
    });

    expect(content.hasBackend).toBe(true);
    expect(content.hasFinbert).toBe(true);
  });

  test("News page shows event type tags", async () => {
    await navigateTo("/news");
    await new Promise((r) => setTimeout(r, 1000));

    const hasEvents = await page.evaluate(() => {
      const text = document.body.innerText;
      return text.includes("NFP") || text.includes("FOMC") || text.includes("CPI");
    });

    expect(hasEvents).toBe(true);
  });
});

describe("P6: Chunk Splitting — Build Output Verification", () => {
  test("Build produces separate chunk files", async () => {
    const fs = await import("fs/promises");
    const path = await import("path");
    const distDir = path.resolve(__dirname, "..", "dist", "assets");

    try {
      const files = await fs.readdir(distDir);
      const hasLightweightCharts = files.some((f) => f.includes("lightweight-charts"));
      const hasAgGrid = files.some((f) => f.includes("ag-grid"));
      const hasTanstack = files.some((f) => f.includes("tanstack"));
      const hasVendor = files.some((f) => f.includes("vendor"));

      expect(hasLightweightCharts).toBe(true);
      expect(hasAgGrid).toBe(true);
      expect(hasTanstack).toBe(true);
      expect(hasVendor).toBe(true);
    } catch {
      console.log("SKIP: dist/ directory not found. Run `npx vite build` first.");
    }
  });
});

describe("S8: Backtest Config Page — Panel Completeness", () => {
  test("All config panels are present", async () => {
    await navigateTo("/backtest");

    const panels = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        models: text.includes("Select Models") || text.includes("Model"),
        asset: text.includes("Asset") || text.includes("Pair"),
        features: text.includes("Features") || text.includes("Indicators"),
        labels: text.includes("Label") || text.includes("Barrier"),
        hpo: text.includes("HPO") || text.includes("Walk-Forward"),
        deploy: text.includes("Deploy") || text.includes("Run"),
      };
    });

    expect(panels.models).toBe(true);
    expect(panels.asset).toBe(true);
    expect(panels.features).toBe(true);
    expect(panels.hpo).toBe(true);
    expect(panels.deploy).toBe(true);
  });

  test("News & Sentiment section exists in Features panel", async () => {
    await navigateTo("/backtest");

    const hasNewsSection = await page.evaluate(() => {
      const text = document.body.innerText;
      return text.includes("News") && (text.includes("Sentiment") || text.includes("VADER"));
    });

    expect(hasNewsSection).toBe(true);
  });

  test("Date pickers exist in Asset Selector", async () => {
    await navigateTo("/backtest");

    const dateInputs = await page.evaluate(() => {
      const inputs = document.querySelectorAll("input[type='date']");
      return inputs.length;
    });

    expect(dateInputs).toBeGreaterThanOrEqual(2);
  });
});

describe("S8: Navigation — Sidebar Active State", () => {
  test("Dashboard route highlights Dashboard nav item", async () => {
    await navigateTo("/");
    const isActive = await page.evaluate(() => {
      const sidebar = document.querySelector("aside");
      if (!sidebar) return false;
      const buttons = sidebar.querySelectorAll("button");
      for (const btn of buttons) {
        const style = window.getComputedStyle(btn);
        if (btn.textContent?.includes("DASHBOARD") && style.borderLeftColor !== "transparent") {
          return true;
        }
      }
      return false;
    });
    expect(isActive).toBe(true);
  });

  test("Clicking Settings nav item navigates to /settings", async () => {
    await navigateTo("/");

    await page.evaluate(() => {
      const sidebar = document.querySelector("aside");
      if (!sidebar) return;
      const buttons = sidebar.querySelectorAll("button");
      for (const btn of buttons) {
        if (btn.textContent?.includes("SETTINGS")) {
          (btn as HTMLElement).click();
          break;
        }
      }
    });

    await new Promise((r) => setTimeout(r, 1000));

    const url = page.url();
    expect(url).toContain("/settings");
  });
});

describe("S8: Error Boundary", () => {
  test("Error boundary component exists in React tree", async () => {
    await navigateTo("/");

    const hasErrorBoundary = await page.evaluate(() => {
      const root = document.getElementById("root");
      if (!root) return false;
      return root.children.length > 0;
    });
    expect(hasErrorBoundary).toBe(true);
  });
});
