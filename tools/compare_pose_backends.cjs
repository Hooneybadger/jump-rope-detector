/* npm install --prefix .benchmark-node playwright@1.58.2 */
const fs = require("node:fs");
const path = require("node:path");
let chromium;
try {
  ({ chromium } = require("playwright"));
} catch {
  ({ chromium } = require("../.benchmark-node/node_modules/playwright"));
}
const output = path.resolve(
  __dirname,
  process.env.BENCHMARK_OUTPUT || "../benchmark_results/60fps",
);
fs.mkdirSync(output, { recursive: true });
const origin = process.env.BENCHMARK_ORIGIN || "http://127.0.0.1:8766";
(async () => {
  const browser = await chromium.launch({
    executablePath:
      process.env.CHROME_PATH ||
      "C:/Program Files/Google/Chrome/Application/chrome.exe",
    headless: true,
    args: ["--use-angle=d3d11", "--ignore-gpu-blocklist"],
  });
  try {
    const page = await browser.newPage();
    page.on("pageerror", (e) => console.error("PAGE ERROR", e.message));
    await page.goto(origin);
    await page.exposeFunction("benchmarkProgress", (text) => console.log(text));
    await page.exposeFunction("benchmarkSave", (clip, result) =>
      fs.writeFileSync(
        path.join(output, clip + "_browser.json"),
        JSON.stringify(result, null, 2),
      ),
    );
    const completed = process.argv.includes("--resume")
      ? Object.fromEntries(
          fs
            .readdirSync(output)
            .filter((f) => f.endsWith("_browser.json"))
            .map((f) => {
              const r = JSON.parse(fs.readFileSync(path.join(output, f)));
              return [r.id, r];
            }),
        )
      : {};
    const results = await page.evaluate(async (completed) => {
      const { BrowserPose } = await import("/pose-client.js");
      async function post(url, value) {
        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(value || {}),
        });
        if (!response.ok)
          throw new Error(
            url + ": " + response.status + " " + (await response.text()),
          );
        return response.json();
      }
      const records = await (await fetch("/benchmark/manifest")).json();
      const results = [];
      for (const record of records) {
        if (completed[record.id]) {
          results.push(completed[record.id]);
          await window.benchmarkProgress("RESUME " + record.id);
          continue;
        }
        await window.benchmarkProgress(
          record.id + " legacy CPU: " + record.indices.length + " frames",
        );
        const legacy = await post("/benchmark/legacy/" + record.id);
        const pose = new BrowserPose();
        const loadStart = performance.now();
        try {
          const info = await pose.init();
          if (/swiftshader|llvmpipe|software/i.test(info.renderer))
            throw new Error(
              "Software renderer is not GPU evidence: " + info.renderer,
            );
          const initMs = performance.now() - loadStart;
          const canvas = document.createElement("canvas");
          canvas.width = 512;
          canvas.height = 288;
          const context = canvas.getContext("2d", { alpha: false });
          await post("/benchmark/start/" + record.id);
          const timings = [];
          for (let sample = 0; sample < record.indices.length; sample++) {
            const response = await fetch(
              "/benchmark/frame/" + record.id + "/" + sample,
            );
            if (!response.ok) throw new Error("Frame read failed");
            const bitmap = await createImageBitmap(await response.blob());
            context.drawImage(bitmap, 0, 0);
            bitmap.close();
            const timestampMs = (sample / (record.analysis_fps || 60)) * 1000;
            const start = performance.now();
            const result = await pose.detect(canvas, timestampMs);
            const workerMs = performance.now() - start;
            const body = {
              type: "pose",
              timestampMs,
              inferenceMs: result.inferenceMs,
              landmarks: result.landmarks,
            };
            const sendStart = performance.now();
            const state = await post(
              "/benchmark/pose/" + record.id + "/" + sample,
              body,
            );
            timings.push({
              inference_ms: result.inferenceMs,
              worker_ms: workerMs,
              server_ms: state.serverMs,
              compute_ms: workerMs + state.serverMs,
              roundtrip_ms: performance.now() - sendStart,
              payload_bytes: new TextEncoder().encode(JSON.stringify(body))
                .length,
            });
            if (sample && sample % 500 === 0)
              await window.benchmarkProgress(
                record.id + " GPU: " + sample + "/" + record.indices.length,
              );
          }
          const gpu = await post("/benchmark/finish/" + record.id);
          const result = {
            id: record.id,
            mode: record.mode,
            gt: record.gt,
            input: record,
            legacy,
            gpu,
            info,
            initMs,
            timings,
          };
          results.push(result);
          await window.benchmarkSave(record.id, result);
          await window.benchmarkProgress(
            "RESULT " +
              record.id +
              " GT=" +
              record.gt +
              " CPU=" +
              legacy.count +
              " GPU=" +
              gpu.count,
          );
        } finally {
          pose.close();
        }
      }
      return results;
    }, completed);
    fs.writeFileSync(
      path.join(output, "comparison.json"),
      JSON.stringify(
        {
          browser: await browser.version(),
          date: new Date().toISOString(),
          results,
        },
        null,
        2,
      ),
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
