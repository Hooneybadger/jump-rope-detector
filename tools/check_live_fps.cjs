const fs = require("node:fs");
const path = require("node:path");
let chromium;
try {
  ({ chromium } = require("playwright"));
} catch {
  ({ chromium } = require("../.benchmark-node/node_modules/playwright"));
}
function stats(v) {
  const a = [...v].sort((a, b) => a - b);
  return {
    mean: v.reduce((a, b) => a + b, 0) / v.length,
    p50: a[Math.floor((a.length - 1) * 0.5)],
    p95: a[Math.floor((a.length - 1) * 0.95)],
  };
}
(async () => {
  const browser = await chromium.launch({
    executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
    headless: true,
    args: [
      "--use-angle=d3d11",
      "--ignore-gpu-blocklist",
      "--use-fake-ui-for-media-stream",
      "--use-fake-device-for-media-stream",
      "--use-file-for-fake-video-capture=" +
        path.resolve(
          process.env.FPS_CAMERA_Y4M || ".benchmark-node/camera60.y4m",
        ),
    ],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    });
    const results = [];
    let active;
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("websocket", (ws) => {
      active = { url: ws.url(), sent: [], received: [], complete: null };
      ws.on("framesent", (f) => {
        if (typeof f.payload !== "string")
          throw new Error("JPEG upload on GPU path");
        const p = JSON.parse(f.payload);
        if (p.type === "pose")
          active.sent.push({
            ...p,
            landmarks: undefined,
            sentAt: performance.now(),
          });
      });
      ws.on("framereceived", (f) => {
        const p = JSON.parse(String(f.payload));
        if (p.type === "state")
          active.received.push({
            ...p,
            landmarks: undefined,
            receivedAt: performance.now(),
          });
        if (p.type === "complete") active.complete = p;
      });
    });
    await page.goto(process.env.FPS_ORIGIN || "http://127.0.0.1:8765/");
    await page.evaluate(
      async (credentials) => {
        const r = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(credentials),
        });
        if (!r.ok) throw new Error("Login failed");
        showApp(await r.json());
        const nativeFrameCallback =
          HTMLVideoElement.prototype.requestVideoFrameCallback;
        HTMLVideoElement.prototype.requestVideoFrameCallback = function (
          callback,
        ) {
          return nativeFrameCallback.call(this, (now, meta) => {
            if (this.id === "camera-source")
              window.videoCallbacks.push({
                now,
                mediaTime: meta.mediaTime,
                busy: state.sending,
              });
            callback(now, meta);
          });
        };
        const { BrowserPose } = await import("/pose-client.js");
        const detect = BrowserPose.prototype.detect;
        BrowserPose.prototype.detect = function (source, ts) {
          const record = { ts, mediaTime: state.lastAnalyzedMediaTime };
          window.poseCaptures.push(record);
          const start = performance.now();
          return detect.call(this, source, ts).then((result) => {
            record.workerMs = performance.now() - start;
            return result;
          });
        };
      },
      {
        username: process.env.FPS_USERNAME || "admin",
        password: process.env.FPS_PASSWORD || "browser-gpu-local-test",
      },
    );
    for (const mode of ["basic", "alternating", "double"]) {
      await page.evaluate(async (mode) => {
        closeSocket();
        stopMedia();
        window.poseCaptures = [];
        window.videoCallbacks = [];
        state.mode = mode;
        state.duration = 10;
        state.countdown = 3;
        await startWorkout();
      }, mode);
      await page.waitForFunction(
        () => state.phase === "COUNTING",
        {},
        { timeout: 60000 },
      );
      const settings = await page.evaluate(() =>
        state.stream.getVideoTracks()[0].getSettings(),
      );
      if (mode === "basic")
        await page.screenshot({
          path: path.resolve(
            __dirname,
            "../benchmark_results/60fps/smoke-live.png",
          ),
        });
      await page.waitForFunction(
        () => state.resultShown,
        {},
        { timeout: 20000 },
      );
      const captures = await page.evaluate(() => window.poseCaptures);
      const callbacks = await page.evaluate(() => window.videoCallbacks);
      const samples = active.received.flatMap((r, i) =>
        r.phase === "COUNTING" && r.elapsed >= 2 && r.elapsed < 9
          ? [{ capture: active.sent[i], reply: r }]
          : [],
      );
      const intervals = samples
        .slice(1)
        .map((s, i) => s.capture.timestampMs - samples[i].capture.timestampMs);
      const first = samples[0],
        last = samples.at(-1);
      const low = first.capture.timestampMs,
        high = last.capture.timestampMs;
      const media = captures
        .filter((c) => c.ts >= low && c.ts <= high)
        .map((c) => c.mediaTime);
      const unique = new Set(media).size;
      const delivered = callbacks.filter((c) => c.now >= low && c.now <= high);
      const timedCaptures = captures.filter((c) => c.ts >= low && c.ts <= high);
      const seconds = (high - low) / 1000;
      const result = {
        mode,
        settings,
        analysisFps: (samples.length - 1) / seconds,
        cameraCallbackFps: (delivered.length - 1) / seconds,
        busyCameraFrames: delivered.filter((c) => c.busy).length,
        workerMs: stats(timedCaptures.map((c) => c.workerMs)),
        uniqueAnalyzedFrames: unique,
        uniqueAnalysisFps: (unique - 1) / seconds,
        frameIntervalMs: stats(intervals),
        inferenceMs: stats(samples.map((s) => s.reply.inferenceMs)),
        serverMs: stats(samples.map((s) => s.reply.processingMs)),
        roundtripMs: stats(
          samples.map((s) => s.reply.receivedAt - s.capture.sentAt),
        ),
        samples: samples.length,
        seconds,
        complete: active.complete,
        captures,
      };
      results.push(result);
      console.log(JSON.stringify({ ...result, captures: undefined }));
    }
    fs.writeFileSync(
      path.resolve(__dirname, "../benchmark_results/60fps/live-fps.json"),
      JSON.stringify(
        {
          browser: browser.version(),
          source:
            "60 fps original AlternativeJump/05.mp4, seconds 29-49 replayed as fake webcam; not a physical webcam/network deployment",
          errors,
          results,
        },
        null,
        2,
      ),
    );
    if (errors.length) throw new Error(errors.join(";"));
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
