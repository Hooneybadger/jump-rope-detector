/* Classic worker: MediaPipe 0.10.21 loads its WASM glue with importScripts. */
let landmarker;
let lastTimestamp = -1;
self.onmessage = async ({ data }) => {
  const { id, type, bitmap, timestampMs } = data;
  try {
    if (type === "init") {
      const { FilesetResolver, PoseLandmarker } = await import("./vendor/mediapipe/vision_bundle.mjs");
      const vision = await FilesetResolver.forVisionTasks(new URL("./vendor/mediapipe/wasm", self.location.href).href);
      const canvas = new OffscreenCanvas(512, 288);
      const gl = canvas.getContext("webgl2");
      if (!gl) throw new Error("WebGL2 is unavailable");
      const info = gl.getExtension("WEBGL_debug_renderer_info");
      const renderer = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
      landmarker = await PoseLandmarker.createFromOptions(vision, {
        canvas,
        baseOptions: { modelAssetPath: new URL("./vendor/mediapipe/pose_landmarker_full.task", self.location.href).href, delegate: "GPU" },
        runningMode: "VIDEO", numPoses: 1,
        minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5, outputSegmentationMasks: false,
      });
      self.postMessage({ id, type: "ready", delegate: "GPU", renderer });
    } else if (type === "frame") {
      if (!landmarker || !Number.isFinite(timestampMs) || timestampMs <= lastTimestamp) throw new Error("Invalid pose frame timestamp");
      lastTimestamp = timestampMs;
      const start = performance.now();
      const result = landmarker.detectForVideo(bitmap, timestampMs);
      const inferenceMs = performance.now() - start;
      const landmarks = (result.landmarks[0] || []).map(p => [p.x, p.y, p.z, p.visibility ?? 0, p.presence ?? 0]);
      self.postMessage({ id, type: "pose", timestampMs, inferenceMs, landmarks });
    } else {
      throw new Error("Unknown pose worker message");
    }
  } catch (error) {
    self.postMessage({ id, type: "error", message: String(error.message || error) });
  } finally {
    bitmap?.close();
  }
};
