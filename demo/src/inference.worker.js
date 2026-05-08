import { loadModelRuntime, predictWithRuntime } from "./modelRuntime.js";

let runtimePromise = null;

self.onmessage = async (event) => {
  const message = event.data || {};
  try {
    if (message.type === "INIT") {
      runtimePromise = loadModelRuntime();
      const runtime = await runtimePromise;
      self.postMessage({
        type: "LOAD_STATUS",
        requestId: message.requestId,
        status: runtime.status,
        manifest: runtime.manifest
      });
      return;
    }

    if (message.type === "PREDICT") {
      if (!runtimePromise) {
        runtimePromise = loadModelRuntime();
      }
      const runtime = await runtimePromise;
      const start = performance.now();
      const results = await predictWithRuntime(runtime, message.vector || [], message.options || {});
      self.postMessage({
        type: "PREDICT_RESULT",
        requestId: message.requestId,
        results,
        inferenceTimeMs: performance.now() - start
      });
      return;
    }

    self.postMessage({
      type: "ERROR",
      requestId: message.requestId,
      error: `Unknown worker message type: ${message.type}`
    });
  } catch (error) {
    self.postMessage({
      type: "ERROR",
      requestId: message.requestId,
      error: error.message || "Worker inference failed"
    });
  }
};
