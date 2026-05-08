import { loadModelRuntime, predictOptionalModels } from "./modelRuntime.js";

let worker = null;
let workerAvailable = true;
let requestSeq = 0;
const pending = new Map();

export async function initInferenceWorker() {
  if (!workerAvailable) {
    return fallbackStatus();
  }
  try {
    const instance = getWorker();
    const response = await sendWorkerMessage(instance, { type: "INIT" }, 8000);
    return {
      status: response.status,
      manifest: response.manifest,
      worker: true
    };
  } catch {
    workerAvailable = false;
    return fallbackStatus();
  }
}

export async function predictInWorker(vector, options = {}) {
  const requestStartedAt = performance.now();
  if (workerAvailable) {
    try {
      const response = await sendWorkerMessage(getWorker(), {
        type: "PREDICT",
        vector,
        options
      });
      return (response.results || []).map((result) => ({
        ...result,
        uiBlockingTimeMs: performance.now() - requestStartedAt
      }));
    } catch {
      workerAvailable = false;
    }
  }
  const results = await predictOptionalModels(vector, options);
  return results.map((result) => ({
    ...result,
    uiBlockingTimeMs: performance.now() - requestStartedAt,
    payload: {
      ...(result.payload || {}),
      workerFallback: true
    }
  }));
}

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL("./inference.worker.js", import.meta.url), { type: "module" });
    worker.onmessage = (event) => {
      const message = event.data || {};
      const entry = pending.get(message.requestId);
      if (!entry) {
        return;
      }
      pending.delete(message.requestId);
      window.clearTimeout(entry.timeout);
      if (message.type === "ERROR") {
        entry.reject(new Error(message.error || "Worker error"));
      } else {
        entry.resolve(message);
      }
    };
    worker.onerror = (event) => {
      workerAvailable = false;
      for (const [requestId, entry] of pending) {
        pending.delete(requestId);
        window.clearTimeout(entry.timeout);
        entry.reject(new Error(event.message || "Worker error"));
      }
    };
  }
  return worker;
}

function sendWorkerMessage(instance, payload, timeoutMs = 30000) {
  const requestId = `req_${++requestSeq}`;
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      pending.delete(requestId);
      reject(new Error("Worker request timed out"));
    }, timeoutMs);
    pending.set(requestId, { resolve, reject, timeout });
    instance.postMessage({ ...payload, requestId });
  });
}

async function fallbackStatus() {
  const runtime = await loadModelRuntime();
  return {
    status: runtime.status,
    manifest: runtime.manifest,
    worker: false
  };
}
