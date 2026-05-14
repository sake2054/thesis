import { apiGet, apiPost } from "../api.js";

const DATASET_LABELS = {
  dsl: "DSL",
  mmc: "MMC",
  collected_data_analysis: "Collected CSV analysis"
};

export async function runBrowserEvaluation({ dataset, featureMode = "current", adminPin, deviceLabel, runId, onProgress }) {
  if (!dataset || !DATASET_LABELS[dataset]) {
    throw new Error("Unknown evaluation dataset.");
  }
  if (!adminPin) {
    throw new Error("Admin PIN is required.");
  }

  const headers = { "x-admin-pin": adminPin };
  onProgress?.({ stage: "run", message: "Preparing browser evaluation run." });
  const run = await apiPost("/api/admin/evaluation/runs", { runId }, { headers });
  const baseEnvironment = await collectEvaluationEnvironment(deviceLabel);
  const memoryBefore = await collectBrowserMemory("before_worker");
  baseEnvironment.browserMemory = { beforeWorker: memoryBefore };
  const workerResult = await runEvaluationWorker({
    dataset,
    featureMode,
    adminPin,
    deviceLabel,
    baseEnvironment
  }, onProgress);
  const memoryAfter = await collectBrowserMemory("after_worker");
  workerResult.environment = {
    ...workerResult.environment,
    browserMemory: {
      ...(workerResult.environment.browserMemory || {}),
      beforeWorker: memoryBefore,
      afterWorker: memoryAfter
    }
  };
  workerResult.artifacts = replaceEnvironmentManifest(workerResult.artifacts, workerResult.environment);

  const attemptId = makeAttemptId();
  const extraArtifacts = [
    ...makeChartArtifacts(workerResult),
    makeMemoryArtifact(workerResult.dataset || outputDatasetKey(dataset, featureMode), workerResult.environment)
  ].filter(Boolean);
  onProgress?.({ stage: "save", message: `Saving ${workerResult.artifacts.length + extraArtifacts.length} artifacts.` });
  const artifactDataset = workerResult.dataset || outputDatasetKey(dataset, featureMode);
  const saved = await apiPost(`/api/admin/evaluation/runs/${run.runId}/artifacts`, {
    dataset: artifactDataset,
    environmentId: workerResult.environment.environmentId,
    attemptId,
    environment: workerResult.environment,
    artifacts: [...workerResult.artifacts, ...extraArtifacts]
  }, { headers });
  const status = await apiGet(`/api/admin/evaluation/runs/${run.runId}`, { headers });
  onProgress?.({ stage: "done", message: `${DATASET_LABELS[dataset]} evaluation saved.` });

  return {
    runId: run.runId,
    attemptId,
    dataset: artifactDataset,
    sourceDataset: dataset,
    featureMode,
    saved,
    status,
    ...workerResult
  };
}

export async function loadEvaluationRun(runId, adminPin) {
  if (!runId || !adminPin) {
    return null;
  }
  return apiGet(`/api/admin/evaluation/runs/${encodeURIComponent(runId)}`, {
    headers: { "x-admin-pin": adminPin }
  });
}

function runEvaluationWorker(payload, onProgress) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./evaluation.worker.js", import.meta.url), { type: "module" });
    worker.addEventListener("message", (event) => {
      const message = event.data || {};
      if (message.type === "progress") {
        onProgress?.(message);
      } else if (message.type === "done") {
        worker.terminate();
        resolve(message.result);
      } else if (message.type === "error") {
        worker.terminate();
        reject(new Error(message.error || "Evaluation worker failed."));
      }
    });
    worker.addEventListener("error", (event) => {
      worker.terminate();
      reject(new Error(event.message || "Evaluation worker crashed."));
    });
    worker.postMessage({ type: "run", payload });
  });
}

async function collectEvaluationEnvironment(deviceLabel) {
  const userAgentData = await readUserAgentData();
  return {
    deviceLabel: deviceLabel || "not_available",
    timestamp: new Date().toISOString(),
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "not_available",
    language: navigator.language || "not_available",
    userAgent: navigator.userAgent || "not_available",
    platform: navigator.platform || "not_available",
    userAgentData,
    screen: {
      width: screen?.width ?? "not_available",
      height: screen?.height ?? "not_available",
      availWidth: screen?.availWidth ?? "not_available",
      availHeight: screen?.availHeight ?? "not_available",
      colorDepth: screen?.colorDepth ?? "not_available",
      pixelDepth: screen?.pixelDepth ?? "not_available"
    },
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight,
      devicePixelRatio: window.devicePixelRatio || 1
    },
    devicePixelRatio: window.devicePixelRatio || 1,
    maxTouchPoints: navigator.maxTouchPoints ?? "not_available",
    hardwareConcurrency: navigator.hardwareConcurrency ?? "not_available",
    deviceMemory: navigator.deviceMemory ?? "not_available",
    webglAvailable: hasWebGl(),
    wasmAvailable: typeof WebAssembly !== "undefined",
    crossOriginIsolated: Boolean(window.crossOriginIsolated),
    appVersion: import.meta.env?.VITE_APP_VERSION || "keystroke-auth-demo/0.1.0",
    locationOrigin: window.location.origin
  };
}

async function collectBrowserMemory(label) {
  const snapshot = {
    label,
    timestamp: new Date().toISOString(),
    source: "not_available",
    usedJSHeapSize: "not_available",
    totalJSHeapSize: "not_available",
    jsHeapSizeLimit: "not_available",
    bytes: "not_available"
  };
  if (performance.memory) {
    snapshot.source = "performance.memory";
    snapshot.usedJSHeapSize = performance.memory.usedJSHeapSize ?? "not_available";
    snapshot.totalJSHeapSize = performance.memory.totalJSHeapSize ?? "not_available";
    snapshot.jsHeapSizeLimit = performance.memory.jsHeapSizeLimit ?? "not_available";
    snapshot.bytes = snapshot.usedJSHeapSize;
  }
  if (typeof performance.measureUserAgentSpecificMemory === "function") {
    try {
      const measured = await performance.measureUserAgentSpecificMemory();
      snapshot.source = snapshot.source === "not_available"
        ? "measureUserAgentSpecificMemory"
        : `${snapshot.source}+measureUserAgentSpecificMemory`;
      snapshot.bytes = measured.bytes ?? snapshot.bytes;
      snapshot.breakdown = measured.breakdown || null;
    } catch (error) {
      snapshot.measureUserAgentSpecificMemoryError = error.message || String(error);
    }
  }
  return snapshot;
}

function replaceEnvironmentManifest(artifacts, environment) {
  const next = artifacts.filter((artifact) => artifact.name !== "environment_manifest.json");
  next.unshift({
    name: "environment_manifest.json",
    content: JSON.stringify(environment, null, 2),
    encoding: "utf8"
  });
  return next;
}

function makeMemoryArtifact(dataset, environment) {
  const rows = [
    { dataset, environmentId: environment.environmentId, ...(environment.browserMemory?.beforeWorker || {}) },
    { dataset, environmentId: environment.environmentId, ...(environment.browserMemory?.afterWorker || {}) }
  ];
  return {
    name: "browser_memory_log.csv",
    content: toCsv(rows),
    encoding: "utf8"
  };
}

async function readUserAgentData() {
  const data = navigator.userAgentData;
  if (!data) {
    return "not_available";
  }
  try {
    const highEntropy = await data.getHighEntropyValues?.([
      "architecture",
      "bitness",
      "brands",
      "fullVersionList",
      "mobile",
      "model",
      "platform",
      "platformVersion",
      "wow64"
    ]);
    return highEntropy || data.toJSON?.() || "not_available";
  } catch {
    return data.toJSON?.() || "not_available";
  }
}

function hasWebGl() {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl") || canvas.getContext("experimental-webgl"));
  } catch {
    return false;
  }
}

function makeAttemptId() {
  const stamp = new Date().toISOString().replaceAll(":", "-").replace(/\.\d{3}Z$/, "Z");
  const random = crypto.randomUUID?.().slice(0, 8) || Math.random().toString(16).slice(2, 10);
  return `${stamp}-${random}`;
}

function outputDatasetKey(dataset, featureMode) {
  return featureMode === "enriched" ? `${dataset}_enriched_features` : dataset;
}

function makeChartArtifacts(result) {
  return [
    drawRocChart(result.chartData?.rocSeries || {}),
    drawEfficiencyBars(result.chartData?.summaryRows || []),
    drawEerTrainingChart(result.chartData?.dataEfficiencyRows || [])
  ].filter(Boolean);
}

function drawRocChart(rocSeries) {
  const entries = Object.entries(rocSeries).filter(([, points]) => Array.isArray(points) && points.length);
  if (!entries.length) {
    return null;
  }
  return canvasArtifact("roc_curves.png", 900, 620, (ctx, box) => {
    drawChartShell(ctx, box, "ROC Curves", "False Positive Rate", "True Positive Rate");
    const colors = ["#89f6c5", "#ffbf69", "#74c0fc", "#ff8787"];
    entries.forEach(([model, points], index) => {
      ctx.strokeStyle = colors[index % colors.length];
      ctx.lineWidth = 3;
      ctx.beginPath();
      points.forEach((point, pointIndex) => {
        const x = box.x + clamp(Number(point.fpr), 0, 1) * box.w;
        const y = box.y + box.h - clamp(Number(point.tpr), 0, 1) * box.h;
        if (pointIndex === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      drawLegendItem(ctx, model, colors[index % colors.length], box.x + box.w - 260, box.y + 28 + index * 28);
    });
  });
}

function drawEfficiencyBars(summaryRows) {
  const rows = summaryRows.filter((row) => row.Model);
  if (!rows.length) {
    return null;
  }
  const metrics = [
    { key: "EER", label: "EER", scale: 100 },
    { key: "Accuracy", label: "Accuracy", scale: 100 },
    { key: "Inference Time (ms/sample)", label: "ms/sample", scale: 1 },
    { key: "UI Blocking Time (ms/test batch)", label: "latency ms", scale: 1 }
  ];
  return canvasArtifact("efficiency_bars.png", 1020, 660, (ctx) => {
    fillCanvas(ctx, 1020, 660);
    ctx.fillStyle = "#f4f7f5";
    ctx.font = "700 26px system-ui, sans-serif";
    ctx.fillText("Model Efficiency Summary", 42, 52);
    const chartX = 70;
    const rowHeight = 58;
    const groupGap = 44;
    let y = 104;
    const colors = ["#89f6c5", "#ffbf69", "#74c0fc"];
    for (const metric of metrics) {
      const values = rows.map((row) => Number(row[metric.key]) * metric.scale).filter(Number.isFinite);
      const maxValue = Math.max(...values, 1);
      ctx.fillStyle = "#d6ded9";
      ctx.font = "700 16px system-ui, sans-serif";
      ctx.fillText(metric.label, chartX, y - 14);
      rows.forEach((row, index) => {
        const value = Number(row[metric.key]) * metric.scale;
        const width = Math.max(1, (Number.isFinite(value) ? value : 0) / maxValue * 520);
        const barY = y + index * rowHeight;
        ctx.fillStyle = "rgba(255,255,255,0.08)";
        roundRect(ctx, chartX + 230, barY, 560, 28, 6);
        ctx.fill();
        ctx.fillStyle = colors[index % colors.length];
        roundRect(ctx, chartX + 230, barY, width, 28, 6);
        ctx.fill();
        ctx.fillStyle = "#f4f7f5";
        ctx.font = "600 14px system-ui, sans-serif";
        ctx.fillText(row.Model, chartX, barY + 20);
        ctx.fillText(formatChartNumber(value), chartX + 810, barY + 20);
      });
      y += rows.length * rowHeight + groupGap;
    }
  });
}

function drawEerTrainingChart(rows) {
  const entries = rows.filter((row) => Number.isFinite(Number(row.EER)));
  if (!entries.length) {
    return null;
  }
  const models = [...new Set(entries.map((row) => row.Model))];
  const sizes = [...new Set(entries.map((row) => Number(row["Genuine Training Samples"])))].sort((a, b) => a - b);
  if (!models.length || !sizes.length) {
    return null;
  }
  return canvasArtifact("eer_vs_training_size.png", 900, 620, (ctx, box) => {
    drawChartShell(ctx, box, "EER vs Training Size", "Genuine Training Samples", "EER");
    const colors = ["#89f6c5", "#ffbf69", "#74c0fc"];
    const maxSize = Math.max(...sizes);
    models.forEach((model, index) => {
      const modelRows = entries.filter((row) => row.Model === model).sort((a, b) => Number(a["Genuine Training Samples"]) - Number(b["Genuine Training Samples"]));
      ctx.strokeStyle = colors[index % colors.length];
      ctx.fillStyle = colors[index % colors.length];
      ctx.lineWidth = 3;
      ctx.beginPath();
      modelRows.forEach((row, pointIndex) => {
        const size = Number(row["Genuine Training Samples"]);
        const eer = clamp(Number(row.EER), 0, 1);
        const x = box.x + (size / maxSize) * box.w;
        const y = box.y + box.h - eer * box.h;
        if (pointIndex === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      for (const row of modelRows) {
        const x = box.x + (Number(row["Genuine Training Samples"]) / maxSize) * box.w;
        const y = box.y + box.h - clamp(Number(row.EER), 0, 1) * box.h;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
      drawLegendItem(ctx, model, colors[index % colors.length], box.x + box.w - 260, box.y + 28 + index * 28);
    });
  });
}

function canvasArtifact(name, width, height, draw) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return null;
  }
  const box = { x: 92, y: 92, w: width - 150, h: height - 170 };
  draw(ctx, box);
  const dataUrl = canvas.toDataURL("image/png");
  return {
    name,
    encoding: "base64",
    content: dataUrl.slice(dataUrl.indexOf(",") + 1)
  };
}

function drawChartShell(ctx, box, title, xLabel, yLabel) {
  fillCanvas(ctx, box.x + box.w + 58, box.y + box.h + 78);
  ctx.fillStyle = "#f4f7f5";
  ctx.font = "700 26px system-ui, sans-serif";
  ctx.fillText(title, 42, 52);
  ctx.strokeStyle = "rgba(255,255,255,0.26)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(box.x, box.y);
  ctx.lineTo(box.x, box.y + box.h);
  ctx.lineTo(box.x + box.w, box.y + box.h);
  ctx.stroke();
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.fillStyle = "#a7b4ae";
  ctx.font = "12px system-ui, sans-serif";
  for (let i = 0; i <= 5; i += 1) {
    const x = box.x + (i / 5) * box.w;
    const y = box.y + box.h - (i / 5) * box.h;
    ctx.beginPath();
    ctx.moveTo(box.x, y);
    ctx.lineTo(box.x + box.w, y);
    ctx.moveTo(x, box.y);
    ctx.lineTo(x, box.y + box.h);
    ctx.stroke();
    ctx.fillText((i / 5).toFixed(1), box.x - 36, y + 4);
    ctx.fillText((i / 5).toFixed(1), x - 8, box.y + box.h + 22);
  }
  ctx.fillStyle = "#d6ded9";
  ctx.font = "600 14px system-ui, sans-serif";
  ctx.fillText(xLabel, box.x + box.w / 2 - 70, box.y + box.h + 54);
  ctx.save();
  ctx.translate(30, box.y + box.h / 2 + 70);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function drawLegendItem(ctx, label, color, x, y) {
  ctx.fillStyle = color;
  roundRect(ctx, x, y - 12, 18, 10, 3);
  ctx.fill();
  ctx.fillStyle = "#f4f7f5";
  ctx.font = "600 13px system-ui, sans-serif";
  ctx.fillText(label, x + 28, y - 3);
}

function fillCanvas(ctx, width, height) {
  ctx.fillStyle = "#101413";
  ctx.fillRect(0, 0, width, height);
}

function roundRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.min(Math.max(value, min), max);
}

function formatChartNumber(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function toCsv(rows) {
  if (!rows.length) return "";
  const headers = Array.from(rows.reduce((set, row) => {
    Object.keys(row).forEach((key) => set.add(key));
    return set;
  }, new Set()));
  return `${headers.join(",")}\n${rows.map((row) => headers.map((header) => csvCell(row[header])).join(",")).join("\n")}\n`;
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = typeof value === "object" ? JSON.stringify(value) : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}
