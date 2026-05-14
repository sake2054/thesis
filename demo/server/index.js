import crypto from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import express from "express";

import {
  collectAdminMetrics,
  computeCalibrationFromGenuineScores,
  deleteEmptyAttempt,
  exportAttemptFeatures,
  exportEventPairs,
  exportModelResults,
  exportMonitoringWindows,
  exportQualitySummary,
  findParticipantByCodeHash,
  getAttempt,
  insertEvents,
  insertFeatures,
  insertMonitoringWindow,
  insertParticipant,
  insertResults,
  insertSession,
  listCalibrations,
  openDatabase,
  queryTableRows,
  upsertAttempt,
  upsertCalibration
} from "./db.js";
import { loadReferenceMetrics } from "./referenceMetrics.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const workspaceRoot = path.resolve(rootDir, "..");
const evaluationRoot = path.join(workspaceRoot, "evaluation_runs");
const isProduction = process.env.NODE_ENV === "production";
const port = Number(process.env.PORT || 3000);
const host = process.env.HOST || "127.0.0.1";
const adminPin = process.env.ADMIN_PIN || "change-me";
const consentVersion = process.env.CONSENT_VERSION || "2026-04-26";

const envConfig = {
  storeRawText: envBool("STORE_RAW_TEXT", true),
  storeKeyValue: envBool("STORE_KEY_VALUE", true),
  storeIpAddress: envBool("STORE_IP_ADDRESS", true),
  storeParticipantCode: envBool("STORE_PARTICIPANT_CODE", true),
  showDevControls: envBool("SHOW_DEV_CONTROLS", false),
  continuousMonitoringEnabled: envBool("CONTINUOUS_MONITORING_ENABLED", false),
  monitoringWindowChars: Number(process.env.MONITORING_WINDOW_CHARS || 120),
  monitoringStepChars: Number(process.env.MONITORING_STEP_CHARS || 60),
  pastePolicyFixed: process.env.PASTE_POLICY_FIXED || "excluded",
  pastePolicyFree: process.env.PASTE_POLICY_FREE || "low_quality",
  calibrationGenuineQuantile: Number(process.env.CALIBRATION_GENUINE_QUANTILE || 0.05),
  promptSetPath: process.env.PROMPT_SET_PATH || "server/prompts.default.json"
};

const app = express();
const db = openDatabase();
const referenceMetrics = loadReferenceMetrics();
const promptSetsPayload = loadPromptSets();

app.set("trust proxy", true);
app.use(express.json({ limit: "100mb" }));

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, node: process.version });
});

app.get("/api/config", (_req, res) => {
  res.json({
    consentVersion,
    referenceMetrics,
    rawTextStorageEnabled: envConfig.storeRawText,
    keyValueStorageEnabled: envConfig.storeKeyValue,
    ipAddressStorageEnabled: envConfig.storeIpAddress,
    participantCodeStorageEnabled: envConfig.storeParticipantCode,
    showDevControls: envConfig.showDevControls,
    continuousMonitoringEnabled: envConfig.continuousMonitoringEnabled,
    monitoringWindowChars: envConfig.monitoringWindowChars,
    monitoringStepChars: envConfig.monitoringStepChars,
    pastePolicies: {
      fixed: envConfig.pastePolicyFixed,
      free: envConfig.pastePolicyFree
    },
    roles: ["genuine", "imposter"],
    deviceClasses: ["desktop", "mobile", "tablet", "unknown"],
    promptSets: promptSetsPayload.promptSets
  });
});

app.get("/api/prompts", (_req, res) => {
  res.json(promptSetsPayload);
});

app.post("/api/consent", (req, res, next) => {
  try {
    const body = req.body || {};
    if (body.accepted !== true) {
      return res.status(400).json({ error: "Consent must be accepted before collection." });
    }

    const participantCode = normalizeOptionalText(body.participantCode);
    const participantCodeHash = participantCode ? hashCode(participantCode) : null;
    const existing = participantCodeHash ? findParticipantByCodeHash(db, participantCodeHash) : null;
    const userAgent = req.get("user-agent") || body.userAgent || "";
    const consentTimestamp = body.consentTimestamp || new Date().toISOString();

    if (existing) {
      return res.status(200).json({
        participantId: existing.id,
        participantCode: existing.participant_code,
        participantCodeHash: existing.participant_code_hash,
        reused: true,
        consentVersion,
        consentTimestamp
      });
    }

    const id = crypto.randomUUID();
    insertParticipant(db, {
      id,
      consentVersion,
      consentTimestamp,
      participantCode: envConfig.storeParticipantCode ? participantCode : null,
      participantCodeHash,
      ipAddress: envConfig.storeIpAddress ? req.ip : null,
      userAgent,
      deviceClass: normalizeDeviceClass(body.deviceClass),
      metadata: body.metadata
    });
    return res.status(201).json({
      participantId: id,
      participantCode: envConfig.storeParticipantCode ? participantCode : null,
      participantCodeHash,
      reused: false,
      consentVersion,
      consentTimestamp
    });
  } catch (error) {
    return next(error);
  }
});

app.post("/api/session", (req, res, next) => {
  try {
    const body = req.body || {};
    requireString(body.participantId, "participantId");
    const id = crypto.randomUUID();
    insertSession(db, {
      id,
      participantId: body.participantId,
      startedAt: body.startedAt,
      deviceClass: normalizeDeviceClass(body.deviceClass),
      userAgent: req.get("user-agent") || body.userAgent || "",
      viewport: body.viewport,
      screen: body.screen,
      navigator: body.navigator,
      timezone: body.timezone,
      language: body.language,
      touchSupport: body.touchSupport,
      metadata: body.metadata,
      experimentId: normalizeOptionalText(body.experimentId) || "pilot",
      sessionNo: integerOrNull(body.sessionNo) || 1
    });
    return res.status(201).json({ sessionId: id, sessionNo: integerOrNull(body.sessionNo) || 1 });
  } catch (error) {
    return next(error);
  }
});

app.post("/api/attempt", (req, res, next) => {
  try {
    const attempt = normalizeAttempt(req.body || {});
    upsertAttempt(db, attempt);
    return res.status(201).json({ attemptId: attempt.id });
  } catch (error) {
    return next(error);
  }
});

app.patch("/api/attempt/:id", (req, res, next) => {
  try {
    const current = getAttempt(db, req.params.id);
    if (!current) {
      return res.status(404).json({ error: "Attempt not found." });
    }
    const merged = {
      id: current.id,
      sessionId: current.session_id,
      participantId: current.participant_id,
      inputMode: req.body.inputMode || current.input_mode,
      roleLabel: req.body.roleLabel || current.role_label,
      promptText: req.body.promptText ?? current.prompt_text,
      rawText: envConfig.storeRawText ? (req.body.rawText ?? current.raw_text) : "",
      startedAt: current.started_at,
      endedAt: req.body.endedAt ?? current.ended_at,
      deviceClass: current.device_class,
      featureQuality: req.body.featureQuality ?? current.feature_quality,
      summary: req.body.summary ?? parseJson(current.summary_json),
      trialNo: req.body.trialNo ?? current.trial_no,
      promptId: req.body.promptId ?? current.prompt_id,
      promptSetId: req.body.promptSetId ?? current.prompt_set_id,
      targetParticipantCode: req.body.targetParticipantCode ?? current.target_participant_code,
      status: normalizeAttemptStatus(req.body.status || current.status),
      submittedAt: req.body.submittedAt ?? current.submitted_at,
      qualityStatus: req.body.qualityStatus ?? current.quality_status,
      exclusionReason: req.body.exclusionReason ?? current.exclusion_reason,
      pasteCount: req.body.pasteCount ?? current.paste_count,
      fixedPromptMatch: req.body.fixedPromptMatch ?? current.fixed_prompt_match,
      fixedPromptEditDistance: req.body.fixedPromptEditDistance ?? current.fixed_prompt_edit_distance,
      suggestionShown: req.body.suggestionShown ?? current.suggestion_shown,
      suggestionId: req.body.suggestionId ?? current.suggestion_id
    };
    upsertAttempt(db, merged);
    return res.json({ attemptId: current.id, status: merged.status });
  } catch (error) {
    return next(error);
  }
});

app.post("/api/attempt/:id/delete-empty", (req, res, next) => {
  try {
    res.json(deleteEmptyAttempt(db, req.params.id));
  } catch (error) {
    next(error);
  }
});

app.post("/api/events/bulk", (req, res, next) => {
  try {
    const attempt = findAttemptFromBody(req.body);
    const events = Array.isArray(req.body.events) ? req.body.events : [];
    if (events.length > 0) {
      insertEvents(db, attempt, events, { storeKeyValue: envConfig.storeKeyValue });
    }
    return res.status(201).json({ inserted: events.length });
  } catch (error) {
    return next(error);
  }
});

app.post("/api/features/bulk", (req, res, next) => {
  try {
    const attempt = findAttemptFromBody(req.body);
    const features = Array.isArray(req.body.features) ? req.body.features : [];
    insertFeatures(db, attempt, features);
    return res.status(201).json({ inserted: features.length });
  } catch (error) {
    return next(error);
  }
});

app.post("/api/results", (req, res, next) => {
  try {
    const attempt = findAttemptFromBody(req.body);
    const results = Array.isArray(req.body.results) ? req.body.results : [];
    insertResults(db, attempt, results);
    return res.status(201).json({ inserted: results.length });
  } catch (error) {
    return next(error);
  }
});

app.post("/api/monitoring-windows", (req, res, next) => {
  try {
    const attempt = findAttemptFromBody(req.body);
    const windows = Array.isArray(req.body.windows) ? req.body.windows : [req.body.window].filter(Boolean);
    for (const windowRow of windows) {
      insertMonitoringWindow(db, attempt, windowRow);
    }
    return res.status(201).json({ inserted: windows.length });
  } catch (error) {
    return next(error);
  }
});

app.get("/api/calibrations", (req, res, next) => {
  try {
    res.json({
      calibrations: listCalibrations(db, {
        participantId: normalizeOptionalText(req.query.participantId),
        participantCode: normalizeOptionalText(req.query.participantCode)
      })
    });
  } catch (error) {
    next(error);
  }
});

app.get("/api/admin/metrics", requireAdmin, (_req, res, next) => {
  try {
    res.json({
      referenceMetrics,
      prompts: promptSetsPayload,
      modelManifestPath: "/models/manifest.json",
      ...collectAdminMetrics(db)
    });
  } catch (error) {
    next(error);
  }
});

app.get("/api/admin/prompts", requireAdmin, (_req, res) => {
  res.json(promptSetsPayload);
});

app.get("/api/admin/calibrations", requireAdmin, (req, res, next) => {
  try {
    res.json({ calibrations: listCalibrations(db, req.query || {}) });
  } catch (error) {
    next(error);
  }
});

app.post("/api/admin/calibrate", requireAdmin, (req, res, next) => {
  try {
    const body = req.body || {};
    if (Number.isFinite(Number(body.threshold))) {
      const row = upsertCalibration(db, {
        participantId: normalizeOptionalText(body.participantId),
        participantCode: normalizeOptionalText(body.participantCode),
        modelName: normalizeOptionalText(body.modelName) || "LightGBM",
        modelScope: normalizeOptionalText(body.modelScope),
        inputMode: normalizeOptionalText(body.inputMode),
        deviceClass: normalizeDeviceClass(body.deviceClass || "unknown"),
        threshold: Number(body.threshold),
        calibrationMethod: normalizeOptionalText(body.calibrationMethod) || "manual",
        referenceAttemptCount: integerOrNull(body.referenceAttemptCount) || 0,
        metrics: body.metrics || null
      });
      return res.status(201).json({ calibration: row });
    }
    const row = computeCalibrationFromGenuineScores(db, {
      participantId: normalizeOptionalText(body.participantId),
      participantCode: normalizeOptionalText(body.participantCode),
      modelName: normalizeOptionalText(body.modelName),
      inputMode: normalizeOptionalText(body.inputMode),
      deviceClass: normalizeOptionalText(body.deviceClass),
      quantile: Number(body.quantile ?? envConfig.calibrationGenuineQuantile)
    });
    return res.status(201).json({ calibration: row });
  } catch (error) {
    next(error);
  }
});

app.get("/api/admin/export/:name.csv", requireAdmin, (req, res, next) => {
  try {
    const name = req.params.name;
    const rows = exportRowsByName(name);
    res.setHeader("Content-Type", "text/csv; charset=utf-8");
    res.setHeader("Content-Disposition", `attachment; filename="${name}.csv"`);
    res.send(toCsv(rows));
  } catch (error) {
    next(error);
  }
});

app.post("/api/admin/evaluation/runs", requireAdmin, (req, res, next) => {
  try {
    const runId = normalizeRunId(req.body?.runId) || makeRunId();
    const runDir = ensureEvaluationRun(runId);
    const manifestPath = path.join(runDir, "run_manifest.json");
    const existing = existsSync(manifestPath) ? parseJson(readFileSync(manifestPath, "utf8")) : null;
    const manifest = existing || {
      runId,
      createdAt: new Date().toISOString(),
      appVersion: "keystroke-auth-demo/0.1.0",
      notes: "Browser-side evaluation run. Server stores browser-generated artifacts only.",
      artifacts: []
    };
    if (!existing) {
      writeJson(manifestPath, manifest);
    }
    res.status(existing ? 200 : 201).json({ runId, createdAt: manifest.createdAt });
  } catch (error) {
    next(error);
  }
});

app.get("/api/admin/evaluation/runs/:runId", requireAdmin, (req, res, next) => {
  try {
    const runId = normalizeRunId(req.params.runId);
    if (!runId) {
      return res.status(400).json({ error: "Invalid run id." });
    }
    const runDir = path.join(evaluationRoot, runId);
    if (!existsSync(runDir)) {
      return res.status(404).json({ error: "Evaluation run not found." });
    }
    const artifacts = listEvaluationArtifacts(runDir);
    res.json({
      runId,
      manifest: readJsonIfExists(path.join(runDir, "run_manifest.json")),
      environments: readCsvIfExists(path.join(runDir, "environments.csv")),
      artifacts,
      summaries: readEvaluationSummaries(runDir, artifacts)
    });
  } catch (error) {
    next(error);
  }
});

app.post("/api/admin/evaluation/runs/:runId/artifacts", requireAdmin, (req, res, next) => {
  try {
    const runId = normalizeRunId(req.params.runId);
    if (!runId) {
      return res.status(400).json({ error: "Invalid run id." });
    }
    const body = req.body || {};
    const dataset = normalizeEvaluationDataset(body.dataset);
    const environmentId = normalizePathSegment(body.environmentId);
    const attemptId = normalizePathSegment(body.attemptId || "attempt-001");
    const artifacts = Array.isArray(body.artifacts) ? body.artifacts : [];
    if (!dataset || !environmentId || !artifacts.length) {
      return res.status(400).json({ error: "dataset, environmentId, and artifacts are required." });
    }

    const runDir = ensureEvaluationRun(runId);
    const targetDir = path.join(runDir, dataset, environmentId, attemptId);
    mkdirSync(targetDir, { recursive: true });

    const written = [];
    for (const artifact of artifacts) {
      const relativeName = normalizeArtifactPath(artifact?.name);
      if (!relativeName) {
        continue;
      }
      const outputPath = path.join(targetDir, relativeName);
      if (!isInside(targetDir, outputPath)) {
        continue;
      }
      mkdirSync(path.dirname(outputPath), { recursive: true });
      if (artifact.encoding === "base64") {
        writeFileSync(outputPath, Buffer.from(String(artifact.content || ""), "base64"));
      } else {
        writeFileSync(outputPath, String(artifact.content ?? ""), "utf8");
      }
      written.push(path.relative(runDir, outputPath));
    }

    upsertEvaluationEnvironment(runDir, body.environment || {}, {
      dataset,
      environmentId,
      attemptId,
      savedAt: new Date().toISOString()
    });
    updateEvaluationManifest(runDir, {
      dataset,
      environmentId,
      attemptId,
      savedAt: new Date().toISOString(),
      artifacts: written
    });
    res.status(201).json({ runId, dataset, environmentId, attemptId, written });
  } catch (error) {
    next(error);
  }
});

app.get("/api/admin/evaluation/data/:source/:file", requireAdmin, (req, res, next) => {
  try {
    const source = req.params.source;
    const fileName = normalizeArtifactPath(req.params.file);
    const filePath = resolveEvaluationDataPath(source, fileName);
    if (!filePath || !existsSync(filePath)) {
      return res.status(404).json({ error: "Evaluation data file not found." });
    }
    res.sendFile(filePath);
  } catch (error) {
    next(error);
  }
});

if (isProduction) {
  const distDir = path.join(rootDir, "dist");
  app.use(express.static(distDir));
  app.get(/.*/, (_req, res) => {
    res.sendFile(path.join(distDir, "index.html"));
  });
} else {
  const { createServer } = await import("vite");
  const vite = await createServer({
    root: rootDir,
    server: { middlewareMode: true },
    appType: "spa"
  });
  app.use(vite.middlewares);
}

app.use((error, _req, res, _next) => {
  console.error(error);
  res.status(error.statusCode || 500).json({
    error: error.message || "Internal server error"
  });
});

const server = app.listen(port, host, () => {
  const dbPath = process.env.DATABASE_PATH || "web_demo_data/keystroke_demo.sqlite";
  const mode = isProduction ? "production" : "development";
  const distNote = isProduction && !existsSync(path.join(rootDir, "dist"))
    ? " (run npm run build first)"
    : "";
  console.log(`Keystroke auth demo listening at http://${host}:${port} in ${mode}${distNote}`);
  console.log(`SQLite database: ${dbPath}`);
});

const keepAlive = setInterval(() => {}, 60 * 60 * 1000);

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    clearInterval(keepAlive);
    server.close(() => {
      process.exit(0);
    });
  });
}

function loadPromptSets() {
  const promptPath = path.resolve(rootDir, envConfig.promptSetPath);
  try {
    return JSON.parse(readFileSync(promptPath, "utf8"));
  } catch (error) {
    console.warn(`Could not load prompt set file ${promptPath}: ${error.message}`);
    return { promptSets: [] };
  }
}

function exportRowsByName(name) {
  switch (name) {
    case "attempt_features":
      return exportAttemptFeatures(db);
    case "event_pairs":
      return exportEventPairs(db);
    case "model_results":
      return exportModelResults(db);
    case "quality_summary":
      return exportQualitySummary(db);
    case "monitoring_windows":
      return exportMonitoringWindows(db);
    default:
      return queryTableRows(db, name);
  }
}

function requireAdmin(req, res, next) {
  const providedPin = req.get("x-admin-pin") || req.query.pin;
  if (!providedPin || providedPin !== adminPin) {
    return res.status(401).json({ error: "Admin PIN required." });
  }
  return next();
}

function normalizeAttempt(body) {
  const id = body.id || crypto.randomUUID();
  requireString(body.sessionId, "sessionId");
  requireString(body.participantId, "participantId");
  return {
    id,
    sessionId: body.sessionId,
    participantId: body.participantId,
    inputMode: body.inputMode === "free" ? "free" : "fixed",
    roleLabel: normalizeRole(body.roleLabel),
    promptText: body.promptText || "",
    rawText: envConfig.storeRawText ? (body.rawText || "") : "",
    startedAt: body.startedAt,
    endedAt: body.endedAt,
    deviceClass: normalizeDeviceClass(body.deviceClass),
    featureQuality: body.featureQuality,
    summary: body.summary,
    trialNo: integerOrNull(body.trialNo) || 1,
    promptId: normalizeOptionalText(body.promptId),
    promptSetId: normalizeOptionalText(body.promptSetId),
    targetParticipantCode: normalizeOptionalText(body.targetParticipantCode),
    status: normalizeAttemptStatus(body.status),
    submittedAt: body.submittedAt || null,
    qualityStatus: body.qualityStatus || null,
    exclusionReason: body.exclusionReason || null,
    pasteCount: integerOrNull(body.pasteCount) || 0,
    fixedPromptMatch: body.fixedPromptMatch,
    fixedPromptEditDistance: body.fixedPromptEditDistance,
    suggestionShown: Boolean(body.suggestionShown),
    suggestionId: normalizeOptionalText(body.suggestionId)
  };
}

function findAttemptFromBody(body = {}) {
  requireString(body.attemptId, "attemptId");
  const attempt = getAttempt(db, body.attemptId);
  if (!attempt) {
    const error = new Error("Attempt not found.");
    error.statusCode = 404;
    throw error;
  }
  return attempt;
}

function normalizeRole(role) {
  return ["genuine", "imposter"].includes(role) ? role : "genuine";
}

function normalizeAttemptStatus(status) {
  return ["in_progress", "submitted", "cancelled", "excluded"].includes(status)
    ? status
    : "in_progress";
}

function normalizeDeviceClass(deviceClass) {
  return ["desktop", "mobile", "tablet", "unknown"].includes(deviceClass)
    ? deviceClass
    : "unknown";
}

function requireString(value, name) {
  if (!value || typeof value !== "string") {
    const error = new Error(`${name} is required.`);
    error.statusCode = 400;
    throw error;
  }
}

function normalizeOptionalText(value) {
  if (value === undefined || value === null) {
    return null;
  }
  const text = String(value).trim();
  return text ? text : null;
}

function integerOrNull(value) {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? Math.trunc(number) : null;
}

function envBool(name, defaultValue) {
  const value = process.env[name];
  if (value === undefined) {
    return defaultValue;
  }
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function hashCode(value) {
  return crypto.createHash("sha256").update(String(value)).digest("hex");
}

function parseJson(value) {
  if (!value) {
    return null;
  }
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function toCsv(rows) {
  if (!rows.length) {
    return "";
  }
  const headers = Array.from(rows.reduce((set, row) => {
    Object.keys(row).forEach((key) => set.add(key));
    return set;
  }, new Set()));
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((header) => csvCell(row[header])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function csvCell(value) {
  if (value === null || value === undefined) {
    return "";
  }
  const text = typeof value === "object" ? JSON.stringify(value) : String(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function makeRunId() {
  const stamp = new Date().toISOString().replaceAll(":", "-").replace(/\.\d{3}Z$/, "Z");
  return `${stamp}-${crypto.randomUUID().slice(0, 8)}`;
}

function normalizeRunId(value) {
  const text = normalizeOptionalText(value);
  if (!text || !/^[A-Za-z0-9._-]+$/.test(text)) {
    return null;
  }
  return text;
}

function normalizePathSegment(value) {
  const text = normalizeOptionalText(value);
  if (!text) {
    return null;
  }
  return text.replace(/[^A-Za-z0-9._-]/g, "_").slice(0, 120) || null;
}

function normalizeEvaluationDataset(value) {
  const text = normalizeOptionalText(value);
  if ([
    "dsl",
    "dsl_enriched_features",
    "mmc",
    "mmc_enriched_features",
    "collected_data_analysis",
    "collected_data_analysis_enriched_features"
  ].includes(text)) {
    return text;
  }
  return null;
}

function normalizeArtifactPath(value) {
  const text = normalizeOptionalText(value);
  if (!text) {
    return null;
  }
  const parts = text.split("/").map(normalizePathSegment).filter(Boolean);
  return parts.length ? parts.join("/") : null;
}

function ensureEvaluationRun(runId) {
  mkdirSync(evaluationRoot, { recursive: true });
  const runDir = path.join(evaluationRoot, runId);
  if (!isInside(evaluationRoot, runDir)) {
    throw new Error("Invalid evaluation run path.");
  }
  mkdirSync(runDir, { recursive: true });
  const manifestPath = path.join(runDir, "run_manifest.json");
  if (!existsSync(manifestPath)) {
    writeJson(manifestPath, {
      runId,
      createdAt: new Date().toISOString(),
      appVersion: "keystroke-auth-demo/0.1.0",
      notes: "Browser-side evaluation run. Server stores browser-generated artifacts only.",
      artifacts: []
    });
  }
  return runDir;
}

function resolveEvaluationDataPath(source, fileName) {
  const safeName = normalizeArtifactPath(fileName);
  if (!safeName) {
    return null;
  }
  const sourceRoots = {
    dsl: workspaceRoot,
    mmc: path.join(workspaceRoot, "ScienceDirect_files_20Apr2026_10-05-23.390"),
    collected: path.join(rootDir, "web_demo_data"),
    models: path.join(rootDir, "public", "models")
  };
  const base = sourceRoots[source];
  if (!base) {
    return null;
  }
  const resolved = path.join(base, safeName);
  return isInside(base, resolved) ? resolved : null;
}

function isInside(baseDir, targetPath) {
  const relative = path.relative(path.resolve(baseDir), path.resolve(targetPath));
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function readJsonIfExists(filePath) {
  if (!existsSync(filePath)) {
    return null;
  }
  return parseJson(readFileSync(filePath, "utf8"));
}

function readCsvIfExists(filePath) {
  if (!existsSync(filePath)) {
    return "";
  }
  return readFileSync(filePath, "utf8");
}

function readEvaluationSummaries(runDir, artifacts) {
  const rows = [];
  for (const artifact of artifacts) {
    if (!artifact.path.endsWith("summary_metrics.csv")) {
      continue;
    }
    const parts = artifact.path.split(path.sep);
    if (parts.length < 4) {
      continue;
    }
    const [dataset, environmentId, attemptId] = parts;
    const filePath = path.join(runDir, artifact.path);
    if (!isInside(runDir, filePath) || !existsSync(filePath)) {
      continue;
    }
    for (const row of parseCsv(readFileSync(filePath, "utf8"))) {
      rows.push({
        dataset,
        environmentId,
        attemptId,
        ...row
      });
    }
  }
  return rows;
}

function parseCsv(text) {
  const clean = String(text || "").replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < clean.length; i += 1) {
    const char = clean[i];
    if (char === '"' && clean[i + 1] === '"') {
      cell += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && clean[i + 1] === "\n") {
        i += 1;
      }
      row.push(cell);
      if (row.some((value) => value !== "")) {
        rows.push(row);
      }
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  if (rows.length < 2) {
    return [];
  }
  const headers = rows[0];
  return rows.slice(1).map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

function writeJson(filePath, payload) {
  writeFileSync(filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

function updateEvaluationManifest(runDir, entry) {
  const manifestPath = path.join(runDir, "run_manifest.json");
  const manifest = readJsonIfExists(manifestPath) || {};
  const artifacts = Array.isArray(manifest.artifacts) ? manifest.artifacts : [];
  artifacts.push(entry);
  writeJson(manifestPath, {
    ...manifest,
    updatedAt: new Date().toISOString(),
    artifacts
  });
}

function upsertEvaluationEnvironment(runDir, environment, row) {
  const envPath = path.join(runDir, "environments.csv");
  const headers = [
    "savedAt",
    "dataset",
    "environmentId",
    "attemptId",
    "deviceLabel",
    "userAgent",
    "platform",
    "language",
    "timezone",
    "hardwareConcurrency",
    "deviceMemory",
    "tfjsBackend",
    "tfjsVersion",
    "webglAvailable",
    "wasmAvailable",
    "wasmSimdAvailable",
    "wasmThreadsAvailable",
    "crossOriginIsolated",
    "viewportWidth",
    "viewportHeight",
    "devicePixelRatio"
  ];
  const values = {
    savedAt: row.savedAt,
    dataset: row.dataset,
    environmentId: row.environmentId,
    attemptId: row.attemptId,
    deviceLabel: environment.deviceLabel,
    userAgent: environment.userAgent,
    platform: environment.platform,
    language: environment.language,
    timezone: environment.timezone,
    hardwareConcurrency: environment.hardwareConcurrency,
    deviceMemory: environment.deviceMemory,
    tfjsBackend: environment.tfjsBackend,
    tfjsVersion: environment.tfjsVersion,
    webglAvailable: environment.webglAvailable,
    wasmAvailable: environment.wasmAvailable,
    wasmSimdAvailable: environment.wasmSimdAvailable,
    wasmThreadsAvailable: environment.wasmThreadsAvailable,
    crossOriginIsolated: environment.crossOriginIsolated,
    viewportWidth: environment.viewport?.width,
    viewportHeight: environment.viewport?.height,
    devicePixelRatio: environment.viewport?.devicePixelRatio
  };
  const line = headers.map((header) => csvCell(values[header] ?? "not_available")).join(",");
  if (!existsSync(envPath)) {
    writeFileSync(envPath, `${headers.join(",")}\n${line}\n`, "utf8");
  } else {
    writeFileSync(envPath, `${readFileSync(envPath, "utf8")}${line}\n`, "utf8");
  }
}

function listEvaluationArtifacts(runDir) {
  const output = [];
  walkEvaluationFiles(runDir, runDir, output);
  return output;
}

function walkEvaluationFiles(root, current, output) {
  for (const entry of readdirSync(current)) {
    const filePath = path.join(current, entry);
    const stat = statSync(filePath);
    if (stat.isDirectory()) {
      walkEvaluationFiles(root, filePath, output);
    } else {
      output.push({
        path: path.relative(root, filePath),
        bytes: stat.size,
        updatedAt: stat.mtime.toISOString()
      });
    }
  }
}
