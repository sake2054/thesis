import crypto from "node:crypto";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import express from "express";

import {
  collectAdminMetrics,
  deleteEmptyAttempt,
  getAttempt,
  insertEvents,
  insertFeatures,
  insertParticipant,
  insertResults,
  insertSession,
  openDatabase,
  queryTableRows,
  upsertAttempt
} from "./db.js";
import { loadReferenceMetrics } from "./referenceMetrics.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const isProduction = process.env.NODE_ENV === "production";
const port = Number(process.env.PORT || 3000);
const host = process.env.HOST || "127.0.0.1";
const adminPin = process.env.ADMIN_PIN || "change-me";
const consentVersion = process.env.CONSENT_VERSION || "2026-04-26";
const fixedPromptText =
  process.env.FIXED_PROMPT_TEXT || "Type the assigned research phrase exactly as shown.";

const app = express();
const db = openDatabase();
const referenceMetrics = loadReferenceMetrics();

app.set("trust proxy", true);
app.use(express.json({ limit: "10mb" }));

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, node: process.version });
});

app.get("/api/config", (_req, res) => {
  res.json({
    fixedPromptText,
    consentVersion,
    referenceMetrics,
    rawTextStorageEnabled: true,
    roles: ["genuine", "imposter"],
    deviceClasses: ["desktop", "mobile", "tablet", "unknown"]
  });
});

app.post("/api/consent", (req, res, next) => {
  try {
    const body = req.body || {};
    if (body.accepted !== true) {
      return res.status(400).json({ error: "Consent must be accepted before collection." });
    }
    const id = crypto.randomUUID();
    const userAgent = req.get("user-agent") || body.userAgent || "";
    insertParticipant(db, {
      id,
      consentVersion,
      consentTimestamp: body.consentTimestamp,
      ipAddress: req.ip,
      userAgent,
      deviceClass: normalizeDeviceClass(body.deviceClass),
      metadata: body.metadata
    });
    return res.status(201).json({
      participantId: id,
      consentVersion,
      consentTimestamp: body.consentTimestamp || new Date().toISOString()
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
      metadata: body.metadata
    });
    return res.status(201).json({ sessionId: id });
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
      rawText: req.body.rawText ?? current.raw_text,
      startedAt: current.started_at,
      endedAt: req.body.endedAt ?? current.ended_at,
      deviceClass: current.device_class,
      featureQuality: req.body.featureQuality ?? current.feature_quality,
      summary: req.body.summary ?? parseJson(current.summary_json)
    };
    upsertAttempt(db, merged);
    return res.json({ attemptId: current.id });
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
      insertEvents(db, attempt, events);
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

app.get("/api/admin/metrics", requireAdmin, (_req, res, next) => {
  try {
    res.json({
      referenceMetrics,
      ...collectAdminMetrics(db)
    });
  } catch (error) {
    next(error);
  }
});

app.get("/api/admin/export/:table.csv", requireAdmin, (req, res, next) => {
  try {
    const rows = queryTableRows(db, req.params.table);
    res.setHeader("Content-Type", "text/csv; charset=utf-8");
    res.setHeader(
      "Content-Disposition",
      `attachment; filename="${req.params.table}.csv"`
    );
    res.send(toCsv(rows));
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
    rawText: body.rawText || "",
    startedAt: body.startedAt,
    endedAt: body.endedAt,
    deviceClass: normalizeDeviceClass(body.deviceClass),
    featureQuality: body.featureQuality,
    summary: body.summary
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
  const headers = Object.keys(rows[0]);
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
  const text = String(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}
