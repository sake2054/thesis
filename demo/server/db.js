import { mkdirSync } from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

const DEFAULT_DB_PATH = "web_demo_data/keystroke_demo.sqlite";

function nowIso() {
  return new Date().toISOString();
}

function ensureParentDir(filePath) {
  mkdirSync(path.dirname(filePath), { recursive: true });
}

function jsonString(value) {
  if (value === undefined || value === null) {
    return null;
  }
  return JSON.stringify(value);
}

export function openDatabase(dbPath = process.env.DATABASE_PATH || DEFAULT_DB_PATH) {
  ensureParentDir(dbPath);
  const db = new DatabaseSync(dbPath);
  db.exec("PRAGMA journal_mode = WAL;");
  db.exec("PRAGMA foreign_keys = ON;");
  migrate(db);
  return db;
}

function migrate(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS participants (
      id TEXT PRIMARY KEY,
      consent_version TEXT NOT NULL,
      consent_timestamp TEXT NOT NULL,
      ip_address TEXT,
      user_agent TEXT,
      device_class TEXT NOT NULL,
      metadata_json TEXT,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      participant_id TEXT NOT NULL,
      started_at TEXT NOT NULL,
      device_class TEXT NOT NULL,
      user_agent TEXT,
      viewport_json TEXT,
      screen_json TEXT,
      navigator_json TEXT,
      timezone TEXT,
      language TEXT,
      touch_support INTEGER NOT NULL DEFAULT 0,
      metadata_json TEXT,
      FOREIGN KEY (participant_id) REFERENCES participants(id)
    );

    CREATE TABLE IF NOT EXISTS attempts (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      participant_id TEXT NOT NULL,
      input_mode TEXT NOT NULL,
      role_label TEXT NOT NULL,
      prompt_text TEXT,
      raw_text TEXT,
      started_at TEXT NOT NULL,
      ended_at TEXT,
      device_class TEXT NOT NULL,
      feature_quality TEXT,
      summary_json TEXT,
      FOREIGN KEY (session_id) REFERENCES sessions(id),
      FOREIGN KEY (participant_id) REFERENCES participants(id)
    );

    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      attempt_id TEXT NOT NULL,
      session_id TEXT NOT NULL,
      participant_id TEXT NOT NULL,
      event_type TEXT NOT NULL,
      event_time REAL,
      relative_time REAL,
      key_value TEXT,
      code TEXT,
      input_type TEXT,
      data TEXT,
      value_length INTEGER,
      is_composing INTEGER NOT NULL DEFAULT 0,
      repeat INTEGER NOT NULL DEFAULT 0,
      device_class TEXT NOT NULL,
      payload_json TEXT,
      created_at TEXT NOT NULL,
      FOREIGN KEY (attempt_id) REFERENCES attempts(id),
      FOREIGN KEY (session_id) REFERENCES sessions(id),
      FOREIGN KEY (participant_id) REFERENCES participants(id)
    );

    CREATE TABLE IF NOT EXISTS features (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      attempt_id TEXT NOT NULL,
      session_id TEXT NOT NULL,
      participant_id TEXT NOT NULL,
      model_scope TEXT NOT NULL,
      feature_name TEXT NOT NULL,
      feature_index INTEGER NOT NULL,
      feature_value REAL,
      payload_json TEXT,
      created_at TEXT NOT NULL,
      FOREIGN KEY (attempt_id) REFERENCES attempts(id),
      FOREIGN KEY (session_id) REFERENCES sessions(id),
      FOREIGN KEY (participant_id) REFERENCES participants(id)
    );

    CREATE TABLE IF NOT EXISTS results (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      attempt_id TEXT NOT NULL,
      session_id TEXT NOT NULL,
      participant_id TEXT NOT NULL,
      model_name TEXT NOT NULL,
      score REAL,
      threshold REAL,
      decision TEXT,
      inference_time_ms REAL,
      ui_blocking_time_ms REAL,
      reference_metrics_json TEXT,
      collected_metrics_json TEXT,
      payload_json TEXT,
      created_at TEXT NOT NULL,
      FOREIGN KEY (attempt_id) REFERENCES attempts(id),
      FOREIGN KEY (session_id) REFERENCES sessions(id),
      FOREIGN KEY (participant_id) REFERENCES participants(id)
    );

    CREATE INDEX IF NOT EXISTS idx_attempts_session ON attempts(session_id);
    CREATE INDEX IF NOT EXISTS idx_attempts_role_device ON attempts(role_label, device_class);
    CREATE INDEX IF NOT EXISTS idx_events_attempt ON events(attempt_id);
    CREATE INDEX IF NOT EXISTS idx_features_attempt ON features(attempt_id);
    CREATE INDEX IF NOT EXISTS idx_results_attempt_model ON results(attempt_id, model_name);
  `);
}

export function insertParticipant(db, row) {
  const timestamp = row.consentTimestamp || nowIso();
  db.prepare(`
    INSERT INTO participants (
      id, consent_version, consent_timestamp, ip_address, user_agent,
      device_class, metadata_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    row.id,
    row.consentVersion,
    timestamp,
    row.ipAddress || null,
    row.userAgent || null,
    row.deviceClass,
    jsonString(row.metadata),
    nowIso()
  );
}

export function insertSession(db, row) {
  db.prepare(`
    INSERT INTO sessions (
      id, participant_id, started_at, device_class, user_agent, viewport_json,
      screen_json, navigator_json, timezone, language, touch_support, metadata_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    row.id,
    row.participantId,
    row.startedAt || nowIso(),
    row.deviceClass,
    row.userAgent || null,
    jsonString(row.viewport),
    jsonString(row.screen),
    jsonString(row.navigator),
    row.timezone || null,
    row.language || null,
    row.touchSupport ? 1 : 0,
    jsonString(row.metadata)
  );
}

export function upsertAttempt(db, row) {
  db.prepare(`
    INSERT INTO attempts (
      id, session_id, participant_id, input_mode, role_label, prompt_text, raw_text,
      started_at, ended_at, device_class, feature_quality, summary_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      input_mode = excluded.input_mode,
      role_label = excluded.role_label,
      prompt_text = excluded.prompt_text,
      raw_text = excluded.raw_text,
      ended_at = excluded.ended_at,
      feature_quality = excluded.feature_quality,
      summary_json = excluded.summary_json
  `).run(
    row.id,
    row.sessionId,
    row.participantId,
    row.inputMode,
    row.roleLabel,
    row.promptText || null,
    row.rawText || "",
    row.startedAt || nowIso(),
    row.endedAt || null,
    row.deviceClass,
    row.featureQuality || null,
    jsonString(row.summary)
  );
}

export function getAttempt(db, attemptId) {
  return db.prepare("SELECT * FROM attempts WHERE id = ?").get(attemptId);
}

export function deleteEmptyAttempt(db, attemptId) {
  const row = db.prepare(`
    SELECT
      a.id,
      length(COALESCE(a.raw_text, '')) AS rawTextLength,
      (SELECT COUNT(*) FROM events WHERE attempt_id = a.id) AS eventCount,
      (SELECT COUNT(*) FROM features WHERE attempt_id = a.id) AS featureCount,
      (SELECT COUNT(*) FROM results WHERE attempt_id = a.id) AS resultCount
    FROM attempts a
    WHERE a.id = ?
  `).get(attemptId);

  if (!row) {
    return { deleted: false, reason: "not_found" };
  }
  if (row.rawTextLength > 0 || row.eventCount > 0 || row.featureCount > 0 || row.resultCount > 0) {
    return { deleted: false, reason: "not_empty" };
  }

  db.prepare("DELETE FROM attempts WHERE id = ?").run(attemptId);
  return { deleted: true, reason: "empty_attempt" };
}

export function insertEvents(db, attempt, events) {
  const stmt = db.prepare(`
    INSERT INTO events (
      attempt_id, session_id, participant_id, event_type, event_time, relative_time,
      key_value, code, input_type, data, value_length, is_composing, repeat,
      device_class, payload_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  db.exec("BEGIN");
  try {
    for (const event of events) {
      stmt.run(
        attempt.id,
        attempt.session_id,
        attempt.participant_id,
        String(event.type || "unknown"),
        numberOrNull(event.eventTime),
        numberOrNull(event.relativeTime),
        valueOrNull(event.key),
        valueOrNull(event.code),
        valueOrNull(event.inputType),
        valueOrNull(event.data),
        integerOrNull(event.valueLength),
        event.isComposing ? 1 : 0,
        event.repeat ? 1 : 0,
        attempt.device_class,
        jsonString(event.payload || event),
        nowIso()
      );
    }
    db.exec("COMMIT");
  } catch (error) {
    db.exec("ROLLBACK");
    throw error;
  }
}

export function insertFeatures(db, attempt, features) {
  const deleteStmt = db.prepare("DELETE FROM features WHERE attempt_id = ? AND model_scope = ?");
  const insertStmt = db.prepare(`
    INSERT INTO features (
      attempt_id, session_id, participant_id, model_scope, feature_name,
      feature_index, feature_value, payload_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const scopes = new Set(features.map((feature) => String(feature.modelScope || "browser")));
  db.exec("BEGIN");
  try {
    for (const scope of scopes) {
      deleteStmt.run(attempt.id, scope);
    }
    features.forEach((feature, index) => {
      insertStmt.run(
        attempt.id,
        attempt.session_id,
        attempt.participant_id,
        String(feature.modelScope || "browser"),
        String(feature.name || `feature_${index}`),
        Number.isInteger(feature.index) ? feature.index : index,
        numberOrNull(feature.value),
        jsonString(feature.payload || null),
        nowIso()
      );
    });
    db.exec("COMMIT");
  } catch (error) {
    db.exec("ROLLBACK");
    throw error;
  }
}

export function insertResults(db, attempt, results) {
  const stmt = db.prepare(`
    INSERT INTO results (
      attempt_id, session_id, participant_id, model_name, score, threshold,
      decision, inference_time_ms, ui_blocking_time_ms, reference_metrics_json,
      collected_metrics_json, payload_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  db.exec("BEGIN");
  try {
    for (const result of results) {
      stmt.run(
        attempt.id,
        attempt.session_id,
        attempt.participant_id,
        String(result.modelName || "unknown"),
        numberOrNull(result.score),
        numberOrNull(result.threshold),
        valueOrNull(result.decision),
        numberOrNull(result.inferenceTimeMs),
        numberOrNull(result.uiBlockingTimeMs),
        jsonString(result.referenceMetrics || null),
        jsonString(result.collectedMetrics || null),
        jsonString(result.payload || result),
        nowIso()
      );
    }
    db.exec("COMMIT");
  } catch (error) {
    db.exec("ROLLBACK");
    throw error;
  }
}

export function queryTableRows(db, tableName) {
  const allowed = new Set([
    "participants",
    "sessions",
    "attempts",
    "events",
    "features",
    "results"
  ]);
  if (!allowed.has(tableName)) {
    const error = new Error(`Unsupported export table: ${tableName}`);
    error.statusCode = 404;
    throw error;
  }
  return db.prepare(`SELECT * FROM ${tableName} ORDER BY rowid`).all();
}

export function collectAdminMetrics(db) {
  const totals = {
    participants: db.prepare("SELECT COUNT(*) AS count FROM participants").get().count,
    sessions: db.prepare("SELECT COUNT(*) AS count FROM sessions").get().count,
    attempts: db.prepare("SELECT COUNT(*) AS count FROM attempts").get().count,
    events: db.prepare("SELECT COUNT(*) AS count FROM events").get().count,
    results: db.prepare("SELECT COUNT(*) AS count FROM results").get().count
  };
  const byDevice = db.prepare(`
    SELECT device_class AS deviceClass, COUNT(*) AS attempts
    FROM attempts
    GROUP BY device_class
    ORDER BY attempts DESC
  `).all();
  const byRole = db.prepare(`
    SELECT role_label AS roleLabel, COUNT(*) AS attempts
    FROM attempts
    GROUP BY role_label
    ORDER BY attempts DESC
  `).all();
  const recentAttempts = db.prepare(`
    SELECT id, input_mode AS inputMode, role_label AS roleLabel, device_class AS deviceClass,
           length(COALESCE(raw_text, '')) AS rawTextLength, started_at AS startedAt,
           ended_at AS endedAt, feature_quality AS featureQuality
    FROM attempts
    ORDER BY started_at DESC
    LIMIT 25
  `).all();
  const resultRows = db.prepare(`
    SELECT r.model_name AS modelName, r.score, r.threshold, a.role_label AS roleLabel,
           a.device_class AS deviceClass
    FROM results r
    JOIN attempts a ON a.id = r.attempt_id
    WHERE r.score IS NOT NULL AND a.role_label IN ('genuine', 'imposter')
  `).all();
  return {
    totals,
    byDevice,
    byRole,
    recentAttempts,
    collectedMetrics: buildCollectedMetrics(resultRows)
  };
}

function buildCollectedMetrics(rows) {
  const groups = new Map();
  for (const row of rows) {
    const labels = ["all", row.deviceClass || "unknown"];
    for (const device of labels) {
      const key = `${row.modelName}::${device}`;
      if (!groups.has(key)) {
        groups.set(key, {
          modelName: row.modelName,
          deviceClass: device,
          scores: [],
          labels: []
        });
      }
      groups.get(key).scores.push(Number(row.score));
      groups.get(key).labels.push(row.roleLabel === "genuine" ? 1 : 0);
    }
  }
  return Array.from(groups.values()).map((group) => ({
    modelName: group.modelName,
    deviceClass: group.deviceClass,
    ...computeBinaryMetrics(group.labels, group.scores)
  }));
}

function computeBinaryMetrics(labels, scores) {
  const positives = labels.filter((label) => label === 1).length;
  const negatives = labels.length - positives;
  if (positives < 2 || negatives < 2) {
    return {
      status: "insufficient labeled attempts",
      samples: labels.length,
      positives,
      negatives
    };
  }

  const thresholds = Array.from(new Set(scores)).sort((a, b) => b - a);
  thresholds.unshift(Math.max(...scores) + 1e-9);
  thresholds.push(Math.min(...scores) - 1e-9);

  let best = null;
  for (const threshold of thresholds) {
    let tp = 0;
    let tn = 0;
    let fp = 0;
    let fn = 0;
    scores.forEach((score, index) => {
      const pred = score >= threshold ? 1 : 0;
      const label = labels[index];
      if (pred === 1 && label === 1) tp += 1;
      else if (pred === 0 && label === 0) tn += 1;
      else if (pred === 1 && label === 0) fp += 1;
      else fn += 1;
    });
    const far = fp / Math.max(fp + tn, 1);
    const frr = fn / Math.max(fn + tp, 1);
    const eer = (far + frr) / 2;
    const diff = Math.abs(far - frr);
    if (!best || diff < best.diff) {
      best = { threshold, far, frr, eer, accuracy: (tp + tn) / labels.length, diff };
    }
  }

  return {
    status: "ready",
    samples: labels.length,
    positives,
    negatives,
    threshold: best.threshold,
    far: best.far,
    frr: best.frr,
    eer: best.eer,
    accuracy: best.accuracy
  };
}

function valueOrNull(value) {
  return value === undefined || value === null ? null : String(value);
}

function numberOrNull(value) {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function integerOrNull(value) {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? Math.trunc(number) : null;
}
