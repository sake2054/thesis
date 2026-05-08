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
  `);

  ensureColumn(db, "participants", "participant_code", "TEXT");
  ensureColumn(db, "participants", "participant_code_hash", "TEXT");
  ensureColumn(db, "sessions", "experiment_id", "TEXT");
  ensureColumn(db, "sessions", "session_no", "INTEGER");
  ensureColumn(db, "attempts", "trial_no", "INTEGER");
  ensureColumn(db, "attempts", "prompt_id", "TEXT");
  ensureColumn(db, "attempts", "prompt_set_id", "TEXT");
  ensureColumn(db, "attempts", "target_participant_code", "TEXT");
  ensureColumn(db, "attempts", "status", "TEXT DEFAULT 'in_progress'");
  ensureColumn(db, "attempts", "submitted_at", "TEXT");
  ensureColumn(db, "attempts", "quality_status", "TEXT");
  ensureColumn(db, "attempts", "exclusion_reason", "TEXT");
  ensureColumn(db, "attempts", "paste_count", "INTEGER DEFAULT 0");
  ensureColumn(db, "attempts", "fixed_prompt_match", "INTEGER");
  ensureColumn(db, "attempts", "fixed_prompt_edit_distance", "INTEGER");
  ensureColumn(db, "attempts", "suggestion_shown", "INTEGER DEFAULT 0");
  ensureColumn(db, "attempts", "suggestion_id", "TEXT");
  ensureColumn(db, "results", "model_scope", "TEXT");
  ensureColumn(db, "results", "device_class", "TEXT");
  ensureColumn(db, "results", "calibration_json", "TEXT");

  db.exec(`
    CREATE TABLE IF NOT EXISTS monitoring_windows (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      window_id TEXT NOT NULL,
      attempt_id TEXT NOT NULL,
      session_id TEXT NOT NULL,
      participant_id TEXT NOT NULL,
      participant_code TEXT,
      window_start_time REAL,
      window_end_time REAL,
      window_char_start INTEGER,
      window_char_end INTEGER,
      feature_json TEXT,
      result_json TEXT,
      quality_status TEXT,
      inference_time_ms REAL,
      created_at TEXT NOT NULL,
      FOREIGN KEY (attempt_id) REFERENCES attempts(id),
      FOREIGN KEY (session_id) REFERENCES sessions(id),
      FOREIGN KEY (participant_id) REFERENCES participants(id)
    );

    CREATE TABLE IF NOT EXISTS model_calibrations (
      id TEXT PRIMARY KEY,
      participant_id TEXT,
      participant_code TEXT,
      model_name TEXT NOT NULL,
      model_scope TEXT,
      input_mode TEXT,
      device_class TEXT,
      threshold REAL NOT NULL,
      calibration_method TEXT NOT NULL,
      reference_attempt_count INTEGER NOT NULL DEFAULT 0,
      metrics_json TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_participants_code_hash ON participants(participant_code_hash);
    CREATE INDEX IF NOT EXISTS idx_sessions_participant_no ON sessions(participant_id, session_no);
    CREATE INDEX IF NOT EXISTS idx_attempts_session ON attempts(session_id);
    CREATE INDEX IF NOT EXISTS idx_attempts_role_device ON attempts(role_label, device_class);
    CREATE INDEX IF NOT EXISTS idx_attempts_quality ON attempts(quality_status, status);
    CREATE INDEX IF NOT EXISTS idx_events_attempt ON events(attempt_id);
    CREATE INDEX IF NOT EXISTS idx_features_attempt ON features(attempt_id);
    CREATE INDEX IF NOT EXISTS idx_results_attempt_model ON results(attempt_id, model_name);
    CREATE INDEX IF NOT EXISTS idx_monitoring_attempt ON monitoring_windows(attempt_id);
    CREATE INDEX IF NOT EXISTS idx_calibrations_lookup
      ON model_calibrations(participant_id, model_name, input_mode, device_class);
  `);
}

function ensureColumn(db, tableName, columnName, definition) {
  const columns = db.prepare(`PRAGMA table_info(${tableName})`).all();
  if (!columns.some((column) => column.name === columnName)) {
    db.exec(`ALTER TABLE ${tableName} ADD COLUMN ${columnName} ${definition};`);
  }
}

export function findParticipantByCodeHash(db, participantCodeHash) {
  if (!participantCodeHash) {
    return null;
  }
  return db.prepare(`
    SELECT * FROM participants
    WHERE participant_code_hash = ?
    ORDER BY created_at ASC
    LIMIT 1
  `).get(participantCodeHash);
}

export function getParticipant(db, participantId) {
  return db.prepare("SELECT * FROM participants WHERE id = ?").get(participantId);
}

export function insertParticipant(db, row) {
  const timestamp = row.consentTimestamp || nowIso();
  db.prepare(`
    INSERT INTO participants (
      id, consent_version, consent_timestamp, ip_address, user_agent,
      device_class, metadata_json, created_at, participant_code, participant_code_hash
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    row.id,
    row.consentVersion,
    timestamp,
    row.ipAddress || null,
    row.userAgent || null,
    row.deviceClass,
    jsonString(row.metadata),
    nowIso(),
    row.participantCode || null,
    row.participantCodeHash || null
  );
}

export function insertSession(db, row) {
  db.prepare(`
    INSERT INTO sessions (
      id, participant_id, started_at, device_class, user_agent, viewport_json,
      screen_json, navigator_json, timezone, language, touch_support, metadata_json,
      experiment_id, session_no
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    jsonString(row.metadata),
    row.experimentId || null,
    integerOrNull(row.sessionNo)
  );
}

export function upsertAttempt(db, row) {
  db.prepare(`
    INSERT INTO attempts (
      id, session_id, participant_id, input_mode, role_label, prompt_text, raw_text,
      started_at, ended_at, device_class, feature_quality, summary_json,
      trial_no, prompt_id, prompt_set_id, target_participant_code, status,
      submitted_at, quality_status, exclusion_reason, paste_count,
      fixed_prompt_match, fixed_prompt_edit_distance, suggestion_shown, suggestion_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      input_mode = excluded.input_mode,
      role_label = excluded.role_label,
      prompt_text = excluded.prompt_text,
      raw_text = excluded.raw_text,
      ended_at = excluded.ended_at,
      feature_quality = excluded.feature_quality,
      summary_json = excluded.summary_json,
      trial_no = excluded.trial_no,
      prompt_id = excluded.prompt_id,
      prompt_set_id = excluded.prompt_set_id,
      target_participant_code = excluded.target_participant_code,
      status = excluded.status,
      submitted_at = excluded.submitted_at,
      quality_status = excluded.quality_status,
      exclusion_reason = excluded.exclusion_reason,
      paste_count = excluded.paste_count,
      fixed_prompt_match = excluded.fixed_prompt_match,
      fixed_prompt_edit_distance = excluded.fixed_prompt_edit_distance,
      suggestion_shown = excluded.suggestion_shown,
      suggestion_id = excluded.suggestion_id
  `).run(
    row.id,
    row.sessionId,
    row.participantId,
    row.inputMode,
    row.roleLabel,
    row.promptText || null,
    row.rawText ?? "",
    row.startedAt || nowIso(),
    row.endedAt || null,
    row.deviceClass,
    row.featureQuality || null,
    jsonString(row.summary),
    integerOrNull(row.trialNo),
    row.promptId || null,
    row.promptSetId || null,
    row.targetParticipantCode || null,
    row.status || "in_progress",
    row.submittedAt || null,
    row.qualityStatus || null,
    row.exclusionReason || null,
    integerOrNull(row.pasteCount) || 0,
    booleanOrNull(row.fixedPromptMatch),
    integerOrNull(row.fixedPromptEditDistance),
    row.suggestionShown ? 1 : 0,
    row.suggestionId || null
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

export function insertEvents(db, attempt, events, options = {}) {
  const storeKeyValue = options.storeKeyValue !== false;
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
        storeKeyValue ? valueOrNull(event.key) : null,
        storeKeyValue ? valueOrNull(event.code) : null,
        valueOrNull(event.inputType),
        storeKeyValue ? valueOrNull(event.data) : null,
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
      attempt_id, session_id, participant_id, model_name, model_scope, device_class,
      score, threshold, decision, inference_time_ms, ui_blocking_time_ms,
      reference_metrics_json, collected_metrics_json, calibration_json,
      payload_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  db.exec("BEGIN");
  try {
    for (const result of results) {
      stmt.run(
        attempt.id,
        attempt.session_id,
        attempt.participant_id,
        String(result.modelName || "unknown"),
        valueOrNull(result.modelScope),
        valueOrNull(result.deviceClass || attempt.device_class),
        numberOrNull(result.score),
        numberOrNull(result.threshold),
        valueOrNull(result.decision),
        numberOrNull(result.inferenceTimeMs),
        numberOrNull(result.uiBlockingTimeMs),
        jsonString(result.referenceMetrics || null),
        jsonString(result.collectedMetrics || null),
        jsonString(result.calibration || null),
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

export function insertMonitoringWindow(db, attempt, row) {
  db.prepare(`
    INSERT INTO monitoring_windows (
      window_id, attempt_id, session_id, participant_id, participant_code,
      window_start_time, window_end_time, window_char_start, window_char_end,
      feature_json, result_json, quality_status, inference_time_ms, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    row.windowId,
    attempt.id,
    attempt.session_id,
    attempt.participant_id,
    row.participantCode || null,
    numberOrNull(row.windowStartTime),
    numberOrNull(row.windowEndTime),
    integerOrNull(row.windowCharStart),
    integerOrNull(row.windowCharEnd),
    jsonString(row.features || row.featureSummary || null),
    jsonString(row.results || row.result || null),
    row.qualityStatus || null,
    numberOrNull(row.inferenceTimeMs),
    nowIso()
  );
}

export function listCalibrations(db, filters = {}) {
  const clauses = [];
  const params = [];
  if (filters.participantId) {
    clauses.push("(participant_id = ? OR participant_id IS NULL)");
    params.push(filters.participantId);
  }
  if (filters.participantCode) {
    clauses.push("(participant_code = ? OR participant_code IS NULL)");
    params.push(filters.participantCode);
  }
  const where = clauses.length ? `WHERE ${clauses.join(" AND ")}` : "";
  return db.prepare(`
    SELECT id, participant_id AS participantId, participant_code AS participantCode,
           model_name AS modelName, model_scope AS modelScope, input_mode AS inputMode,
           device_class AS deviceClass, threshold, calibration_method AS calibrationMethod,
           reference_attempt_count AS referenceAttemptCount, metrics_json AS metricsJson,
           created_at AS createdAt, updated_at AS updatedAt
    FROM model_calibrations
    ${where}
    ORDER BY updated_at DESC
  `).all(...params).map((row) => ({ ...row, metrics: parseJson(row.metricsJson) }));
}

export function upsertCalibration(db, row) {
  const id = row.id || `${row.participantId || "global"}:${row.modelName}:${row.modelScope || "any"}:${row.inputMode || "any"}:${row.deviceClass || "any"}`;
  const timestamp = nowIso();
  db.prepare(`
    INSERT INTO model_calibrations (
      id, participant_id, participant_code, model_name, model_scope, input_mode,
      device_class, threshold, calibration_method, reference_attempt_count,
      metrics_json, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      participant_id = excluded.participant_id,
      participant_code = excluded.participant_code,
      model_name = excluded.model_name,
      model_scope = excluded.model_scope,
      input_mode = excluded.input_mode,
      device_class = excluded.device_class,
      threshold = excluded.threshold,
      calibration_method = excluded.calibration_method,
      reference_attempt_count = excluded.reference_attempt_count,
      metrics_json = excluded.metrics_json,
      updated_at = excluded.updated_at
  `).run(
    id,
    row.participantId || null,
    row.participantCode || null,
    row.modelName,
    row.modelScope || null,
    row.inputMode || null,
    row.deviceClass || null,
    Number(row.threshold),
    row.calibrationMethod || "manual",
    integerOrNull(row.referenceAttemptCount) || 0,
    jsonString(row.metrics || null),
    row.createdAt || timestamp,
    timestamp
  );
  return db.prepare("SELECT * FROM model_calibrations WHERE id = ?").get(id);
}

export function computeCalibrationFromGenuineScores(db, options = {}) {
  const quantile = Number(options.quantile ?? 0.05);
  const clauses = ["a.role_label = 'genuine'", "r.score IS NOT NULL"];
  const params = [];
  if (options.participantId) {
    clauses.push("a.participant_id = ?");
    params.push(options.participantId);
  }
  if (options.participantCode) {
    clauses.push("p.participant_code = ?");
    params.push(options.participantCode);
  }
  if (options.modelName) {
    clauses.push("r.model_name = ?");
    params.push(options.modelName);
  }
  if (options.inputMode) {
    clauses.push("a.input_mode = ?");
    params.push(options.inputMode);
  }
  if (options.deviceClass) {
    clauses.push("a.device_class = ?");
    params.push(options.deviceClass);
  }
  const rows = db.prepare(`
    SELECT r.score, r.model_name AS modelName, r.model_scope AS modelScope,
           a.input_mode AS inputMode, a.device_class AS deviceClass,
           a.participant_id AS participantId, p.participant_code AS participantCode
    FROM results r
    JOIN attempts a ON a.id = r.attempt_id
    JOIN participants p ON p.id = a.participant_id
    WHERE ${clauses.join(" AND ")}
    ORDER BY r.score ASC
  `).all(...params);
  if (!rows.length) {
    const error = new Error("No genuine scores available for calibration.");
    error.statusCode = 400;
    throw error;
  }
  const index = Math.max(0, Math.min(rows.length - 1, Math.floor((rows.length - 1) * quantile)));
  const threshold = Number(rows[index].score);
  return upsertCalibration(db, {
    participantId: options.participantId || rows[0].participantId || null,
    participantCode: options.participantCode || rows[0].participantCode || null,
    modelName: options.modelName || rows[0].modelName,
    modelScope: options.modelScope || rows[0].modelScope || null,
    inputMode: options.inputMode || rows[0].inputMode,
    deviceClass: options.deviceClass || rows[0].deviceClass,
    threshold,
    calibrationMethod: `genuine_q${quantile}`,
    referenceAttemptCount: rows.length,
    metrics: { quantile, scoreCount: rows.length, minScore: rows[0].score, maxScore: rows[rows.length - 1].score }
  });
}

export function queryTableRows(db, tableName) {
  const allowed = new Set([
    "participants",
    "sessions",
    "attempts",
    "events",
    "features",
    "results",
    "monitoring_windows",
    "model_calibrations"
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
    participants: scalar(db, "SELECT COUNT(*) FROM participants"),
    sessions: scalar(db, "SELECT COUNT(*) FROM sessions"),
    attempts: scalar(db, "SELECT COUNT(*) FROM attempts"),
    submittedAttempts: scalar(db, "SELECT COUNT(*) FROM attempts WHERE status = 'submitted'"),
    excludedAttempts: scalar(db, "SELECT COUNT(*) FROM attempts WHERE status = 'excluded' OR quality_status = 'excluded'"),
    lowQualityAttempts: scalar(db, "SELECT COUNT(*) FROM attempts WHERE quality_status = 'low_quality'"),
    pasteDetectedAttempts: scalar(db, "SELECT COUNT(*) FROM attempts WHERE COALESCE(paste_count, 0) > 0"),
    events: scalar(db, "SELECT COUNT(*) FROM events"),
    results: scalar(db, "SELECT COUNT(*) FROM results"),
    monitoringWindows: scalar(db, "SELECT COUNT(*) FROM monitoring_windows")
  };
  const qualityAverages = db.prepare(`
    SELECT
      AVG(json_extract(summary_json, '$.keyupCoverage')) AS avgKeyupCoverage,
      AVG(json_extract(summary_json, '$.compositionRatio')) AS avgCompositionRatio,
      AVG(json_extract(summary_json, '$.timingCoverage')) AS avgTimingCoverage,
      AVG(length(COALESCE(raw_text, ''))) AS avgTextLength
    FROM attempts
  `).get();
  const byDevice = groupedCount(db, "device_class", "deviceClass");
  const byInputMode = groupedCount(db, "input_mode", "inputMode");
  const byRole = groupedCount(db, "role_label", "roleLabel");
  const byPrompt = groupedCount(db, "COALESCE(prompt_id, 'none')", "promptId");
  const qualityByDevice = db.prepare(`
    SELECT device_class AS deviceClass, COALESCE(quality_status, 'unknown') AS qualityStatus,
           COUNT(*) AS attempts
    FROM attempts
    GROUP BY device_class, quality_status
    ORDER BY device_class, attempts DESC
  `).all();
  const exclusionReasons = db.prepare(`
    SELECT COALESCE(exclusion_reason, 'none') AS exclusionReason, COUNT(*) AS attempts
    FROM attempts
    WHERE quality_status IN ('excluded', 'low_quality') OR status = 'excluded'
    GROUP BY exclusion_reason
    ORDER BY attempts DESC
  `).all();
  const recentLowQuality = db.prepare(`
    SELECT a.id, p.participant_code AS participantCode, s.session_no AS sessionNo,
           a.trial_no AS trialNo, a.input_mode AS inputMode, a.role_label AS roleLabel,
           a.prompt_id AS promptId, a.device_class AS deviceClass,
           length(COALESCE(a.raw_text, '')) AS rawTextLength,
           a.started_at AS startedAt, a.submitted_at AS submittedAt,
           a.status, a.quality_status AS qualityStatus,
           a.exclusion_reason AS exclusionReason, a.paste_count AS pasteCount
    FROM attempts a
    JOIN participants p ON p.id = a.participant_id
    JOIN sessions s ON s.id = a.session_id
    WHERE a.quality_status IN ('excluded', 'low_quality') OR a.status = 'excluded'
    ORDER BY COALESCE(a.submitted_at, a.started_at) DESC
    LIMIT 25
  `).all();
  const recentAttempts = db.prepare(`
    SELECT a.id, p.participant_code AS participantCode, s.session_no AS sessionNo,
           a.trial_no AS trialNo, a.input_mode AS inputMode, a.role_label AS roleLabel,
           a.device_class AS deviceClass, length(COALESCE(a.raw_text, '')) AS rawTextLength,
           a.started_at AS startedAt, a.ended_at AS endedAt, a.status,
           a.feature_quality AS featureQuality, a.quality_status AS qualityStatus
    FROM attempts a
    JOIN participants p ON p.id = a.participant_id
    JOIN sessions s ON s.id = a.session_id
    ORDER BY a.started_at DESC
    LIMIT 25
  `).all();
  const resultRows = db.prepare(`
    SELECT r.model_name AS modelName, r.model_scope AS modelScope, r.score, r.threshold,
           a.role_label AS roleLabel, a.device_class AS deviceClass, a.input_mode AS inputMode
    FROM results r
    JOIN attempts a ON a.id = r.attempt_id
    WHERE r.score IS NOT NULL AND a.role_label IN ('genuine', 'imposter')
  `).all();
  const calibrationStatus = db.prepare(`
    SELECT participant_code AS participantCode, model_name AS modelName,
           input_mode AS inputMode, device_class AS deviceClass, threshold,
           reference_attempt_count AS referenceAttemptCount, updated_at AS updatedAt
    FROM model_calibrations
    ORDER BY updated_at DESC
    LIMIT 50
  `).all();
  return {
    totals,
    qualityOverview: { ...qualityAverages },
    byDevice,
    byInputMode,
    byRole,
    byPrompt,
    qualityByDevice,
    exclusionReasons,
    recentLowQuality,
    recentAttempts,
    calibrationStatus,
    collectedMetrics: buildCollectedMetrics(resultRows)
  };
}

export function exportAttemptFeatures(db) {
  const attempts = db.prepare(`
    SELECT a.*, p.participant_code, p.participant_code_hash, p.user_agent AS participant_user_agent,
           s.session_no, s.viewport_json, s.screen_json, s.navigator_json,
           s.timezone, s.language, s.user_agent AS session_user_agent
    FROM attempts a
    JOIN participants p ON p.id = a.participant_id
    JOIN sessions s ON s.id = a.session_id
    ORDER BY p.participant_code, s.session_no, a.trial_no, a.started_at
  `).all();
  const rows = [];
  const featureRows = db.prepare("SELECT attempt_id, feature_name, feature_value FROM features ORDER BY feature_index").all();
  const featuresByAttempt = groupBy(featureRows, "attempt_id");
  const eventStats = buildEventStats(db);
  for (const attempt of attempts) {
    const viewport = parseJson(attempt.viewport_json) || {};
    const navigatorJson = parseJson(attempt.navigator_json) || {};
    const summary = parseJson(attempt.summary_json) || {};
    const stats = eventStats.get(attempt.id) || {};
    const row = {
      participant_id: attempt.participant_id,
      participant_code: attempt.participant_code,
      participant_code_hash: attempt.participant_code_hash,
      session_id: attempt.session_id,
      session_no: attempt.session_no,
      trial_no: attempt.trial_no,
      attempt_id: attempt.id,
      role_label: attempt.role_label,
      target_participant_code: attempt.target_participant_code,
      input_mode: attempt.input_mode,
      prompt_set_id: attempt.prompt_set_id,
      prompt_id: attempt.prompt_id,
      prompt_text: attempt.prompt_text,
      raw_text: attempt.raw_text,
      raw_text_length: String(attempt.raw_text || "").length,
      started_at: attempt.started_at,
      submitted_at: attempt.submitted_at,
      device_class: attempt.device_class,
      browser_user_agent: attempt.session_user_agent || attempt.participant_user_agent,
      viewport_width: viewport.width,
      viewport_height: viewport.height,
      language: attempt.language || navigatorJson.language,
      timezone: attempt.timezone,
      quality_status: attempt.quality_status,
      exclusion_reason: attempt.exclusion_reason,
      paste_count: attempt.paste_count,
      fixed_prompt_match: attempt.fixed_prompt_match,
      fixed_prompt_edit_distance: attempt.fixed_prompt_edit_distance,
      suggestion_shown: attempt.suggestion_shown,
      suggestion_id: attempt.suggestion_id,
      event_count: stats.eventCount ?? summary.eventCount,
      keydown_count: stats.keydownCount ?? summary.keydownCount,
      keyup_count: stats.keyupCount ?? summary.keyupCount,
      paired_key_count: summary.pairedKeyCount,
      keyup_coverage: summary.keyupCoverage,
      composition_ratio: summary.compositionRatio,
      timing_coverage: summary.timingCoverage,
      feature_quality: attempt.feature_quality || summary.featureQuality
    };
    for (const feature of featuresByAttempt.get(attempt.id) || []) {
      row[feature.feature_name] = feature.feature_value;
    }
    rows.push(row);
  }
  return rows;
}

export function exportEventPairs(db) {
  const attempts = db.prepare(`
    SELECT a.id, a.role_label, a.input_mode, a.prompt_id, a.trial_no,
           p.participant_code, s.session_no
    FROM attempts a
    JOIN participants p ON p.id = a.participant_id
    JOIN sessions s ON s.id = a.session_id
    ORDER BY a.started_at
  `).all();
  const events = db.prepare(`
    SELECT attempt_id, event_type, relative_time, key_value, code, repeat
    FROM events
    WHERE event_type IN ('keydown', 'keyup')
    ORDER BY attempt_id, relative_time, id
  `).all();
  const byAttempt = groupBy(events, "attempt_id");
  const rows = [];
  for (const attempt of attempts) {
    const pairs = pairKeyEvents(byAttempt.get(attempt.id) || []);
    for (let index = 0; index < pairs.length; index += 1) {
      const current = pairs[index];
      const next = pairs[index + 1] || {};
      rows.push({
        attempt_id: attempt.id,
        participant_code: attempt.participant_code,
        session_no: attempt.session_no,
        trial_no: attempt.trial_no,
        role_label: attempt.role_label,
        input_mode: attempt.input_mode,
        prompt_id: attempt.prompt_id,
        pair_index: index,
        key_value: current.key,
        code: current.code,
        key_class: keyClass(current.key),
        down_time: current.down,
        up_time: current.up,
        hold_ms: saneTiming(current.up - current.down) ? current.up - current.down : null,
        next_key_value: next.key || null,
        next_code: next.code || null,
        dd_ms: saneTiming(next.down - current.down) ? next.down - current.down : null,
        ud_ms: saneTiming(next.down - current.up) ? next.down - current.up : null,
        uu_ms: saneTiming(next.up - current.up) ? next.up - current.up : null,
        du_ms: saneTiming(next.up - current.down) ? next.up - current.down : null,
        is_sane_timing: saneTiming(current.up - current.down) ? 1 : 0
      });
    }
  }
  return rows;
}

export function exportModelResults(db) {
  return db.prepare(`
    SELECT p.participant_code AS participant_code, s.session_no AS session_no,
           a.trial_no AS trial_no, a.role_label AS role_label, a.input_mode AS input_mode,
           a.prompt_id AS prompt_id, a.id AS attempt_id, r.model_name AS model_name,
           r.model_scope AS model_scope, COALESCE(r.device_class, a.device_class) AS device_class,
           r.score AS score, r.threshold AS threshold, r.decision AS decision,
           r.inference_time_ms AS inference_time_ms,
           r.ui_blocking_time_ms AS ui_blocking_time_ms,
           a.quality_status AS quality_status, a.exclusion_reason AS exclusion_reason
    FROM results r
    JOIN attempts a ON a.id = r.attempt_id
    JOIN participants p ON p.id = a.participant_id
    JOIN sessions s ON s.id = a.session_id
    ORDER BY p.participant_code, s.session_no, a.trial_no, r.model_name
  `).all();
}

export function exportQualitySummary(db) {
  return db.prepare(`
    SELECT a.device_class AS device_class, a.input_mode AS input_mode,
           a.role_label AS role_label, COALESCE(a.prompt_id, 'none') AS prompt_id,
           COUNT(*) AS attempt_count,
           SUM(CASE WHEN a.status = 'submitted' THEN 1 ELSE 0 END) AS submitted_count,
           SUM(CASE WHEN a.status = 'excluded' OR a.quality_status = 'excluded' THEN 1 ELSE 0 END) AS excluded_count,
           SUM(CASE WHEN a.quality_status = 'low_quality' THEN 1 ELSE 0 END) AS low_quality_count,
           SUM(CASE WHEN COALESCE(a.paste_count, 0) > 0 THEN 1 ELSE 0 END) AS paste_count,
           AVG(json_extract(a.summary_json, '$.keyupCoverage')) AS avg_keyup_coverage,
           AVG(json_extract(a.summary_json, '$.compositionRatio')) AS avg_composition_ratio,
           AVG(json_extract(a.summary_json, '$.timingCoverage')) AS avg_timing_coverage,
           AVG(length(COALESCE(a.raw_text, ''))) AS avg_text_length
    FROM attempts a
    GROUP BY a.device_class, a.input_mode, a.role_label, a.prompt_id
    ORDER BY a.device_class, a.input_mode, a.role_label, a.prompt_id
  `).all();
}

export function exportMonitoringWindows(db) {
  const rows = db.prepare(`
    SELECT p.participant_code, mw.session_id, mw.attempt_id, mw.window_id,
           mw.window_start_time, mw.window_end_time,
           mw.window_char_start, mw.window_char_end, mw.feature_json,
           mw.result_json, mw.inference_time_ms, mw.quality_status
    FROM monitoring_windows mw
    JOIN participants p ON p.id = mw.participant_id
    ORDER BY mw.created_at, mw.window_id
  `).all();
  return rows.map((row) => {
    const features = parseJson(row.feature_json) || {};
    const result = Array.isArray(parseJson(row.result_json))
      ? parseJson(row.result_json)[0] || {}
      : parseJson(row.result_json) || {};
    return {
      participant_code: row.participant_code,
      session_id: row.session_id,
      attempt_id: row.attempt_id,
      window_id: row.window_id,
      window_start_time: row.window_start_time,
      window_end_time: row.window_end_time,
      window_char_start: row.window_char_start,
      window_char_end: row.window_char_end,
      hold_mean_ms: features.hold_mean_ms,
      hold_std_ms: features.hold_std_ms,
      down_down_mean_ms: features.down_down_mean_ms,
      up_down_mean_ms: features.up_down_mean_ms,
      model_name: result.modelName,
      model_scope: result.modelScope,
      score: result.score,
      threshold: result.threshold,
      decision: result.decision,
      inference_time_ms: row.inference_time_ms || result.inferenceTimeMs,
      quality_status: row.quality_status
    };
  });
}

function buildEventStats(db) {
  const rows = db.prepare(`
    SELECT attempt_id,
           COUNT(*) AS eventCount,
           SUM(CASE WHEN event_type = 'keydown' THEN 1 ELSE 0 END) AS keydownCount,
           SUM(CASE WHEN event_type = 'keyup' THEN 1 ELSE 0 END) AS keyupCount,
           SUM(CASE WHEN event_type = 'paste' OR input_type = 'insertFromPaste' THEN 1 ELSE 0 END) AS pasteCount
    FROM events
    GROUP BY attempt_id
  `).all();
  return new Map(rows.map((row) => [row.attempt_id, row]));
}

function pairKeyEvents(events) {
  const keydowns = [];
  const active = new Map();
  for (const event of events) {
    const time = Number(event.relative_time);
    if (!Number.isFinite(time)) {
      continue;
    }
    const identity = event.code || event.key_value || "unknown";
    if (event.event_type === "keydown") {
      if (event.repeat) {
        continue;
      }
      const entry = { key: event.key_value, code: event.code, down: time, up: null };
      keydowns.push(entry);
      if (!active.has(identity)) {
        active.set(identity, []);
      }
      active.get(identity).push(entry);
    } else if (event.event_type === "keyup") {
      const stack = active.get(identity) || [];
      const entry = stack.find((candidate) => candidate.up === null);
      if (entry) {
        entry.up = time;
      }
    }
  }
  return keydowns.filter((entry) => Number.isFinite(entry.down) && Number.isFinite(entry.up));
}

function buildCollectedMetrics(rows) {
  const groups = new Map();
  for (const row of rows) {
    const labels = ["all", row.deviceClass || "unknown"];
    for (const device of labels) {
      const key = `${row.modelName}::${device}::${row.inputMode || "all"}`;
      if (!groups.has(key)) {
        groups.set(key, {
          modelName: row.modelName,
          modelScope: row.modelScope,
          deviceClass: device,
          inputMode: row.inputMode || "all",
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
    modelScope: group.modelScope,
    deviceClass: group.deviceClass,
    inputMode: group.inputMode,
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

function groupedCount(db, expression, alias) {
  return db.prepare(`
    SELECT ${expression} AS ${alias}, COUNT(*) AS attempts
    FROM attempts
    GROUP BY ${expression}
    ORDER BY attempts DESC
  `).all();
}

function scalar(db, sql) {
  return db.prepare(sql).get()["COUNT(*)"];
}

function groupBy(rows, key) {
  const groups = new Map();
  for (const row of rows) {
    const value = row[key];
    if (!groups.has(value)) {
      groups.set(value, []);
    }
    groups.get(value).push(row);
  }
  return groups;
}

function keyClass(key) {
  if (!key) return "unknown";
  if (key.length === 1) return "character";
  if (key === "Backspace" || key === "Delete") return "edit";
  if (key === "Shift" || key === "Control" || key === "Alt" || key === "Meta") return "modifier";
  if (key === "Enter" || key === "Tab" || key === " ") return "control";
  return "special";
}

function saneTiming(value) {
  return Number.isFinite(value) && value >= -1000 && value <= 10000;
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

function booleanOrNull(value) {
  if (value === undefined || value === null) {
    return null;
  }
  return value ? 1 : 0;
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
