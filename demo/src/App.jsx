import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiGet, apiPatch, apiPost, downloadAdminCsv } from "./api.js";
import {
  collectClientMetadata,
  detectDeviceClass,
  extractFeatureBundle,
  getTemplateCounts,
  makeEventPayload,
  scoreInstantBaseline
} from "./keystroke.js";
import { loadModelRuntime, predictOptionalModels } from "./modelRuntime.js";

const EXPORT_TABLES = ["participants", "sessions", "attempts", "events", "features", "results"];

export default function App() {
  const isAdmin = window.location.pathname.startsWith("/admin");
  if (isAdmin) {
    return <AdminApp />;
  }
  return <ParticipantApp />;
}

function ParticipantApp() {
  const [config, setConfig] = useState(null);
  const [deviceClass] = useState(() => detectDeviceClass());
  const [participantId, setParticipantId] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [attemptId, setAttemptId] = useState("");
  const [inputMode, setInputMode] = useState("fixed");
  const [roleLabel, setRoleLabel] = useState("genuine");
  const [rawText, setRawText] = useState("");
  const [error, setError] = useState("");
  const [isConsenting, setIsConsenting] = useState(false);
  const [isStartingAttempt, setIsStartingAttempt] = useState(false);
  const [results, setResults] = useState([]);
  const [quality, setQuality] = useState(null);
  const [templateCounts, setTemplateCounts] = useState(() => getTemplateCounts());
  const [savedCounts, setSavedCounts] = useState({ queuedEvents: 0, savedEvents: 0, analyses: 0 });
  const [artifactState, setArtifactState] = useState("checking");

  const attemptIdRef = useRef("");
  const attemptStartedAtRef = useRef(performance.now());
  const eventQueueRef = useRef([]);
  const allEventsRef = useRef([]);
  const rawTextRef = useRef("");
  const lastAttemptKeyRef = useRef("");
  const flushPromiseRef = useRef(Promise.resolve());
  const analysisSeqRef = useRef(0);

  useEffect(() => {
    apiGet("/api/config")
      .then(setConfig)
      .catch((apiError) => setError(apiError.message));
    loadModelRuntime().then((runtime) => {
      setArtifactState(runtime?.manifest?.artifactAvailable ? "ready" : "optional artifacts unavailable");
    });
  }, []);

  useEffect(() => {
    attemptIdRef.current = attemptId;
  }, [attemptId]);

  useEffect(() => {
    rawTextRef.current = rawText;
  }, [rawText]);

  const fixedPromptText = config?.fixedPromptText || "";
  const metadata = useMemo(() => collectClientMetadata(deviceClass), [deviceClass]);

  const flushEvents = useCallback(() => {
    const currentAttemptId = attemptIdRef.current;
    const batch = eventQueueRef.current.splice(0);
    if (!currentAttemptId || batch.length === 0) {
      return flushPromiseRef.current;
    }
    setSavedCounts((current) => ({ ...current, queuedEvents: 0 }));
    flushPromiseRef.current = flushPromiseRef.current
      .then(() => apiPost("/api/events/bulk", { attemptId: currentAttemptId, events: batch }))
      .then((payload) => {
        setSavedCounts((current) => ({
          ...current,
          savedEvents: current.savedEvents + (payload?.inserted || 0)
        }));
      })
      .catch((apiError) => {
        eventQueueRef.current.unshift(...batch);
        setSavedCounts((current) => ({ ...current, queuedEvents: eventQueueRef.current.length }));
        setError(apiError.message);
      });
    return flushPromiseRef.current;
  }, []);

  const cleanupEmptyAttempt = useCallback(async (id = attemptIdRef.current) => {
    if (!id) {
      return;
    }
    if (rawTextRef.current.length > 0 || allEventsRef.current.length > 0 || eventQueueRef.current.length > 0) {
      return;
    }
    try {
      await apiPost(`/api/attempt/${id}/delete-empty`, {});
    } catch (apiError) {
      setError(apiError.message);
    }
  }, []);

  const createAttempt = useCallback(async () => {
    if (!participantId || !sessionId || !config) {
      return;
    }
    const attemptKey = `${participantId}:${sessionId}:${inputMode}:${roleLabel}:${fixedPromptText}`;
    if (lastAttemptKeyRef.current === attemptKey && attemptIdRef.current) {
      return;
    }
    lastAttemptKeyRef.current = attemptKey;
    setIsStartingAttempt(true);
    await cleanupEmptyAttempt();
    await flushEvents();

    const nextAttemptId = randomId();
    const startedAt = new Date().toISOString();
    attemptStartedAtRef.current = performance.now();
    eventQueueRef.current = [];
    allEventsRef.current = [];
    setRawText("");
    setResults([]);
    setQuality(null);
    setSavedCounts({ queuedEvents: 0, savedEvents: 0, analyses: 0 });

    await apiPost("/api/attempt", {
      id: nextAttemptId,
      participantId,
      sessionId,
      inputMode,
      roleLabel,
      promptText: inputMode === "fixed" ? fixedPromptText : "",
      rawText: "",
      startedAt,
      deviceClass,
      summary: { artifactState }
    });
    setAttemptId(nextAttemptId);
    setIsStartingAttempt(false);
  }, [
    artifactState,
    config,
    cleanupEmptyAttempt,
    deviceClass,
    fixedPromptText,
    flushEvents,
    inputMode,
    participantId,
    roleLabel,
    sessionId
  ]);

  useEffect(() => {
    createAttempt().catch((apiError) => {
      setIsStartingAttempt(false);
      setError(apiError.message);
    });
  }, [createAttempt]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      flushEvents();
    }, 1500);
    return () => window.clearInterval(interval);
  }, [flushEvents]);

  useEffect(() => {
    const cleanupOnPageHide = () => {
      const id = attemptIdRef.current;
      if (!id || rawTextRef.current.length > 0 || allEventsRef.current.length > 0 || eventQueueRef.current.length > 0) {
        return;
      }
      const body = new Blob(["{}"], { type: "application/json" });
      navigator.sendBeacon?.(`/api/attempt/${id}/delete-empty`, body);
    };
    window.addEventListener("pagehide", cleanupOnPageHide);
    return () => window.removeEventListener("pagehide", cleanupOnPageHide);
  }, []);

  useEffect(() => {
    if (!attemptId) {
      return undefined;
    }
    const timeout = window.setTimeout(() => {
      apiPatch(`/api/attempt/${attemptId}`, {
        rawText,
        inputMode,
        roleLabel,
        promptText: inputMode === "fixed" ? fixedPromptText : "",
        featureQuality: quality?.featureQuality || null,
        summary: quality
      }).catch((apiError) => setError(apiError.message));
    }, 550);
    return () => window.clearTimeout(timeout);
  }, [attemptId, fixedPromptText, inputMode, quality, rawText, roleLabel]);

  useEffect(() => {
    if (!attemptId || rawText.length === 0) {
      return undefined;
    }
    const timeout = window.setTimeout(() => {
      runAnalysis();
    }, 420);
    return () => window.clearTimeout(timeout);
  }, [attemptId, rawText, inputMode, roleLabel]); // eslint-disable-line react-hooks/exhaustive-deps

  async function acceptConsent() {
    setIsConsenting(true);
    setError("");
    try {
      const consent = await apiPost("/api/consent", {
        accepted: true,
        consentTimestamp: new Date().toISOString(),
        deviceClass,
        metadata
      });
      const session = await apiPost("/api/session", {
        participantId: consent.participantId,
        startedAt: new Date().toISOString(),
        deviceClass,
        ...metadata
      });
      setParticipantId(consent.participantId);
      setSessionId(session.sessionId);
    } catch (apiError) {
      setError(apiError.message);
    } finally {
      setIsConsenting(false);
    }
  }

  function recordEvent(event, type, value = rawTextRef.current) {
    if (!attemptIdRef.current) {
      return;
    }
    const payload = makeEventPayload(event.nativeEvent || event, type, value, attemptStartedAtRef.current);
    eventQueueRef.current.push(payload);
    allEventsRef.current.push(payload);
    setSavedCounts((current) => ({ ...current, queuedEvents: eventQueueRef.current.length }));
  }

  async function runAnalysis() {
    const currentAttemptId = attemptIdRef.current;
    if (!currentAttemptId) {
      return;
    }
    const seq = analysisSeqRef.current + 1;
    analysisSeqRef.current = seq;
    const analysisStartedAt = performance.now();
    const bundle = extractFeatureBundle(allEventsRef.current, rawTextRef.current, inputMode, deviceClass);
    setQuality(bundle.quality);

    const baselineStartedAt = performance.now();
    const baseline = scoreInstantBaseline(inputMode, roleLabel, bundle.vector, currentAttemptId);
    const baselineResult = {
      ...baseline,
      inferenceTimeMs: performance.now() - baselineStartedAt,
      uiBlockingTimeMs: performance.now() - analysisStartedAt
    };
    const optionalResults = await predictOptionalModels(bundle.vector, inputMode);
    if (seq !== analysisSeqRef.current) {
      return;
    }
    const enrichedResults = [baselineResult, ...optionalResults].map((result) => ({
      ...result,
      referenceMetrics: findReferenceMetrics(config, inputMode, result.modelName),
      payload: {
        featureQuality: bundle.quality,
        templateCounts: getTemplateCounts()
      }
    }));

    setResults(enrichedResults);
    setTemplateCounts(getTemplateCounts());
    await Promise.all([
      flushEvents(),
      apiPatch(`/api/attempt/${currentAttemptId}`, {
        rawText: rawTextRef.current,
        endedAt: new Date().toISOString(),
        inputMode,
        roleLabel,
        promptText: inputMode === "fixed" ? fixedPromptText : "",
        featureQuality: bundle.quality.featureQuality,
        summary: bundle.quality
      }),
      apiPost("/api/features/bulk", {
        attemptId: currentAttemptId,
        features: bundle.features
      }),
      apiPost("/api/results", {
        attemptId: currentAttemptId,
        results: enrichedResults
      })
    ]).catch((apiError) => setError(apiError.message));
    setSavedCounts((current) => ({ ...current, analyses: current.analyses + 1 }));
  }

  function startNewAttempt() {
    lastAttemptKeyRef.current = "";
    createAttempt().catch((apiError) => setError(apiError.message));
  }

  const consentAccepted = Boolean(participantId && sessionId);
  const inputDisabled = !consentAccepted || !attemptId || isStartingAttempt;
  const verdict = bestVerdict(results);

  return (
    <div className="app-shell">
      {!consentAccepted ? (
        <ConsentModal
          consentVersion={config?.consentVersion}
          deviceClass={deviceClass}
          isSubmitting={isConsenting}
          onAccept={acceptConsent}
        />
      ) : null}

      <header className="topbar">
        <div>
          <p className="eyebrow">Continuous Authentication</p>
          <h1>Keystroke Security Check</h1>
        </div>
        <div className="topbar-meta">
          <span>{deviceClass}</span>
          <span>{artifactState}</span>
          <a href="/admin">Admin</a>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <main className="demo-grid" aria-hidden={!consentAccepted}>
        <section className="input-panel">
          <div className="toolbar-row">
            <SegmentedControl
              label="Role"
              value={roleLabel}
              options={[
                ["genuine", "Genuine"],
                ["imposter", "Imposter"]
              ]}
              onChange={setRoleLabel}
            />
            <SegmentedControl
              label="Text"
              value={inputMode}
              options={[
                ["fixed", "Fixed"],
                ["free", "Free"]
              ]}
              onChange={setInputMode}
            />
            <button className="secondary-button" type="button" onClick={startNewAttempt} disabled={!consentAccepted}>
              New attempt
            </button>
          </div>

          <div className="entry-wrap">
            {inputMode === "fixed" ? <div className="ghost-text">{fixedPromptText}</div> : null}
            <textarea
              className="typing-area"
              value={rawText}
              disabled={inputDisabled}
              autoCapitalize="off"
              autoComplete="off"
              spellCheck="false"
              inputMode="text"
              aria-label={inputMode === "fixed" ? "Fixed text input" : "Free text input"}
              onChange={(event) => setRawText(event.target.value)}
              onKeyDown={(event) => recordEvent(event, "keydown")}
              onKeyUp={(event) => recordEvent(event, "keyup")}
              onBeforeInput={(event) => recordEvent(event, "beforeinput")}
              onInput={(event) => recordEvent(event, "input", event.currentTarget.value)}
              onCompositionStart={(event) => recordEvent(event, "compositionstart")}
              onCompositionUpdate={(event) => recordEvent(event, "compositionupdate")}
              onCompositionEnd={(event) => recordEvent(event, "compositionend", event.currentTarget.value)}
              onPaste={(event) => recordEvent(event, "paste", event.currentTarget.value)}
            />
          </div>

          <div className="status-strip">
            <span>Attempt {shortId(attemptId)}</span>
            <span>Queued {savedCounts.queuedEvents}</span>
            <span>Saved {savedCounts.savedEvents}</span>
            <span>Analyses {savedCounts.analyses}</span>
          </div>
        </section>

        <aside className="result-panel">
          <div className={`verdict ${verdict.className}`}>
            <span>{verdict.label}</span>
            <strong>{verdict.detail}</strong>
          </div>

          <div className="score-list">
            {["Instant Baseline", "LightGBM", "1D-CNN"].map((modelName) => (
              <ModelResult
                key={modelName}
                result={results.find((item) => item.modelName === modelName)}
                fallbackName={modelName}
              />
            ))}
          </div>

          <div className="metrics-block">
            <h2>Reference Metrics</h2>
            <ReferenceMetrics metrics={config?.referenceMetrics} mode={inputMode} />
          </div>

          <div className="metrics-block">
            <h2>Collection Quality</h2>
            <div className="quality-grid">
              <Metric label="Quality" value={quality?.featureQuality || "waiting"} />
              <Metric label="Key coverage" value={formatPercent(quality?.keyupCoverage)} />
              <Metric label="Composition" value={formatPercent(quality?.compositionRatio)} />
              <Metric label="Templates" value={`${templateCounts.fixed}/${templateCounts.free}`} />
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

function ConsentModal({ consentVersion, deviceClass, isSubmitting, onAccept }) {
  return (
    <div className="consent-backdrop">
      <section className="consent-dialog">
        <p className="eyebrow">Research Consent</p>
        <h2>개인정보 및 키 입력 데이터 수집 동의</h2>
        <p>
          이 데모는 연구 분석을 위해 자유 텍스트 원문, 지정 텍스트 입력값, 키 입력 timing,
          IP 주소, user agent, 기기/화면 정보, 모델 결과와 역할 라벨을 저장합니다.
        </p>
        <p>
          저장된 데이터는 모바일과 데스크톱을 분리해 분석하며, 관리자 화면에서 CSV로
          내보낼 수 있습니다.
        </p>
        <div className="consent-facts">
          <span>Version {consentVersion || "loading"}</span>
          <span>{deviceClass}</span>
          <span>Raw text stored</span>
        </div>
        <button className="primary-button" type="button" onClick={onAccept} disabled={isSubmitting}>
          {isSubmitting ? "Starting..." : "Agree and start"}
        </button>
      </section>
    </div>
  );
}

function AdminApp() {
  const [pin, setPin] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function loadMetrics(event) {
    event?.preventDefault();
    setLoading(true);
    setError("");
    try {
      const payload = await apiGet("/api/admin/metrics", {
        headers: { "x-admin-pin": pin }
      });
      setMetrics(payload);
    } catch (apiError) {
      setError(apiError.message);
    } finally {
      setLoading(false);
    }
  }

  async function exportTable(table) {
    try {
      await downloadAdminCsv(table, pin);
    } catch (apiError) {
      setError(apiError.message);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>Research Data Console</h1>
        </div>
        <div className="topbar-meta">
          <a href="/">Demo</a>
        </div>
      </header>

      <form className="admin-login" onSubmit={loadMetrics}>
        <input
          value={pin}
          onChange={(event) => setPin(event.target.value)}
          type="password"
          placeholder="ADMIN_PIN"
          aria-label="Admin PIN"
        />
        <button className="primary-button" type="submit" disabled={!pin || loading}>
          {loading ? "Loading..." : "Open console"}
        </button>
      </form>

      {error ? <div className="error-banner">{error}</div> : null}

      {metrics ? (
        <main className="admin-grid">
          <section className="admin-panel">
            <h2>Totals</h2>
            <div className="quality-grid">
              {Object.entries(metrics.totals).map(([label, value]) => (
                <Metric key={label} label={label} value={value} />
              ))}
            </div>
          </section>

          <section className="admin-panel">
            <h2>Collected Metrics</h2>
            <MetricsTable rows={metrics.collectedMetrics} />
          </section>

          <section className="admin-panel">
            <h2>Exports</h2>
            <div className="export-row">
              {EXPORT_TABLES.map((table) => (
                <button key={table} className="secondary-button" type="button" onClick={() => exportTable(table)}>
                  {table}.csv
                </button>
              ))}
            </div>
          </section>

          <section className="admin-panel">
            <h2>Recent Attempts</h2>
            <div className="recent-list">
              {metrics.recentAttempts.map((attempt) => (
                <div className="recent-item" key={attempt.id}>
                  <strong>{shortId(attempt.id)}</strong>
                  <span>{attempt.inputMode} / {attempt.roleLabel} / {attempt.deviceClass}</span>
                  <span>{attempt.rawTextLength} chars / {attempt.featureQuality || "pending"}</span>
                </div>
              ))}
            </div>
          </section>
        </main>
      ) : null}
    </div>
  );
}

function SegmentedControl({ label, value, options, onChange }) {
  return (
    <div className="segmented" aria-label={label}>
      {options.map(([optionValue, optionLabel]) => (
        <button
          key={optionValue}
          type="button"
          className={value === optionValue ? "active" : ""}
          onClick={() => onChange(optionValue)}
        >
          {optionLabel}
        </button>
      ))}
    </div>
  );
}

function ModelResult({ result, fallbackName }) {
  const scoreText = result?.score === null || result?.score === undefined
    ? "--"
    : result.score.toFixed(3);
  return (
    <div className="model-row">
      <div>
        <strong>{result?.modelName || fallbackName}</strong>
        <span>{result?.decision || "waiting"}</span>
      </div>
      <div>
        <strong>{scoreText}</strong>
        <span>{formatMs(result?.inferenceTimeMs)}</span>
      </div>
    </div>
  );
}

function ReferenceMetrics({ metrics, mode }) {
  const key = mode === "fixed" ? "dsl" : "mmcTiming";
  const rows = metrics?.[key] || [];
  return (
    <div className="reference-list">
      {rows.slice(0, 3).map((row) => (
        <div className="reference-row" key={row.model}>
          <span>{normalizeModelName(row.model)}</span>
          <span>EER {formatPercent(row.eer)}</span>
          <span>Acc {formatPercent(row.accuracy)}</span>
        </div>
      ))}
    </div>
  );
}

function MetricsTable({ rows }) {
  if (!rows.length) {
    return <p className="muted">insufficient labeled attempts</p>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Device</th>
            <th>Status</th>
            <th>EER</th>
            <th>FAR</th>
            <th>FRR</th>
            <th>Samples</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.modelName}-${row.deviceClass}`}>
              <td>{row.modelName}</td>
              <td>{row.deviceClass}</td>
              <td>{row.status}</td>
              <td>{formatPercent(row.eer)}</td>
              <td>{formatPercent(row.far)}</td>
              <td>{formatPercent(row.frr)}</td>
              <td>{row.samples}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric-tile">
      <span>{label}</span>
      <strong>{value ?? "--"}</strong>
    </div>
  );
}

function findReferenceMetrics(config, mode, modelName) {
  const key = mode === "fixed" ? "dsl" : "mmcTiming";
  const rows = config?.referenceMetrics?.[key] || [];
  const normalized = normalizeModelName(modelName);
  return rows.find((row) => normalizeModelName(row.model) === normalized) || null;
}

function normalizeModelName(name = "") {
  if (name.includes("Manhattan") || name.includes("Baseline")) {
    return "Baseline";
  }
  if (name.includes("LightGBM")) {
    return "LightGBM";
  }
  if (name.includes("CNN")) {
    return "1D-CNN";
  }
  return name;
}

function bestVerdict(results) {
  const baseline = results.find((result) => result.modelName === "Instant Baseline");
  if (!baseline) {
    return { label: "Ready", detail: "waiting for input", className: "neutral" };
  }
  if (baseline.decision?.startsWith("accept")) {
    return { label: "Accepted", detail: baseline.decision, className: "accept" };
  }
  if (baseline.decision?.includes("profile")) {
    return { label: "Accepted", detail: baseline.decision, className: "accept" };
  }
  if (baseline.decision === "reject") {
    return { label: "Review", detail: "baseline mismatch", className: "reject" };
  }
  return { label: "Learning", detail: baseline.decision, className: "neutral" };
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatMs(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-- ms";
  }
  return `${Number(value).toFixed(2)} ms`;
}

function shortId(id) {
  return id ? id.slice(0, 8) : "pending";
}

function randomId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
}
