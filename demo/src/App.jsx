import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiGet, apiPatch, apiPost, downloadAdminCsv } from "./api.js";
import { loadEvaluationRun, runBrowserEvaluation } from "./evaluation/evaluationClient.js";
import { initInferenceWorker, predictInWorker } from "./inferenceClient.js";
import {
  collectClientMetadata,
  detectDeviceClass,
  extractFeatureBundle,
  featureVectorToObject,
  getTemplateCounts,
  makeEventPayload,
  scoreInstantBaseline
} from "./keystroke.js";

const RAW_EXPORTS = ["participants", "sessions", "attempts", "events", "features", "results", "monitoring_windows", "model_calibrations"];
const ANALYSIS_EXPORTS = ["attempt_features", "event_pairs", "model_results", "quality_summary", "monitoring_windows"];

export default function App() {
  const isEvaluationAdmin = window.location.pathname.startsWith("/admin/evaluation");
  if (isEvaluationAdmin) {
    return <EvaluationAdminApp />;
  }
  const isAdmin = window.location.pathname.startsWith("/admin");
  if (isAdmin) {
    return <AdminApp />;
  }
  return <ParticipantApp />;
}

function ParticipantApp() {
  const query = useMemo(() => new URLSearchParams(window.location.search), []);
  const [config, setConfig] = useState(null);
  const [deviceClass] = useState(() => detectDeviceClass());
  const [participantId, setParticipantId] = useState("");
  const [participantCode, setParticipantCode] = useState(query.get("participantCode") || "");
  const [participantCodeHash, setParticipantCodeHash] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [sessionNo, setSessionNo] = useState(numberFromQuery(query, "sessionNo", 1));
  const [trialNo, setTrialNo] = useState(numberFromQuery(query, "trialNo", 1));
  const [inputMode, setInputMode] = useState(query.get("inputMode") === "free" ? "free" : "fixed");
  const [roleLabel, setRoleLabel] = useState(normalizeRole(query.get("role")));
  const [targetParticipantCode] = useState(query.get("targetParticipantCode") || "");
  const [selectedPromptSetId, setSelectedPromptSetId] = useState(query.get("promptSetId") || "");
  const [rawText, setRawText] = useState("");
  const [typingStartedAt, setTypingStartedAt] = useState(null);
  const [typingNow, setTypingNow] = useState(() => performance.now());
  const [error, setError] = useState("");
  const [isConsenting, setIsConsenting] = useState(false);
  const [isWorking, setIsWorking] = useState(false);
  const [attemptId, setAttemptId] = useState("");
  const [attemptStatus, setAttemptStatus] = useState("idle");
  const [results, setResults] = useState([]);
  const [quality, setQuality] = useState(null);
  const [templateCounts, setTemplateCounts] = useState(() => getTemplateCounts());
  const [savedCounts, setSavedCounts] = useState({ queuedEvents: 0, savedEvents: 0, analyses: 0, windows: 0 });
  const [artifactState, setArtifactState] = useState("checking");
  const [workerEnabled, setWorkerEnabled] = useState(false);
  const [calibrations, setCalibrations] = useState([]);

  const attemptIdRef = useRef("");
  const attemptStartedAtRef = useRef(performance.now());
  const eventQueueRef = useRef([]);
  const allEventsRef = useRef([]);
  const rawTextRef = useRef("");
  const flushPromiseRef = useRef(Promise.resolve());
  const lastMonitoringCharRef = useRef(0);

  useEffect(() => {
    apiGet("/api/config")
      .then((payload) => {
        setConfig(payload);
        if (!query.get("inputMode")) {
          const firstFixed = payload.promptSets?.find((set) => set.inputMode === "fixed");
          if (firstFixed) setSelectedPromptSetId(firstFixed.id);
        }
      })
      .catch((apiError) => setError(apiError.message));
    initInferenceWorker().then((runtime) => {
      setArtifactState(runtime?.status || "optional artifacts unavailable");
      setWorkerEnabled(Boolean(runtime?.worker));
    });
  }, [query]);

  useEffect(() => {
    attemptIdRef.current = attemptId;
  }, [attemptId]);

  useEffect(() => {
    rawTextRef.current = rawText;
  }, [rawText]);

  useEffect(() => {
    if (rawText.length > 0 && typingStartedAt === null) {
      const now = performance.now();
      setTypingStartedAt(now);
      setTypingNow(now);
    }
  }, [rawText.length, typingStartedAt]);

  useEffect(() => {
    if (!typingStartedAt || attemptStatus !== "in_progress") {
      return undefined;
    }
    const interval = window.setInterval(() => {
      setTypingNow(performance.now());
    }, 1000);
    return () => window.clearInterval(interval);
  }, [attemptStatus, typingStartedAt]);

  const devControls = query.get("devControls") === "1" || Boolean(config?.showDevControls);
  const monitoringEnabled = query.get("monitoring") === "1" || Boolean(config?.continuousMonitoringEnabled);
  const metadata = useMemo(() => collectClientMetadata(deviceClass), [deviceClass]);
  const promptSets = config?.promptSets || [];
  const fixedPromptSet = useMemo(
    () => findPromptSet(promptSets, selectedPromptSetId, "fixed"),
    [promptSets, selectedPromptSetId]
  );
  const freePromptSet = useMemo(
    () => findPromptSet(promptSets, selectedPromptSetId, "free"),
    [promptSets, selectedPromptSetId]
  );
  const activePrompt = useMemo(
    () => selectPromptForFlow(inputMode, trialNo, fixedPromptSet, freePromptSet),
    [fixedPromptSet, freePromptSet, inputMode, trialNo]
  );
  const activePromptSetId = inputMode === "fixed" ? fixedPromptSet?.id : freePromptSet?.id;
  const nextPromptStep = useMemo(
    () => getNextPromptStep(inputMode, trialNo, fixedPromptSet, freePromptSet),
    [fixedPromptSet, freePromptSet, inputMode, trialNo]
  );
  const consentAccepted = Boolean(participantId && sessionId);
  const inputDisabled = !consentAccepted || !attemptId || attemptStatus !== "in_progress" || isWorking;
  const canFinish = consentAccepted && attemptStatus === "in_progress" && !isWorking;
  const canNext = consentAccepted && ["submitted", "cancelled", "excluded"].includes(attemptStatus) && !isWorking && Boolean(nextPromptStep);
  const shouldEmphasizeNext = consentAccepted && ["submitted", "excluded"].includes(attemptStatus) && !isWorking && Boolean(nextPromptStep);
  const collectionComplete = consentAccepted && ["submitted", "excluded"].includes(attemptStatus) && !nextPromptStep;
  const typingMetric = useMemo(
    () => formatTypingRate(rawText.length, typingStartedAt, typingNow),
    [rawText.length, typingNow, typingStartedAt]
  );

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

  useEffect(() => {
    const interval = window.setInterval(() => {
      flushEvents();
    }, 1500);
    return () => window.clearInterval(interval);
  }, [flushEvents]);

  useEffect(() => {
    if (!attemptId || attemptStatus !== "in_progress") {
      return undefined;
    }
    const timeout = window.setTimeout(() => {
      apiPatch(`/api/attempt/${attemptId}`, {
        rawText,
        inputMode,
        roleLabel,
        trialNo,
        promptId: activePrompt?.id || "",
        promptSetId: activePromptSetId || "",
        promptText: activePrompt?.text || "",
        targetParticipantCode,
        suggestionShown: inputMode === "free" && Boolean(activePrompt),
        suggestionId: inputMode === "free" ? activePrompt?.id || "" : ""
      }).catch((apiError) => setError(apiError.message));
    }, 700);
    return () => window.clearTimeout(timeout);
  }, [activePrompt, activePromptSetId, attemptId, attemptStatus, inputMode, rawText, roleLabel, targetParticipantCode, trialNo]);

  useEffect(() => {
    if (!monitoringEnabled || !attemptId || attemptStatus !== "in_progress") {
      return;
    }
    processMonitoringWindows().catch((apiError) => setError(apiError.message));
  }, [rawText, monitoringEnabled, attemptId, attemptStatus]); // eslint-disable-line react-hooks/exhaustive-deps

  async function acceptConsent() {
    setIsConsenting(true);
    setError("");
    try {
      const consent = await apiPost("/api/consent", {
        accepted: true,
        participantCode,
        consentTimestamp: new Date().toISOString(),
        deviceClass,
        metadata
      });
      const session = await apiPost("/api/session", {
        participantId: consent.participantId,
        sessionNo,
        startedAt: new Date().toISOString(),
        deviceClass,
        ...metadata
      });
      setParticipantId(consent.participantId);
      setParticipantCode(consent.participantCode || participantCode);
      setParticipantCodeHash(consent.participantCodeHash || "");
      setSessionId(session.sessionId);
      setSessionNo(session.sessionNo || sessionNo);
      const calibrationPayload = await apiGet(`/api/calibrations?participantId=${encodeURIComponent(consent.participantId)}&participantCode=${encodeURIComponent(consent.participantCode || participantCode)}`);
      setCalibrations(calibrationPayload.calibrations || []);
      await createAttempt({
        participantId: consent.participantId,
        sessionId: session.sessionId,
        nextTrialNo: trialNo,
        nextInputMode: inputMode,
        nextRoleLabel: roleLabel
      });
    } catch (apiError) {
      setError(apiError.message);
    } finally {
      setIsConsenting(false);
    }
  }

  async function createAttempt({ participantId: pid = participantId, sessionId: sid = sessionId, nextTrialNo = trialNo, nextInputMode = inputMode, nextRoleLabel = roleLabel, nextPrompt = null } = {}) {
    if (!pid || !sid || !config) {
      return;
    }
    setIsWorking(true);
    try {
      await flushEvents();
      const nextAttemptId = randomId();
      const prompt = nextPrompt || selectPromptForFlow(nextInputMode, nextTrialNo, fixedPromptSet, freePromptSet);
      attemptStartedAtRef.current = performance.now();
      eventQueueRef.current = [];
      allEventsRef.current = [];
      lastMonitoringCharRef.current = 0;
      setRawText("");
      setTypingStartedAt(null);
      setTypingNow(performance.now());
      setResults([]);
      setQuality(null);
      setSavedCounts({ queuedEvents: 0, savedEvents: 0, analyses: 0, windows: 0 });

      await apiPost("/api/attempt", {
        id: nextAttemptId,
        participantId: pid,
        sessionId: sid,
        inputMode: nextInputMode,
        roleLabel: nextRoleLabel,
        trialNo: nextTrialNo,
        promptId: prompt?.id || "",
        promptSetId: nextInputMode === "fixed" ? fixedPromptSet?.id : freePromptSet?.id,
        promptText: prompt?.text || "",
        targetParticipantCode,
        rawText: "",
        startedAt: new Date().toISOString(),
        deviceClass,
        status: "in_progress",
        suggestionShown: nextInputMode === "free" && Boolean(prompt),
        suggestionId: nextInputMode === "free" ? prompt?.id || "" : "",
        summary: { artifactState, workerEnabled }
      });
      setAttemptId(nextAttemptId);
      setAttemptStatus("in_progress");
    } catch (apiError) {
      setError(apiError.message);
    } finally {
      setIsWorking(false);
    }
  }

  function recordEvent(event, type, value = rawTextRef.current) {
    if (!attemptIdRef.current || attemptStatus !== "in_progress") {
      return;
    }
    const payload = makeEventPayload(event.nativeEvent || event, type, value, attemptStartedAtRef.current);
    eventQueueRef.current.push(payload);
    allEventsRef.current.push(payload);
    setSavedCounts((current) => ({ ...current, queuedEvents: eventQueueRef.current.length }));
  }

  async function submitAttempt() {
    if (!canFinish) {
      return;
    }
    setIsWorking(true);
    setError("");
    try {
      await flushEvents();
      const submittedAt = new Date().toISOString();
      const bundle = extractCurrentBundle();
      setQuality(bundle.quality);
      const baselineStartedAt = performance.now();
      const baseline = scoreInstantBaseline(inputMode, roleLabel, bundle.vector, attemptId);
      const baselineResult = {
        ...baseline,
        modelScope: `Instant Baseline::${deviceClass}::${inputMode}`,
        deviceClass,
        inferenceTimeMs: performance.now() - baselineStartedAt,
        uiBlockingTimeMs: performance.now() - baselineStartedAt
      };
      const optionalResults = await predictInWorker(bundle.vector, {
        inputMode,
        deviceClass,
        participantId,
        participantCode,
        calibrations
      });
      const enrichedResults = [baselineResult, ...optionalResults].map((result) => ({
        ...result,
        referenceMetrics: findReferenceMetrics(config, inputMode, result.modelName),
        payload: {
          ...(result.payload || {}),
          featureQuality: bundle.quality,
          templateCounts: getTemplateCounts(),
          participantCodeHash
        }
      }));
      const finalStatus = bundle.quality.qualityStatus === "excluded" ? "excluded" : "submitted";
      await Promise.all([
        apiPatch(`/api/attempt/${attemptId}`, {
          rawText: rawTextRef.current,
          endedAt: submittedAt,
          submittedAt,
          inputMode,
          roleLabel,
          trialNo,
          promptId: activePrompt?.id || "",
          promptSetId: activePromptSetId || "",
          promptText: activePrompt?.text || "",
          targetParticipantCode,
          featureQuality: bundle.quality.featureQuality,
          qualityStatus: bundle.quality.qualityStatus,
          exclusionReason: bundle.quality.exclusionReason,
          pasteCount: bundle.quality.pasteCount,
          fixedPromptMatch: bundle.quality.fixedPromptMatch,
          fixedPromptEditDistance: bundle.quality.fixedPromptEditDistance,
          suggestionShown: inputMode === "free" && Boolean(activePrompt),
          suggestionId: inputMode === "free" ? activePrompt?.id || "" : "",
          status: finalStatus,
          summary: bundle.quality
        }),
        apiPost("/api/features/bulk", {
          attemptId,
          features: bundle.features
        }),
        apiPost("/api/results", {
          attemptId,
          results: enrichedResults
        })
      ]);
      setResults(enrichedResults);
      setTemplateCounts(getTemplateCounts());
      setAttemptStatus(finalStatus);
      setSavedCounts((current) => ({ ...current, analyses: current.analyses + 1 }));
    } catch (apiError) {
      setError(apiError.message);
    } finally {
      setIsWorking(false);
    }
  }

  async function cancelAttempt() {
    if (!attemptId || attemptStatus !== "in_progress") {
      return;
    }
    setIsWorking(true);
    try {
      await flushEvents();
      await apiPatch(`/api/attempt/${attemptId}`, {
        rawText: rawTextRef.current,
        endedAt: new Date().toISOString(),
        status: "cancelled",
        summary: { ...(quality || {}), cancelled: true }
      });
      setAttemptStatus("cancelled");
    } catch (apiError) {
      setError(apiError.message);
    } finally {
      setIsWorking(false);
    }
  }

  async function nextPrompt() {
    if (!nextPromptStep) {
      return;
    }
    setTrialNo(nextPromptStep.trialNo);
    setInputMode(nextPromptStep.inputMode);
    await createAttempt({
      nextTrialNo: nextPromptStep.trialNo,
      nextInputMode: nextPromptStep.inputMode,
      nextPrompt: nextPromptStep.prompt
    });
  }

  function extractCurrentBundle(windowEvents = allEventsRef.current, windowText = rawTextRef.current) {
    return extractFeatureBundle(windowEvents, windowText, inputMode, deviceClass, {
      promptText: inputMode === "fixed" ? activePrompt?.text || "" : "",
      minChars: inputMode === "fixed" ? activePrompt?.minChars : 80,
      pastePolicyFixed: config?.pastePolicies?.fixed,
      pastePolicyFree: config?.pastePolicies?.free,
      suggestionShown: inputMode === "free" && Boolean(activePrompt),
      suggestionId: inputMode === "free" ? activePrompt?.id : null
    });
  }

  async function processMonitoringWindows() {
    const windowSize = Number(config?.monitoringWindowChars || 120);
    const step = Number(config?.monitoringStepChars || 60);
    const windows = [];
    while (rawTextRef.current.length - lastMonitoringCharRef.current >= windowSize) {
      const charStart = lastMonitoringCharRef.current;
      const charEnd = charStart + windowSize;
      const windowText = rawTextRef.current.slice(charStart, charEnd);
      const windowEvents = allEventsRef.current.filter(
        (event) => Number(event.valueLength) >= charStart && Number(event.valueLength) <= charEnd
      );
      const bundle = extractCurrentBundle(windowEvents, windowText);
      const prediction = await predictInWorker(bundle.vector, {
        inputMode,
        deviceClass,
        participantId,
        participantCode,
        calibrations
      });
      const times = windowEvents.map((event) => Number(event.relativeTime)).filter(Number.isFinite);
      windows.push({
        windowId: `${attemptId}-${charStart}-${charEnd}`,
        participantCode,
        windowStartTime: times.length ? Math.min(...times) : null,
        windowEndTime: times.length ? Math.max(...times) : null,
        windowCharStart: charStart,
        windowCharEnd: charEnd,
        features: featureVectorToObject(bundle.features),
        results: prediction,
        qualityStatus: bundle.quality.qualityStatus,
        inferenceTimeMs: prediction.reduce((sum, result) => sum + Number(result.inferenceTimeMs || 0), 0)
      });
      lastMonitoringCharRef.current += step;
    }
    if (windows.length) {
      await apiPost("/api/monitoring-windows", { attemptId, windows });
      setSavedCounts((current) => ({ ...current, windows: current.windows + windows.length }));
      if (devControls) {
        setResults(windows[windows.length - 1].results);
        setQuality(extractCurrentBundle().quality);
      }
    }
  }

  const verdict = bestVerdict(results);

  return (
    <div className="app-shell">
      {!consentAccepted ? (
        <ConsentModal
          consentVersion={config?.consentVersion}
          deviceClass={deviceClass}
          participantCode={participantCode}
          setParticipantCode={setParticipantCode}
          sessionNo={sessionNo}
          setSessionNo={setSessionNo}
          isSubmitting={isConsenting}
          onAccept={acceptConsent}
        />
      ) : null}

      <header className="topbar">
        <div>
          <p className="eyebrow">Continuous Authentication Research</p>
          <h1>Keystroke Data Collection</h1>
        </div>
        <div className="topbar-meta">
          <span>{deviceClass}</span>
          <span>{workerEnabled ? "worker inference" : "main-thread fallback"}</span>
          <span>{artifactState}</span>
          <a href="/admin">Admin</a>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <main className="demo-grid" aria-hidden={!consentAccepted}>
        <section className="input-panel">
          <div className="condition-grid">
            <Metric label="Participant" value={participantCode || shortId(participantId)} />
            <Metric label="Session" value={sessionNo} />
            <Metric label="Trial" value={trialNo} />
            <Metric label="Role" value={roleLabel} />
            <Metric label="Prompt ID" value={activePrompt?.id || "free"} />
            <Metric label="Status" value={attemptStatus} />
          </div>

          {devControls ? (
            <div className="toolbar-row">
              <SegmentedControl
                label="Role"
                value={roleLabel}
                options={[["genuine", "Genuine"], ["imposter", "Imposter"]]}
                onChange={setRoleLabel}
              />
              <SegmentedControl
                label="Text"
                value={inputMode}
                options={[["fixed", "Fixed"], ["free", "Free"]]}
                onChange={setInputMode}
              />
              <select value={selectedPromptSetId} onChange={(event) => setSelectedPromptSetId(event.target.value)}>
                {promptSets.map((set) => (
                  <option key={set.id} value={set.id}>{set.label}</option>
                ))}
              </select>
            </div>
          ) : null}

          {inputMode === "fixed" ? (
            <div className="prompt-card">
              <strong>{activePrompt?.id}</strong>
              <p>{activePrompt?.text || "No fixed prompt configured."}</p>
            </div>
          ) : (
            <FreeSuggestion
              prompt={activePrompt}
            />
          )}

          {collectionComplete ? (
            <div className="completion-banner">
              <strong>데이터 수집이 완료되었습니다.</strong>
              <span>모든 프롬프트 제출이 끝났습니다.</span>
            </div>
          ) : null}

          <div className="entry-wrap">
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
            <div className="typing-meter" aria-live="polite">
              <span>{typingMetric.rate} 타/분</span>
              <span>{typingMetric.length}자</span>
              <span>{typingMetric.elapsed}</span>
            </div>
          </div>

          <div className="action-row">
            <button className="primary-button" type="button" onClick={submitAttempt} disabled={!canFinish}>
              {isWorking ? "Saving..." : "Submit attempt"}
            </button>
            <button className="secondary-button" type="button" onClick={cancelAttempt} disabled={!canFinish}>
              Cancel attempt
            </button>
            <button className="secondary-button" type="button" onClick={() => createAttempt()} disabled={!canNext}>
              New attempt
            </button>
            <button
              className={shouldEmphasizeNext ? "primary-button next-button" : "secondary-button"}
              type="button"
              onClick={nextPrompt}
              disabled={!canNext}
            >
              Next prompt
            </button>
          </div>

          <div className="status-strip">
            <span>Attempt {shortId(attemptId)}</span>
            <span>Queued {savedCounts.queuedEvents}</span>
            <span>Saved {savedCounts.savedEvents}</span>
            <span>Submitted analyses {savedCounts.analyses}</span>
            {monitoringEnabled ? <span>Monitoring windows {savedCounts.windows}</span> : null}
          </div>
        </section>

        <aside className="result-panel">
          {devControls ? (
            <>
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
            </>
          ) : (
            <div className="privacy-note">
              <strong>Collection mode</strong>
              <span>Model scores are hidden from participants. The attempt is analyzed after submission.</span>
            </div>
          )}

          <div className="metrics-block">
            <h2>Collection Quality</h2>
            <div className="quality-grid">
              <Metric label="Quality" value={quality?.featureQuality || "waiting"} />
              <Metric label="Status" value={quality?.qualityStatus || "waiting"} />
              <Metric label="Paste count" value={quality?.pasteCount ?? 0} />
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

function ConsentModal({ consentVersion, deviceClass, participantCode, setParticipantCode, sessionNo, setSessionNo, isSubmitting, onAccept }) {
  return (
    <div className="consent-backdrop">
      <section className="consent-dialog">
        <p className="eyebrow">Research Consent</p>
        <h2>개인정보 및 키 입력 데이터 수집 동의</h2>
        <p>
          이 데모는 연구용 키스트로크 데이터 수집 시스템입니다. 원문 raw text, key value,
          key code, keydown/keyup/beforeinput/input/composition/paste 이벤트, IP address,
          user agent, device/screen/browser metadata, 모델 결과와 CSV export 가능한 분석
          데이터를 저장합니다.
        </p>
        <p>
          입력 중 이름, 전화번호, 주소, 계정, 비밀번호, 민감정보를 쓰지 마세요. 철회나
          삭제 요청이 필요한 경우 연구자에게 문의하세요.
        </p>
        <label className="field-label">
          Participant code
          <input value={participantCode} onChange={(event) => setParticipantCode(event.target.value)} placeholder="P001" />
        </label>
        <label className="field-label">
          Session no.
          <input value={sessionNo} onChange={(event) => setSessionNo(Number(event.target.value) || 1)} type="number" min="1" />
        </label>
        <div className="consent-facts">
          <span>Version {consentVersion || "loading"}</span>
          <span>{deviceClass}</span>
          <span>Raw text stored</span>
          <span>Key values stored</span>
        </div>
        <button className="primary-button" type="button" onClick={onAccept} disabled={isSubmitting || !participantCode.trim()}>
          {isSubmitting ? "Starting..." : "Agree and start"}
        </button>
      </section>
    </div>
  );
}

function FreeSuggestion({ prompt }) {
  return (
    <div className="prompt-card">
      <strong>{prompt?.id || "free"}</strong>
      <p>
        아래 주제에 맞춰 자유롭게 작성해 주세요. 이름, 전화번호, 주소, 계정, 비밀번호는
        입력하지 마세요.
      </p>
      {prompt ? <blockquote>{prompt.text}</blockquote> : <span className="muted">No free prompt configured.</span>}
    </div>
  );
}

function AdminApp() {
  const [pin, setPin] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [calibrations, setCalibrations] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function loadMetrics(event) {
    event?.preventDefault();
    setLoading(true);
    setError("");
    try {
      const [payload, calibrationPayload] = await Promise.all([
        apiGet("/api/admin/metrics", { headers: { "x-admin-pin": pin } }),
        apiGet("/api/admin/calibrations", { headers: { "x-admin-pin": pin } })
      ]);
      setMetrics(payload);
      setCalibrations(calibrationPayload.calibrations || []);
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
          <a href="/admin/evaluation">Evaluation</a>
          <a href="/">Demo</a>
        </div>
      </header>

      <form className="admin-login" onSubmit={loadMetrics}>
        <input value={pin} onChange={(event) => setPin(event.target.value)} type="password" placeholder="ADMIN_PIN" aria-label="Admin PIN" />
        <button className="primary-button" type="submit" disabled={!pin || loading}>
          {loading ? "Loading..." : "Open console"}
        </button>
      </form>

      {error ? <div className="error-banner">{error}</div> : null}

      {metrics ? (
        <main className="admin-grid">
          <section className="admin-panel">
            <h2>Quality Overview</h2>
            <div className="quality-grid">
              {Object.entries(metrics.totals).map(([label, value]) => (
                <Metric key={label} label={label} value={value} />
              ))}
              <Metric label="avg keyup coverage" value={formatPercent(metrics.qualityOverview?.avgKeyupCoverage)} />
              <Metric label="avg composition" value={formatPercent(metrics.qualityOverview?.avgCompositionRatio)} />
              <Metric label="avg timing coverage" value={formatPercent(metrics.qualityOverview?.avgTimingCoverage)} />
            </div>
          </section>

          <section className="admin-panel">
            <h2>Attempts by Condition</h2>
            <div className="three-tables">
              <SmallCountTable title="Device" rows={metrics.byDevice} labelKey="deviceClass" />
              <SmallCountTable title="Input mode" rows={metrics.byInputMode} labelKey="inputMode" />
              <SmallCountTable title="Role" rows={metrics.byRole} labelKey="roleLabel" />
            </div>
          </section>

          <section className="admin-panel">
            <h2>Prompt Coverage</h2>
            <SmallCountTable rows={metrics.byPrompt} labelKey="promptId" />
          </section>

          <section className="admin-panel">
            <h2>Exclusion Reasons</h2>
            <SmallCountTable rows={metrics.exclusionReasons} labelKey="exclusionReason" />
          </section>

          <section className="admin-panel">
            <h2>Collected Metrics</h2>
            <MetricsTable rows={metrics.collectedMetrics || []} />
          </section>

          <section className="admin-panel">
            <h2>Prompt Sets</h2>
            <div className="recent-list">
              {metrics.prompts?.promptSets?.map((set) => (
                <div className="recent-item" key={set.id}>
                  <strong>{set.id}</strong>
                  <span>{set.label} / {set.inputMode} / {set.language}</span>
                  <span>{set.prompts?.length || 0} prompts</span>
                </div>
              ))}
            </div>
          </section>

          <section className="admin-panel">
            <h2>Exports</h2>
            <h3>Analysis-ready</h3>
            <div className="export-row">
              {ANALYSIS_EXPORTS.map((table) => (
                <button key={table} className="secondary-button" type="button" onClick={() => exportTable(table)}>
                  {table}.csv
                </button>
              ))}
            </div>
            <h3>Raw tables</h3>
            <div className="export-row">
              {RAW_EXPORTS.map((table) => (
                <button key={table} className="secondary-button" type="button" onClick={() => exportTable(table)}>
                  {table}.csv
                </button>
              ))}
            </div>
          </section>

          <section className="admin-panel">
            <h2>Calibrations</h2>
            <div className="recent-list">
              {calibrations.slice(0, 20).map((row) => (
                <div className="recent-item" key={row.id}>
                  <strong>{row.participantCode || "global"} / {row.modelName}</strong>
                  <span>{row.inputMode || "any"} / {row.deviceClass || "any"} / {row.modelScope || "any"}</span>
                  <span>threshold {Number(row.threshold).toFixed(3)} / refs {row.referenceAttemptCount}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="admin-panel">
            <h2>Recent Low-quality Attempts</h2>
            <div className="recent-list">
              {(metrics.recentLowQuality || []).map((attempt) => (
                <div className="recent-item" key={attempt.id}>
                  <strong>{shortId(attempt.id)} / {attempt.participantCode}</strong>
                  <span>{attempt.inputMode} / {attempt.roleLabel} / {attempt.deviceClass} / prompt {attempt.promptId || "none"}</span>
                  <span>{attempt.qualityStatus} / {attempt.exclusionReason || "no reason"} / paste {attempt.pasteCount}</span>
                </div>
              ))}
            </div>
          </section>
        </main>
      ) : null}
    </div>
  );
}

const EVALUATION_DATASETS = [
  {
    id: "dsl",
    label: "DSL",
    currentDescription: "현재 특징: H/DD/UD timing columns",
    enrichedDescription: "확장 특징: key sequence + trigraph + timing distribution + adaptive feature"
  },
  {
    id: "mmc",
    label: "MMC",
    currentDescription: "현재 특징: timing channels 14..19 flattened sequence",
    enrichedDescription: "확장 특징: timing distribution + delta/trigraph curvature + adaptive feature"
  },
  {
    id: "collected_data_analysis",
    label: "Collected CSV",
    currentDescription: "현재 특징: features.csv browser_baseline aggregate features",
    enrichedDescription: "확장 특징: events.csv key sequence + digraph/trigraph + distribution + adaptive feature"
  }
];

function EvaluationAdminApp() {
  const [pin, setPin] = useState("");
  const [deviceLabel, setDeviceLabel] = useState(() => defaultDeviceLabel());
  const [runId, setRunId] = useState("");
  const [status, setStatus] = useState(null);
  const [progress, setProgress] = useState([]);
  const [error, setError] = useState("");
  const [runningDataset, setRunningDataset] = useState("");
  const [activeView, setActiveView] = useState("run");

  const environments = useMemo(() => parseCsvText(status?.environments || ""), [status]);
  const summaries = status?.summaries || [];
  const artifacts = status?.artifacts || [];

  async function runDataset(dataset, featureMode) {
    const runKey = `${dataset}:${featureMode}`;
    setError("");
    setRunningDataset(runKey);
    setProgress([]);
    try {
      const result = await runBrowserEvaluation({
        dataset,
        featureMode,
        adminPin: pin,
        deviceLabel,
        runId: runId.trim(),
        onProgress: appendProgress
      });
      setRunId(result.runId);
      setStatus(result.status);
      setActiveView("compare");
    } catch (apiError) {
      setError(apiError.message);
    } finally {
      setRunningDataset("");
    }
  }

  async function loadRun(event) {
    event?.preventDefault();
    setError("");
    try {
      const payload = await loadEvaluationRun(runId.trim(), pin);
      setStatus(payload);
      setActiveView("compare");
    } catch (apiError) {
      setError(apiError.message);
    }
  }

  function appendProgress(item) {
    setProgress((current) => [...current.slice(-80), {
      ...item,
      at: item.at || new Date().toISOString()
    }]);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Admin Evaluation</p>
          <h1>Browser-side Model Evaluation</h1>
        </div>
        <div className="topbar-meta">
          <a href="/admin">Admin</a>
          <a href="/">Demo</a>
        </div>
      </header>

      <form className="admin-login evaluation-login" onSubmit={loadRun}>
        <input value={pin} onChange={(event) => setPin(event.target.value)} type="password" placeholder="ADMIN_PIN" aria-label="Admin PIN" />
        <input value={deviceLabel} onChange={(event) => setDeviceLabel(event.target.value)} placeholder="Environment label" aria-label="Environment label" />
        <input value={runId} onChange={(event) => setRunId(event.target.value)} placeholder="Run ID (blank creates new)" aria-label="Evaluation run ID" />
        <button className="secondary-button" type="submit" disabled={!pin || !runId.trim() || Boolean(runningDataset)}>
          Load run
        </button>
      </form>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="evaluation-tabs">
        <SegmentedControl
          label="Evaluation view"
          value={activeView}
          options={[["run", "Run"], ["compare", "Environment comparison"], ["artifacts", "Artifacts"]]}
          onChange={setActiveView}
        />
      </div>

      {activeView === "run" ? (
        <main className="evaluation-grid">
          <section className="admin-panel">
            <h2>평가 실행</h2>
            <p className="muted">
              모든 학습, 추론, 지표 계산, 시간 측정은 브라우저 Web Worker에서 수행됩니다. 서버는 데이터 제공과 산출물 저장만 합니다.
            </p>
            <div className="dataset-actions">
              {EVALUATION_DATASETS.map((dataset) => (
                <div className="evaluation-dataset-row" key={dataset.id}>
                  <div>
                    <strong>{dataset.label}</strong>
                    <span>학습은 과거 데이터, 테스트는 미래 데이터로 수행합니다.</span>
                  </div>
                  <div className="evaluation-mode-actions">
                    <button
                      className="primary-button evaluation-run-button"
                      type="button"
                      disabled={!pin || Boolean(runningDataset)}
                      onClick={() => runDataset(dataset.id, "current")}
                    >
                      <span>{runningDataset === `${dataset.id}:current` ? "실행 중..." : "현재 데이터 평가"}</span>
                      <small>{dataset.currentDescription}</small>
                    </button>
                    <button
                      className="primary-button evaluation-run-button"
                      type="button"
                      disabled={!pin || Boolean(runningDataset)}
                      onClick={() => runDataset(dataset.id, "enriched")}
                    >
                      <span>{runningDataset === `${dataset.id}:enriched` ? "실행 중..." : "확장 데이터 평가"}</span>
                      <small>{dataset.enrichedDescription}</small>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="admin-panel">
            <h2>Run Manifest</h2>
            <div className="quality-grid">
              <Metric label="Run ID" value={runId || "new run"} />
              <Metric label="Environment label" value={deviceLabel || "not_available"} />
              <Metric label="Saved environments" value={environments.length} />
              <Metric label="Saved artifacts" value={artifacts.length} />
            </div>
          </section>

          <section className="admin-panel evaluation-progress">
            <h2>Progress</h2>
            {progress.length ? (
              <div className="progress-log">
                {progress.map((item, index) => (
                  <div key={`${item.at}-${index}`}>
                    <span>{item.stage}</span>
                    <strong>{item.message}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">평가 버튼을 누르면 진행 로그가 여기에 표시됩니다.</p>
            )}
          </section>
        </main>
      ) : null}

      {activeView === "compare" ? (
        <main className="admin-grid">
          <section className="admin-panel">
            <h2>환경 비교</h2>
            <EvaluationComparisonTable rows={summaries} />
          </section>
          <section className="admin-panel">
            <h2>Captured Environments</h2>
            <EnvironmentTable rows={environments} />
          </section>
        </main>
      ) : null}

      {activeView === "artifacts" ? (
        <main className="admin-grid">
          <section className="admin-panel">
            <h2>Artifacts</h2>
            <ArtifactTable rows={artifacts} />
          </section>
        </main>
      ) : null}
    </div>
  );
}

function EvaluationComparisonTable({ rows }) {
  if (!rows.length) {
    return <p className="muted">아직 저장된 `summary_metrics.csv`가 없습니다.</p>;
  }
  return (
    <div className="table-wrap evaluation-table">
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Features</th>
            <th>Model</th>
            <th>Environment</th>
            <th>Attempt</th>
            <th>EER</th>
            <th>Accuracy</th>
            <th>Inference</th>
            <th>Latency</th>
            <th>Memory</th>
            <th>Min data</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.dataset}-${row.environmentId}-${row.attemptId}-${row.Model}-${index}`}>
              <td>{row.dataset}</td>
              <td>{row.feature_mode || featureModeFromDataset(row.dataset)}</td>
              <td>{row.Model}</td>
              <td>{shortId(row.environmentId)}</td>
              <td>{shortId(row.attemptId)}</td>
              <td>{formatPercent(row.EER)}</td>
              <td>{formatPercent(row.Accuracy)}</td>
              <td>{formatMs(row["Inference Time (ms/sample)"])}</td>
              <td>{formatMs(row["UI Blocking Time (ms/test batch)"])}</td>
              <td>{formatBytes(row["Browser Memory After (bytes)"])}</td>
              <td>{row.min_genuine_samples_for_eer_lt_10pct || row["Min Genuine Samples for EER < 10%"] || "--"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EnvironmentTable({ rows }) {
  if (!rows.length) {
    return <p className="muted">저장된 환경 manifest가 없습니다.</p>;
  }
  return (
    <div className="table-wrap evaluation-table">
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Environment</th>
            <th>Label</th>
            <th>Backend</th>
            <th>Cores</th>
            <th>Memory</th>
            <th>WebGL</th>
            <th>WASM SIMD</th>
            <th>Viewport</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.dataset}-${row.environmentId}-${row.attemptId}-${index}`}>
              <td>{row.dataset}</td>
              <td>{shortId(row.environmentId)}</td>
              <td>{row.deviceLabel || "not_available"}</td>
              <td>{row.tfjsBackend || "not_available"}</td>
              <td>{row.hardwareConcurrency || "not_available"}</td>
              <td>{row.deviceMemory || "not_available"}</td>
              <td>{String(row.webglAvailable || "not_available")}</td>
              <td>{String(row.wasmSimdAvailable || "not_available")}</td>
              <td>{row.viewportWidth} x {row.viewportHeight} @ {row.devicePixelRatio}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ArtifactTable({ rows }) {
  if (!rows.length) {
    return <p className="muted">저장된 산출물이 없습니다.</p>;
  }
  return (
    <div className="table-wrap evaluation-table">
      <table>
        <thead>
          <tr>
            <th>Path</th>
            <th>Bytes</th>
            <th>Updated</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.path}>
              <td>{row.path}</td>
              <td>{row.bytes}</td>
              <td>{row.updatedAt}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SegmentedControl({ label, value, options, onChange }) {
  return (
    <div className="segmented" aria-label={label}>
      {options.map(([optionValue, optionLabel]) => (
        <button key={optionValue} type="button" className={value === optionValue ? "active" : ""} onClick={() => onChange(optionValue)}>
          {optionLabel}
        </button>
      ))}
    </div>
  );
}

function ModelResult({ result, fallbackName }) {
  const scoreText = result?.score === null || result?.score === undefined ? "--" : Number(result.score).toFixed(3);
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
            <th>Input</th>
            <th>Status</th>
            <th>EER</th>
            <th>FAR</th>
            <th>FRR</th>
            <th>Samples</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.modelName}-${row.deviceClass}-${row.inputMode}`}>
              <td>{row.modelName}</td>
              <td>{row.deviceClass}</td>
              <td>{row.inputMode}</td>
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

function SmallCountTable({ title, rows = [], labelKey }) {
  return (
    <div className="mini-table">
      {title ? <h3>{title}</h3> : null}
      <table>
        <tbody>
          {rows.map((row) => (
            <tr key={`${labelKey}-${row[labelKey]}`}>
              <td>{row[labelKey] ?? "unknown"}</td>
              <td>{row.attempts}</td>
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
  if (name.includes("Manhattan") || name.includes("Baseline")) return "Baseline";
  if (name.includes("LightGBM")) return "LightGBM";
  if (name.includes("CNN")) return "1D-CNN";
  return name;
}

function featureModeFromDataset(dataset = "") {
  return String(dataset).endsWith("_enriched_features") ? "enriched" : "current";
}

function bestVerdict(results) {
  const baseline = results.find((result) => result.modelName === "Instant Baseline");
  if (!baseline) return { label: "Ready", detail: "waiting for submitted attempt", className: "neutral" };
  if (baseline.decision?.startsWith("accept") || baseline.decision?.includes("profile")) {
    return { label: "Accepted", detail: baseline.decision, className: "accept" };
  }
  if (baseline.decision === "reject") {
    return { label: "Review", detail: "baseline mismatch", className: "reject" };
  }
  return { label: "Learning", detail: baseline.decision, className: "neutral" };
}

function findPromptSet(promptSets, selectedId, mode) {
  return promptSets.find((set) => set.id === selectedId && set.inputMode === mode)
    || promptSets.find((set) => set.inputMode === mode && set.language === "ko")
    || promptSets.find((set) => set.inputMode === mode)
    || null;
}

function selectPromptForFlow(inputMode, trialNo, fixedPromptSet, freePromptSet) {
  const promptSet = inputMode === "free" ? freePromptSet : fixedPromptSet;
  const promptIndex = getPromptIndex(inputMode, trialNo, fixedPromptSet?.prompts?.length || 0);
  return selectPromptByIndex(promptSet, promptIndex);
}

function getNextPromptStep(inputMode, trialNo, fixedPromptSet, freePromptSet) {
  const fixedCount = fixedPromptSet?.prompts?.length || 0;
  const freeCount = freePromptSet?.prompts?.length || 0;
  const currentIndex = getPromptIndex(inputMode, trialNo, fixedCount);

  if (inputMode === "fixed") {
    if (currentIndex + 1 < fixedCount) {
      const nextTrialNo = Math.max(1, Number(trialNo) || 1) + 1;
      return {
        inputMode: "fixed",
        trialNo: nextTrialNo,
        prompt: selectPromptByIndex(fixedPromptSet, currentIndex + 1)
      };
    }
    if (freeCount > 0) {
      const nextTrialNo = Math.max(Math.max(1, Number(trialNo) || 1) + 1, fixedCount + 1);
      return {
        inputMode: "free",
        trialNo: nextTrialNo,
        prompt: selectPromptByIndex(freePromptSet, 0)
      };
    }
    return null;
  }

  if (currentIndex + 1 < freeCount) {
    const nextTrialNo = Math.max(1, Number(trialNo) || 1) + 1;
    return {
      inputMode: "free",
      trialNo: nextTrialNo,
      prompt: selectPromptByIndex(freePromptSet, currentIndex + 1)
    };
  }
  return null;
}

function getPromptIndex(inputMode, trialNo, fixedCount) {
  const safeTrialNo = Math.max(1, Number(trialNo) || 1);
  if (inputMode === "free") {
    return Math.max(0, safeTrialNo > fixedCount ? safeTrialNo - fixedCount - 1 : safeTrialNo - 1);
  }
  return safeTrialNo - 1;
}

function selectPromptByIndex(promptSet, index) {
  const prompts = promptSet?.prompts || [];
  return prompts[index] || null;
}

function normalizeRole(role) {
  return ["genuine", "imposter"].includes(role) ? role : "genuine";
}

function numberFromQuery(query, name, fallback) {
  const value = Number(query.get(name));
  return Number.isFinite(value) && value > 0 ? Math.trunc(value) : fallback;
}

function formatTypingRate(length, startedAt, now) {
  if (!startedAt || length <= 0) {
    return { rate: 0, length, elapsed: "0초" };
  }
  const elapsedSeconds = Math.max(1, (now - startedAt) / 1000);
  const rate = Math.round(length / (elapsedSeconds / 60));
  return {
    rate,
    length,
    elapsed: `${Math.floor(elapsedSeconds)}초`
  };
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatMs(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-- ms";
  return `${Number(value).toFixed(2)} ms`;
}

function formatBytes(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "not_available";
  if (number >= 1024 * 1024) return `${(number / (1024 * 1024)).toFixed(1)} MB`;
  if (number >= 1024) return `${(number / 1024).toFixed(1)} KB`;
  return `${number.toFixed(0)} B`;
}

function shortId(id) {
  return id ? id.slice(0, 8) : "pending";
}

function randomId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
}

function defaultDeviceLabel() {
  const platform = navigator.platform || "browser";
  const language = navigator.language || "unknown";
  return `${platform}-${language}`;
}

function parseCsvText(text) {
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
      if (char === "\r" && clean[i + 1] === "\n") i += 1;
      row.push(cell);
      if (row.some((value) => value !== "")) rows.push(row);
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
  if (rows.length < 2) return [];
  const headers = rows[0];
  return rows.slice(1).map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}
