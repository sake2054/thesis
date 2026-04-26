const TEMPLATE_KEY = "keystrokeDemo.enrollmentTemplates.v1";

export function detectDeviceClass() {
  const ua = navigator.userAgent || "";
  const hasTouch = navigator.maxTouchPoints > 0 || "ontouchstart" in window;
  const width = window.innerWidth || 0;
  if (/iPad|Tablet|PlayBook|Silk/i.test(ua) || (hasTouch && width >= 700 && width <= 1200)) {
    return "tablet";
  }
  if (/Android|webOS|iPhone|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua) || (hasTouch && width < 700)) {
    return "mobile";
  }
  if (ua) {
    return "desktop";
  }
  return "unknown";
}

export function collectClientMetadata(deviceClass) {
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || "";
  return {
    deviceClass,
    userAgent: navigator.userAgent,
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight,
      devicePixelRatio: window.devicePixelRatio
    },
    screen: {
      width: window.screen?.width,
      height: window.screen?.height,
      availWidth: window.screen?.availWidth,
      availHeight: window.screen?.availHeight,
      colorDepth: window.screen?.colorDepth
    },
    navigator: {
      platform: navigator.platform,
      language: navigator.language,
      languages: navigator.languages,
      hardwareConcurrency: navigator.hardwareConcurrency,
      maxTouchPoints: navigator.maxTouchPoints,
      cookieEnabled: navigator.cookieEnabled,
      vendor: navigator.vendor
    },
    timezone,
    language: navigator.language,
    touchSupport: navigator.maxTouchPoints > 0 || "ontouchstart" in window
  };
}

export function makeEventPayload(event, type, value, attemptStartedAt) {
  const relativeTime = performance.now() - attemptStartedAt;
  const payload = {
    type,
    eventTime: Date.now(),
    relativeTime,
    valueLength: value.length,
    payload: {
      timeStamp: event.timeStamp,
      altKey: Boolean(event.altKey),
      ctrlKey: Boolean(event.ctrlKey),
      metaKey: Boolean(event.metaKey),
      shiftKey: Boolean(event.shiftKey)
    }
  };

  if ("key" in event) {
    payload.key = event.key;
    payload.code = event.code;
    payload.repeat = Boolean(event.repeat);
    payload.isComposing = Boolean(event.isComposing);
  }
  if ("inputType" in event) {
    payload.inputType = event.inputType;
    payload.data = event.data;
    payload.isComposing = Boolean(event.isComposing);
  }
  if (type.startsWith("composition")) {
    payload.data = event.data;
  }
  return payload;
}

export function extractFeatureBundle(events, rawText, inputMode, deviceClass) {
  const sorted = [...events].sort((a, b) => (a.relativeTime || 0) - (b.relativeTime || 0));
  const keydowns = [];
  const active = new Map();
  const compositionEvents = sorted.filter((event) => event.type?.startsWith("composition"));
  let backspaces = 0;
  let repeated = 0;

  for (const event of sorted) {
    const time = Number(event.relativeTime);
    if (!Number.isFinite(time)) {
      continue;
    }
    const identity = event.code || event.key || "unknown";
    if (event.type === "keydown") {
      if (event.key === "Backspace") {
        backspaces += 1;
      }
      if (event.repeat) {
        repeated += 1;
        continue;
      }
      const entry = {
        key: event.key,
        code: event.code,
        identity,
        down: time,
        up: null
      };
      keydowns.push(entry);
      if (!active.has(identity)) {
        active.set(identity, []);
      }
      active.get(identity).push(entry);
    } else if (event.type === "keyup") {
      const stack = active.get(identity) || [];
      const entry = stack.find((candidate) => candidate.up === null);
      if (entry) {
        entry.up = time;
      }
    }
  }

  const paired = keydowns.filter((entry) => Number.isFinite(entry.down) && Number.isFinite(entry.up));
  const holds = paired.map((entry) => entry.up - entry.down).filter(isSaneTiming);
  const dd = [];
  const ud = [];
  const uu = [];
  const du = [];
  for (let index = 0; index < paired.length - 1; index += 1) {
    const current = paired[index];
    const next = paired[index + 1];
    dd.push(next.down - current.down);
    ud.push(next.down - current.up);
    uu.push(next.up - current.up);
    du.push(next.up - current.down);
  }

  const durationMs = sorted.length > 1
    ? Math.max(1, sorted[sorted.length - 1].relativeTime - sorted[0].relativeTime)
    : 1;
  const keyupCoverage = keydowns.length ? paired.length / keydowns.length : 0;
  const compositionRatio = sorted.length ? compositionEvents.length / sorted.length : 0;
  const timingCoverage = deviceClass === "desktop"
    ? keyupCoverage
    : Math.max(0.15, keyupCoverage * 0.6 + compositionRatio * 0.4);
  const featureQuality = classifyFeatureQuality(deviceClass, keyupCoverage, compositionRatio);

  const values = [
    mean(holds),
    std(holds),
    mean(dd.filter(isSaneTiming)),
    std(dd.filter(isSaneTiming)),
    mean(ud.filter(isSaneTiming)),
    std(ud.filter(isSaneTiming)),
    mean(uu.filter(isSaneTiming)),
    mean(du.filter(isSaneTiming)),
    rawText.length / (durationMs / 1000),
    backspaces,
    repeated,
    compositionRatio,
    keyupCoverage,
    timingCoverage,
    rawText.length
  ];

  const names = [
    "hold_mean_ms",
    "hold_std_ms",
    "down_down_mean_ms",
    "down_down_std_ms",
    "up_down_mean_ms",
    "up_down_std_ms",
    "up_up_mean_ms",
    "down_up_mean_ms",
    "chars_per_second",
    "backspace_count",
    "repeat_count",
    "composition_ratio",
    "keyup_coverage",
    "timing_coverage",
    "text_length"
  ];

  return {
    vector: values.map((value) => (Number.isFinite(value) ? value : 0)),
    features: names.map((name, index) => ({
      modelScope: "browser_baseline",
      name,
      index,
      value: Number.isFinite(values[index]) ? values[index] : 0
    })),
    quality: {
      inputMode,
      deviceClass,
      featureQuality,
      eventCount: sorted.length,
      keydownCount: keydowns.length,
      pairedKeyCount: paired.length,
      keyupCoverage,
      compositionRatio,
      timingCoverage,
      textLength: rawText.length
    }
  };
}

export function scoreInstantBaseline(inputMode, roleLabel, vector, attemptId = cryptoRandomId()) {
  const templates = loadTemplates();
  const modeTemplates = templates[inputMode] || [];
  const previousTemplates = modeTemplates.filter((sample) => sample.attemptId !== attemptId);
  const threshold = 0.5;

  if (roleLabel === "genuine") {
    const comparison = previousTemplates.length
      ? scoreAgainstTemplates(previousTemplates, vector, threshold)
      : {
          score: 1,
          threshold,
          decision: "profile started",
          templateSamples: 0
        };
    const updated = {
      ...templates,
      [inputMode]: [
        ...previousTemplates,
        { attemptId, vector, enrolledAt: new Date().toISOString() }
      ].slice(-25)
    };
    saveTemplates(updated);
    return {
      modelName: "Instant Baseline",
      score: comparison.score,
      threshold,
      decision: comparison.decision === "profile started"
        ? "profile started"
        : "accept + profile updated",
      templateSamples: updated[inputMode].length
    };
  }

  if (!previousTemplates.length) {
    return {
      modelName: "Instant Baseline",
      score: null,
      threshold,
      decision: "needs genuine profile",
      templateSamples: 0
    };
  }

  return {
    modelName: "Instant Baseline",
    ...scoreAgainstTemplates(previousTemplates, vector, threshold)
  };
}

function scoreAgainstTemplates(samples, vector, threshold) {
  const matrix = samples.map((sample) => sample.vector);
  const center = vector.map((_, index) => mean(matrix.map((sample) => sample[index] || 0)));
  const scale = vector.map((_, index) => Math.max(std(matrix.map((sample) => sample[index] || 0)), 1));
  const distance = mean(vector.map((value, index) => Math.abs((value - center[index]) / scale[index])));
  const score = 1 / (1 + distance);
  return {
    score,
    threshold,
    decision: score >= threshold ? "accept" : "reject",
    templateSamples: samples.length
  };
}

export function getTemplateCounts() {
  const templates = loadTemplates();
  return {
    fixed: templates.fixed?.length || 0,
    free: templates.free?.length || 0
  };
}

function loadTemplates() {
  try {
    return JSON.parse(localStorage.getItem(TEMPLATE_KEY)) || { fixed: [], free: [] };
  } catch {
    return { fixed: [], free: [] };
  }
}

function saveTemplates(templates) {
  localStorage.setItem(TEMPLATE_KEY, JSON.stringify(templates));
}

function cryptoRandomId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
}

function classifyFeatureQuality(deviceClass, keyupCoverage, compositionRatio) {
  if (deviceClass === "desktop" && keyupCoverage >= 0.75) {
    return "high";
  }
  if (keyupCoverage >= 0.4 || compositionRatio > 0) {
    return "medium";
  }
  return "low";
}

function isSaneTiming(value) {
  return Number.isFinite(value) && value >= -1000 && value <= 10000;
}

function mean(values) {
  const clean = values.filter(Number.isFinite);
  if (!clean.length) {
    return 0;
  }
  return clean.reduce((sum, value) => sum + value, 0) / clean.length;
}

function std(values) {
  const clean = values.filter(Number.isFinite);
  if (clean.length < 2) {
    return 0;
  }
  const avg = mean(clean);
  return Math.sqrt(mean(clean.map((value) => (value - avg) ** 2)));
}
