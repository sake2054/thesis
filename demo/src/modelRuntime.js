import * as tf from "@tensorflow/tfjs";

let runtimePromise = null;

export function loadModelRuntime() {
  if (!runtimePromise) {
    runtimePromise = fetch("/models/manifest.json")
      .then((response) => (response.ok ? response.json() : null))
      .then(async (manifest) => {
        if (!manifest) {
          return { manifest: null, cache: new Map(), status: "manifest unavailable" };
        }
        return { manifest, cache: new Map(), status: manifest.artifactAvailable ? "ready" : "optional artifacts unavailable" };
      })
      .catch(() => ({ manifest: null, cache: new Map(), status: "manifest unavailable" }));
  }
  return runtimePromise;
}

export async function predictOptionalModels(vector, options = {}) {
  const runtime = await loadModelRuntime();
  return predictWithRuntime(runtime, vector, options);
}

export async function predictWithRuntime(runtime, vector, options = {}) {
  const start = performance.now();
  const manifest = runtime?.manifest;
  const deviceClass = normalizeDeviceClass(options.deviceClass);
  const inputMode = options.inputMode === "free" ? "free" : "fixed";
  const calibrations = Array.isArray(options.calibrations) ? options.calibrations : [];
  const results = [];

  for (const modelKey of ["lightgbm", "cnn1d"]) {
    const selected = selectModelConfig(manifest, deviceClass, inputMode, modelKey);
    const displayName = modelKey === "lightgbm" ? "LightGBM" : "1D-CNN";
    if (!selected?.config) {
      results.push({
        modelName: displayName,
        modelScope: `${displayName}::unavailable::${inputMode}::v${manifest?.version || "none"}`,
        deviceClass,
        score: null,
        threshold: null,
        decision: "artifact unavailable",
        inferenceTimeMs: null,
        uiBlockingTimeMs: performance.now() - start,
        payload: { fallback: true, reason: "artifact unavailable" }
      });
      continue;
    }

    try {
      const scoreStart = performance.now();
      const score = modelKey === "lightgbm"
        ? await predictLightGbm(runtime, selected, vector)
        : await predictCnn(runtime, selected, vector);
      const modelScope = `${displayName}::${selected.scopeDevice}::${selected.scopeInputMode}::v${manifest?.version || "unknown"}`;
      const thresholdInfo = resolveThreshold({
        defaultThreshold: Number(selected.config.threshold ?? 0.5),
        calibrations,
        modelName: displayName,
        modelScope,
        inputMode,
        deviceClass,
        participantId: options.participantId,
        participantCode: options.participantCode
      });
      results.push({
        modelName: displayName,
        modelScope,
        deviceClass,
        score,
        threshold: thresholdInfo.threshold,
        decision: score >= thresholdInfo.threshold ? "accept" : "reject",
        inferenceTimeMs: performance.now() - scoreStart,
        uiBlockingTimeMs: performance.now() - start,
        calibration: thresholdInfo.calibration || null,
        payload: {
          fallback: selected.fallback,
          requestedDeviceClass: deviceClass,
          requestedInputMode: inputMode,
          scopeDevice: selected.scopeDevice,
          scopeInputMode: selected.scopeInputMode,
          featureNames: selected.config.featureNames || null
        }
      });
    } catch (error) {
      results.push({
        modelName: displayName,
        modelScope: `${displayName}::error::${inputMode}::v${manifest?.version || "none"}`,
        deviceClass,
        score: null,
        threshold: Number(selected.config.threshold ?? 0.5),
        decision: "artifact error",
        inferenceTimeMs: null,
        uiBlockingTimeMs: performance.now() - start,
        payload: { error: error.message, fallback: selected.fallback }
      });
    }
  }
  return results;
}

export function selectModelConfig(manifest, deviceClass, inputMode, modelKey) {
  const models = manifest?.models || {};
  const candidates = [
    [deviceClass, inputMode],
    ["default", inputMode],
    ["desktop", inputMode]
  ];
  for (const [candidateDevice, candidateMode] of candidates) {
    const config = models?.[candidateDevice]?.[candidateMode]?.[modelKey];
    if (config) {
      return {
        config,
        scopeDevice: candidateDevice,
        scopeInputMode: candidateMode,
        fallback: candidateDevice !== deviceClass || candidateMode !== inputMode
      };
    }
  }
  for (const [scopeDevice, byMode] of Object.entries(models)) {
    for (const [scopeInputMode, byModel] of Object.entries(byMode || {})) {
      if (byModel?.[modelKey]) {
        return {
          config: byModel[modelKey],
          scopeDevice,
          scopeInputMode,
          fallback: true
        };
      }
    }
  }
  return null;
}

async function predictLightGbm(runtime, selected, vector) {
  const model = await getCached(runtime, selected.config.path, async () => {
    const response = await fetch(selected.config.path);
    if (!response.ok) {
      throw new Error(`LightGBM artifact failed: ${response.status}`);
    }
    return response.json();
  });
  const prepared = prepareVector(vector, selected.config);
  let rawScore = Number(model.average_output || 0);
  for (const tree of model.tree_info || []) {
    rawScore += evaluateTree(tree.tree_structure, prepared);
  }
  if (model.objective === "binary sigmoid" || selected.config.output === "raw") {
    return sigmoid(rawScore);
  }
  return sigmoid(rawScore);
}

async function predictCnn(runtime, selected, vector) {
  const model = await getCached(runtime, selected.config.modelJson, async () => tf.loadLayersModel(selected.config.modelJson));
  const prepared = prepareVector(vector, selected.config);
  const shape = selected.config.inputShape || [prepared.length, 1];
  const expected = shape.reduce((product, value) => product * Number(value), 1);
  if (expected !== prepared.length) {
    throw new Error(`CNN inputShape ${shape.join("x")} expects ${expected} values, got ${prepared.length}`);
  }
  const tensor = tf.tensor(prepared, [1, ...shape]);
  let prediction;
  try {
    prediction = model.predict(tensor);
    const output = Array.isArray(prediction) ? prediction[0] : prediction;
    const values = await output.data();
    return Number(values[0]);
  } finally {
    tensor.dispose();
    if (Array.isArray(prediction)) {
      prediction.forEach((item) => item.dispose?.());
    } else {
      prediction?.dispose?.();
    }
  }
}

async function getCached(runtime, key, loader) {
  if (!key) {
    throw new Error("Missing model artifact path.");
  }
  if (!runtime.cache) {
    runtime.cache = new Map();
  }
  if (!runtime.cache.has(key)) {
    runtime.cache.set(key, loader());
  }
  return runtime.cache.get(key);
}

function resolveThreshold({
  defaultThreshold,
  calibrations,
  modelName,
  modelScope,
  inputMode,
  deviceClass,
  participantId,
  participantCode
}) {
  const normalized = calibrations.filter((row) => row && row.modelName === modelName);
  const tiers = [
    (row) => matchesParticipant(row, participantId, participantCode) && row.modelScope === modelScope,
    (row) => matchesParticipant(row, participantId, participantCode) && matchesContext(row, inputMode, deviceClass),
    (row) => !row.participantId && !row.participantCode && matchesContext(row, inputMode, deviceClass)
  ];
  for (const predicate of tiers) {
    const match = normalized.find(predicate);
    if (match && Number.isFinite(Number(match.threshold))) {
      return { threshold: Number(match.threshold), calibration: match };
    }
  }
  return { threshold: Number.isFinite(defaultThreshold) ? defaultThreshold : 0.5, calibration: null };
}

function matchesParticipant(row, participantId, participantCode) {
  return (participantId && row.participantId === participantId)
    || (participantCode && row.participantCode === participantCode);
}

function matchesContext(row, inputMode, deviceClass) {
  const inputMatches = !row.inputMode || row.inputMode === inputMode;
  const deviceMatches = !row.deviceClass || row.deviceClass === deviceClass || row.deviceClass === "unknown";
  return inputMatches && deviceMatches;
}

function evaluateTree(node, vector) {
  if (!node) {
    return 0;
  }
  if (Object.hasOwn(node, "leaf_value")) {
    return Number(node.leaf_value);
  }
  const value = vector[Number(node.split_feature)];
  const missing = value === undefined || value === null || Number.isNaN(Number(value));
  const defaultLeft = node.default_left !== false;
  if (missing) {
    return evaluateTree(defaultLeft ? node.left_child : node.right_child, vector);
  }
  const threshold = Number(node.threshold);
  let goesLeft = Number(value) <= threshold;
  if (node.decision_type === "==") {
    goesLeft = String(value) === String(node.threshold);
  }
  return evaluateTree(goesLeft ? node.left_child : node.right_child, vector);
}

function prepareVector(vector, config) {
  const length = Number(config.featureCount || vector.length);
  const prepared = Array.from({ length }, (_, index) => Number(vector[index] || 0));
  if (Array.isArray(config.mean) && Array.isArray(config.scale) && config.mean.length && config.scale.length) {
    return prepared.map((value, index) => {
      const scale = Number(config.scale[index] || 1);
      return (value - Number(config.mean[index] || 0)) / (scale || 1);
    });
  }
  return prepared;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function normalizeDeviceClass(deviceClass) {
  return ["desktop", "mobile", "tablet", "unknown"].includes(deviceClass)
    ? deviceClass
    : "unknown";
}
