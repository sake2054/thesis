import * as tf from "@tensorflow/tfjs";

let runtimePromise = null;

export function loadModelRuntime() {
  if (!runtimePromise) {
    runtimePromise = fetch("/models/manifest.json")
      .then((response) => (response.ok ? response.json() : null))
      .then(async (manifest) => {
        if (!manifest || manifest.artifactAvailable === false) {
          return { manifest, lightgbm: null, cnn1d: null };
        }
        const lightgbm = manifest.models?.lightgbm
          ? await loadLightGbm(manifest.models.lightgbm)
          : null;
        const cnn1d = manifest.models?.cnn1d
          ? await loadCnn(manifest.models.cnn1d)
          : null;
        return { manifest, lightgbm, cnn1d };
      })
      .catch(() => ({ manifest: null, lightgbm: null, cnn1d: null }));
  }
  return runtimePromise;
}

export async function predictOptionalModels(vector, mode) {
  const start = performance.now();
  const runtime = await loadModelRuntime();
  const results = [];

  if (runtime.lightgbm) {
    const scoreStart = performance.now();
    const score = runtime.lightgbm.predict(vector, mode);
    results.push({
      modelName: "LightGBM",
      score,
      threshold: runtime.lightgbm.threshold,
      decision: score >= runtime.lightgbm.threshold ? "accept" : "reject",
      inferenceTimeMs: performance.now() - scoreStart,
      uiBlockingTimeMs: performance.now() - start
    });
  } else {
    results.push({
      modelName: "LightGBM",
      score: null,
      threshold: null,
      decision: "artifact unavailable",
      inferenceTimeMs: null,
      uiBlockingTimeMs: null
    });
  }

  if (runtime.cnn1d) {
    const scoreStart = performance.now();
    const score = await runtime.cnn1d.predict(vector, mode);
    results.push({
      modelName: "1D-CNN",
      score,
      threshold: runtime.cnn1d.threshold,
      decision: score >= runtime.cnn1d.threshold ? "accept" : "reject",
      inferenceTimeMs: performance.now() - scoreStart,
      uiBlockingTimeMs: performance.now() - start
    });
  } else {
    results.push({
      modelName: "1D-CNN",
      score: null,
      threshold: null,
      decision: "artifact unavailable",
      inferenceTimeMs: null,
      uiBlockingTimeMs: null
    });
  }

  return results;
}

async function loadLightGbm(config) {
  const response = await fetch(config.path);
  if (!response.ok) {
    return null;
  }
  const model = await response.json();
  return {
    threshold: Number(config.threshold ?? 0.5),
    predict(vector) {
      const prepared = prepareVector(vector, config);
      let rawScore = Number(model.average_output || 0);
      for (const tree of model.tree_info || []) {
        rawScore += evaluateTree(tree.tree_structure, prepared);
      }
      return sigmoid(rawScore);
    }
  };
}

async function loadCnn(config) {
  const model = await tf.loadLayersModel(config.modelJson);
  return {
    threshold: Number(config.threshold ?? 0.5),
    async predict(vector) {
      const prepared = prepareVector(vector, config);
      const shape = config.inputShape || [prepared.length, 1];
      const tensor = tf.tensor(prepared, [1, ...shape]);
      const prediction = model.predict(tensor);
      const values = await prediction.data();
      tensor.dispose();
      prediction.dispose();
      return Number(values[0]);
    }
  };
}

function evaluateTree(node, vector) {
  if (!node) {
    return 0;
  }
  if (Object.hasOwn(node, "leaf_value")) {
    return Number(node.leaf_value);
  }
  const feature = vector[Number(node.split_feature)] ?? 0;
  const threshold = Number(node.threshold);
  const goesLeft = node.decision_type === "<="
    ? feature <= threshold
    : feature <= threshold;
  return evaluateTree(goesLeft ? node.left_child : node.right_child, vector);
}

function prepareVector(vector, config) {
  const length = Number(config.featureCount || vector.length);
  const prepared = Array.from({ length }, (_, index) => Number(vector[index] || 0));
  if (Array.isArray(config.mean) && Array.isArray(config.scale)) {
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
