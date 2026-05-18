import * as tf from "@tensorflow/tfjs";

const DSL_FILE = "DSL-StrongPasswordData.csv";
const MMC_PAIRS = [
  { feature: "mmc1.npy", label: "mmc5.npy", partition: "reference" },
  { feature: "mmc2.npy", label: "mmc6.npy", partition: "reference" },
  { feature: "mmc3.npy", label: "mmc7.npy", partition: "test" },
  { feature: "mmc4.npy", label: "mmc8.npy", partition: "test" }
];
const COLLECTED_FILES = [
  "participants.csv",
  "sessions.csv",
  "attempts.csv",
  "events.csv",
  "features.csv",
  "results.csv",
  "monitoring_windows.csv",
  "model_calibrations.csv"
];
const MMC_TIMING_CHANNELS = [14, 15, 16, 17, 18, 19];
const EER_TARGET = 0.10;
const LIGHTGBM_RUNTIME = "browser_lightgbm_histogram_leafwise";
const FEATURE_MODE_CURRENT = "current";
const FEATURE_MODE_ENRICHED = "enriched";
const ADAPTIVE_FEATURE_CAP = 384;
const COLLECTED_SEQUENCE_CAP = 180;
const HASH_BUCKETS = 64;

self.addEventListener("message", (event) => {
  const message = event.data || {};
  if (message.type === "run") {
    runEvaluation(message.payload || {}).catch((error) => {
      self.postMessage({ type: "error", error: error.message || String(error) });
    });
  }
});

async function runEvaluation(options) {
  const startedAt = performance.now();
  await tf.ready();
  options.featureMode = normalizeFeatureMode(options.featureMode);
  options.dataset = baseDatasetKey(options.dataset);
  const environment = await buildEnvironment(options.baseEnvironment || {}, options);
  progress("environment", `Environment ${environment.environmentId} ready (${environment.tfjsBackend})`);

  if (options.dataset === "collected_data_analysis") {
    const result = await runCollectedAnalysisAndEvaluation(options, environment, startedAt);
    self.postMessage({ type: "done", result });
    return;
  }

  const data = options.dataset === "mmc"
    ? await loadMmcDataset(options)
    : await loadDslDataset(options);
  const result = await runModelEvaluation(options, environment, data, startedAt);
  self.postMessage({ type: "done", result });
}

async function runModelEvaluation(options, environment, data, startedAt, extraArtifacts = []) {
  progress("dataset", `${data.datasetLabel} loaded (${data.samples.length} samples)`);
  const modelConfig = buildModelConfig(options, data);
  const resultDataset = outputDatasetKey(options.dataset, options.featureMode);
  const environmentWithHashes = {
    ...environment,
    datasetFileHashes: data.datasetFileHashes,
    modelConfig,
    modelConfigHash: await stableHash(JSON.stringify(modelConfig))
  };

  const subjects = unique(data.samples.map((sample) => sample.subject)).sort(subjectSort);
  const summaryRows = [];
  const userwiseRows = [];
  const dataEfficiencyRows = [];
  const timingRows = [];
  const rocSeries = {};

  const modelSpecs = [
    { key: "manhattan", name: "Scaled Manhattan Distance" },
    { key: "lightgbm", name: "LightGBM Classifier", runtime: LIGHTGBM_RUNTIME },
    { key: "cnn1d", name: "1D-CNN TFJS" }
  ];

  for (const spec of modelSpecs) {
    const modelRows = [];
    const allScores = [];
    const allLabels = [];
    for (let subjectIndex = 0; subjectIndex < subjects.length; subjectIndex += 1) {
      const subject = subjects[subjectIndex];
      progress("model", `${data.datasetLabel}: ${spec.name} subject ${subjectIndex + 1}/${subjects.length}`);
      const split = makeAuthSplit(data, subject);
      if (!split) {
        continue;
      }
      const trained = await trainAndScore(spec, split, data.sequenceShape);
      const metrics = computeMetrics(split.testY, trained.scores);
      const row = {
        dataset: resultDataset,
        source_dataset: options.dataset,
        feature_mode: options.featureMode,
        feature_set: data.featureSet || "current_stored_features",
        model: spec.name,
        model_runtime: spec.runtime || "browser_worker",
        subject,
        train_genuine_count: split.trainY.filter((value) => value === 1).length,
        test_genuine_count: split.testY.filter((value) => value === 1).length,
        train_impostor_count: split.trainY.filter((value) => value === 0).length,
        test_impostor_count: split.testY.filter((value) => value === 0).length,
        FAR: metrics.far,
        FRR: metrics.frr,
        EER: metrics.eer,
        Accuracy: metrics.accuracy,
        AUC: metrics.auc,
        threshold: metrics.threshold,
        train_time_ms: trained.trainTimeMs,
        inference_time_ms_per_sample: trained.inferenceTimeMsPerSample,
        ui_blocking_time_ms: trained.inferenceBatchMs,
        memory_before_bytes: trained.memoryBefore,
        memory_after_bytes: trained.memoryAfter,
        tf_num_tensors: trained.tfMemory?.numTensors ?? "not_available",
        tf_num_bytes: trained.tfMemory?.numBytes ?? "not_available",
        adaptive_selected_feature_count: split.adaptiveSelectedFeatureCount ?? "not_enabled"
      };
      userwiseRows.push(row);
      modelRows.push(row);
      timingRows.push({
        dataset: resultDataset,
        source_dataset: options.dataset,
        feature_mode: options.featureMode,
        model: spec.name,
        subject,
        phase: "train_and_infer",
        elapsed_ms: trained.trainTimeMs + trained.inferenceBatchMs,
        train_time_ms: trained.trainTimeMs,
        inference_batch_ms: trained.inferenceBatchMs,
        inference_time_ms_per_sample: trained.inferenceTimeMsPerSample,
        memory_before_bytes: trained.memoryBefore,
        memory_after_bytes: trained.memoryAfter,
        adaptive_selected_feature_count: split.adaptiveSelectedFeatureCount ?? "not_enabled"
      });
      for (let i = 0; i < trained.scores.length; i += 1) {
        allScores.push(trained.scores[i]);
        allLabels.push(split.testY[i]);
      }
    }

    progress("aggregate", `${data.datasetLabel}: ${spec.name} aggregating metrics`);
    const aggregate = averageMetricRows(modelRows, resultDataset, spec);
    aggregate.source_dataset = options.dataset;
    aggregate.feature_mode = options.featureMode;
    aggregate.feature_set = data.featureSet || "current_stored_features";
    summaryRows.push(aggregate);
    if (allScores.length) {
      rocSeries[spec.name] = makeRocSeries(allLabels, allScores);
    }

    progress("data_efficiency", `${data.datasetLabel}: ${spec.name} evaluating minimum training data`);
    const efficiencyRows = await evaluateDataEfficiency(spec, data, subjects, timingRows, resultDataset, options);
    dataEfficiencyRows.push(...efficiencyRows);
    aggregate.min_genuine_samples_for_eer_lt_10pct = minSamplesForTarget(efficiencyRows, spec.name);
    progress("model_done", `${data.datasetLabel}: ${spec.name} complete`);
  }

  const finishedAt = performance.now();
  const runRows = [
    {
      dataset: options.dataset,
      source_dataset: options.dataset,
      feature_mode: options.featureMode,
      environmentId: environmentWithHashes.environmentId,
      perceived_latency_ms: finishedAt - startedAt,
      worker_elapsed_ms: finishedAt - startedAt,
      completed_at: new Date().toISOString()
    }
  ];

  const artifacts = [
    textArtifact("environment_manifest.json", JSON.stringify(environmentWithHashes, null, 2)),
    ...extraArtifacts,
    textArtifact("summary_metrics.csv", toCsv(summaryRows)),
    textArtifact("userwise_metrics.csv", toCsv(userwiseRows)),
    textArtifact("data_efficiency_eer.csv", toCsv(dataEfficiencyRows)),
    textArtifact("browser_timing_log.csv", toCsv([...timingRows, ...runRows]))
  ];
  if (options.dataset === "mmc") {
    artifacts.push(textArtifact("pair_manifest.csv", toCsv(data.pairManifest)));
  }

  return {
    dataset: resultDataset,
    sourceDataset: options.dataset,
    featureMode: options.featureMode,
    environment: environmentWithHashes,
    artifacts,
    chartData: {
      summaryRows,
      dataEfficiencyRows,
      rocSeries
    },
    summaryRows,
    elapsedMs: finishedAt - startedAt
  };
}

function buildModelConfig(options, data) {
  const modelConfig = {
    featureMode: normalizeFeatureMode(options.featureMode),
    featureSet: data.featureSet || "current_stored_features",
    splitRule: options.dataset === "mmc"
      ? "mmc1->mmc5 and mmc2->mmc6 reference; mmc3->mmc7 and mmc4->mmc8 future test"
      : options.dataset === "collected_data_analysis"
        ? "submitted usable collected attempts sorted per participant; earlier attempts reference, later attempts future test"
        : "sessionIndex <= 4 reference; sessionIndex > 4 future test",
    models: [
      { key: "manhattan", name: "Scaled Manhattan Distance" },
      { key: "lightgbm", name: "LightGBM Classifier", runtime: LIGHTGBM_RUNTIME },
      { key: "cnn1d", name: "1D-CNN TFJS", epochs: 4, batchSize: 128 }
    ],
    lightgbm: {
      objective: "binary",
      boosting: "gbdt",
      treeLearner: "serial",
      splitSearch: "histogram",
      growth: "leaf-wise",
      learningRate: 0.05,
      numLeaves: 15,
      featureFraction: 0.9,
      baggingFraction: 0.9,
      baggingFreq: 1
    },
    sequenceShape: data.sequenceShape,
    timingChannels: options.dataset === "mmc" ? MMC_TIMING_CHANNELS : null,
    featureNames: data.featureNames || [],
    featureCount: data.featureNames?.length || data.samples?.[0]?.vector?.length || 0,
    adaptiveFeatureSelection: Boolean(data.adaptiveFeatureSelection),
    adaptiveFeatureCap: data.adaptiveFeatureSelection ? ADAPTIVE_FEATURE_CAP : "not_enabled",
    dataEfficiencyTargetEer: EER_TARGET
  };
  return modelConfig;
}

async function loadDslDataset(options) {
  const text = await fetchText(options, "dsl", DSL_FILE);
  const rows = parseCsv(text);
  const baseFeatureNames = Object.keys(rows[0] || {}).filter((key) => /^(H|DD|UD)\./.test(key));
  const featureNames = options.featureMode === FEATURE_MODE_ENRICHED
    ? buildDslEnrichedFeatureNames(baseFeatureNames)
    : baseFeatureNames;
  const samples = rows.map((row, index) => ({
    subject: String(row.subject),
    order: index,
    session: Number(row.sessionIndex),
    partition: Number(row.sessionIndex) <= 4 ? "reference" : "test",
    vector: options.featureMode === FEATURE_MODE_ENRICHED
      ? buildDslEnrichedVector(row, baseFeatureNames, featureNames)
      : featureNames.map((name) => numberValue(row[name]))
  }));
  return {
    datasetLabel: "DSL",
    featureNames,
    sequenceShape: [featureNames.length, 1],
    samples,
    datasetFileHashes: { [DSL_FILE]: await sha256Text(text) },
    featureSet: options.featureMode === FEATURE_MODE_ENRICHED
      ? "dsl_current_h_dd_ud_plus_sequence_trigraph_distribution_adaptive"
      : "dsl_current_h_dd_ud",
    adaptiveFeatureSelection: options.featureMode === FEATURE_MODE_ENRICHED
  };
}

async function loadMmcDataset(options) {
  const samples = [];
  const pairManifest = [];
  const fileHashes = {};
  let orderBase = 0;
  const baseFeatureNames = makeMmcFlatFeatureNames();
  const featureNames = options.featureMode === FEATURE_MODE_ENRICHED
    ? buildMmcEnrichedFeatureNames(baseFeatureNames)
    : baseFeatureNames;
  for (const pair of MMC_PAIRS) {
    const [featureBuffer, labelBuffer] = await Promise.all([
      fetchBuffer(options, "mmc", pair.feature),
      fetchBuffer(options, "mmc", pair.label)
    ]);
    fileHashes[pair.feature] = await sha256Buffer(featureBuffer);
    fileHashes[pair.label] = await sha256Buffer(labelBuffer);
    const labels = parseUnicodeNpy(labelBuffer);
    const timing = parseObjectNpyTiming(featureBuffer, MMC_TIMING_CHANNELS, 1_000_000);
    if (timing.shape[0] !== labels.shape[0]) {
      throw new Error(`${pair.feature}/${pair.label} sample count mismatch`);
    }
    for (let i = 0; i < timing.shape[0]; i += 1) {
      samples.push({
        subject: String(labels.values[i * labels.shape[1]]),
        order: orderBase + i,
        partition: pair.partition,
        vector: options.featureMode === FEATURE_MODE_ENRICHED
          ? buildMmcEnrichedVector(timing.values[i], featureNames)
          : timing.values[i]
      });
    }
    pairManifest.push({
      feature_file: pair.feature,
      label_file: pair.label,
      partition: pair.partition,
      samples: timing.shape[0],
      sequence_length: timing.shape[1],
      timing_channels: MMC_TIMING_CHANNELS.join("|")
    });
    orderBase += 1_000_000;
  }
  return {
    datasetLabel: "MMC",
    featureNames,
    sequenceShape: options.featureMode === FEATURE_MODE_ENRICHED
      ? [featureNames.length, 1]
      : [642, MMC_TIMING_CHANNELS.length],
    samples,
    pairManifest,
    datasetFileHashes: fileHashes,
    featureSet: options.featureMode === FEATURE_MODE_ENRICHED
      ? "mmc_flat_timing_channels_plus_distribution_delta_trigraph_adaptive"
      : "mmc_flat_timing_channels_14_19",
    adaptiveFeatureSelection: options.featureMode === FEATURE_MODE_ENRICHED
  };
}

async function runCollectedAnalysisAndEvaluation(options, environment, startedAt) {
  const tableTexts = {};
  const hashes = {};
  for (const fileName of COLLECTED_FILES) {
    const text = await fetchText(options, "collected", fileName);
    tableTexts[fileName] = text;
    hashes[fileName] = await sha256Text(text);
  }
  const tables = Object.fromEntries(
    Object.entries(tableTexts).map(([name, text]) => [name.replace(".csv", ""), parseCsv(text)])
  );
  progress("dataset", `Collected CSV tables loaded (${Object.keys(tables).length} tables)`);

  const canonicalAttempts = buildCollectedAttempts(tables);
  const eventPairs = buildCollectedEventPairs(tables.events || []);
  const featureMatrix = buildCollectedFeatureMatrix(tables.features || []);
  const enrichedFeatureMatrix = options.featureMode === FEATURE_MODE_ENRICHED
    ? buildCollectedEnrichedFeatureMatrix(canonicalAttempts, eventPairs, featureMatrix)
    : [];
  const qualitySummary = buildCollectedQualitySummary(canonicalAttempts, eventPairs);
  const filterSummary = buildCollectedFilterSummary(canonicalAttempts);
  const analysisArtifacts = [
    textArtifact("collected_attempts_canonical.csv", toCsv(canonicalAttempts)),
    textArtifact("collected_event_pairs.csv", toCsv(eventPairs)),
    textArtifact("collected_feature_matrix.csv", toCsv(featureMatrix)),
    ...(options.featureMode === FEATURE_MODE_ENRICHED
      ? [textArtifact("collected_enriched_feature_matrix.csv", toCsv(enrichedFeatureMatrix))]
      : []),
    textArtifact("collected_quality_summary.csv", toCsv(qualitySummary)),
    textArtifact("collected_filter_summary.csv", toCsv(filterSummary))
  ];
  const data = buildCollectedEvaluationDataset(
    canonicalAttempts,
    options.featureMode === FEATURE_MODE_ENRICHED ? enrichedFeatureMatrix : featureMatrix,
    hashes,
    options.featureMode
  );
  return runModelEvaluation(options, environment, data, startedAt, analysisArtifacts);
}

function buildCollectedEvaluationDataset(canonicalAttempts, featureMatrix, hashes, featureMode = FEATURE_MODE_CURRENT) {
  const attempts = new Map(canonicalAttempts.map((attempt) => [attempt.attempt_id, attempt]));
  const featureNames = Array.from(featureMatrix.reduce((set, row) => {
    Object.keys(row).forEach((key) => {
      if (key !== "attempt_id") set.add(key);
    });
    return set;
  }, new Set())).sort();

  const rows = [];
  for (const featureRow of featureMatrix) {
    const attempt = attempts.get(featureRow.attempt_id);
    if (!attempt || !isCollectedTrainingCandidate(attempt)) {
      continue;
    }
    const vector = featureNames.map((name) => numberValue(featureRow[name]));
    if (!vector.some((value) => value !== 0)) {
      continue;
    }
    rows.push({
      subject: canonicalCollectedSubject(attempt.participant_code || attempt.participant_id),
      attemptId: attempt.attempt_id,
      timestamp: attempt.submitted_at || attempt.ended_at || attempt.started_at || "",
      vector
    });
  }

  const bySubject = groupBy(rows, "subject");
  const samples = [];
  for (const [, subjectRows] of bySubject.entries()) {
    const sorted = subjectRows.slice().sort((a, b) => String(a.timestamp).localeCompare(String(b.timestamp)));
    if (sorted.length < 2) {
      continue;
    }
    const referenceCount = Math.min(sorted.length - 1, Math.max(1, Math.floor(sorted.length * 0.6)));
    for (let index = 0; index < sorted.length; index += 1) {
      samples.push({
        subject: sorted[index].subject,
        order: Date.parse(sorted[index].timestamp) || index,
        partition: index < referenceCount ? "reference" : "test",
        vector: sorted[index].vector,
        attemptId: sorted[index].attemptId
      });
    }
  }

  return {
    datasetLabel: "Collected CSV",
    featureNames,
    sequenceShape: [featureNames.length, 1],
    samples: samples.sort((a, b) => a.order - b.order),
    datasetFileHashes: hashes,
    dataEfficiencySizes: [1, 2, 3, 5, 10],
    featureSet: featureMode === FEATURE_MODE_ENRICHED
      ? "collected_stored_features_plus_key_sequence_digraph_trigraph_distribution_adaptive"
      : "collected_stored_browser_baseline_features",
    adaptiveFeatureSelection: featureMode === FEATURE_MODE_ENRICHED
  };
}

function makeAuthSplit(data, genuineSubject, { adaptive = true } = {}) {
  const trainSamples = [];
  const testSamples = [];
  for (const sample of data.samples) {
    if (sample.partition === "reference") {
      trainSamples.push(sample);
    } else if (sample.partition === "test") {
      testSamples.push(sample);
    }
  }
  const trainY = trainSamples.map((sample) => (sample.subject === genuineSubject ? 1 : 0));
  const testY = testSamples.map((sample) => (sample.subject === genuineSubject ? 1 : 0));
  if (!trainY.includes(1) || !trainY.includes(0) || !testY.includes(1) || !testY.includes(0)) {
    return null;
  }
  const split = {
    trainX: trainSamples.map((sample) => sample.vector),
    testX: testSamples.map((sample) => sample.vector),
    trainY,
    testY,
    trainOrder: trainSamples.map((sample) => sample.order),
    sequenceShape: data.sequenceShape,
    featureNames: data.featureNames || []
  };
  return adaptive && data.adaptiveFeatureSelection ? applyAdaptiveFeatureSelection(split) : split;
}

async function trainAndScore(spec, split, sequenceShape) {
  const resolvedShape = split.sequenceShape || sequenceShape;
  if (spec.key === "manhattan") {
    return trainAndScoreManhattan(split);
  }
  if (spec.key === "lightgbm") {
    return trainAndScoreLightGbm(split);
  }
  return trainAndScoreCnn(split, resolvedShape);
}

function trainAndScoreManhattan(split) {
  const memoryBefore = browserMemory();
  const startTrain = performance.now();
  const model = fitManhattan(split.trainX, split.trainY);
  const trainTimeMs = performance.now() - startTrain;
  const startInfer = performance.now();
  const scores = split.testX.map((row) => scoreManhattan(model, row));
  const inferenceBatchMs = performance.now() - startInfer;
  return {
    scores,
    trainTimeMs,
    inferenceBatchMs,
    inferenceTimeMsPerSample: inferenceBatchMs / Math.max(scores.length, 1),
    memoryBefore,
    memoryAfter: browserMemory(),
    tfMemory: null
  };
}

function fitManhattan(X, y) {
  const genuine = X.filter((_, index) => y[index] === 1);
  const width = X[0].length;
  const center = Array(width).fill(0);
  for (const row of genuine) {
    for (let i = 0; i < width; i += 1) center[i] += row[i] || 0;
  }
  for (let i = 0; i < width; i += 1) center[i] /= genuine.length;
  const scale = Array(width).fill(0);
  for (const row of genuine) {
    for (let i = 0; i < width; i += 1) scale[i] += ((row[i] || 0) - center[i]) ** 2;
  }
  for (let i = 0; i < width; i += 1) scale[i] = Math.max(Math.sqrt(scale[i] / Math.max(genuine.length - 1, 1)), 1e-6);
  return { center, scale };
}

function scoreManhattan(model, row) {
  let distance = 0;
  for (let i = 0; i < row.length; i += 1) {
    distance += Math.abs(((row[i] || 0) - model.center[i]) / model.scale[i]);
  }
  return -distance;
}

function trainAndScoreLightGbm(split) {
  const memoryBefore = browserMemory();
  const startTrain = performance.now();
  const model = fitBrowserLightGbm(split.trainX, split.trainY, resolveLightGbmOptions(split));
  const trainTimeMs = performance.now() - startTrain;
  const startInfer = performance.now();
  const scores = split.testX.map((row) => predictBrowserLightGbm(model, row));
  const inferenceBatchMs = performance.now() - startInfer;
  return {
    scores,
    trainTimeMs,
    inferenceBatchMs,
    inferenceTimeMsPerSample: inferenceBatchMs / Math.max(scores.length, 1),
    memoryBefore,
    memoryAfter: browserMemory(),
    tfMemory: null
  };
}

function resolveLightGbmOptions(split) {
  const featureCount = split.trainX[0]?.length || 0;
  const isWideTemporal = featureCount > 1000;
  return {
    objective: "binary",
    boosting: "gbdt",
    learningRate: 0.05,
    numIterations: isWideTemporal ? 80 : 150,
    numLeaves: isWideTemporal ? 11 : 15,
    maxDepth: isWideTemporal ? 5 : 6,
    minDataInLeaf: isWideTemporal ? 10 : 20,
    minSumHessianInLeaf: 1e-3,
    lambdaL2: 1,
    minGainToSplit: 1e-7,
    maxBin: isWideTemporal ? 15 : 31,
    featureFraction: 0.9,
    baggingFraction: 0.9,
    baggingFreq: 1,
    featureCap: isWideTemporal ? 256 : featureCount,
    scoreClip: 10,
    leafValueClip: 6
  };
}

function fitBrowserLightGbm(X, y, options) {
  const n = X.length;
  const positiveRate = Math.min(Math.max(mean(y), 1e-4), 1 - 1e-4);
  const baseScore = Math.log(positiveRate / (1 - positiveRate));
  const rawScores = Array(n).fill(baseScore);
  const features = selectLightGbmFeatures(X, options);
  const thresholdsByFeature = makeLightGbmThresholds(X, features, options.maxBin);
  const trees = [];

  for (let iteration = 0; iteration < options.numIterations; iteration += 1) {
    const gradients = Array(n);
    const hessians = Array(n);
    for (let i = 0; i < n; i += 1) {
      const probability = sigmoid(rawScores[i]);
      gradients[i] = probability - y[i];
      hessians[i] = Math.max(probability * (1 - probability), 1e-6);
    }

    const treeFeatures = selectLightGbmTreeFeatures(features, options.featureFraction, iteration);
    const rootIndices = makeLightGbmBag(options, n, iteration);
    const tree = buildLightGbmTree(X, gradients, hessians, rootIndices, treeFeatures, thresholdsByFeature, options);
    if (!tree) {
      break;
    }
    trees.push(tree);
    for (let i = 0; i < n; i += 1) {
      rawScores[i] = clampNumber(
        rawScores[i] + options.learningRate * evaluateLightGbmTree(tree.root, X[i]),
        -options.scoreClip,
        options.scoreClip
      );
    }
  }

  return {
    runtime: LIGHTGBM_RUNTIME,
    objective: options.objective,
    boosting: options.boosting,
    baseScore,
    trees,
    learningRate: options.learningRate,
    featureCount: X[0]?.length || 0,
    trainedFeatureCount: features.length,
    options
  };
}

function predictBrowserLightGbm(model, row) {
  let value = model.baseScore;
  for (const tree of model.trees) {
    value += model.learningRate * evaluateLightGbmTree(tree.root, row);
  }
  return sigmoid(value);
}

function buildLightGbmTree(X, gradients, hessians, rootIndices, features, thresholdsByFeature, options) {
  if (!rootIndices.length || !features.length) {
    return null;
  }
  const root = makeLightGbmLeaf(rootIndices, gradients, hessians, options, 0);
  const leaves = [root];
  let splitCount = 0;

  while (leaves.length < options.numLeaves) {
    let best = null;
    for (const leaf of leaves) {
      if (leaf.depth >= options.maxDepth || leaf.indices.length < options.minDataInLeaf * 2) {
        continue;
      }
      const split = findLightGbmSplit(X, gradients, hessians, leaf, features, thresholdsByFeature, options);
      if (split && (!best || split.gain > best.gain)) {
        best = { leaf, split };
      }
    }
    if (!best || best.split.gain <= options.minGainToSplit) {
      break;
    }

    const left = makeLightGbmLeaf(best.split.leftIndices, gradients, hessians, options, best.leaf.depth + 1);
    const right = makeLightGbmLeaf(best.split.rightIndices, gradients, hessians, options, best.leaf.depth + 1);
    best.leaf.splitFeature = best.split.feature;
    best.leaf.threshold = best.split.threshold;
    best.leaf.defaultLeft = true;
    best.leaf.left = left;
    best.leaf.right = right;
    delete best.leaf.indices;
    const leafIndex = leaves.indexOf(best.leaf);
    leaves.splice(leafIndex, 1, left, right);
    splitCount += 1;
  }

  return splitCount || Number.isFinite(root.value) ? { root, leaves: leaves.length, splits: splitCount } : null;
}

function makeLightGbmLeaf(indices, gradients, hessians, options, depth) {
  let gradSum = 0;
  let hessSum = 0;
  for (const index of indices) {
    gradSum += gradients[index];
    hessSum += hessians[index];
  }
  const rawValue = -gradSum / Math.max(hessSum + options.lambdaL2, 1e-9);
  return {
    indices,
    depth,
    gradSum,
    hessSum,
    count: indices.length,
    value: clampNumber(rawValue, -options.leafValueClip, options.leafValueClip)
  };
}

function findLightGbmSplit(X, gradients, hessians, leaf, features, thresholdsByFeature, options) {
  const parentGain = lightGbmGain(leaf.gradSum, leaf.hessSum, options.lambdaL2);
  let best = null;
  for (const feature of features) {
    const thresholds = thresholdsByFeature.get(feature);
    if (!thresholds?.length) {
      continue;
    }
    const binCount = thresholds.length + 1;
    const gradBins = Array(binCount).fill(0);
    const hessBins = Array(binCount).fill(0);
    const countBins = Array(binCount).fill(0);
    for (const rowIndex of leaf.indices) {
      const bin = lightGbmBinIndex(X[rowIndex][feature], thresholds);
      gradBins[bin] += gradients[rowIndex];
      hessBins[bin] += hessians[rowIndex];
      countBins[bin] += 1;
    }

    let leftGrad = 0;
    let leftHess = 0;
    let leftCount = 0;
    for (let bin = 0; bin < thresholds.length; bin += 1) {
      leftGrad += gradBins[bin];
      leftHess += hessBins[bin];
      leftCount += countBins[bin];
      const rightCount = leaf.count - leftCount;
      const rightHess = leaf.hessSum - leftHess;
      if (
        leftCount < options.minDataInLeaf
        || rightCount < options.minDataInLeaf
        || leftHess < options.minSumHessianInLeaf
        || rightHess < options.minSumHessianInLeaf
      ) {
        continue;
      }
      const rightGrad = leaf.gradSum - leftGrad;
      const gain = lightGbmGain(leftGrad, leftHess, options.lambdaL2)
        + lightGbmGain(rightGrad, rightHess, options.lambdaL2)
        - parentGain;
      if (!best || gain > best.gain) {
        best = { feature, threshold: thresholds[bin], gain };
      }
    }
  }

  if (!best) {
    return null;
  }
  const leftIndices = [];
  const rightIndices = [];
  for (const rowIndex of leaf.indices) {
    const value = X[rowIndex][best.feature];
    if (!Number.isFinite(Number(value)) || Number(value) <= best.threshold) {
      leftIndices.push(rowIndex);
    } else {
      rightIndices.push(rowIndex);
    }
  }
  if (leftIndices.length < options.minDataInLeaf || rightIndices.length < options.minDataInLeaf) {
    return null;
  }
  return { ...best, leftIndices, rightIndices };
}

function evaluateLightGbmTree(node, row) {
  if (!node.left || !node.right) {
    return Number(node.value || 0);
  }
  const value = Number(row[node.splitFeature]);
  if (!Number.isFinite(value)) {
    return evaluateLightGbmTree(node.defaultLeft ? node.left : node.right, row);
  }
  return evaluateLightGbmTree(value <= node.threshold ? node.left : node.right, row);
}

function selectLightGbmFeatures(X, options) {
  const width = X[0]?.length || 0;
  if (!width) {
    return [];
  }
  const cap = Math.max(1, Math.min(width, options.featureCap || width));
  if (width <= cap) {
    return Array.from({ length: width }, (_, index) => index);
  }
  return featureVariances(X)
    .sort((a, b) => b.variance - a.variance)
    .slice(0, cap)
    .map((row) => row.feature);
}

function selectLightGbmTreeFeatures(features, fraction, iteration) {
  const count = Math.max(1, Math.ceil(features.length * Math.min(Math.max(fraction || 1, 0.01), 1)));
  if (count >= features.length) {
    return features;
  }
  return features
    .map((feature) => ({ feature, key: seededUnit(feature + 1, iteration + 17) }))
    .sort((a, b) => a.key - b.key)
    .slice(0, count)
    .map((row) => row.feature);
}

function makeLightGbmThresholds(X, features, maxBin) {
  const thresholdsByFeature = new Map();
  for (const feature of features) {
    const values = [];
    for (const row of X) {
      const value = Number(row[feature]);
      if (Number.isFinite(value)) {
        values.push(value);
      }
    }
    if (values.length < 2) {
      thresholdsByFeature.set(feature, []);
      continue;
    }
    values.sort((a, b) => a - b);
    const thresholds = [];
    const bins = Math.min(maxBin, values.length - 1);
    for (let bin = 1; bin <= bins; bin += 1) {
      const index = Math.min(values.length - 2, Math.max(0, Math.floor((bin / (bins + 1)) * values.length)));
      const threshold = values[index];
      if (threshold < values[values.length - 1] && thresholds[thresholds.length - 1] !== threshold) {
        thresholds.push(threshold);
      }
    }
    thresholdsByFeature.set(feature, thresholds);
  }
  return thresholdsByFeature;
}

function makeLightGbmBag(options, n, iteration) {
  if (!options.baggingFreq || iteration % options.baggingFreq !== 0 || options.baggingFraction >= 0.999) {
    return Array.from({ length: n }, (_, index) => index);
  }
  const indices = [];
  for (let index = 0; index < n; index += 1) {
    if (seededUnit(index + 1, iteration + 101) <= options.baggingFraction) {
      indices.push(index);
    }
  }
  return indices.length >= options.minDataInLeaf * 2 ? indices : Array.from({ length: n }, (_, index) => index);
}

function lightGbmBinIndex(value, thresholds) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return 0;
  }
  let low = 0;
  let high = thresholds.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (number <= thresholds[mid]) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  return low;
}

function lightGbmGain(gradSum, hessSum, lambdaL2) {
  return (gradSum * gradSum) / Math.max(hessSum + lambdaL2, 1e-9);
}

async function trainAndScoreCnn(split, sequenceShape) {
  const memoryBefore = browserMemory();
  const scaled = standardizeTrainTest(split.trainX, split.testX);
  const trainTensor = tf.tensor(scaled.train, [scaled.train.length, ...sequenceShape]);
  const testTensor = tf.tensor(scaled.test, [scaled.test.length, ...sequenceShape]);
  const yTensor = tf.tensor2d(split.trainY, [split.trainY.length, 1]);
  const model = tf.sequential();
  model.add(tf.layers.conv1d({ inputShape: sequenceShape, filters: 16, kernelSize: 3, padding: "same", activation: "relu" }));
  model.add(tf.layers.maxPooling1d({ poolSize: 2, strides: 2 }));
  model.add(tf.layers.conv1d({ filters: 24, kernelSize: 3, padding: "same", activation: "relu" }));
  model.add(tf.layers.globalAveragePooling1d({}));
  model.add(tf.layers.dense({ units: 16, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: "binaryCrossentropy" });
  const positives = split.trainY.filter((value) => value === 1).length;
  const negatives = split.trainY.length - positives;
  const classWeight = {
    0: split.trainY.length / Math.max(2 * negatives, 1),
    1: split.trainY.length / Math.max(2 * positives, 1)
  };
  const startTrain = performance.now();
  await model.fit(trainTensor, yTensor, {
    epochs: 4,
    batchSize: 128,
    shuffle: false,
    classWeight,
    verbose: 0
  });
  const trainTimeMs = performance.now() - startTrain;
  const startInfer = performance.now();
  const prediction = model.predict(testTensor);
  const values = Array.from(await prediction.data());
  const inferenceBatchMs = performance.now() - startInfer;
  const tfMemory = tf.memory();
  trainTensor.dispose();
  testTensor.dispose();
  yTensor.dispose();
  prediction.dispose();
  model.dispose();
  return {
    scores: values,
    trainTimeMs,
    inferenceBatchMs,
    inferenceTimeMsPerSample: inferenceBatchMs / Math.max(values.length, 1),
    memoryBefore,
    memoryAfter: browserMemory(),
    tfMemory
  };
}

async function evaluateDataEfficiency(spec, data, subjects, timingRows, datasetKey, options = {}) {
  const sizes = data.dataEfficiencySizes || (data.datasetLabel === "MMC" ? [1, 2, 5, 10, 20] : [5, 10, 25, 50, 100, 200]);
  const output = [];
  for (const size of sizes) {
    progress("data_efficiency", `${data.datasetLabel}: ${spec.name} genuine prefix ${size} (${subjects.length} subjects)`);
    const startedAt = performance.now();
    const eers = [];
    for (let subjectIndex = 0; subjectIndex < subjects.length; subjectIndex += 1) {
      const subject = subjects[subjectIndex];
      if (subjectIndex > 0 && subjectIndex % 10 === 0) {
        progress("data_efficiency", `${data.datasetLabel}: ${spec.name} prefix ${size}, subject ${subjectIndex + 1}/${subjects.length}`);
      }
      const split = makeAuthSplit(data, subject, { adaptive: false });
      if (!split) continue;
      const genuineIndices = split.trainY
        .map((label, index) => ({ label, index, order: split.trainOrder[index] }))
        .filter((row) => row.label === 1)
        .sort((a, b) => a.order - b.order)
        .slice(0, size)
        .map((row) => row.index);
      if (genuineIndices.length < size) continue;
      const impostorIndices = split.trainY.map((label, index) => ({ label, index })).filter((row) => row.label === 0).map((row) => row.index);
      const chosen = new Set([...genuineIndices, ...impostorIndices]);
      const reduced = {
        ...split,
        trainX: split.trainX.filter((_, index) => chosen.has(index)),
        trainY: split.trainY.filter((_, index) => chosen.has(index))
      };
      const prepared = data.adaptiveFeatureSelection ? applyAdaptiveFeatureSelection(reduced) : reduced;
      const scored = await trainAndScore(spec, prepared, data.sequenceShape);
      eers.push(computeMetrics(prepared.testY, scored.scores).eer);
    }
    if (eers.length) {
      const meanEer = mean(eers);
      const elapsedMs = performance.now() - startedAt;
      output.push({
        dataset: datasetKey || data.datasetLabel.toLowerCase(),
        source_dataset: options.dataset || datasetKey || data.datasetLabel.toLowerCase(),
        feature_mode: options.featureMode || FEATURE_MODE_CURRENT,
        feature_set: data.featureSet || "current_stored_features",
        Model: spec.name,
        "Genuine Training Samples": size,
        EER: meanEer,
        evaluated_subjects: eers.length,
        elapsed_ms: elapsedMs,
        stopped_after_target_reached: meanEer < EER_TARGET
      });
      timingRows?.push({
        dataset: datasetKey,
        source_dataset: options.dataset || datasetKey,
        feature_mode: options.featureMode || FEATURE_MODE_CURRENT,
        model: spec.name,
        subject: "all",
        phase: "data_efficiency",
        genuine_training_samples: size,
        elapsed_ms: elapsedMs,
        evaluated_subjects: eers.length,
        EER: meanEer
      });
      if (meanEer < EER_TARGET) {
        progress("data_efficiency", `${data.datasetLabel}: ${spec.name} reached EER < 10% at prefix ${size}; stopping larger prefixes`);
        break;
      }
    }
  }
  return output;
}

function computeMetrics(yTrue, scores) {
  const prepared = prepareScoreScan(yTrue, scores);
  const positives = prepared.positives;
  const negatives = prepared.negatives;
  if (!positives || !negatives) {
    return {
      far: 0,
      frr: 0,
      eer: 0,
      accuracy: 0,
      auc: 0,
      threshold: 0
    };
  }

  let tp = 0;
  let fp = 0;
  let fn = positives;
  let tn = negatives;
  let best = metricCandidate(Number.POSITIVE_INFINITY, tp, fp, tn, fn, positives, negatives);
  const pairs = prepared.pairs;
  for (let index = 0; index < pairs.length;) {
    const score = pairs[index].score;
    let groupPositives = 0;
    let groupNegatives = 0;
    while (index < pairs.length && pairs[index].score === score) {
      if (pairs[index].label === 1) groupPositives += 1;
      else groupNegatives += 1;
      index += 1;
    }
    tp += groupPositives;
    fn -= groupPositives;
    fp += groupNegatives;
    tn -= groupNegatives;
    const candidate = metricCandidate(score, tp, fp, tn, fn, positives, negatives);
    if (candidate.distance < best.distance) {
      best = candidate;
    }
  }
  const allPositive = metricCandidate(Number.NEGATIVE_INFINITY, positives, negatives, 0, 0, positives, negatives);
  if (allPositive.distance < best.distance) {
    best = allPositive;
  }

  return {
    far: best.fpr,
    frr: best.fnr,
    eer: (best.fpr + best.fnr) / 2,
    accuracy: best.accuracy,
    auc: aucFromPrepared(prepared),
    threshold: best.threshold
  };
}

function metricCandidate(threshold, tp, fp, tn, fn, positives, negatives) {
  const fpr = fp / Math.max(negatives, 1);
  const fnr = fn / Math.max(positives, 1);
  return {
    threshold,
    fpr,
    fnr,
    distance: Math.abs(fpr - fnr),
    accuracy: (tp + tn) / Math.max(positives + negatives, 1)
  };
}

function aucScore(yTrue, scores) {
  return aucFromPrepared(prepareScoreScan(yTrue, scores));
}

function aucFromPrepared(prepared) {
  const pairs = prepared.pairs.slice().sort((a, b) => a.score - b.score);
  let rankSum = 0;
  let positives = 0;
  let negatives = 0;
  for (let i = 0; i < pairs.length;) {
    let j = i + 1;
    while (j < pairs.length && pairs[j].score === pairs[i].score) {
      j += 1;
    }
    const averageRank = (i + 1 + j) / 2;
    for (let k = i; k < j; k += 1) {
      if (pairs[k].label === 1) {
        positives += 1;
        rankSum += averageRank;
      } else {
        negatives += 1;
      }
    }
    i = j;
  }
  if (!positives || !negatives) return 0;
  return (rankSum - (positives * (positives + 1)) / 2) / (positives * negatives);
}

function makeRocSeries(yTrue, scores) {
  const prepared = prepareScoreScan(yTrue, scores);
  const positives = prepared.positives;
  const negatives = prepared.negatives;
  if (!positives || !negatives) {
    return [];
  }
  const points = [{ fpr: 0, tpr: 0 }];
  let tp = 0;
  let fp = 0;
  const pairs = prepared.pairs;
  for (let index = 0; index < pairs.length;) {
    const score = pairs[index].score;
    while (index < pairs.length && pairs[index].score === score) {
      if (pairs[index].label === 1) {
        tp += 1;
      } else {
        fp += 1;
      }
      index += 1;
    }
    points.push({
      fpr: fp / Math.max(negatives, 1),
      tpr: tp / Math.max(positives, 1)
    });
  }
  return downsampleRoc(points, 1200);
}

function prepareScoreScan(yTrue, scores) {
  const pairs = [];
  let positives = 0;
  let negatives = 0;
  for (let index = 0; index < scores.length; index += 1) {
    const score = Number(scores[index]);
    if (!Number.isFinite(score)) {
      continue;
    }
    const label = yTrue[index] === 1 ? 1 : 0;
    if (label === 1) {
      positives += 1;
    } else {
      negatives += 1;
    }
    pairs.push({ score, label });
  }
  pairs.sort((a, b) => b.score - a.score);
  return { pairs, positives, negatives };
}

function downsampleRoc(points, limit) {
  if (points.length <= limit) {
    return points;
  }
  const output = [];
  for (let i = 0; i < limit; i += 1) {
    const index = Math.round((i / (limit - 1)) * (points.length - 1));
    output.push(points[index]);
  }
  return output;
}

function averageMetricRows(rows, dataset, spec) {
  const metric = (name) => mean(rows.map((row) => Number(row[name])).filter(Number.isFinite));
  const metricOrUnavailable = (name) => {
    const values = rows.map((row) => Number(row[name])).filter(Number.isFinite);
    return values.length ? mean(values) : "not_available";
  };
  return {
    dataset,
    Model: spec.name,
    model_runtime: spec.runtime || "browser_worker",
    FAR: metric("FAR"),
    FRR: metric("FRR"),
    EER: metric("EER"),
    Accuracy: metric("Accuracy"),
    AUC: metric("AUC"),
    "Inference Time (ms/sample)": metric("inference_time_ms_per_sample"),
    "UI Blocking Time (ms/test batch)": metric("ui_blocking_time_ms"),
    "Train Time (ms/subject)": metric("train_time_ms"),
    "Browser Memory Before (bytes)": metricOrUnavailable("memory_before_bytes"),
    "Browser Memory After (bytes)": metricOrUnavailable("memory_after_bytes"),
    "Min Genuine Samples for EER < 10%": "Not reached",
    subjects: rows.length
  };
}

function minSamplesForTarget(rows, modelName) {
  const matches = rows
    .filter((row) => row.Model === modelName && Number(row.EER) < EER_TARGET)
    .sort((a, b) => Number(a["Genuine Training Samples"]) - Number(b["Genuine Training Samples"]));
  return matches[0]?.["Genuine Training Samples"] ?? "Not reached";
}

function standardizeTrainTest(train, test) {
  const width = train[0].length;
  const center = Array(width).fill(0);
  const scale = Array(width).fill(0);
  for (const row of train) {
    for (let i = 0; i < width; i += 1) center[i] += row[i] || 0;
  }
  for (let i = 0; i < width; i += 1) center[i] /= train.length;
  for (const row of train) {
    for (let i = 0; i < width; i += 1) scale[i] += ((row[i] || 0) - center[i]) ** 2;
  }
  for (let i = 0; i < width; i += 1) scale[i] = Math.max(Math.sqrt(scale[i] / Math.max(train.length - 1, 1)), 1e-6);
  const transform = (row) => row.map((value, index) => ((value || 0) - center[index]) / scale[index]);
  return { train: train.map(transform), test: test.map(transform) };
}

function parseObjectNpyTiming(buffer, selectedChannels, timeScale) {
  const view = new DataView(buffer);
  const headerLength = view.getUint16(8, true);
  const headerText = asciiDecode(new Uint8Array(buffer, 10, headerLength));
  const shape = parseNpyShape(headerText);
  const [samples, timesteps, channels] = shape;
  const out = Array.from({ length: samples }, () => Array(timesteps * selectedChannels.length).fill(0));
  const selected = new Map(selectedChannels.map((channel, index) => [channel, index]));
  const bytes = new Uint8Array(buffer);
  let pos = findPickleDataStart(bytes, 10 + headerLength);
  let valueIndex = 0;
  const expectedValues = samples * timesteps * channels;
  while (pos < bytes.length && valueIndex < expectedValues) {
    const op = bytes[pos];
    if (op === 0x47) {
      const value = view.getFloat64(pos + 1, false);
      recordMmcValue(out, valueIndex, channels, timesteps, selected, value / timeScale);
      valueIndex += 1;
      pos += 9;
    } else if (op === 0x58) {
      const length = view.getUint32(pos + 1, true);
      recordMmcValue(out, valueIndex, channels, timesteps, selected, 0);
      valueIndex += 1;
      pos += 5 + length;
    } else if (op === 0x55) {
      const length = bytes[pos + 1];
      recordMmcValue(out, valueIndex, channels, timesteps, selected, 0);
      valueIndex += 1;
      pos += 2 + length;
    } else if (op === 0x68) {
      recordMmcValue(out, valueIndex, channels, timesteps, selected, 0);
      valueIndex += 1;
      pos += 2;
    } else if (op === 0x6a) {
      recordMmcValue(out, valueIndex, channels, timesteps, selected, 0);
      valueIndex += 1;
      pos += 5;
    } else if (op === 0x71) {
      pos += 2;
    } else if (op === 0x72) {
      pos += 5;
    } else if (op === 0x80 || op === 0x4b) {
      pos += 2;
    } else if (op === 0x4d) {
      pos += 3;
    } else if (op === 0x4a) {
      pos += 5;
    } else if (op === 0x63) {
      pos = skipPickleGlobal(bytes, pos);
    } else if (op === 0x2e) {
      break;
    } else {
      pos += 1;
    }
  }
  if (valueIndex < expectedValues) {
    throw new Error(`MMC object npy parse stopped at ${valueIndex}/${expectedValues} values`);
  }
  return { shape: [samples, timesteps, selectedChannels.length], values: out };
}

function skipPickleGlobal(bytes, pos) {
  let newlineCount = 0;
  let next = pos + 1;
  while (next < bytes.length) {
    if (bytes[next] === 0x0a) {
      newlineCount += 1;
      if (newlineCount === 2) {
        return next + 1;
      }
    }
    next += 1;
  }
  return next;
}

function findPickleDataStart(bytes, start) {
  for (let i = start; i < Math.min(bytes.length - 3, start + 512); i += 1) {
    if (bytes[i] === 0x5d && bytes[i + 1] === 0x71) {
      for (let j = i + 2; j < i + 16; j += 1) {
        if (bytes[j] === 0x28) return j + 1;
      }
    }
  }
  throw new Error("Could not find pickle object-list data start in MMC npy file");
}

function recordMmcValue(out, valueIndex, channels, timesteps, selected, value) {
  const sample = Math.floor(valueIndex / (timesteps * channels));
  const within = valueIndex % (timesteps * channels);
  const timestep = Math.floor(within / channels);
  const channel = within % channels;
  const selectedIndex = selected.get(channel);
  if (selectedIndex !== undefined) {
    out[sample][timestep * selected.size + selectedIndex] = Number.isFinite(value) ? value : 0;
  }
}

function parseUnicodeNpy(buffer) {
  const view = new DataView(buffer);
  const headerLength = view.getUint16(8, true);
  const headerText = asciiDecode(new Uint8Array(buffer, 10, headerLength));
  const shape = parseNpyShape(headerText);
  const dtype = /'descr':\s*'([^']+)'/.exec(headerText)?.[1] || "";
  const charsPerValue = Number(dtype.match(/U(\d+)/)?.[1] || 0);
  if (!charsPerValue) throw new Error(`Unsupported label dtype ${dtype}`);
  const values = [];
  let offset = 10 + headerLength;
  const count = shape.reduce((product, value) => product * value, 1);
  for (let i = 0; i < count; i += 1) {
    let text = "";
    for (let c = 0; c < charsPerValue; c += 1) {
      const code = view.getUint32(offset, true);
      if (code) text += String.fromCodePoint(code);
      offset += 4;
    }
    values.push(text);
  }
  return { shape, values };
}

function parseNpyShape(headerText) {
  const match = /\(([^)]*)\)/.exec(headerText);
  if (!match) throw new Error(`Could not parse npy shape: ${headerText}`);
  return match[1].split(",").map((item) => item.trim()).filter(Boolean).map(Number);
}

function parseCsv(text) {
  const clean = text.replace(/^\uFEFF/, "");
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

function buildCollectedAttempts(tables) {
  const participants = new Map((tables.participants || []).map((row) => [row.id, row]));
  const sessions = new Map((tables.sessions || []).map((row) => [row.id, row]));
  return (tables.attempts || []).map((attempt) => {
    const session = sessions.get(attempt.session_id) || {};
    const participant = participants.get(attempt.participant_id) || {};
    return {
      attempt_id: attempt.id,
      participant_id: attempt.participant_id,
      participant_code: participant.participant_code || "",
      session_id: attempt.session_id,
      session_no: session.session_no,
      input_mode: attempt.input_mode,
      role_label: attempt.role_label,
      status: attempt.status,
      quality_status: attempt.quality_status,
      summary_quality_status: attempt.summary_qualityStatus,
      feature_quality: attempt.feature_quality,
      summary_feature_quality: attempt.summary_featureQuality,
      device_class: attempt.device_class || session.device_class,
      prompt_id: attempt.prompt_id,
      prompt_set_id: attempt.prompt_set_id,
      started_at: attempt.started_at,
      ended_at: attempt.ended_at,
      submitted_at: attempt.submitted_at,
      raw_text_length: attempt.summary_rawTextLength || attempt.raw_text?.length || 0,
      event_count: attempt.summary_eventCount,
      paired_key_count: attempt.summary_pairedKeyCount,
      keyup_coverage: attempt.summary_keyupCoverage,
      paste_count: attempt.paste_count,
      summary_paste_count: attempt.summary_pasteCount,
      paste_detected: truthyValue(attempt.summary_pasteDetected)
        || numberValue(attempt.paste_count) > 0
        || numberValue(attempt.summary_pasteCount) > 0,
      fixed_prompt_match: attempt.fixed_prompt_match
    };
  });
}

function buildCollectedEventPairs(events) {
  const byAttempt = groupBy(events, "attempt_id");
  const rows = [];
  for (const [attemptId, attemptEvents] of byAttempt.entries()) {
    const sorted = attemptEvents.slice().sort((a, b) => numberValue(a.relative_time) - numberValue(b.relative_time));
    const active = new Map();
    const pairs = [];
    for (const event of sorted) {
      const key = event.code || event.key_value || "unknown";
      if (event.event_type === "keydown") {
        if (!active.has(key)) active.set(key, []);
        active.get(key).push(event);
      } else if (event.event_type === "keyup") {
        const stack = active.get(key) || [];
        const down = stack.shift();
        if (down) {
          pairs.push({
            down,
            up: event,
            hold_ms: numberValue(event.relative_time) - numberValue(down.relative_time)
          });
        }
      }
    }
    for (let i = 0; i < pairs.length; i += 1) {
      rows.push({
        attempt_id: attemptId,
        pair_index: i,
        key_value: pairs[i].down.key_value,
        code: pairs[i].down.code,
        down_time_ms: pairs[i].down.relative_time,
        up_time_ms: pairs[i].up.relative_time,
        hold_ms: pairs[i].hold_ms,
        down_down_ms: i > 0 ? numberValue(pairs[i].down.relative_time) - numberValue(pairs[i - 1].down.relative_time) : "",
        up_up_ms: i > 0 ? numberValue(pairs[i].up.relative_time) - numberValue(pairs[i - 1].up.relative_time) : "",
        up_down_ms: i > 0 ? numberValue(pairs[i].down.relative_time) - numberValue(pairs[i - 1].up.relative_time) : ""
      });
    }
  }
  return rows;
}

function buildCollectedFeatureMatrix(features) {
  const byAttempt = groupBy(features, "attempt_id");
  const rows = [];
  for (const [attemptId, featureRows] of byAttempt.entries()) {
    const row = { attempt_id: attemptId };
    for (const feature of featureRows) {
      row[feature.feature_name] = feature.feature_value;
    }
    rows.push(row);
  }
  return rows;
}

function buildCollectedQualitySummary(attempts, eventPairs) {
  const pairsByAttempt = groupBy(eventPairs, "attempt_id");
  return attempts.map((attempt) => {
    const pairs = pairsByAttempt.get(attempt.attempt_id) || [];
    return {
      attempt_id: attempt.attempt_id,
      participant_code: attempt.participant_code,
      input_mode: attempt.input_mode,
      role_label: attempt.role_label,
      status: attempt.status,
      quality_status: attempt.quality_status,
      summary_quality_status: attempt.summary_quality_status,
      feature_quality: attempt.feature_quality,
      device_class: attempt.device_class,
      event_count: attempt.event_count,
      paired_key_count: pairs.length,
      hold_mean_ms: mean(pairs.map((row) => numberValue(row.hold_ms))),
      down_down_mean_ms: mean(pairs.map((row) => numberValue(row.down_down_ms)).filter(Number.isFinite)),
      up_down_mean_ms: mean(pairs.map((row) => numberValue(row.up_down_ms)).filter(Number.isFinite)),
      raw_text_length: attempt.raw_text_length,
      paste_count: attempt.paste_count,
      summary_paste_count: attempt.summary_paste_count,
      paste_detected: attempt.paste_detected,
      fixed_prompt_match: attempt.fixed_prompt_match
    };
  });
}

function buildCollectedFilterSummary(attempts) {
  const checks = [
    ["total_attempts", () => true],
    ["submitted", (attempt) => normalizedLower(attempt.status) === "submitted"],
    ["quality_status_usable", (attempt) => normalizedLower(attempt.quality_status) === "usable"],
    ["summary_quality_status_usable_or_blank", (attempt) => ["", "nan", "usable"].includes(normalizedLower(attempt.summary_quality_status))],
    ["not_low_feature_quality", (attempt) => normalizedLower(attempt.feature_quality) !== "low"],
    ["not_low_summary_feature_quality", (attempt) => normalizedLower(attempt.summary_feature_quality) !== "low"],
    ["paste_not_detected", (attempt) => !truthyValue(attempt.paste_detected)],
    ["not_test_participant", (attempt) => canonicalCollectedSubject(attempt.participant_code || attempt.participant_id) !== "TEST"],
    ["included_for_training_and_test", isCollectedTrainingCandidate]
  ];
  return checks.map(([filter, predicate]) => ({
    filter,
    attempts: attempts.filter(predicate).length
  }));
}

function isCollectedTrainingCandidate(attempt) {
  return normalizedLower(attempt.status) === "submitted"
    && normalizedLower(attempt.quality_status) === "usable"
    && ["", "nan", "usable"].includes(normalizedLower(attempt.summary_quality_status))
    && normalizedLower(attempt.feature_quality) !== "low"
    && normalizedLower(attempt.summary_feature_quality) !== "low"
    && !truthyValue(attempt.paste_detected)
    && canonicalCollectedSubject(attempt.participant_code || attempt.participant_id) !== "TEST";
}

function buildDslEnrichedFeatureNames(baseFeatureNames) {
  const keySequence = dslKeySequence(baseFeatureNames);
  const names = [...baseFeatureNames];
  for (let index = 0; index < keySequence.length; index += 1) {
    names.push(`sequence.key_bucket_pos_${pad3(index)}`);
  }
  names.push(...statsNames("timing_distribution.hold"));
  names.push(...statsNames("timing_distribution.down_down"));
  names.push(...statsNames("timing_distribution.up_down"));
  for (let index = 0; index < keySequence.length - 2; index += 1) {
    const [a, b, c] = keySequence.slice(index, index + 3);
    names.push(`trigraph.${a}.${b}.${c}.down_down_sum`);
    names.push(`trigraph.${a}.${b}.${c}.up_down_sum`);
  }
  return names;
}

function buildDslEnrichedVector(row, baseFeatureNames) {
  const keySequence = dslKeySequence(baseFeatureNames);
  const holdValues = baseFeatureNames.filter((name) => name.startsWith("H.")).map((name) => numberValue(row[name]));
  const downDownValues = baseFeatureNames.filter((name) => name.startsWith("DD.")).map((name) => numberValue(row[name]));
  const upDownValues = baseFeatureNames.filter((name) => name.startsWith("UD.")).map((name) => numberValue(row[name]));
  const vector = baseFeatureNames.map((name) => numberValue(row[name]));
  for (const key of keySequence) {
    vector.push(normalizedHashBucket(key, HASH_BUCKETS));
  }
  vector.push(...statsValues(holdValues));
  vector.push(...statsValues(downDownValues));
  vector.push(...statsValues(upDownValues));
  for (let index = 0; index < keySequence.length - 2; index += 1) {
    const [a, b, c] = keySequence.slice(index, index + 3);
    vector.push(numberValue(row[`DD.${a}.${b}`]) + numberValue(row[`DD.${b}.${c}`]));
    vector.push(numberValue(row[`UD.${a}.${b}`]) + numberValue(row[`UD.${b}.${c}`]));
  }
  return vector;
}

function dslKeySequence(baseFeatureNames) {
  return baseFeatureNames
    .filter((name) => name.startsWith("H."))
    .map((name) => name.slice(2));
}

function makeMmcFlatFeatureNames() {
  const names = [];
  for (let timestep = 0; timestep < 642; timestep += 1) {
    for (const channel of MMC_TIMING_CHANNELS) {
      names.push(`sequence.t${pad3(timestep)}.timing_channel_${channel}`);
    }
  }
  return names;
}

function buildMmcEnrichedFeatureNames(baseFeatureNames) {
  const names = [...baseFeatureNames];
  names.push("sequence_length");
  names.push("timing_channel_count");
  names.push(...statsNames("timing_distribution.all_channels"));
  for (const channel of MMC_TIMING_CHANNELS) {
    names.push(...statsNames(`timing_distribution.channel_${channel}`));
    names.push(...statsNames(`digraph_delta.channel_${channel}`));
    names.push(...statsNames(`trigraph_curvature.channel_${channel}`));
  }
  return names;
}

function buildMmcEnrichedVector(baseVector) {
  const vector = baseVector.slice();
  vector.push(642);
  vector.push(MMC_TIMING_CHANNELS.length);
  vector.push(...statsValues(baseVector));
  for (let channelIndex = 0; channelIndex < MMC_TIMING_CHANNELS.length; channelIndex += 1) {
    const values = [];
    for (let timestep = 0; timestep < 642; timestep += 1) {
      values.push(baseVector[timestep * MMC_TIMING_CHANNELS.length + channelIndex] || 0);
    }
    const deltas = [];
    const curvatures = [];
    for (let timestep = 1; timestep < values.length; timestep += 1) {
      deltas.push(values[timestep] - values[timestep - 1]);
    }
    for (let timestep = 2; timestep < values.length; timestep += 1) {
      curvatures.push(values[timestep] - 2 * values[timestep - 1] + values[timestep - 2]);
    }
    vector.push(...statsValues(values));
    vector.push(...statsValues(deltas));
    vector.push(...statsValues(curvatures));
  }
  return vector;
}

function buildCollectedEnrichedFeatureMatrix(canonicalAttempts, eventPairs, featureMatrix) {
  const baseByAttempt = new Map(featureMatrix.map((row) => [row.attempt_id, row]));
  const pairsByAttempt = groupBy(eventPairs, "attempt_id");
  return canonicalAttempts.map((attempt) => {
    const row = { attempt_id: attempt.attempt_id };
    const base = baseByAttempt.get(attempt.attempt_id) || {};
    for (const [key, value] of Object.entries(base)) {
      if (key !== "attempt_id") row[key] = value;
    }
    const pairs = (pairsByAttempt.get(attempt.attempt_id) || [])
      .slice()
      .sort((a, b) => numberValue(a.pair_index) - numberValue(b.pair_index));
    addCollectedSequenceFeatures(row, pairs);
    addCollectedDistributionFeatures(row, pairs);
    addCollectedHashedNgramFeatures(row, pairs);
    return row;
  });
}

function addCollectedSequenceFeatures(row, pairs) {
  const capped = pairs.slice(0, COLLECTED_SEQUENCE_CAP);
  row["sequence.length"] = pairs.length;
  row["sequence.capped_length"] = capped.length;
  row["sequence.unique_key_ratio"] = unique(capped.map((pair) => normalizedKeyCode(pair))).length / Math.max(capped.length, 1);
  const classCounts = new Map();
  for (let index = 0; index < capped.length; index += 1) {
    const pair = capped[index];
    row[`sequence.key_bucket_pos_${pad3(index)}`] = normalizedHashBucket(normalizedKeyCode(pair), HASH_BUCKETS);
    row[`sequence.hold_pos_${pad3(index)}`] = numberValue(pair.hold_ms);
    if (index > 0) {
      row[`digraph.down_down_pos_${pad3(index)}`] = numberValue(pair.down_down_ms);
      row[`digraph.up_down_pos_${pad3(index)}`] = numberValue(pair.up_down_ms);
      row[`digraph.up_up_pos_${pad3(index)}`] = numberValue(pair.up_up_ms);
    }
    const keyClass = classifyKey(pair);
    classCounts.set(keyClass, (classCounts.get(keyClass) || 0) + 1);
  }
  for (const keyClass of ["letter", "digit", "space", "modifier", "control", "punctuation", "other"]) {
    row[`sequence.key_class_${keyClass}_ratio`] = (classCounts.get(keyClass) || 0) / Math.max(capped.length, 1);
  }
}

function addCollectedDistributionFeatures(row, pairs) {
  const hold = pairs.map((pair) => finiteNumber(pair.hold_ms)).filter(Number.isFinite);
  const downDown = pairs.map((pair) => finiteNumber(pair.down_down_ms)).filter(Number.isFinite);
  const upDown = pairs.map((pair) => finiteNumber(pair.up_down_ms)).filter(Number.isFinite);
  const upUp = pairs.map((pair) => finiteNumber(pair.up_up_ms)).filter(Number.isFinite);
  const trigraphDuration = [];
  for (let index = 2; index < pairs.length; index += 1) {
    const start = finiteNumber(pairs[index - 2].down_time_ms);
    const end = finiteNumber(pairs[index].up_time_ms);
    if (Number.isFinite(start) && Number.isFinite(end)) {
      trigraphDuration.push(end - start);
    }
  }
  addStatsToRow(row, "timing_distribution.hold", hold);
  addStatsToRow(row, "timing_distribution.down_down", downDown);
  addStatsToRow(row, "timing_distribution.up_down", upDown);
  addStatsToRow(row, "timing_distribution.up_up", upUp);
  addStatsToRow(row, "timing_distribution.trigraph_duration", trigraphDuration);
}

function addCollectedHashedNgramFeatures(row, pairs) {
  const unigram = makeBucketAccumulator();
  const digraph = makeBucketAccumulator();
  const trigraph = makeBucketAccumulator();
  for (let index = 0; index < pairs.length; index += 1) {
    const key = normalizedKeyCode(pairs[index]);
    const keyBucket = hashBucket(key, HASH_BUCKETS);
    addBucketValue(unigram, keyBucket, numberValue(pairs[index].hold_ms));
    if (index > 0) {
      const previous = normalizedKeyCode(pairs[index - 1]);
      const bucket = hashBucket(`${previous}>${key}`, HASH_BUCKETS);
      addBucketValue(digraph, bucket, numberValue(pairs[index].down_down_ms));
    }
    if (index > 1) {
      const first = normalizedKeyCode(pairs[index - 2]);
      const second = normalizedKeyCode(pairs[index - 1]);
      const bucket = hashBucket(`${first}>${second}>${key}`, HASH_BUCKETS);
      const start = finiteNumber(pairs[index - 2].down_time_ms);
      const end = finiteNumber(pairs[index].up_time_ms);
      addBucketValue(trigraph, bucket, Number.isFinite(start) && Number.isFinite(end) ? end - start : 0);
    }
  }
  flushBucketAccumulator(row, "key_sequence.unigram", unigram);
  flushBucketAccumulator(row, "digraph.timing", digraph);
  flushBucketAccumulator(row, "trigraph.timing", trigraph);
}

function makeBucketAccumulator() {
  return Array.from({ length: HASH_BUCKETS }, () => ({ count: 0, sum: 0 }));
}

function addBucketValue(accumulator, bucket, value) {
  const slot = accumulator[bucket] || accumulator[0];
  slot.count += 1;
  slot.sum += Number.isFinite(value) ? value : 0;
}

function flushBucketAccumulator(row, prefix, accumulator) {
  for (let bucket = 0; bucket < accumulator.length; bucket += 1) {
    const slot = accumulator[bucket];
    if (!slot.count) continue;
    const id = `b${String(bucket).padStart(2, "0")}`;
    row[`${prefix}.${id}.count`] = slot.count;
    row[`${prefix}.${id}.mean_ms`] = slot.sum / slot.count;
  }
}

function normalizedKeyCode(pair) {
  return String(pair.code || pair.key_value || "unknown").trim() || "unknown";
}

function classifyKey(pair) {
  const code = normalizedKeyCode(pair);
  if (/^Key[A-Z]$/.test(code)) return "letter";
  if (/^Digit\d$|^Numpad\d$/.test(code)) return "digit";
  if (code === "Space") return "space";
  if (/Shift|Control|Alt|Meta/.test(code)) return "modifier";
  if (/Enter|Backspace|Delete|Tab|Escape|Arrow/.test(code)) return "control";
  if (/Comma|Period|Slash|Quote|Bracket|Minus|Equal|Semicolon|Backquote/.test(code)) return "punctuation";
  return "other";
}

function applyAdaptiveFeatureSelection(split) {
  const width = split.trainX[0]?.length || 0;
  if (!width) return split;
  const positiveIndices = split.trainY.map((label, index) => (label === 1 ? index : -1)).filter((index) => index >= 0);
  const negativeIndices = split.trainY.map((label, index) => (label === 0 ? index : -1)).filter((index) => index >= 0);
  if (!positiveIndices.length || !negativeIndices.length) return split;
  const scored = [];
  for (let feature = 0; feature < width; feature += 1) {
    const positive = statsForIndices(split.trainX, positiveIndices, feature);
    const negative = statsForIndices(split.trainX, negativeIndices, feature);
    const separation = Math.abs(positive.mean - negative.mean) / (positive.std + negative.std + 1e-6);
    const stability = 1 / (positive.std + 1e-6);
    const variance = positive.std + negative.std;
    const score = Number.isFinite(separation + 0.02 * stability) && variance > 0
      ? separation + 0.02 * stability
      : 0;
    scored.push({
      feature,
      score,
      center: positive.mean,
      scale: Math.max(positive.std, 1e-6)
    });
  }
  const selected = scored
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.min(ADAPTIVE_FEATURE_CAP, width))
    .sort((a, b) => a.feature - b.feature);
  if (!selected.length) return split;
  const transform = (row) => selected.map((item) => ((row[item.feature] || 0) - item.center) / item.scale);
  return {
    ...split,
    trainX: split.trainX.map(transform),
    testX: split.testX.map(transform),
    sequenceShape: [selected.length, 1],
    adaptiveSelectedFeatureCount: selected.length,
    adaptiveFeatureNames: selected.map((item) => split.featureNames[item.feature] || `feature_${item.feature}`)
  };
}

function statsForIndices(matrix, indices, feature) {
  let sum = 0;
  for (const index of indices) {
    sum += matrix[index][feature] || 0;
  }
  const meanValue = sum / Math.max(indices.length, 1);
  let variance = 0;
  for (const index of indices) {
    variance += ((matrix[index][feature] || 0) - meanValue) ** 2;
  }
  return {
    mean: meanValue,
    std: Math.sqrt(variance / Math.max(indices.length - 1, 1))
  };
}

function addStatsToRow(row, prefix, values) {
  const names = statsNames(prefix);
  const stats = statsValues(values);
  for (let index = 0; index < names.length; index += 1) {
    row[names[index]] = stats[index];
  }
}

function statsNames(prefix) {
  return [
    `${prefix}.count`,
    `${prefix}.mean`,
    `${prefix}.std`,
    `${prefix}.min`,
    `${prefix}.p10`,
    `${prefix}.p25`,
    `${prefix}.median`,
    `${prefix}.p75`,
    `${prefix}.p90`,
    `${prefix}.max`,
    `${prefix}.iqr`,
    `${prefix}.cv`
  ];
}

function statsValues(values) {
  const clean = values.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  if (!clean.length) {
    return Array(statsNames("x").length).fill(0);
  }
  const avg = mean(clean);
  let variance = 0;
  for (const value of clean) {
    variance += (value - avg) ** 2;
  }
  const std = Math.sqrt(variance / Math.max(clean.length - 1, 1));
  const p25 = percentile(clean, 0.25);
  const p75 = percentile(clean, 0.75);
  return [
    clean.length,
    avg,
    std,
    clean[0],
    percentile(clean, 0.10),
    p25,
    percentile(clean, 0.50),
    p75,
    percentile(clean, 0.90),
    clean[clean.length - 1],
    p75 - p25,
    avg === 0 ? 0 : std / Math.abs(avg)
  ];
}

function percentile(sortedValues, p) {
  if (!sortedValues.length) return 0;
  const index = (sortedValues.length - 1) * p;
  const low = Math.floor(index);
  const high = Math.ceil(index);
  if (low === high) return sortedValues[low];
  const weight = index - low;
  return sortedValues[low] * (1 - weight) + sortedValues[high] * weight;
}

function finiteNumber(value) {
  if (value === "" || value === null || value === undefined) return Number.NaN;
  const number = Number(value);
  return Number.isFinite(number) ? number : Number.NaN;
}

function hashBucket(value, bucketCount) {
  let hash = 2166136261;
  const text = String(value);
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) % bucketCount;
}

function normalizedHashBucket(value, bucketCount) {
  return hashBucket(value, bucketCount) / Math.max(bucketCount - 1, 1);
}

function pad3(value) {
  return String(value).padStart(3, "0");
}

function normalizeFeatureMode(value) {
  return value === FEATURE_MODE_ENRICHED ? FEATURE_MODE_ENRICHED : FEATURE_MODE_CURRENT;
}

function baseDatasetKey(value) {
  const text = String(value || "");
  return text.replace(/_enriched_features$/, "");
}

function outputDatasetKey(dataset, featureMode) {
  const base = baseDatasetKey(dataset);
  return normalizeFeatureMode(featureMode) === FEATURE_MODE_ENRICHED ? `${base}_enriched_features` : base;
}

async function buildEnvironment(base, options) {
  const capabilities = await detectCapabilities();
  const env = {
    ...base,
    deviceLabel: options.deviceLabel || base.deviceLabel || "not_available",
    timestamp: new Date().toISOString(),
    tfjsBackend: tf.getBackend(),
    tfjsVersion: tf.version?.tfjs || "not_available",
    workerType: "module_worker",
    ...capabilities,
    webglAvailable: base.webglAvailable ?? capabilities.webglAvailable
  };
  env.environmentId = await stableHash(JSON.stringify({
    deviceLabel: env.deviceLabel,
    userAgent: env.userAgent,
    platform: env.platform,
    hardwareConcurrency: env.hardwareConcurrency,
    deviceMemory: env.deviceMemory,
    tfjsBackend: env.tfjsBackend,
    screen: env.screen,
    viewport: env.viewport
  }));
  return env;
}

async function detectCapabilities() {
  const wasmAvailable = typeof WebAssembly !== "undefined";
  return {
    wasmAvailable,
    wasmSimdAvailable: wasmAvailable ? await validateWasmFeature("simd") : false,
    wasmThreadsAvailable: typeof SharedArrayBuffer !== "undefined" && Boolean(self.crossOriginIsolated),
    webglAvailable: false,
    crossOriginIsolated: Boolean(self.crossOriginIsolated)
  };
}

async function validateWasmFeature(feature) {
  if (feature !== "simd") return false;
  const simdModule = new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 26, 11
  ]);
  try {
    return WebAssembly.validate(simdModule);
  } catch {
    return false;
  }
}

async function fetchText(options, source, fileName) {
  const response = await fetch(`/api/admin/evaluation/data/${source}/${encodeURIComponent(fileName)}`, {
    headers: { "x-admin-pin": options.adminPin || "" }
  });
  if (!response.ok) throw new Error(`Failed to load ${source}/${fileName}: ${response.status}`);
  return response.text();
}

async function fetchBuffer(options, source, fileName) {
  const response = await fetch(`/api/admin/evaluation/data/${source}/${encodeURIComponent(fileName)}`, {
    headers: { "x-admin-pin": options.adminPin || "" }
  });
  if (!response.ok) throw new Error(`Failed to load ${source}/${fileName}: ${response.status}`);
  return response.arrayBuffer();
}

async function sha256Text(text) {
  return sha256Buffer(new TextEncoder().encode(text).buffer);
}

async function sha256Buffer(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer.slice(0));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function stableHash(text) {
  return (await sha256Text(text)).slice(0, 16);
}

function textArtifact(name, content) {
  return { name, content: content || "", encoding: "utf8" };
}

function progress(stage, message) {
  self.postMessage({ type: "progress", stage, message, at: new Date().toISOString() });
}

function browserMemory() {
  return performance.memory?.usedJSHeapSize ?? "not_available";
}

function asciiDecode(bytes) {
  return Array.from(bytes, (byte) => String.fromCharCode(byte)).join("");
}

function numberValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function truthyValue(value) {
  if (value === true) return true;
  return ["1", "true", "yes", "y"].includes(String(value ?? "").trim().toLowerCase());
}

function normalizedLower(value) {
  return String(value ?? "").trim().toLowerCase();
}

function canonicalCollectedSubject(value) {
  const text = String(value ?? "").trim();
  const match = /^(.+)_\d+$/.exec(text);
  return match ? match[1] : text;
}

function unique(values) {
  return Array.from(new Set(values));
}

function mean(values) {
  const clean = values.filter(Number.isFinite);
  return clean.length ? clean.reduce((sum, value) => sum + value, 0) / clean.length : 0;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function clampNumber(value, min, max) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(Math.max(value, min), max);
}

function subjectSort(a, b) {
  const na = Number(a);
  const nb = Number(b);
  if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb;
  return String(a).localeCompare(String(b));
}

function columnMean(X, feature) {
  let sum = 0;
  for (const row of X) sum += row[feature] || 0;
  return sum / Math.max(X.length, 1);
}

function featureVariances(X) {
  const width = X[0].length;
  const variances = [];
  for (let feature = 0; feature < width; feature += 1) {
    const avg = columnMean(X, feature);
    let variance = 0;
    for (const row of X) variance += ((row[feature] || 0) - avg) ** 2;
    variances.push({ feature, variance });
  }
  return variances;
}

function seededUnit(a, b) {
  let value = Math.imul(a | 0, 0x45d9f3b) ^ Math.imul(b | 0, 0x119de1f3);
  value ^= value >>> 16;
  value = Math.imul(value, 0x45d9f3b);
  value ^= value >>> 16;
  return ((value >>> 0) / 0xffffffff);
}

function groupBy(rows, key) {
  const map = new Map();
  for (const row of rows) {
    const value = row[key] || "";
    if (!map.has(value)) map.set(value, []);
    map.get(value).push(row);
  }
  return map;
}
