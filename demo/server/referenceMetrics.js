import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const demoRoot = path.resolve(__dirname, "..");
const workspaceRoot = path.resolve(demoRoot, "..");

const SUMMARY_FILES = [
  ["dsl", path.join(workspaceRoot, "benchmark_outputs/summary_metrics.csv")],
  ["mmcTiming", path.join(workspaceRoot, "benchmark_outputs_mmc_web_temporal/summary_metrics.csv")]
];

export function loadReferenceMetrics() {
  const result = {};
  for (const [key, filePath] of SUMMARY_FILES) {
    result[key] = existsSync(filePath) ? parseSummaryCsv(readFileSync(filePath, "utf8")) : [];
  }
  return result;
}

function parseSummaryCsv(text) {
  const rows = parseCsv(text);
  return rows.map((row) => ({
    model: row.Model,
    far: toNumber(row.FAR),
    frr: toNumber(row.FRR),
    eer: toNumber(row.EER),
    accuracy: toNumber(row.Accuracy),
    auc: toNumber(row.AUC),
    inferenceTimeMs: toNumber(row["Inference Time (ms/sample)"]),
    uiBlockingTimeMs: toNumber(row["UI Blocking Time (ms/test batch)"]),
    memoryMb: toNumber(row["Peak Memory During Inference (MB)"]),
    minTrainingSamples: row["Min Genuine Samples for EER < 10%"]
  }));
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) {
    return [];
  }
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function splitCsvLine(line) {
  const values = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"' && line[index + 1] === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  return values;
}

function toNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
