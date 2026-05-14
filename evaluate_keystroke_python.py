#!/usr/bin/env python3
"""Run keystroke model evaluations locally in Python.

The browser evaluator is still the reference for device-dependent latency.
This script is for faster local model/feature experiments on the same datasets.
It preserves the temporal rule: reference/past samples train, future samples test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import resource
import sys
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / "keystroke_python_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional at runtime
    lgb = None

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception:  # pragma: no cover - optional at runtime
    roc_auc_score = None
    roc_curve = None

tf = None
plt = None


ROOT = Path(__file__).resolve().parent
DSL_FILE = ROOT / "DSL-StrongPasswordData.csv"
MMC_DIR = ROOT / "ScienceDirect_files_20Apr2026_10-05-23.390"
COLLECTED_DIR = ROOT / "demo" / "web_demo_data"
MMC_PAIRS = [
    ("mmc1.npy", "mmc5.npy", "reference"),
    ("mmc2.npy", "mmc6.npy", "reference"),
    ("mmc3.npy", "mmc7.npy", "test"),
    ("mmc4.npy", "mmc8.npy", "test"),
]
MMC_TIMING_CHANNELS = [14, 15, 16, 17, 18, 19]
EER_TARGET = 0.10
HASH_BUCKETS = 64
COLLECTED_SEQUENCE_CAP = 180
ADAPTIVE_FEATURE_CAP = 384


@dataclass
class Dataset:
    key: str
    label: str
    feature_mode: str
    feature_set: str
    x: np.ndarray
    subjects: np.ndarray
    partitions: np.ndarray
    orders: np.ndarray
    feature_names: list[str]
    sequence_shape: tuple[int, int]
    file_hashes: dict[str, str]
    data_efficiency_sizes: list[int]
    adaptive: bool
    extra_artifacts: dict[str, pd.DataFrame] | None = None


@dataclass
class Split:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    train_order: np.ndarray
    sequence_shape: tuple[int, int]
    adaptive_selected_feature_count: int | str = "not_enabled"


def main() -> int:
    args = parse_args()
    datasets = expand_choice(args.dataset, ["dsl", "mmc", "collected"])
    feature_modes = expand_choice(args.feature_mode, ["current", "enriched"])
    models = expand_choice(args.models, ["manhattan", "lightgbm", "cnn"])

    run_id = args.run_id or make_run_id()
    run_root = Path(args.output_dir) / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "runId": run_id,
        "createdAt": utc_now(),
        "script": str(Path(__file__).name),
        "temporalRule": "reference/past samples train; future/test samples evaluate",
        "python": sys.version,
        "platform": platform.platform(),
        "models": models,
        "datasets": datasets,
        "featureModes": feature_modes,
        "notes": "Python-side model/feature evaluation. Browser latency should still be measured in browser.",
    }
    write_json(run_root / "run_manifest.json", manifest)

    for dataset_key in datasets:
        for feature_mode in feature_modes:
            dataset = load_dataset(dataset_key, feature_mode)
            if args.max_subjects:
                dataset = limit_subjects(dataset, args.max_subjects)
            output_key = output_dataset_key(dataset.key, dataset.feature_mode)
            output_dir = run_root / output_key
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[dataset] {output_key}: {len(dataset.x)} samples, {len(unique_sorted(dataset.subjects))} subjects, {len(dataset.feature_names)} features")
            run_dataset_evaluation(dataset, models, output_dir, args)

    print(f"[done] results saved to {run_root}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Python keystroke evaluation")
    parser.add_argument("--dataset", default="collected", choices=["dsl", "mmc", "collected", "all"], help="Dataset to evaluate")
    parser.add_argument("--feature-mode", default="enriched", choices=["current", "enriched", "both"], help="Feature set to use")
    parser.add_argument("--models", default="manhattan,lightgbm", help="Comma list: manhattan,lightgbm,cnn,all")
    parser.add_argument("--output-dir", default="evaluation_runs_python", help="Output root directory")
    parser.add_argument("--run-id", default="", help="Optional run id")
    parser.add_argument("--max-subjects", type=int, default=0, help="Optional smoke-test subject limit")
    parser.add_argument("--skip-data-efficiency", action="store_true", help="Skip minimum training data experiment")
    parser.add_argument("--cnn-epochs", type=int, default=4, help="CNN epochs")
    parser.add_argument("--cnn-batch-size", type=int, default=128, help="CNN batch size")
    parser.add_argument("--lightgbm-estimators", type=int, default=120, help="LightGBM boosting rounds")
    parser.add_argument("--lightgbm-n-jobs", type=int, default=max(os.cpu_count() or 1, 1), help="LightGBM threads")
    parser.add_argument("--eer-target", type=float, default=EER_TARGET, help="Stop data-efficiency after mean EER is below this")
    return parser.parse_args()


def expand_choice(value: str, all_values: list[str]) -> list[str]:
    if value in {"all", "both"}:
        return all_values
    values = [part.strip() for part in value.split(",") if part.strip()]
    if "all" in values:
        return all_values
    unknown = sorted(set(values) - set(all_values))
    if unknown:
        raise SystemExit(f"Unknown choice(s): {', '.join(unknown)}")
    return values


def load_dataset(dataset_key: str, feature_mode: str) -> Dataset:
    if dataset_key == "dsl":
        return load_dsl(feature_mode)
    if dataset_key == "mmc":
        return load_mmc(feature_mode)
    if dataset_key == "collected":
        return load_collected(feature_mode)
    raise ValueError(dataset_key)


def load_dsl(feature_mode: str) -> Dataset:
    df = pd.read_csv(DSL_FILE)
    base_names = [col for col in df.columns if col.startswith(("H.", "DD.", "UD."))]
    x_base = df[base_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    feature_names = list(base_names)
    x = x_base
    feature_set = "dsl_current_h_dd_ud"
    if feature_mode == "enriched":
        extra_df = build_dsl_enriched_features(df, base_names)
        feature_names = base_names + list(extra_df.columns)
        x = np.column_stack([x_base, extra_df.to_numpy(dtype=np.float32)])
        feature_set = "dsl_current_h_dd_ud_plus_sequence_trigraph_distribution_adaptive"

    sessions = pd.to_numeric(df["sessionIndex"], errors="coerce").fillna(0).to_numpy()
    return Dataset(
        key="dsl",
        label="DSL",
        feature_mode=feature_mode,
        feature_set=feature_set,
        x=finite_matrix(x),
        subjects=df["subject"].astype(str).to_numpy(),
        partitions=np.where(sessions <= 4, "reference", "test"),
        orders=np.arange(len(df), dtype=np.float64),
        feature_names=feature_names,
        sequence_shape=(x.shape[1], 1),
        file_hashes={DSL_FILE.name: sha256_file(DSL_FILE)},
        data_efficiency_sizes=[5, 10, 25, 50, 100, 200],
        adaptive=feature_mode == "enriched",
    )


def build_dsl_enriched_features(df: pd.DataFrame, base_names: list[str]) -> pd.DataFrame:
    key_sequence = [name[2:] for name in base_names if name.startswith("H.")]
    out: dict[str, np.ndarray] = {}
    for index, key in enumerate(key_sequence):
        out[f"sequence.key_bucket_pos_{index:03d}"] = np.full(len(df), normalized_hash_bucket(key), dtype=np.float32)

    groups = {
        "timing_distribution.hold": [name for name in base_names if name.startswith("H.")],
        "timing_distribution.down_down": [name for name in base_names if name.startswith("DD.")],
        "timing_distribution.up_down": [name for name in base_names if name.startswith("UD.")],
    }
    for prefix, cols in groups.items():
        stats = row_stats(df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
        for name, values in stats.items():
            out[f"{prefix}.{name}"] = values

    for index in range(max(len(key_sequence) - 2, 0)):
        a, b, c = key_sequence[index : index + 3]
        dd_sum = numeric_series(df.get(f"DD.{a}.{b}", 0), len(df)) + numeric_series(df.get(f"DD.{b}.{c}", 0), len(df))
        ud_sum = numeric_series(df.get(f"UD.{a}.{b}", 0), len(df)) + numeric_series(df.get(f"UD.{b}.{c}", 0), len(df))
        out[f"trigraph.{a}.{b}.{c}.down_down_sum"] = dd_sum
        out[f"trigraph.{a}.{b}.{c}.up_down_sum"] = ud_sum
    return pd.DataFrame(out)


def load_mmc(feature_mode: str) -> Dataset:
    samples: list[np.ndarray] = []
    subjects: list[str] = []
    partitions: list[str] = []
    orders: list[float] = []
    hashes: dict[str, str] = {}
    pair_rows = []
    order_base = 0
    for feature_file, label_file, partition in MMC_PAIRS:
        feature_path = MMC_DIR / feature_file
        label_path = MMC_DIR / label_file
        hashes[feature_file] = sha256_file(feature_path)
        hashes[label_file] = sha256_file(label_path)
        raw = np.load(feature_path, allow_pickle=True)
        labels = np.load(label_path, allow_pickle=True)
        timing = object_timing_to_float(raw[:, :, MMC_TIMING_CHANNELS]) / 1_000_000.0
        flat = timing.reshape(timing.shape[0], -1).astype(np.float32)
        if feature_mode == "enriched":
            x_pair = np.column_stack([flat, build_mmc_enriched_features(timing)])
        else:
            x_pair = flat
        samples.append(x_pair)
        subjects.extend(labels[:, 0].astype(str).tolist())
        partitions.extend([partition] * len(labels))
        orders.extend((order_base + np.arange(len(labels))).astype(float).tolist())
        pair_rows.append({
            "feature_file": feature_file,
            "label_file": label_file,
            "partition": partition,
            "samples": len(labels),
            "sequence_length": timing.shape[1],
            "timing_channels": "|".join(map(str, MMC_TIMING_CHANNELS)),
        })
        order_base += 1_000_000

    x = np.vstack(samples).astype(np.float32)
    base_names = [f"sequence.t{t:03d}.timing_channel_{ch}" for t in range(642) for ch in MMC_TIMING_CHANNELS]
    feature_names = base_names
    feature_set = "mmc_flat_timing_channels_14_19"
    sequence_shape = (642, len(MMC_TIMING_CHANNELS))
    if feature_mode == "enriched":
        feature_names = base_names + mmc_enriched_feature_names()
        feature_set = "mmc_flat_timing_channels_plus_distribution_delta_trigraph_adaptive"
        sequence_shape = (x.shape[1], 1)

    return Dataset(
        key="mmc",
        label="MMC",
        feature_mode=feature_mode,
        feature_set=feature_set,
        x=finite_matrix(x),
        subjects=np.asarray(subjects),
        partitions=np.asarray(partitions),
        orders=np.asarray(orders, dtype=np.float64),
        feature_names=feature_names,
        sequence_shape=sequence_shape,
        file_hashes=hashes,
        data_efficiency_sizes=[1, 2, 5, 10, 20],
        adaptive=feature_mode == "enriched",
        extra_artifacts={"pair_manifest.csv": pd.DataFrame(pair_rows)},
    )


def object_timing_to_float(values: np.ndarray) -> np.ndarray:
    flat = pd.to_numeric(pd.Series(values.reshape(-1)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return flat.reshape(values.shape)


def build_mmc_enriched_features(timing: np.ndarray) -> np.ndarray:
    blocks = [
        np.full((timing.shape[0], 1), timing.shape[1], dtype=np.float32),
        np.full((timing.shape[0], 1), timing.shape[2], dtype=np.float32),
        stats_matrix(timing.reshape(timing.shape[0], -1)),
    ]
    for channel_index in range(timing.shape[2]):
        values = timing[:, :, channel_index]
        deltas = np.diff(values, axis=1)
        curvature = np.diff(values, n=2, axis=1)
        blocks.extend([stats_matrix(values), stats_matrix(deltas), stats_matrix(curvature)])
    return np.column_stack(blocks).astype(np.float32)


def mmc_enriched_feature_names() -> list[str]:
    names = ["sequence_length", "timing_channel_count"]
    names.extend(stats_names("timing_distribution.all_channels"))
    for channel in MMC_TIMING_CHANNELS:
        names.extend(stats_names(f"timing_distribution.channel_{channel}"))
        names.extend(stats_names(f"digraph_delta.channel_{channel}"))
        names.extend(stats_names(f"trigraph_curvature.channel_{channel}"))
    return names


def load_collected(feature_mode: str) -> Dataset:
    attempts = read_collected_csv("attempts.csv")
    participants = read_collected_csv("participants.csv")
    sessions = read_collected_csv("sessions.csv")
    features = read_collected_csv("features.csv")
    events = read_collected_csv("events.csv")

    canonical = build_collected_attempts(attempts, participants, sessions)
    event_pairs = build_collected_event_pairs(events)
    current_features = build_collected_feature_matrix(features)
    feature_matrix = current_features
    feature_set = "collected_stored_browser_baseline_features"
    extra_artifacts = {
        "collected_attempts_canonical.csv": canonical,
        "collected_event_pairs.csv": event_pairs,
        "collected_feature_matrix.csv": current_features,
        "collected_quality_summary.csv": build_collected_quality_summary(canonical, event_pairs),
        "collected_filter_summary.csv": build_collected_filter_summary(canonical),
    }
    if feature_mode == "enriched":
        feature_matrix = build_collected_enriched_feature_matrix(canonical, event_pairs, current_features)
        extra_artifacts["collected_enriched_feature_matrix.csv"] = feature_matrix
        feature_set = "collected_stored_features_plus_key_sequence_digraph_trigraph_distribution_adaptive"

    usable = canonical[collected_training_mask(canonical)].copy()
    merged = usable.merge(feature_matrix, left_on="attempt_id", right_on="attempt_id", how="inner")
    subject_col = merged["participant_code"].where(merged["participant_code"].astype(str).str.len() > 0, merged["participant_id"])
    meta_cols = set(canonical.columns) | {"attempt_id"}
    feature_names = [col for col in feature_matrix.columns if col not in meta_cols and col != "attempt_id"]
    x = merged[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    nonzero = np.any(x != 0, axis=1)
    merged = merged.loc[nonzero].copy()
    x = x[nonzero]
    subject_col = subject_col.loc[nonzero]

    partitions = np.empty(len(merged), dtype=object)
    orders = np.empty(len(merged), dtype=np.float64)
    keep = np.zeros(len(merged), dtype=bool)
    for _, idx in merged.groupby(subject_col).groups.items():
        subject_indices = list(idx)
        subject_indices.sort(key=lambda i: timestamp_order(merged.at[i, "timestamp"]))
        if len(subject_indices) < 2:
            continue
        reference_count = min(len(subject_indices) - 1, max(1, math.floor(len(subject_indices) * 0.6)))
        for offset, row_index in enumerate(subject_indices):
            pos = merged.index.get_loc(row_index)
            partitions[pos] = "reference" if offset < reference_count else "test"
            orders[pos] = timestamp_order(merged.at[row_index, "timestamp"]) or float(offset)
            keep[pos] = True

    x = x[keep]
    subjects = subject_col.to_numpy()[keep].astype(str)
    partitions = partitions[keep].astype(str)
    orders = orders[keep]

    hashes = {name: sha256_file(COLLECTED_DIR / name) for name in [
        "participants.csv",
        "sessions.csv",
        "attempts.csv",
        "events.csv",
        "features.csv",
        "results.csv",
        "monitoring_windows.csv",
        "model_calibrations.csv",
    ] if (COLLECTED_DIR / name).exists()}

    return Dataset(
        key="collected_data_analysis",
        label="Collected CSV",
        feature_mode=feature_mode,
        feature_set=feature_set,
        x=finite_matrix(x),
        subjects=subjects,
        partitions=partitions,
        orders=orders,
        feature_names=feature_names,
        sequence_shape=(x.shape[1], 1),
        file_hashes=hashes,
        data_efficiency_sizes=[1, 2, 3, 5, 10],
        adaptive=feature_mode == "enriched",
        extra_artifacts=extra_artifacts,
    )


def read_collected_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(COLLECTED_DIR / name, encoding="utf-8-sig")


def build_collected_attempts(attempts: pd.DataFrame, participants: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    participant_map = participants.set_index("id") if "id" in participants.columns else pd.DataFrame()
    session_map = sessions.set_index("id") if "id" in sessions.columns else pd.DataFrame()
    rows = []
    for _, attempt in attempts.iterrows():
        participant = participant_map.loc[attempt.get("participant_id")] if attempt.get("participant_id") in participant_map.index else {}
        session = session_map.loc[attempt.get("session_id")] if attempt.get("session_id") in session_map.index else {}
        rows.append({
            "attempt_id": attempt.get("id", ""),
            "participant_id": attempt.get("participant_id", ""),
            "participant_code": safe_get(participant, "participant_code", ""),
            "session_id": attempt.get("session_id", ""),
            "session_no": safe_get(session, "session_no", ""),
            "input_mode": attempt.get("input_mode", ""),
            "role_label": attempt.get("role_label", ""),
            "status": attempt.get("status", ""),
            "quality_status": attempt.get("quality_status", ""),
            "summary_quality_status": attempt.get("summary_qualityStatus", ""),
            "feature_quality": attempt.get("feature_quality", ""),
            "summary_feature_quality": attempt.get("summary_featureQuality", ""),
            "device_class": attempt.get("device_class", safe_get(session, "device_class", "")),
            "prompt_id": attempt.get("prompt_id", ""),
            "prompt_set_id": attempt.get("prompt_set_id", ""),
            "started_at": attempt.get("started_at", ""),
            "ended_at": attempt.get("ended_at", ""),
            "submitted_at": attempt.get("submitted_at", ""),
            "timestamp": attempt.get("submitted_at", "") or attempt.get("ended_at", "") or attempt.get("started_at", ""),
            "raw_text_length": attempt.get("summary_rawTextLength", len(str(attempt.get("raw_text", "")))),
            "event_count": attempt.get("summary_eventCount", ""),
            "paired_key_count": attempt.get("summary_pairedKeyCount", ""),
            "keyup_coverage": attempt.get("summary_keyupCoverage", ""),
            "paste_count": attempt.get("paste_count", ""),
            "summary_paste_count": attempt.get("summary_pasteCount", ""),
            "paste_detected": boolish(attempt.get("summary_pasteDetected", False))
            or num(attempt.get("paste_count", 0)) > 0
            or num(attempt.get("summary_pasteCount", 0)) > 0,
            "fixed_prompt_match": attempt.get("fixed_prompt_match", ""),
        })
    return pd.DataFrame(rows)


def build_collected_event_pairs(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attempt_id, group in events.groupby("attempt_id", sort=False):
        sorted_events = group.sort_values("relative_time")
        active: dict[str, list[pd.Series]] = {}
        pairs = []
        for _, event in sorted_events.iterrows():
            key = str(event.get("code") or event.get("key_value") or "unknown")
            event_type = event.get("event_type")
            if event_type == "keydown":
                active.setdefault(key, []).append(event)
            elif event_type == "keyup":
                stack = active.get(key, [])
                if stack:
                    down = stack.pop(0)
                    pairs.append((down, event))
        for index, (down, up) in enumerate(pairs):
            prev_down, prev_up = pairs[index - 1] if index > 0 else (None, None)
            down_time = num(down.get("relative_time"))
            up_time = num(up.get("relative_time"))
            rows.append({
                "attempt_id": attempt_id,
                "pair_index": index,
                "key_value": down.get("key_value", ""),
                "code": down.get("code", ""),
                "down_time_ms": down_time,
                "up_time_ms": up_time,
                "hold_ms": up_time - down_time,
                "down_down_ms": down_time - num(prev_down.get("relative_time")) if prev_down is not None else np.nan,
                "up_up_ms": up_time - num(prev_up.get("relative_time")) if prev_up is not None else np.nan,
                "up_down_ms": down_time - num(prev_up.get("relative_time")) if prev_up is not None else np.nan,
            })
    return pd.DataFrame(rows)


def build_collected_feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["attempt_id"])
    matrix = features.pivot_table(
        index="attempt_id",
        columns="feature_name",
        values="feature_value",
        aggfunc="first",
    ).reset_index()
    matrix.columns = [str(col) for col in matrix.columns]
    return matrix.fillna(0.0)


def build_collected_enriched_feature_matrix(
    canonical: pd.DataFrame,
    event_pairs: pd.DataFrame,
    current_features: pd.DataFrame,
) -> pd.DataFrame:
    base = current_features.set_index("attempt_id").to_dict("index") if not current_features.empty else {}
    grouped = {attempt_id: group.sort_values("pair_index") for attempt_id, group in event_pairs.groupby("attempt_id", sort=False)}
    rows = []
    for attempt_id in canonical["attempt_id"]:
        row = {"attempt_id": attempt_id}
        row.update(base.get(attempt_id, {}))
        pairs = grouped.get(attempt_id, pd.DataFrame())
        row.update(collected_sequence_features(pairs))
        row.update(collected_distribution_features(pairs))
        row.update(collected_hashed_ngram_features(pairs))
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def collected_sequence_features(pairs: pd.DataFrame) -> dict[str, float]:
    row: dict[str, float] = {
        "sequence.length": float(len(pairs)),
        "sequence.capped_length": float(min(len(pairs), COLLECTED_SEQUENCE_CAP)),
    }
    if pairs.empty:
        row["sequence.unique_key_ratio"] = 0.0
        return row
    capped = pairs.head(COLLECTED_SEQUENCE_CAP).reset_index(drop=True)
    keys = [normalized_key_code(record) for _, record in capped.iterrows()]
    row["sequence.unique_key_ratio"] = len(set(keys)) / max(len(keys), 1)
    class_counts = {name: 0 for name in ["letter", "digit", "space", "modifier", "control", "punctuation", "other"]}
    for index, record in capped.iterrows():
        row[f"sequence.key_bucket_pos_{index:03d}"] = normalized_hash_bucket(keys[index])
        row[f"sequence.hold_pos_{index:03d}"] = num(record.get("hold_ms"))
        if index > 0:
            row[f"digraph.down_down_pos_{index:03d}"] = num(record.get("down_down_ms"))
            row[f"digraph.up_down_pos_{index:03d}"] = num(record.get("up_down_ms"))
            row[f"digraph.up_up_pos_{index:03d}"] = num(record.get("up_up_ms"))
        class_counts[classify_key(keys[index])] += 1
    for key_class, count in class_counts.items():
        row[f"sequence.key_class_{key_class}_ratio"] = count / max(len(capped), 1)
    return row


def collected_distribution_features(pairs: pd.DataFrame) -> dict[str, float]:
    row: dict[str, float] = {}
    for prefix, column in [
        ("timing_distribution.hold", "hold_ms"),
        ("timing_distribution.down_down", "down_down_ms"),
        ("timing_distribution.up_down", "up_down_ms"),
        ("timing_distribution.up_up", "up_up_ms"),
    ]:
        add_stats(row, prefix, pd.to_numeric(pairs.get(column, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy())
    trigram = []
    if len(pairs) >= 3:
        reset = pairs.reset_index(drop=True)
        for index in range(2, len(reset)):
            start = num(reset.at[index - 2, "down_time_ms"])
            end = num(reset.at[index, "up_time_ms"])
            if np.isfinite(start) and np.isfinite(end):
                trigram.append(end - start)
    add_stats(row, "timing_distribution.trigraph_duration", np.asarray(trigram, dtype=np.float32))
    return row


def collected_hashed_ngram_features(pairs: pd.DataFrame) -> dict[str, float]:
    row: dict[str, float] = {}
    unigram = make_bucket_accumulator()
    digraph = make_bucket_accumulator()
    trigraph = make_bucket_accumulator()
    reset = pairs.reset_index(drop=True)
    keys = [normalized_key_code(record) for _, record in reset.iterrows()]
    for index, record in reset.iterrows():
        add_bucket_value(unigram, hash_bucket(keys[index]), num(record.get("hold_ms")))
        if index > 0:
            add_bucket_value(digraph, hash_bucket(f"{keys[index - 1]}>{keys[index]}"), num(record.get("down_down_ms")))
        if index > 1:
            start = num(reset.at[index - 2, "down_time_ms"])
            end = num(reset.at[index, "up_time_ms"])
            value = end - start if np.isfinite(start) and np.isfinite(end) else 0.0
            add_bucket_value(trigraph, hash_bucket(f"{keys[index - 2]}>{keys[index - 1]}>{keys[index]}"), value)
    flush_bucket_accumulator(row, "key_sequence.unigram", unigram)
    flush_bucket_accumulator(row, "digraph.timing", digraph)
    flush_bucket_accumulator(row, "trigraph.timing", trigraph)
    return row


def build_collected_quality_summary(canonical: pd.DataFrame, event_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = {attempt_id: group for attempt_id, group in event_pairs.groupby("attempt_id", sort=False)}
    for _, attempt in canonical.iterrows():
        pairs = grouped.get(attempt["attempt_id"], pd.DataFrame())
        rows.append({
            "attempt_id": attempt["attempt_id"],
            "participant_code": attempt["participant_code"],
            "input_mode": attempt["input_mode"],
            "role_label": attempt["role_label"],
            "status": attempt["status"],
            "quality_status": attempt["quality_status"],
            "summary_quality_status": attempt["summary_quality_status"],
            "feature_quality": attempt["feature_quality"],
            "device_class": attempt["device_class"],
            "event_count": attempt["event_count"],
            "paired_key_count": len(pairs),
            "hold_mean_ms": mean_array(pairs.get("hold_ms", pd.Series(dtype=float))),
            "down_down_mean_ms": mean_array(pairs.get("down_down_ms", pd.Series(dtype=float))),
            "up_down_mean_ms": mean_array(pairs.get("up_down_ms", pd.Series(dtype=float))),
            "raw_text_length": attempt["raw_text_length"],
            "paste_count": attempt["paste_count"],
            "summary_paste_count": attempt["summary_paste_count"],
            "paste_detected": attempt["paste_detected"],
            "fixed_prompt_match": attempt["fixed_prompt_match"],
        })
    return pd.DataFrame(rows)


def build_collected_filter_summary(canonical: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("total_attempts", pd.Series([True] * len(canonical), index=canonical.index)),
        ("submitted", canonical["status"].astype(str).str.lower().eq("submitted")),
        ("quality_status_usable", canonical["quality_status"].astype(str).str.lower().eq("usable")),
        ("summary_quality_status_usable_or_blank", canonical["summary_quality_status"].astype(str).str.lower().isin(["", "nan", "usable"])),
        ("not_low_feature_quality", ~canonical["feature_quality"].astype(str).str.lower().eq("low")),
        ("not_low_summary_feature_quality", ~canonical["summary_feature_quality"].astype(str).str.lower().eq("low")),
        ("paste_not_detected", ~canonical["paste_detected"].astype(bool)),
        ("included_for_training_and_test", collected_training_mask(canonical)),
    ]
    return pd.DataFrame({
        "filter": [name for name, _ in checks],
        "attempts": [int(mask.sum()) for _, mask in checks],
    })


def collected_training_mask(canonical: pd.DataFrame) -> pd.Series:
    status = canonical["status"].astype(str).str.lower()
    quality = canonical["quality_status"].astype(str).str.lower()
    summary_quality = canonical["summary_quality_status"].astype(str).str.lower()
    feature_quality = canonical["feature_quality"].astype(str).str.lower()
    summary_feature_quality = canonical["summary_feature_quality"].astype(str).str.lower()
    paste_detected = canonical["paste_detected"].astype(bool)
    return (
        status.eq("submitted")
        & quality.eq("usable")
        & summary_quality.isin(["", "nan", "usable"])
        & ~feature_quality.eq("low")
        & ~summary_feature_quality.eq("low")
        & ~paste_detected
    )


def run_dataset_evaluation(dataset: Dataset, models: list[str], output_dir: Path, args: argparse.Namespace) -> None:
    started = time.perf_counter()
    environment = build_environment(dataset, args)
    summary_rows = []
    userwise_rows = []
    timing_rows = []
    data_efficiency_rows = []
    roc_payload: dict[str, tuple[list[int], list[float]]] = {}
    subjects = unique_sorted(dataset.subjects)

    for model_key in models:
        if model_key == "lightgbm" and lgb is None:
            print("[skip] lightgbm package is not available")
            continue
        if model_key == "cnn" and package_version("tensorflow") == "not_available":
            print("[skip] tensorflow package is not available")
            continue
        model_name = model_display_name(model_key)
        all_y: list[int] = []
        all_scores: list[float] = []
        model_rows = []
        for index, subject in enumerate(subjects, start=1):
            print(f"[model] {dataset.label} {dataset.feature_mode}: {model_name} subject {index}/{len(subjects)}")
            split = make_auth_split(dataset, subject)
            if split is None:
                continue
            result = train_and_score(model_key, split, args)
            metrics = compute_metrics(split.test_y, result["scores"])
            row = {
                "dataset": output_dataset_key(dataset.key, dataset.feature_mode),
                "source_dataset": dataset.key,
                "feature_mode": dataset.feature_mode,
                "feature_set": dataset.feature_set,
                "model": model_name,
                "model_runtime": model_runtime(model_key),
                "subject": subject,
                "train_genuine_count": int(np.sum(split.train_y == 1)),
                "test_genuine_count": int(np.sum(split.test_y == 1)),
                "train_impostor_count": int(np.sum(split.train_y == 0)),
                "test_impostor_count": int(np.sum(split.test_y == 0)),
                "FAR": metrics["far"],
                "FRR": metrics["frr"],
                "EER": metrics["eer"],
                "Accuracy": metrics["accuracy"],
                "AUC": metrics["auc"],
                "threshold": metrics["threshold"],
                "train_time_ms": result["train_time_ms"],
                "inference_time_ms_per_sample": result["inference_time_ms_per_sample"],
                "ui_blocking_time_ms": result["inference_batch_ms"],
                "memory_before_bytes": result["memory_before_bytes"],
                "memory_after_bytes": result["memory_after_bytes"],
                "adaptive_selected_feature_count": split.adaptive_selected_feature_count,
            }
            userwise_rows.append(row)
            model_rows.append(row)
            timing_rows.append({
                "dataset": output_dataset_key(dataset.key, dataset.feature_mode),
                "source_dataset": dataset.key,
                "feature_mode": dataset.feature_mode,
                "model": model_name,
                "subject": subject,
                "phase": "train_and_infer",
                "elapsed_ms": result["train_time_ms"] + result["inference_batch_ms"],
                "train_time_ms": result["train_time_ms"],
                "inference_batch_ms": result["inference_batch_ms"],
                "inference_time_ms_per_sample": result["inference_time_ms_per_sample"],
                "memory_before_bytes": result["memory_before_bytes"],
                "memory_after_bytes": result["memory_after_bytes"],
                "adaptive_selected_feature_count": split.adaptive_selected_feature_count,
            })
            all_y.extend(split.test_y.astype(int).tolist())
            all_scores.extend(result["scores"].astype(float).tolist())

        aggregate = average_metric_rows(model_rows, dataset, model_key)
        if not args.skip_data_efficiency:
            efficiency = evaluate_data_efficiency(dataset, model_key, subjects, timing_rows, args)
            data_efficiency_rows.extend(efficiency)
            aggregate["Min Genuine Samples for EER < 10%"] = min_samples_for_target(efficiency, model_name, args.eer_target)
        summary_rows.append(aggregate)
        roc_payload[model_name] = (all_y, all_scores)

    timing_rows.append({
        "dataset": output_dataset_key(dataset.key, dataset.feature_mode),
        "source_dataset": dataset.key,
        "feature_mode": dataset.feature_mode,
        "subject": "all",
        "phase": "run",
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "completed_at": utc_now(),
    })

    write_json(output_dir / "environment_manifest.json", environment)
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(userwise_rows).to_csv(output_dir / "userwise_metrics.csv", index=False)
    pd.DataFrame(data_efficiency_rows).to_csv(output_dir / "data_efficiency_eer.csv", index=False)
    timing_frame = pd.DataFrame(timing_rows)
    timing_frame.to_csv(output_dir / "python_timing_log.csv", index=False)
    timing_frame.to_csv(output_dir / "browser_timing_log.csv", index=False)
    if dataset.extra_artifacts:
        for name, frame in dataset.extra_artifacts.items():
            frame.to_csv(output_dir / name, index=False)
    draw_charts(output_dir, pd.DataFrame(summary_rows), pd.DataFrame(data_efficiency_rows), roc_payload)


def make_auth_split(dataset: Dataset, genuine_subject: str, adaptive: bool = True) -> Split | None:
    train_mask = dataset.partitions == "reference"
    test_mask = dataset.partitions == "test"
    train_y = (dataset.subjects[train_mask] == genuine_subject).astype(np.int8)
    test_y = (dataset.subjects[test_mask] == genuine_subject).astype(np.int8)
    if not (np.any(train_y == 1) and np.any(train_y == 0) and np.any(test_y == 1) and np.any(test_y == 0)):
        return None
    split = Split(
        train_x=finite_matrix(dataset.x[train_mask]),
        train_y=train_y,
        test_x=finite_matrix(dataset.x[test_mask]),
        test_y=test_y,
        train_order=dataset.orders[train_mask],
        sequence_shape=dataset.sequence_shape,
    )
    if adaptive and dataset.adaptive:
        split = apply_adaptive_feature_selection(split)
    return split


def apply_adaptive_feature_selection(split: Split) -> Split:
    positive = finite_matrix(split.train_x[split.train_y == 1])
    negative = finite_matrix(split.train_x[split.train_y == 0])
    if len(positive) == 0 or len(negative) == 0:
        return split
    pos_mean = finite_vector(positive.mean(axis=0))
    neg_mean = finite_vector(negative.mean(axis=0))
    pos_std = safe_column_std(positive)
    neg_std = safe_column_std(negative)
    score = np.abs(pos_mean - neg_mean) / (pos_std + neg_std + 1e-6) + 0.02 / (pos_std + 1e-6)
    score = finite_vector(score)
    count = min(ADAPTIVE_FEATURE_CAP, split.train_x.shape[1])
    selected = np.argsort(score)[-count:]
    selected.sort()
    center = finite_vector(pos_mean[selected])
    scale = np.maximum(finite_vector(pos_std[selected]), 1e-6)
    train_x = finite_matrix((split.train_x[:, selected] - center) / scale)
    test_x = finite_matrix((split.test_x[:, selected] - center) / scale)
    return Split(
        train_x=train_x,
        train_y=split.train_y,
        test_x=test_x,
        test_y=split.test_y,
        train_order=split.train_order,
        sequence_shape=(len(selected), 1),
        adaptive_selected_feature_count=len(selected),
    )


def train_and_score(model_key: str, split: Split, args: argparse.Namespace) -> dict[str, np.ndarray | float | int]:
    if model_key == "manhattan":
        return train_and_score_manhattan(split)
    if model_key == "lightgbm":
        return train_and_score_lightgbm(split, args)
    if model_key == "cnn":
        return train_and_score_cnn(split, args)
    raise ValueError(model_key)


def train_and_score_manhattan(split: Split) -> dict[str, np.ndarray | float | int]:
    memory_before = rss_bytes()
    start_train = time.perf_counter()
    genuine = finite_matrix(split.train_x[split.train_y == 1])
    center = finite_vector(genuine.mean(axis=0))
    scale = np.maximum(safe_column_std(genuine), 1e-6)
    train_time = (time.perf_counter() - start_train) * 1000
    start_infer = time.perf_counter()
    scores = -np.sum(np.abs((finite_matrix(split.test_x) - center) / scale), axis=1)
    infer_time = (time.perf_counter() - start_infer) * 1000
    return result_payload(scores, train_time, infer_time, memory_before)


def train_and_score_lightgbm(split: Split, args: argparse.Namespace) -> dict[str, np.ndarray | float | int]:
    memory_before = rss_bytes()
    positives = int(np.sum(split.train_y == 1))
    negatives = int(np.sum(split.train_y == 0))
    min_child_samples = max(2, min(20, len(split.train_y) // 20))
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=args.lightgbm_estimators,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=6,
        min_child_samples=min_child_samples,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        class_weight={0: len(split.train_y) / max(2 * negatives, 1), 1: len(split.train_y) / max(2 * positives, 1)},
        n_jobs=args.lightgbm_n_jobs,
        random_state=20260513,
        verbosity=-1,
    )
    start_train = time.perf_counter()
    model.fit(finite_matrix(split.train_x), split.train_y)
    train_time = (time.perf_counter() - start_train) * 1000
    start_infer = time.perf_counter()
    scores = model.predict_proba(finite_matrix(split.test_x))[:, 1]
    infer_time = (time.perf_counter() - start_infer) * 1000
    return result_payload(scores, train_time, infer_time, memory_before)


def train_and_score_cnn(split: Split, args: argparse.Namespace) -> dict[str, np.ndarray | float | int]:
    tensorflow = get_tensorflow()
    memory_before = rss_bytes()
    train_x, test_x = standardize_train_test(split.train_x, split.test_x)
    train_x = train_x.reshape((train_x.shape[0], *split.sequence_shape))
    test_x = test_x.reshape((test_x.shape[0], *split.sequence_shape))
    tensorflow.keras.backend.clear_session()
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Input(shape=split.sequence_shape),
        tensorflow.keras.layers.Conv1D(16, 3, padding="same", activation="relu"),
        tensorflow.keras.layers.MaxPooling1D(pool_size=2, strides=2),
        tensorflow.keras.layers.Conv1D(24, 3, padding="same", activation="relu"),
        tensorflow.keras.layers.GlobalAveragePooling1D(),
        tensorflow.keras.layers.Dense(16, activation="relu"),
        tensorflow.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(0.001), loss="binary_crossentropy")
    positives = int(np.sum(split.train_y == 1))
    negatives = len(split.train_y) - positives
    class_weight = {
        0: len(split.train_y) / max(2 * negatives, 1),
        1: len(split.train_y) / max(2 * positives, 1),
    }
    start_train = time.perf_counter()
    model.fit(
        train_x,
        split.train_y,
        epochs=args.cnn_epochs,
        batch_size=args.cnn_batch_size,
        shuffle=False,
        class_weight=class_weight,
        verbose=0,
    )
    train_time = (time.perf_counter() - start_train) * 1000
    start_infer = time.perf_counter()
    scores = model.predict(test_x, batch_size=args.cnn_batch_size, verbose=0).reshape(-1)
    infer_time = (time.perf_counter() - start_infer) * 1000
    tensorflow.keras.backend.clear_session()
    return result_payload(scores, train_time, infer_time, memory_before)


def result_payload(scores: np.ndarray, train_ms: float, infer_ms: float, memory_before: int) -> dict[str, np.ndarray | float | int]:
    safe_scores = sanitize_scores(scores)
    return {
        "scores": safe_scores,
        "train_time_ms": train_ms,
        "inference_batch_ms": infer_ms,
        "inference_time_ms_per_sample": infer_ms / max(len(scores), 1),
        "memory_before_bytes": memory_before,
        "memory_after_bytes": rss_bytes(),
    }


def evaluate_data_efficiency(
    dataset: Dataset,
    model_key: str,
    subjects: list[str],
    timing_rows: list[dict],
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    model_name = model_display_name(model_key)
    for size in dataset.data_efficiency_sizes:
        print(f"[data_efficiency] {dataset.label} {dataset.feature_mode}: {model_name} prefix {size}")
        started = time.perf_counter()
        eers = []
        for index, subject in enumerate(subjects, start=1):
            split = make_auth_split(dataset, subject, adaptive=False)
            if split is None:
                continue
            genuine_indices = np.where(split.train_y == 1)[0]
            genuine_indices = genuine_indices[np.argsort(split.train_order[genuine_indices])][:size]
            if len(genuine_indices) < size:
                continue
            impostor_indices = np.where(split.train_y == 0)[0]
            chosen = np.sort(np.concatenate([genuine_indices, impostor_indices]))
            reduced = Split(
                train_x=split.train_x[chosen],
                train_y=split.train_y[chosen],
                test_x=split.test_x,
                test_y=split.test_y,
                train_order=split.train_order[chosen],
                sequence_shape=split.sequence_shape,
            )
            if dataset.adaptive:
                reduced = apply_adaptive_feature_selection(reduced)
            result = train_and_score(model_key, reduced, args)
            eer = compute_metrics(reduced.test_y, result["scores"])["eer"]
            if is_finite(eer):
                eers.append(eer)
            if index % 10 == 0:
                print(f"  subject {index}/{len(subjects)}")
        if not eers:
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000
        mean_eer = float(np.mean(eers))
        row = {
            "dataset": output_dataset_key(dataset.key, dataset.feature_mode),
            "source_dataset": dataset.key,
            "feature_mode": dataset.feature_mode,
            "feature_set": dataset.feature_set,
            "Model": model_name,
            "Genuine Training Samples": size,
            "EER": mean_eer,
            "evaluated_subjects": len(eers),
            "elapsed_ms": elapsed_ms,
            "stopped_after_target_reached": mean_eer < args.eer_target,
        }
        rows.append(row)
        timing_rows.append({
            "dataset": output_dataset_key(dataset.key, dataset.feature_mode),
            "source_dataset": dataset.key,
            "feature_mode": dataset.feature_mode,
            "model": model_name,
            "subject": "all",
            "phase": "data_efficiency",
            "genuine_training_samples": size,
            "elapsed_ms": elapsed_ms,
            "evaluated_subjects": len(eers),
            "EER": mean_eer,
        })
        if mean_eer < args.eer_target:
            print(f"[data_efficiency] reached EER < {args.eer_target:.2%} at prefix {size}; stopping")
            break
    return rows


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0 or len(s) == 0:
        return {"far": np.nan, "frr": np.nan, "eer": np.nan, "accuracy": np.nan, "auc": np.nan, "threshold": np.nan}

    order = np.argsort(-s)
    sorted_scores = s[order]
    sorted_y = y[order]
    tp = fp = 0
    fn = positives
    tn = negatives
    best = metric_candidate(math.inf, tp, fp, tn, fn, positives, negatives)
    index = 0
    while index < len(sorted_scores):
        threshold = sorted_scores[index]
        # Restrict to the current contiguous tie group in sorted order.
        end = index
        while end < len(sorted_scores) and sorted_scores[end] == threshold:
            end += 1
        labels = sorted_y[index:end]
        group_pos = int(np.sum(labels == 1))
        group_neg = int(np.sum(labels == 0))
        tp += group_pos
        fn -= group_pos
        fp += group_neg
        tn -= group_neg
        candidate = metric_candidate(float(threshold), tp, fp, tn, fn, positives, negatives)
        if candidate["distance"] < best["distance"]:
            best = candidate
        index = end
    all_positive = metric_candidate(-math.inf, positives, negatives, 0, 0, positives, negatives)
    if all_positive["distance"] < best["distance"]:
        best = all_positive
    auc = float(roc_auc_score(y, s)) if roc_auc_score is not None else auc_rank(y, s)
    return {
        "far": best["far"],
        "frr": best["frr"],
        "eer": (best["far"] + best["frr"]) / 2,
        "accuracy": best["accuracy"],
        "auc": auc,
        "threshold": best["threshold"],
    }


def metric_candidate(threshold: float, tp: int, fp: int, tn: int, fn: int, positives: int, negatives: int) -> dict[str, float]:
    far = fp / max(negatives, 1)
    frr = fn / max(positives, 1)
    return {
        "threshold": threshold,
        "far": far,
        "frr": frr,
        "distance": abs(far - frr),
        "accuracy": (tp + tn) / max(positives + negatives, 1),
    }


def average_metric_rows(rows: list[dict], dataset: Dataset, model_key: str) -> dict:
    def avg(name: str):
        values = [float(row[name]) for row in rows if is_finite(row.get(name))]
        return float(np.mean(values)) if values else np.nan

    return {
        "dataset": output_dataset_key(dataset.key, dataset.feature_mode),
        "source_dataset": dataset.key,
        "feature_mode": dataset.feature_mode,
        "feature_set": dataset.feature_set,
        "Model": model_display_name(model_key),
        "model_runtime": model_runtime(model_key),
        "FAR": avg("FAR"),
        "FRR": avg("FRR"),
        "EER": avg("EER"),
        "Accuracy": avg("Accuracy"),
        "AUC": avg("AUC"),
        "Inference Time (ms/sample)": avg("inference_time_ms_per_sample"),
        "UI Blocking Time (ms/test batch)": avg("ui_blocking_time_ms"),
        "Train Time (ms/subject)": avg("train_time_ms"),
        "Browser Memory Before (bytes)": avg("memory_before_bytes"),
        "Browser Memory After (bytes)": avg("memory_after_bytes"),
        "Min Genuine Samples for EER < 10%": "Not reached",
        "subjects": len(rows),
    }


def min_samples_for_target(rows: list[dict], model_name: str, target: float) -> int | str:
    matches = [row for row in rows if row["Model"] == model_name and float(row["EER"]) < target]
    if not matches:
        return "Not reached"
    return min(int(row["Genuine Training Samples"]) for row in matches)


def draw_charts(output_dir: Path, summary: pd.DataFrame, efficiency: pd.DataFrame, roc_payload: dict[str, tuple[list[int], list[float]]]) -> None:
    pyplot = get_pyplot()
    if pyplot is None:
        return
    if not summary.empty:
        fig, ax = pyplot.subplots(figsize=(9, 5))
        labels = summary["Model"].astype(str)
        x = np.arange(len(labels))
        eer_values = pd.to_numeric(summary["EER"], errors="coerce").fillna(0.0)
        accuracy_values = pd.to_numeric(summary["Accuracy"], errors="coerce").fillna(0.0)
        ax.bar(x - 0.2, eer_values, width=0.4, label="EER")
        ax.bar(x + 0.2, accuracy_values, width=0.4, label="Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("Efficiency Metrics")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "efficiency_bars.png", dpi=160)
        pyplot.close(fig)

    if not efficiency.empty:
        fig, ax = pyplot.subplots(figsize=(9, 5))
        for model, group in efficiency.groupby("Model"):
            group = group.sort_values("Genuine Training Samples")
            x_values = pd.to_numeric(group["Genuine Training Samples"], errors="coerce")
            y_values = pd.to_numeric(group["EER"], errors="coerce")
            mask = x_values.notna() & y_values.notna()
            if mask.any():
                ax.plot(x_values[mask], y_values[mask], marker="o", label=model)
        ax.axhline(EER_TARGET, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Genuine Training Samples")
        ax.set_ylabel("EER")
        ax.set_title("EER vs Training Size")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "eer_vs_training_size.png", dpi=160)
        pyplot.close(fig)

    if roc_curve is not None and roc_payload:
        fig, ax = pyplot.subplots(figsize=(8, 6))
        for model, (labels, scores) in roc_payload.items():
            labels_array = np.asarray(labels, dtype=np.int8)
            scores_array = np.asarray(scores, dtype=np.float64)
            mask = np.isfinite(scores_array)
            labels_array = labels_array[mask]
            scores_array = scores_array[mask]
            if len(scores_array) == 0 or len(set(labels_array.tolist())) < 2:
                continue
            fpr, tpr, _ = roc_curve(labels_array, scores_array)
            ax.plot(fpr, tpr, label=model)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curves.png", dpi=160)
        pyplot.close(fig)


def build_environment(dataset: Dataset, args: argparse.Namespace) -> dict:
    packages = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": getattr(lgb, "__version__", "not_available") if lgb is not None else "not_available",
        "tensorflow": package_version("tensorflow"),
    }
    config = {
        "dataset": output_dataset_key(dataset.key, dataset.feature_mode),
        "sourceDataset": dataset.key,
        "featureMode": dataset.feature_mode,
        "featureSet": dataset.feature_set,
        "featureCount": len(dataset.feature_names),
        "sequenceShape": dataset.sequence_shape,
        "adaptiveFeatureSelection": dataset.adaptive,
        "adaptiveFeatureCap": ADAPTIVE_FEATURE_CAP if dataset.adaptive else "not_enabled",
        "models": args.models,
        "dataEfficiencyTargetEer": args.eer_target,
    }
    return {
        "environmentId": hashlib.sha256(json.dumps({
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": packages,
        }, sort_keys=True).encode()).hexdigest()[:16],
        "timestamp": utc_now(),
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "packages": packages,
        "datasetFileHashes": dataset.file_hashes,
        "modelConfig": config,
        "modelConfigHash": hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16],
        "memoryRssBytesAtStart": rss_bytes(),
    }


def limit_subjects(dataset: Dataset, count: int) -> Dataset:
    keep_subjects = set(unique_sorted(dataset.subjects)[:count])
    mask = np.asarray([subject in keep_subjects for subject in dataset.subjects])
    return Dataset(
        **{
            **dataset.__dict__,
            "x": dataset.x[mask],
            "subjects": dataset.subjects[mask],
            "partitions": dataset.partitions[mask],
            "orders": dataset.orders[mask],
        }
    )


def row_stats(matrix: np.ndarray) -> dict[str, np.ndarray]:
    matrix = finite_matrix(matrix)
    if matrix.size == 0:
        zeros = np.zeros(matrix.shape[0], dtype=np.float32)
        return {name: zeros for name in stat_suffixes()}
    std = safe_row_std(matrix)
    mean_values = np.mean(matrix, axis=1)
    return {
        "count": np.full(matrix.shape[0], matrix.shape[1], dtype=np.float32),
        "mean": mean_values,
        "std": std,
        "min": np.min(matrix, axis=1),
        "p10": np.percentile(matrix, 10, axis=1),
        "p25": np.percentile(matrix, 25, axis=1),
        "median": np.percentile(matrix, 50, axis=1),
        "p75": np.percentile(matrix, 75, axis=1),
        "p90": np.percentile(matrix, 90, axis=1),
        "max": np.max(matrix, axis=1),
        "iqr": np.percentile(matrix, 75, axis=1) - np.percentile(matrix, 25, axis=1),
        "cv": std / np.maximum(np.abs(mean_values), 1e-6),
    }


def stats_matrix(matrix: np.ndarray) -> np.ndarray:
    stats = row_stats(np.asarray(matrix, dtype=np.float32))
    return np.column_stack([stats[name] for name in stat_suffixes()]).astype(np.float32)


def stats_names(prefix: str) -> list[str]:
    return [f"{prefix}.{suffix}" for suffix in stat_suffixes()]


def stat_suffixes() -> list[str]:
    return ["count", "mean", "std", "min", "p10", "p25", "median", "p75", "p90", "max", "iqr", "cv"]


def add_stats(row: dict[str, float], prefix: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        for suffix in stat_suffixes():
            row[f"{prefix}.{suffix}"] = 0.0
        return
    percentiles = np.percentile(values, [10, 25, 50, 75, 90])
    avg = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    payload = {
        "count": float(len(values)),
        "mean": avg,
        "std": std,
        "min": float(np.min(values)),
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "median": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "max": float(np.max(values)),
        "iqr": float(percentiles[3] - percentiles[1]),
        "cv": std / max(abs(avg), 1e-6),
    }
    for suffix, value in payload.items():
        row[f"{prefix}.{suffix}"] = value


def make_bucket_accumulator() -> list[dict[str, float]]:
    return [{"count": 0.0, "sum": 0.0} for _ in range(HASH_BUCKETS)]


def add_bucket_value(accumulator: list[dict[str, float]], bucket: int, value: float) -> None:
    slot = accumulator[bucket]
    slot["count"] += 1.0
    slot["sum"] += value if np.isfinite(value) else 0.0


def flush_bucket_accumulator(row: dict[str, float], prefix: str, accumulator: list[dict[str, float]]) -> None:
    for bucket, slot in enumerate(accumulator):
        if slot["count"] <= 0:
            continue
        row[f"{prefix}.b{bucket:02d}.count"] = slot["count"]
        row[f"{prefix}.b{bucket:02d}.mean_ms"] = slot["sum"] / slot["count"]


def normalized_key_code(record: pd.Series) -> str:
    value = str(record.get("code") or record.get("key_value") or "unknown").strip()
    return value or "unknown"


def classify_key(code: str) -> str:
    if code.startswith("Key") and len(code) == 4:
        return "letter"
    if code.startswith("Digit") or code.startswith("Numpad"):
        return "digit"
    if code == "Space":
        return "space"
    if any(part in code for part in ["Shift", "Control", "Alt", "Meta"]):
        return "modifier"
    if any(part in code for part in ["Enter", "Backspace", "Delete", "Tab", "Escape", "Arrow"]):
        return "control"
    if any(part in code for part in ["Comma", "Period", "Slash", "Quote", "Bracket", "Minus", "Equal", "Semicolon", "Backquote"]):
        return "punctuation"
    return "other"


def hash_bucket(value: str, bucket_count: int = HASH_BUCKETS) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % bucket_count


def normalized_hash_bucket(value: str, bucket_count: int = HASH_BUCKETS) -> float:
    return hash_bucket(value, bucket_count) / max(bucket_count - 1, 1)


def finite_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def finite_vector(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def safe_column_std(matrix: np.ndarray) -> np.ndarray:
    matrix = finite_matrix(matrix)
    if matrix.ndim != 2 or matrix.shape[0] <= 1:
        width = matrix.shape[1] if matrix.ndim == 2 else 0
        return np.zeros(width, dtype=np.float32)
    return finite_vector(np.std(matrix, axis=0, ddof=1))


def safe_row_std(matrix: np.ndarray) -> np.ndarray:
    matrix = finite_matrix(matrix)
    if matrix.ndim != 2 or matrix.shape[1] <= 1:
        rows = matrix.shape[0] if matrix.ndim >= 1 else 0
        return np.zeros(rows, dtype=np.float32)
    return finite_vector(np.std(matrix, axis=1, ddof=1))


def sanitize_scores(scores: np.ndarray) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return np.zeros_like(array, dtype=np.float64)
    low = float(np.min(finite) - 1.0)
    high = float(np.max(finite) + 1.0)
    return np.nan_to_num(array, nan=low, neginf=low, posinf=high).astype(np.float64, copy=False)


def standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = finite_matrix(train)
    test = finite_matrix(test)
    center = finite_vector(train.mean(axis=0))
    scale = np.maximum(safe_column_std(train), 1e-6)
    return finite_matrix((train - center) / scale), finite_matrix((test - center) / scale)


def auc_rank(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive_ranks = ranks[y == 1].sum()
    positives = np.sum(y == 1)
    negatives = np.sum(y == 0)
    return float((positive_ranks - positives * (positives + 1) / 2) / max(positives * negatives, 1))


def output_dataset_key(dataset_key: str, feature_mode: str) -> str:
    return f"{dataset_key}_enriched_features" if feature_mode == "enriched" else dataset_key


def model_display_name(model_key: str) -> str:
    return {
        "manhattan": "Scaled Manhattan Distance",
        "lightgbm": "LightGBM Classifier",
        "cnn": "1D-CNN TensorFlow",
    }[model_key]


def model_runtime(model_key: str) -> str:
    return {
        "manhattan": "python_numpy",
        "lightgbm": "python_lightgbm",
        "cnn": "python_tensorflow_keras",
    }[model_key]


def get_tensorflow():
    global tf
    if tf is None:
        import tensorflow as tensorflow

        tensorflow.get_logger().setLevel("ERROR")
        tf = tensorflow
    return tf


def get_pyplot():
    global plt
    if plt is not None:
        return plt
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        plt = pyplot
    except Exception:
        plt = False
    return plt or None


def package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not_available"


def unique_sorted(values: np.ndarray) -> list[str]:
    return sorted(set(values.astype(str).tolist()), key=subject_sort_key)


def subject_sort_key(value: str):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def safe_get(row, key: str, default=""):
    try:
        value = row.get(key, default)
    except AttributeError:
        return default
    if pd.isna(value):
        return default
    return value


def numeric_series(value, length: int) -> np.ndarray:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return np.full(length, num(value), dtype=np.float32)


def num(value) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    return number if math.isfinite(number) else 0.0


def boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def mean_array(values) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    return float(series.mean()) if len(series) else 0.0


def timestamp_order(value) -> float:
    if not value or pd.isna(value):
        return 0.0
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}-{uuid.uuid4().hex[:8]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    return int(value * 1024)


if __name__ == "__main__":
    raise SystemExit(main())
