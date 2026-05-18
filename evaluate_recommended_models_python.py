#!/usr/bin/env python3
"""Evaluate recommended keystroke-authentication models.

This file is intentionally independent from evaluate_keystroke_python.py.
It uses the same benchmark shape: same datasets, temporal train/test splits,
EER/Accuracy/AUC metrics, timing logs, data-efficiency runs, and CSV/PNG
artifacts. The model set focuses on template, novelty, and device-aware
methods that are more suitable for small keystroke datasets than large CNNs.
"""

from __future__ import annotations

import argparse
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

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
except Exception as exc:  # pragma: no cover - runtime environment check
    raise SystemExit(f"scikit-learn is required for this script: {exc}") from exc


ROOT = Path(__file__).resolve().parent
if not os.environ.get("MPLCONFIGDIR"):
    MPL_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "thesis_matplotlib_cache"
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

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
ALL_MODELS = [
    "template_knn",
    "device_aware_knn",
    "pca_mahalanobis",
    "oneclass_svm",
    "isolation_forest",
    "lof",
]
EER_TARGET = 0.10
HASH_BUCKETS = 64
COLLECTED_SEQUENCE_CAP = 180
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


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
    devices: np.ndarray
    feature_names: list[str]
    sequence_shape: tuple[int, int]
    sequence_values: np.ndarray
    file_hashes: dict[str, str]
    data_efficiency_sizes: list[int]
    extra_artifacts: dict[str, pd.DataFrame] | None = None


@dataclass
class Split:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    train_order: np.ndarray
    train_device: np.ndarray
    test_device: np.ndarray
    train_sequence: np.ndarray
    test_sequence: np.ndarray


def main() -> int:
    args = parse_args()
    datasets = expand_dataset(args.dataset)
    feature_modes = ["current", "enriched"] if args.feature_mode == "both" else [args.feature_mode]
    models = expand_models(args.models)

    run_id = args.run_id or make_run_id()
    run_root = Path(args.output_dir) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_manifest.json", {
        "runId": run_id,
        "createdAt": utc_now(),
        "script": Path(__file__).name,
        "benchmark": "same temporal train/test benchmark as evaluate_keystroke_python.py",
        "models": models,
        "datasets": datasets,
        "featureModes": feature_modes,
        "python": sys.version,
        "platform": platform.platform(),
        "notes": "Recommended template, novelty-detection, and device-aware model evaluation.",
    })

    for dataset_key in datasets:
        for feature_mode in feature_modes:
            dataset = load_dataset(dataset_key, feature_mode)
            if args.max_subjects:
                dataset = limit_subjects(dataset, args.max_subjects)
            output_key = output_dataset_key(dataset.key, dataset.feature_mode)
            output_dir = run_root / output_key
            output_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[dataset] {output_key}: {len(dataset.x)} samples, "
                f"{len(unique_sorted(dataset.subjects))} subjects, {len(dataset.feature_names)} features"
            )
            run_dataset(dataset, models, output_dir, args)

    print(f"[done] results saved to {run_root}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommended keystroke model evaluation")
    parser.add_argument("--dataset", default="all", choices=["dsl", "mmc", "collected", "all"])
    parser.add_argument("--feature-mode", default="both", choices=["current", "enriched", "both"])
    parser.add_argument("--models", default="all", help="Comma list, fast, or all. Default: all non-DTW recommended models")
    parser.add_argument("--output-dir", default="evaluation_runs_recommended")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--skip-data-efficiency", action="store_true")
    parser.add_argument("--eer-target", type=float, default=EER_TARGET)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--template-k", type=int, default=3)
    parser.add_argument("--ocsvm-nu", type=float, default=0.10)
    parser.add_argument("--isolation-estimators", type=int, default=150)
    parser.add_argument("--lof-neighbors", type=int, default=10)
    return parser.parse_args()


def expand_dataset(value: str) -> list[str]:
    return ["dsl", "mmc", "collected"] if value == "all" else [value]


def expand_models(value: str) -> list[str]:
    if value == "fast":
        return ALL_MODELS[:]
    if value == "all":
        return ALL_MODELS[:]
    models = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(models) - set(ALL_MODELS))
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(unknown)}")
    return models


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
        extra = build_dsl_enriched_features(df, base_names)
        feature_names = base_names + list(extra.columns)
        x = np.column_stack([x_base, extra.to_numpy(dtype=np.float32)])
        feature_set = "dsl_current_h_dd_ud_plus_sequence_trigraph_distribution"

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
        devices=np.asarray(["unknown"] * len(df)),
        feature_names=feature_names,
        sequence_shape=(x.shape[1], 1),
        sequence_values=finite_matrix(x).reshape((len(x), x.shape[1], 1)),
        file_hashes={DSL_FILE.name: sha256_file(DSL_FILE)},
        data_efficiency_sizes=[5, 10, 25, 50, 100, 200],
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
        a, b, c = key_sequence[index:index + 3]
        out[f"trigraph.{a}.{b}.{c}.down_down_sum"] = numeric_series(df.get(f"DD.{a}.{b}", 0), len(df)) + numeric_series(df.get(f"DD.{b}.{c}", 0), len(df))
        out[f"trigraph.{a}.{b}.{c}.up_down_sum"] = numeric_series(df.get(f"UD.{a}.{b}", 0), len(df)) + numeric_series(df.get(f"UD.{b}.{c}", 0), len(df))
    return pd.DataFrame(out)


def load_mmc(feature_mode: str) -> Dataset:
    blocks: list[np.ndarray] = []
    seq_blocks: list[np.ndarray] = []
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
        x_pair = flat
        if feature_mode == "enriched":
            x_pair = np.column_stack([flat, build_mmc_enriched_features(timing)])
        blocks.append(x_pair)
        seq_blocks.append(timing.astype(np.float32))
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

    x = np.vstack(blocks).astype(np.float32)
    sequences = np.vstack(seq_blocks).astype(np.float32)
    feature_names = [f"sequence.t{t:03d}.timing_channel_{ch}" for t in range(642) for ch in MMC_TIMING_CHANNELS]
    feature_set = "mmc_flat_timing_channels_14_19"
    if feature_mode == "enriched":
        feature_names = feature_names + mmc_enriched_feature_names()
        feature_set = "mmc_flat_timing_channels_plus_distribution_delta_trigraph"
    return Dataset(
        key="mmc",
        label="MMC",
        feature_mode=feature_mode,
        feature_set=feature_set,
        x=finite_matrix(x),
        subjects=np.asarray(subjects),
        partitions=np.asarray(partitions),
        orders=np.asarray(orders, dtype=np.float64),
        devices=np.asarray(["unknown"] * len(subjects)),
        feature_names=feature_names,
        sequence_shape=(sequences.shape[1], sequences.shape[2]),
        sequence_values=finite_matrix(sequences),
        file_hashes=hashes,
        data_efficiency_sizes=[1, 2, 5, 10, 20],
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
        blocks.extend([stats_matrix(values), stats_matrix(np.diff(values, axis=1)), stats_matrix(np.diff(values, n=2, axis=1))])
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
    if feature_mode == "enriched":
        feature_matrix = build_collected_enriched_feature_matrix(canonical, event_pairs, current_features)
        feature_set = "collected_stored_features_plus_key_sequence_digraph_trigraph_distribution"

    included = canonical[collected_training_mask(canonical)].copy()
    merged = included.merge(feature_matrix, on="attempt_id", how="inner")
    subject_col = merged["participant_code"].where(merged["participant_code"].astype(str).str.len() > 0, merged["participant_id"])
    subject_col = subject_col.map(canonical_collected_subject)
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

    x = finite_matrix(x[keep])
    subjects = subject_col.to_numpy()[keep].astype(str)
    partitions = partitions[keep].astype(str)
    orders = orders[keep]
    devices = merged["device_class"].astype(str).to_numpy()[keep]
    sequences = x.reshape((x.shape[0], x.shape[1], 1))
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
        x=x,
        subjects=subjects,
        partitions=partitions,
        orders=orders,
        devices=devices,
        feature_names=feature_names,
        sequence_shape=(x.shape[1], 1),
        sequence_values=sequences,
        file_hashes=hashes,
        data_efficiency_sizes=[1, 2, 3, 5, 10],
        extra_artifacts={
            "collected_attempts_canonical.csv": canonical,
            "collected_event_pairs.csv": event_pairs,
            "collected_feature_matrix.csv": current_features,
            "collected_filter_summary.csv": build_collected_filter_summary(canonical),
            **({"collected_enriched_feature_matrix.csv": feature_matrix} if feature_mode == "enriched" else {}),
        },
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
        participant_code = safe_get(participant, "participant_code", "")
        paste_detected = (
            boolish(attempt.get("summary_pasteDetected", False))
            or num(attempt.get("paste_count", 0)) > 0
            or num(attempt.get("summary_pasteCount", 0)) > 0
        )
        rows.append({
            "attempt_id": attempt.get("id", ""),
            "participant_id": attempt.get("participant_id", ""),
            "participant_code": participant_code,
            "canonical_subject": canonical_collected_subject(participant_code or attempt.get("participant_id", "")),
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
            "started_at": attempt.get("started_at", ""),
            "ended_at": attempt.get("ended_at", ""),
            "submitted_at": attempt.get("submitted_at", ""),
            "timestamp": attempt.get("submitted_at", "") or attempt.get("ended_at", "") or attempt.get("started_at", ""),
            "raw_text_length": attempt.get("summary_rawTextLength", len(str(attempt.get("raw_text", "")))),
            "event_count": attempt.get("summary_eventCount", ""),
            "paired_key_count": attempt.get("summary_pairedKeyCount", ""),
            "paste_count": attempt.get("paste_count", ""),
            "summary_paste_count": attempt.get("summary_pasteCount", ""),
            "paste_detected": paste_detected,
        })
    return pd.DataFrame(rows)


def collected_training_mask(canonical: pd.DataFrame) -> pd.Series:
    return (
        canonical["status"].astype(str).str.lower().eq("submitted")
        & canonical["quality_status"].astype(str).str.lower().eq("usable")
        & canonical["summary_quality_status"].astype(str).str.lower().isin(["", "nan", "usable"])
        & ~canonical["feature_quality"].astype(str).str.lower().eq("low")
        & ~canonical["summary_feature_quality"].astype(str).str.lower().eq("low")
        & ~canonical["paste_detected"].astype(bool)
        & ~canonical["canonical_subject"].eq("TEST")
    )


def build_collected_filter_summary(canonical: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("total_attempts", pd.Series([True] * len(canonical), index=canonical.index)),
        ("submitted", canonical["status"].astype(str).str.lower().eq("submitted")),
        ("quality_status_usable", canonical["quality_status"].astype(str).str.lower().eq("usable")),
        ("paste_not_detected", ~canonical["paste_detected"].astype(bool)),
        ("not_test_participant", ~canonical["canonical_subject"].eq("TEST")),
        ("included_for_training_and_test", collected_training_mask(canonical)),
    ]
    return pd.DataFrame({"filter": [name for name, _ in checks], "attempts": [int(mask.sum()) for _, mask in checks]})


def build_collected_event_pairs(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attempt_id, group in events.groupby("attempt_id", sort=False):
        active: dict[str, list[pd.Series]] = {}
        pairs = []
        for _, event in group.sort_values("relative_time").iterrows():
            key = str(event.get("code") or event.get("key_value") or "unknown")
            if event.get("event_type") == "keydown":
                active.setdefault(key, []).append(event)
            elif event.get("event_type") == "keyup":
                stack = active.get(key, [])
                if stack:
                    pairs.append((stack.pop(0), event))
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
    matrix = features.pivot_table(index="attempt_id", columns="feature_name", values="feature_value", aggfunc="first").reset_index()
    matrix.columns = [str(col) for col in matrix.columns]
    return matrix.fillna(0.0)


def build_collected_enriched_feature_matrix(canonical: pd.DataFrame, event_pairs: pd.DataFrame, current_features: pd.DataFrame) -> pd.DataFrame:
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
    row = {"sequence.length": float(len(pairs)), "sequence.capped_length": float(min(len(pairs), COLLECTED_SEQUENCE_CAP))}
    if pairs.empty:
        row["sequence.unique_key_ratio"] = 0.0
        return row
    capped = pairs.head(COLLECTED_SEQUENCE_CAP).reset_index(drop=True)
    keys = [normalized_key_code(record) for _, record in capped.iterrows()]
    row["sequence.unique_key_ratio"] = len(set(keys)) / max(len(keys), 1)
    for index, record in capped.iterrows():
        row[f"sequence.key_bucket_pos_{index:03d}"] = normalized_hash_bucket(keys[index])
        row[f"sequence.hold_pos_{index:03d}"] = num(record.get("hold_ms"))
        if index > 0:
            row[f"digraph.down_down_pos_{index:03d}"] = num(record.get("down_down_ms"))
            row[f"digraph.up_down_pos_{index:03d}"] = num(record.get("up_down_ms"))
            row[f"digraph.up_up_pos_{index:03d}"] = num(record.get("up_up_ms"))
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
    return row


def collected_hashed_ngram_features(pairs: pd.DataFrame) -> dict[str, float]:
    row: dict[str, float] = {}
    unigram = make_bucket_accumulator()
    digraph = make_bucket_accumulator()
    reset = pairs.reset_index(drop=True)
    keys = [normalized_key_code(record) for _, record in reset.iterrows()]
    for index, record in reset.iterrows():
        add_bucket_value(unigram, hash_bucket(keys[index]), num(record.get("hold_ms")))
        if index > 0:
            add_bucket_value(digraph, hash_bucket(f"{keys[index - 1]}>{keys[index]}"), num(record.get("down_down_ms")))
    flush_bucket_accumulator(row, "key_sequence.unigram", unigram)
    flush_bucket_accumulator(row, "digraph.timing", digraph)
    return row


def run_dataset(dataset: Dataset, models: list[str], output_dir: Path, args: argparse.Namespace) -> None:
    started = time.perf_counter()
    environment = build_environment(dataset, args, models)
    summary_rows: list[dict] = []
    userwise_rows: list[dict] = []
    timing_rows: list[dict] = []
    efficiency_rows: list[dict] = []
    roc_payload: dict[str, tuple[list[int], list[float]]] = {}
    subjects = unique_sorted(dataset.subjects)

    for model_key in models:
        model_rows = []
        all_y: list[int] = []
        all_scores: list[float] = []
        for index, subject in enumerate(subjects, start=1):
            print(f"[model] {dataset.label} {dataset.feature_mode}: {model_display_name(model_key)} subject {index}/{len(subjects)}")
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
                "model": model_display_name(model_key),
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
            }
            userwise_rows.append(row)
            model_rows.append(row)
            timing_rows.append({
                "dataset": output_dataset_key(dataset.key, dataset.feature_mode),
                "source_dataset": dataset.key,
                "feature_mode": dataset.feature_mode,
                "model": model_display_name(model_key),
                "subject": subject,
                "phase": "train_and_infer",
                "elapsed_ms": result["train_time_ms"] + result["inference_batch_ms"],
                "train_time_ms": result["train_time_ms"],
                "inference_batch_ms": result["inference_batch_ms"],
                "inference_time_ms_per_sample": result["inference_time_ms_per_sample"],
                "memory_before_bytes": result["memory_before_bytes"],
                "memory_after_bytes": result["memory_after_bytes"],
            })
            all_y.extend(split.test_y.astype(int).tolist())
            all_scores.extend(np.asarray(result["scores"], dtype=float).tolist())

        aggregate = average_metric_rows(model_rows, dataset, model_key)
        if not args.skip_data_efficiency:
            rows = evaluate_data_efficiency(dataset, model_key, subjects, timing_rows, args)
            efficiency_rows.extend(rows)
            aggregate["Min Genuine Samples for EER < 10%"] = min_samples_for_target(rows, model_display_name(model_key), args.eer_target)
        summary_rows.append(aggregate)
        roc_payload[model_display_name(model_key)] = (all_y, all_scores)

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
    pd.DataFrame(efficiency_rows).to_csv(output_dir / "data_efficiency_eer.csv", index=False)
    timing_frame = pd.DataFrame(timing_rows)
    timing_frame.to_csv(output_dir / "python_timing_log.csv", index=False)
    timing_frame.to_csv(output_dir / "browser_timing_log.csv", index=False)
    if dataset.extra_artifacts:
        for name, frame in dataset.extra_artifacts.items():
            frame.to_csv(output_dir / name, index=False)
    draw_charts(output_dir, pd.DataFrame(summary_rows), pd.DataFrame(efficiency_rows), roc_payload)


def make_auth_split(dataset: Dataset, genuine_subject: str) -> Split | None:
    train_mask = dataset.partitions == "reference"
    test_mask = dataset.partitions == "test"
    train_y = (dataset.subjects[train_mask] == genuine_subject).astype(np.int8)
    test_y = (dataset.subjects[test_mask] == genuine_subject).astype(np.int8)
    if not (np.any(train_y == 1) and np.any(train_y == 0) and np.any(test_y == 1) and np.any(test_y == 0)):
        return None
    return Split(
        train_x=finite_matrix(dataset.x[train_mask]),
        train_y=train_y,
        test_x=finite_matrix(dataset.x[test_mask]),
        test_y=test_y,
        train_order=dataset.orders[train_mask],
        train_device=dataset.devices[train_mask],
        test_device=dataset.devices[test_mask],
        train_sequence=finite_matrix(dataset.sequence_values[train_mask]),
        test_sequence=finite_matrix(dataset.sequence_values[test_mask]),
    )


def train_and_score(model_key: str, split: Split, args: argparse.Namespace) -> dict[str, np.ndarray | float | int]:
    if model_key == "template_knn":
        return score_template_knn(split, args)
    if model_key == "device_aware_knn":
        return score_device_aware_knn(split, args)
    if model_key == "pca_mahalanobis":
        return score_pca_mahalanobis(split, args)
    if model_key == "oneclass_svm":
        return score_oneclass_svm(split, args)
    if model_key == "isolation_forest":
        return score_isolation_forest(split, args)
    if model_key == "lof":
        return score_lof(split, args)
    raise ValueError(model_key)


def score_template_knn(split: Split, args: argparse.Namespace) -> dict[str, np.ndarray | float | int]:
    memory_before = rss_bytes()
    start_train = time.perf_counter()
    train_z, test_z = scale_train_test(split.train_x, split.test_x)
    genuine = train_z[split.train_y == 1]
    train_ms = (time.perf_counter() - start_train) * 1000
    start_infer = time.perf_counter()
    scores = -knn_distance(test_z, genuine, args.template_k)
    infer_ms = (time.perf_counter() - start_infer) * 1000
    return result_payload(scores, train_ms, infer_ms, memory_before)


def score_device_aware_knn(split: Split, args: argparse.Namespace) -> dict[str, np.ndarray | float | int]:
    memory_before = rss_bytes()
    start_train = time.perf_counter()
    train_z, test_z = scale_train_test(split.train_x, split.test_x)
    genuine_mask = split.train_y == 1
    genuine = train_z[genuine_mask]
    genuine_devices = split.train_device[genuine_mask]
    train_ms = (time.perf_counter() - start_train) * 1000
    start_infer = time.perf_counter()
    scores = []
    for row, device in zip(test_z, split.test_device, strict=False):
        global_d = knn_distance(row.reshape(1, -1), genuine, args.template_k)[0]
        same = genuine[genuine_devices == device]
        if len(same):
            same_d = knn_distance(row.reshape(1, -1), same, args.template_k)[0]
            scores.append(-(0.65 * same_d + 0.35 * global_d))
        else:
            scores.append(-global_d)
    infer_ms = (time.perf_counter() - start_infer) * 1000
    return result_payload(np.asarray(scores), train_ms, infer_ms, memory_before)


def score_pca_mahalanobis(split: Split, args: argparse.Namespace) -> dict[str, np.ndarray | float | int]:
    memory_before = rss_bytes()
    start_train = time.perf_counter()
    train_z, test_z = scale_train_test(split.train_x, split.test_x)
    train_r, test_r = reduce_with_pca(train_z, test_z, args.pca_components)
    genuine = train_r[split.train_y == 1]
    center = genuine.mean(axis=0)
    if len(genuine) > 1:
        cov = np.cov(genuine, rowvar=False)
        cov = np.atleast_2d(cov)
        cov += np.eye(cov.shape[0]) * 1e-3
        inv_cov = np.linalg.pinv(cov)
    else:
        inv_cov = np.eye(train_r.shape[1])
    train_ms = (time.perf_counter() - start_train) * 1000
    start_infer = time.perf_counter()
    diff = test_r - center
    scores = -np.sqrt(np.maximum(np.sum((diff @ inv_cov) * diff, axis=1), 0.0))
    infer_ms = (time.perf_counter() - start_infer) * 1000
    return result_payload(scores, train_ms, infer_ms, memory_before)


def score_oneclass_svm(split: Split, args: argparse.Namespace) -> dict[str, np.ndarray | float | int]:
    return score_novelty(split, args, "oneclass_svm")


def score_isolation_forest(split: Split, args: argparse.Namespace) -> dict[str, np.ndarray | float | int]:
    return score_novelty(split, args, "isolation_forest")


def score_lof(split: Split, args: argparse.Namespace) -> dict[str, np.ndarray | float | int]:
    return score_novelty(split, args, "lof")


def score_novelty(split: Split, args: argparse.Namespace, family: str) -> dict[str, np.ndarray | float | int]:
    memory_before = rss_bytes()
    start_train = time.perf_counter()
    train_z, test_z = scale_train_test(split.train_x, split.test_x)
    train_r, test_r = reduce_with_pca(train_z, test_z, args.pca_components)
    genuine = train_r[split.train_y == 1]
    if len(genuine) < 2:
        return score_template_knn(split, args)
    if family == "oneclass_svm":
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=args.ocsvm_nu)
        model.fit(genuine)
        score_fn = model.decision_function
    elif family == "isolation_forest":
        model = IsolationForest(n_estimators=args.isolation_estimators, contamination="auto", random_state=20260518)
        model.fit(genuine)
        score_fn = model.score_samples
    else:
        neighbors = min(args.lof_neighbors, max(1, len(genuine) - 1))
        if neighbors < 1:
            return score_template_knn(split, args)
        model = LocalOutlierFactor(n_neighbors=neighbors, novelty=True)
        model.fit(genuine)
        score_fn = model.score_samples
    train_ms = (time.perf_counter() - start_train) * 1000
    start_infer = time.perf_counter()
    scores = score_fn(test_r)
    infer_ms = (time.perf_counter() - start_infer) * 1000
    return result_payload(scores, train_ms, infer_ms, memory_before)


def evaluate_data_efficiency(dataset: Dataset, model_key: str, subjects: list[str], timing_rows: list[dict], args: argparse.Namespace) -> list[dict]:
    rows = []
    for size in dataset.data_efficiency_sizes:
        print(f"[data_efficiency] {dataset.label} {dataset.feature_mode}: {model_display_name(model_key)} prefix {size}")
        started = time.perf_counter()
        eers = []
        for index, subject in enumerate(subjects, start=1):
            split = make_auth_split(dataset, subject)
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
                train_device=split.train_device[chosen],
                test_device=split.test_device,
                train_sequence=split.train_sequence[chosen],
                test_sequence=split.test_sequence,
            )
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
            "Model": model_display_name(model_key),
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
            "model": model_display_name(model_key),
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
    s = sanitize_scores(scores)
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
    auc = float(roc_auc_score(y, s))
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


def scale_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_z = scaler.fit_transform(finite_matrix(train))
    test_z = scaler.transform(finite_matrix(test))
    return finite_matrix(train_z), finite_matrix(test_z)


def reduce_with_pca(train: np.ndarray, test: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray]:
    max_components = min(components, train.shape[0] - 1, train.shape[1])
    if max_components < 1:
        return train[:, :1], test[:, :1]
    pca = PCA(n_components=max_components, random_state=20260518)
    train_r = pca.fit_transform(train)
    test_r = pca.transform(test)
    return finite_matrix(train_r), finite_matrix(test_r)


def knn_distance(test: np.ndarray, templates: np.ndarray, k: int) -> np.ndarray:
    if len(templates) == 0:
        return np.full(test.shape[0], np.inf)
    diff = test[:, None, :] - templates[None, :, :]
    distances = np.sqrt(np.mean(diff * diff, axis=2))
    kk = max(1, min(k, distances.shape[1]))
    return np.mean(np.partition(distances, kk - 1, axis=1)[:, :kk], axis=1)


def result_payload(scores: np.ndarray, train_ms: float, infer_ms: float, memory_before: int) -> dict[str, np.ndarray | float | int]:
    safe_scores = sanitize_scores(scores)
    return {
        "scores": safe_scores,
        "train_time_ms": train_ms,
        "inference_batch_ms": infer_ms,
        "inference_time_ms_per_sample": infer_ms / max(len(safe_scores), 1),
        "memory_before_bytes": memory_before,
        "memory_after_bytes": rss_bytes(),
    }


def draw_charts(output_dir: Path, summary: pd.DataFrame, efficiency: pd.DataFrame, roc_payload: dict[str, tuple[list[int], list[float]]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = summary["Model"].astype(str)
        x = np.arange(len(labels))
        ax.bar(x - 0.2, pd.to_numeric(summary["EER"], errors="coerce").fillna(0.0), width=0.4, label="EER")
        ax.bar(x + 0.2, pd.to_numeric(summary["Accuracy"], errors="coerce").fillna(0.0), width=0.4, label="Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("Recommended Model Metrics")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "efficiency_bars.png", dpi=160)
        plt.close(fig)
    if not efficiency.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for model, group in efficiency.groupby("Model"):
            group = group.sort_values("Genuine Training Samples")
            ax.plot(group["Genuine Training Samples"], group["EER"], marker="o", label=model)
        ax.axhline(EER_TARGET, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Genuine Training Samples")
        ax.set_ylabel("EER")
        ax.set_title("EER vs Training Size")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "eer_vs_training_size.png", dpi=160)
        plt.close(fig)
    if roc_payload:
        fig, ax = plt.subplots(figsize=(8, 6))
        for model, (labels, scores) in roc_payload.items():
            labels_array = np.asarray(labels, dtype=np.int8)
            scores_array = sanitize_scores(scores)
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
        plt.close(fig)


def build_environment(dataset: Dataset, args: argparse.Namespace, models: list[str]) -> dict:
    config = {
        "dataset": output_dataset_key(dataset.key, dataset.feature_mode),
        "sourceDataset": dataset.key,
        "featureMode": dataset.feature_mode,
        "featureSet": dataset.feature_set,
        "featureCount": len(dataset.feature_names),
        "sequenceShape": dataset.sequence_shape,
        "models": models,
        "benchmark": "temporal reference/train vs future/test",
        "collectedSubjectRule": "exclude TEST; merge suffix IDs such as M001_1 into M001",
        "qualityRule": "exclude low_quality, low feature quality, and paste-detected attempts",
        "dataEfficiencyTargetEer": args.eer_target,
    }
    return {
        "environmentId": hashlib.sha256(json.dumps({
            "python": sys.version,
            "platform": platform.platform(),
            "sklearn": package_version("scikit-learn"),
        }, sort_keys=True).encode()).hexdigest()[:16],
        "timestamp": utc_now(),
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": package_version("scikit-learn"),
        },
        "datasetFileHashes": dataset.file_hashes,
        "modelConfig": config,
        "modelConfigHash": hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16],
        "memoryRssBytesAtStart": rss_bytes(),
    }


def limit_subjects(dataset: Dataset, count: int) -> Dataset:
    keep_subjects = set(unique_sorted(dataset.subjects)[:count])
    mask = np.asarray([subject in keep_subjects for subject in dataset.subjects])
    return Dataset(**{
        **dataset.__dict__,
        "x": dataset.x[mask],
        "subjects": dataset.subjects[mask],
        "partitions": dataset.partitions[mask],
        "orders": dataset.orders[mask],
        "devices": dataset.devices[mask],
        "sequence_values": dataset.sequence_values[mask],
    })


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
    stats = row_stats(matrix)
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


def hash_bucket(value: str, bucket_count: int = HASH_BUCKETS) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % bucket_count


def normalized_hash_bucket(value: str, bucket_count: int = HASH_BUCKETS) -> float:
    return hash_bucket(value, bucket_count) / max(bucket_count - 1, 1)


def finite_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def safe_row_std(matrix: np.ndarray) -> np.ndarray:
    matrix = finite_matrix(matrix)
    if matrix.ndim != 2 or matrix.shape[1] <= 1:
        rows = matrix.shape[0] if matrix.ndim >= 1 else 0
        return np.zeros(rows, dtype=np.float32)
    return np.nan_to_num(np.std(matrix, axis=1, ddof=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def sanitize_scores(scores: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return np.zeros_like(array, dtype=np.float64)
    low = float(np.min(finite) - 1.0)
    high = float(np.max(finite) + 1.0)
    return np.nan_to_num(array, nan=low, neginf=low, posinf=high).astype(np.float64, copy=False)


def auc_rank(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = np.sum(y == 1)
    negatives = np.sum(y == 0)
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / max(positives * negatives, 1))


def output_dataset_key(dataset_key: str, feature_mode: str) -> str:
    return f"{dataset_key}_enriched_features" if feature_mode == "enriched" else dataset_key


def model_display_name(model_key: str) -> str:
    return {
        "template_knn": "Template kNN Distance",
        "device_aware_knn": "Device-Aware Template kNN",
        "pca_mahalanobis": "PCA Mahalanobis Distance",
        "oneclass_svm": "One-Class SVM",
        "isolation_forest": "Isolation Forest",
        "lof": "Local Outlier Factor",
    }[model_key]


def model_runtime(model_key: str) -> str:
    return f"python_sklearn_{model_key}" if model_key not in {"template_knn", "device_aware_knn"} else f"python_numpy_{model_key}"


def unique_sorted(values: np.ndarray) -> list[str]:
    return sorted(set(values.astype(str).tolist()), key=subject_sort_key)


def subject_sort_key(value: str):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def canonical_collected_subject(value) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return text
    parts = text.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return text


def numeric_series(value, length: int) -> np.ndarray:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return np.full(length, num(value), dtype=np.float32)


def safe_get(row, key: str, default=""):
    try:
        value = row.get(key, default)
    except AttributeError:
        return default
    if pd.isna(value):
        return default
    return value


def num(value) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    return number if math.isfinite(number) else 0.0


def boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not_available"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}-{uuid.uuid4().hex[:8]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


if __name__ == "__main__":
    raise SystemExit(main())
