#!/usr/bin/env python3
"""
Browser-compatible MMC `.npy` benchmark for continuous keystroke authentication.

This script analyzes the supplementary NumPy files distributed with:

    Kim, J. and Kang, P. (2020).
    "Freely typed keystroke dynamics-based user authentication for mobile
    devices based on heterogeneous features", Pattern Recognition 108, 107556.

The paper's FACT data contains heterogeneous mobile-only features:
accelerometer, touch coordinates, pressure/geometry-like touch descriptors, and
time features. For a hardware-agnostic web-browser deployment scenario, this
benchmark keeps only keyboard-event timing channels that can be reconstructed
from browser keydown/keyup timestamps. It excludes:

  - accelerometer channels
  - touch coordinate / pressure / contact-geometry channels
  - key identity channels by default, even though key codes are browser-visible,
    so the default benchmark remains timing-only and text/content independent.

Important dataset detail
------------------------
The supplementary materials are organized as four feature tensors and four
label arrays. This script follows the 1:1 pairing provided by file order:

  mmc1.npy -> mmc5.npy
  mmc2.npy -> mmc6.npy
  mmc3.npy -> mmc7.npy
  mmc4.npy -> mmc8.npy

The label arrays have shape (N, 2). Column 0 is the subject/user ID; column 1
is retained as auxiliary metadata and reported in pair_manifest.csv for audit.

The default split follows the paper's reference/test partition rather than a
random split. This avoids temporal leakage and matches a realistic enrollment
then future-authentication workflow.

Dependencies:
  pip install numpy pandas scikit-learn lightgbm tensorflow matplotlib seaborn
"""

from __future__ import annotations

import argparse
import gc
import os
import time
import tracemalloc
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/codex-cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

try:
    import lightgbm as lgb
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow import keras
    from tensorflow.keras import layers
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name!r}. Install the benchmark stack with:\n"
        "  pip install numpy pandas scikit-learn lightgbm tensorflow matplotlib seaborn"
    ) from exc

tf.get_logger().setLevel("ERROR")


RANDOM_STATE = 42
EER_TARGET = 0.10
DEFAULT_DATA_DIR = "ScienceDirect_files_20Apr2026_10-05-23.390"
DEFAULT_OUTPUT_DIR = "benchmark_outputs_mmc_web_temporal"

# Observed MMC channel layout:
#   0-1   : key IDs for current/next key transition (browser-visible, excluded
#           by default for timing-only evaluation)
#   2-13  : accelerometer-derived current/next features (mobile-only)
#   14-19 : timing features derived from key press/release events
#   20-31 : touch coordinate / pressure / geometry features (mobile-only)
#
# Timing channel interpretation inferred from the row-to-row overlap pattern:
#   14 = current hold time
#   15 = next hold time
#   16 = current down -> next down
#   17 = current up   -> next up
#   18 = current down -> next up
#   19 = current up   -> next down
DEFAULT_TEMPORAL_INDICES = (14, 15, 16, 17, 18, 19)
DEFAULT_TIME_SCALE = 1_000_000.0  # Android nanoseconds -> milliseconds.


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedData:
    """Temporal-only MMC dataset.

    X is 3D: (samples, padded_transitions, browser_timing_channels).
    y is binary: 1 = genuine user, 0 = imposter.
    partitions is "reference" or "test" from the paper's data partition.
    """

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    feature_names: List[str]
    genuine_subject: str
    sample_order: np.ndarray
    partitions: np.ndarray
    source_feature_files: np.ndarray
    source_label_files: np.ndarray
    sequence_lengths: np.ndarray
    temporal_feature_indices: Tuple[int, ...]
    pair_manifest: pd.DataFrame


@dataclass(frozen=True)
class PreparedViews:
    """Train/test splits plus model-specific input views."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    X_train_flat_raw: np.ndarray
    X_test_flat_raw: np.ndarray
    X_train_seq_scaled: np.ndarray
    X_test_seq_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    subjects_train: np.ndarray
    subjects_test: np.ndarray
    order_train: np.ndarray
    order_test: np.ndarray
    partitions_train: np.ndarray
    partitions_test: np.ndarray
    sequence_shape: Tuple[int, ...]
    split_strategy: str


@dataclass(frozen=True)
class SecurityMetrics:
    far: float
    frr: float
    eer: float
    accuracy: float
    auc: float
    threshold: float


@dataclass(frozen=True)
class EfficiencyMetrics:
    inference_time_ms: float
    ui_blocking_time_ms: float
    peak_memory_mb: float


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_view: str
    factory: Callable[[], "AuthModel"]


@dataclass(frozen=True)
class ResolvedPair:
    feature_file: str
    label_file: str
    partition: str
    mean_aux_value: float
    mean_abs_nonpad_aux_delta: float
    exact_nonpad_aux_matches: int


# ---------------------------------------------------------------------------
# MMC loading, pairing, and browser-compatible feature filtering
# ---------------------------------------------------------------------------


def load_label_file(path: Path) -> np.ndarray:
    labels = np.load(path, allow_pickle=True)
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError(f"{path.name} must be a 2D label array with at least 2 columns.")
    return labels.astype(str)


def feature_sequence_lengths(
    feature_path: Path,
    length_channel: int = DEFAULT_TEMPORAL_INDICES[0],
) -> np.ndarray:
    """Count non-padded transitions from a timing channel."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.load(feature_path, allow_pickle=True)
    if X.ndim != 3:
        raise ValueError(f"{feature_path.name} must be 3D, got shape {X.shape}.")
    if length_channel >= X.shape[2]:
        raise ValueError(
            f"Length channel {length_channel} is outside {feature_path.name}'s "
            f"feature dimension {X.shape[2]}."
        )

    channel = X[:, :, length_channel].astype(np.float32)
    lengths = (np.abs(channel) > 1e-12).sum(axis=1).astype(np.int32)
    del X, channel
    gc.collect()
    return lengths


def resolve_feature_label_pairs(
    data_dir: Path,
    feature_files: Sequence[str],
    label_files: Sequence[str],
    subject_label_column: int,
    length_label_column: int,
    reference_label_files: Sequence[str],
    test_label_files: Sequence[str],
) -> List[ResolvedPair]:
    """Pair feature and label files by the supplementary-material file order.

    The paper bundle provides four feature tensors followed by four label
    arrays. The intended 1:1 mapping is positional: mmc1->mmc5, mmc2->mmc6,
    mmc3->mmc7, and mmc4->mmc8. The auxiliary metadata column is not used for
    remapping; non-padded length comparisons are only exported as diagnostics.
    """

    if len(feature_files) != len(label_files):
        raise ValueError("feature_files and label_files must have the same length.")
    if subject_label_column == length_label_column:
        raise ValueError("subject_label_column and length_label_column must differ.")

    pairs: List[ResolvedPair] = []
    for feature_name, label_name in zip(feature_files, label_files):
        labels = load_label_file(data_dir / label_name)
        feature_lengths = feature_sequence_lengths(data_dir / feature_name)
        if len(feature_lengths) != len(labels):
            raise ValueError(
                f"Sample mismatch: {feature_name} has {len(feature_lengths)} "
                f"samples but {label_name} has {len(labels)} labels."
            )
        aux_values = labels[:, length_label_column].astype(np.int32)
        diff = aux_values - feature_lengths
        if label_name in reference_label_files:
            partition = "reference"
        elif label_name in test_label_files:
            partition = "test"
        else:
            raise ValueError(
                f"Label file {label_name} is not listed as reference or test."
            )

        pairs.append(
            ResolvedPair(
                feature_file=feature_name,
                label_file=label_name,
                partition=partition,
                mean_aux_value=float(aux_values.mean()),
                mean_abs_nonpad_aux_delta=float(np.mean(np.abs(diff))),
                exact_nonpad_aux_matches=int(np.sum(diff == 0)),
            )
        )

    if len(pairs) != len(feature_files):
        raise ValueError(
            f"Expected {len(feature_files)} feature/label pairs, resolved {len(pairs)}."
        )

    return pairs


def coerce_feature_tensor(path: Path) -> np.ndarray:
    """Load and convert object-typed ScienceDirect tensors to float32."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.load(path, allow_pickle=True)
    try:
        return X.astype(np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path.name} could not be converted to float32.") from exc


def resolve_temporal_indices(
    temporal_feature_indices: Optional[Sequence[int]],
) -> Tuple[int, ...]:
    if temporal_feature_indices:
        indices = tuple(int(idx) for idx in temporal_feature_indices)
    else:
        indices = DEFAULT_TEMPORAL_INDICES

    if not indices:
        raise ValueError("At least one timing feature index must be selected.")
    return indices


def timing_feature_names(indices: Sequence[int]) -> List[str]:
    names = {
        14: "hold_current_ms",
        15: "hold_next_ms",
        16: "down_down_ms",
        17: "up_up_ms",
        18: "down_up_ms",
        19: "up_down_ms",
    }
    return [names.get(idx, f"timing_ch_{idx}_ms") for idx in indices]


def choose_genuine_subject(subjects: np.ndarray) -> Tuple[str, Dict[str, int]]:
    counts = Counter(map(str, subjects))
    if not counts:
        raise ValueError("No subject IDs found.")

    def rank(subject_id: str) -> Tuple[int, object]:
        if subject_id.isdigit():
            return (0, -int(subject_id))
        return (1, subject_id)

    genuine_subject, _ = max(counts.items(), key=lambda item: (item[1], rank(item[0])))
    return str(genuine_subject), {str(k): int(v) for k, v in counts.items()}


def load_npy_data(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    feature_files: Sequence[str] = ("mmc1.npy", "mmc2.npy", "mmc3.npy", "mmc4.npy"),
    label_files: Sequence[str] = ("mmc5.npy", "mmc6.npy", "mmc7.npy", "mmc8.npy"),
    reference_label_files: Sequence[str] = ("mmc5.npy", "mmc6.npy"),
    test_label_files: Sequence[str] = ("mmc7.npy", "mmc8.npy"),
    subject_label_column: int = 0,
    length_label_column: int = 1,
    temporal_feature_indices: Optional[Sequence[int]] = None,
    genuine_subject: str = "auto",
    time_scale: float = DEFAULT_TIME_SCALE,
) -> LoadedData:
    """Load MMC data and retain only browser-reconstructable timing features."""

    data_dir = Path(data_dir)
    missing = [
        name
        for name in [*feature_files, *label_files]
        if not (data_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing required MMC file(s) in {data_dir}: {', '.join(missing)}"
        )

    temporal_indices = resolve_temporal_indices(temporal_feature_indices)
    pairs = resolve_feature_label_pairs(
        data_dir=data_dir,
        feature_files=feature_files,
        label_files=label_files,
        subject_label_column=subject_label_column,
        length_label_column=length_label_column,
        reference_label_files=reference_label_files,
        test_label_files=test_label_files,
    )

    X_parts: List[np.ndarray] = []
    subject_parts: List[np.ndarray] = []
    partition_parts: List[np.ndarray] = []
    source_feature_parts: List[np.ndarray] = []
    source_label_parts: List[np.ndarray] = []
    length_parts: List[np.ndarray] = []
    order_parts: List[np.ndarray] = []

    for pair_rank, pair in enumerate(pairs):
        labels = load_label_file(data_dir / pair.label_file)
        subjects = labels[:, subject_label_column].astype(str)
        lengths = labels[:, length_label_column].astype(np.int32)

        X_full = coerce_feature_tensor(data_dir / pair.feature_file)
        if max(temporal_indices) >= X_full.shape[2]:
            raise ValueError(
                f"Timing indices {temporal_indices} exceed {pair.feature_file}'s "
                f"feature dimension {X_full.shape[2]}."
            )
        if X_full.shape[0] != len(subjects):
            raise ValueError(
                f"Sample mismatch: {pair.feature_file} has {X_full.shape[0]} "
                f"samples but {pair.label_file} has {len(subjects)} labels."
            )

        X_timing = X_full[:, :, temporal_indices].astype(np.float32)
        if time_scale != 1.0:
            X_timing = X_timing / np.float32(time_scale)
        X_timing = np.nan_to_num(X_timing, nan=0.0, posinf=0.0, neginf=0.0)

        # Deterministic order for cold-start prefixes. The first reference file
        # is consumed before the second reference file; rows are already grouped
        # as ten samples per subject in the supplementary arrays.
        order = (pair_rank * 1_000_000 + np.arange(len(subjects))).astype(np.int32)

        X_parts.append(X_timing)
        subject_parts.append(subjects)
        partition_parts.append(np.full(len(subjects), pair.partition, dtype=object))
        source_feature_parts.append(np.full(len(subjects), pair.feature_file, dtype=object))
        source_label_parts.append(np.full(len(subjects), pair.label_file, dtype=object))
        length_parts.append(lengths)
        order_parts.append(order)

        del X_full, X_timing
        gc.collect()

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    subjects_all = np.concatenate(subject_parts).astype(str)
    partitions = np.concatenate(partition_parts).astype(str)
    source_feature_files = np.concatenate(source_feature_parts).astype(str)
    source_label_files = np.concatenate(source_label_parts).astype(str)
    sequence_lengths = np.concatenate(length_parts).astype(np.int32)
    sample_order = np.concatenate(order_parts).astype(np.int32)

    if genuine_subject == "auto":
        selected_genuine, _ = choose_genuine_subject(subjects_all)
    else:
        selected_genuine = str(genuine_subject)
    y = (subjects_all == selected_genuine).astype(np.int32)
    if y.sum() == 0:
        raise ValueError(f"No samples found for genuine subject {selected_genuine!r}.")

    pair_manifest = pd.DataFrame(
        [
            {
                "feature_file": pair.feature_file,
                "label_file": pair.label_file,
                "partition": pair.partition,
                "mean_aux_value": pair.mean_aux_value,
                "mean_abs_nonpad_aux_delta": pair.mean_abs_nonpad_aux_delta,
                "exact_nonpad_aux_matches": pair.exact_nonpad_aux_matches,
            }
            for pair in pairs
        ]
    )

    return LoadedData(
        X=X,
        y=y,
        subjects=subjects_all,
        feature_names=timing_feature_names(temporal_indices),
        genuine_subject=selected_genuine,
        sample_order=sample_order,
        partitions=partitions,
        source_feature_files=source_feature_files,
        source_label_files=source_label_files,
        sequence_lengths=sequence_lengths,
        temporal_feature_indices=temporal_indices,
        pair_manifest=pair_manifest,
    )


# ---------------------------------------------------------------------------
# Train/test preparation
# ---------------------------------------------------------------------------


def flatten_samples(X: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        return X.astype(np.float32, copy=False)
    if X.ndim == 3:
        return X.reshape(X.shape[0], -1).astype(np.float32, copy=False)
    raise ValueError(f"Expected 2D or 3D X, got shape {X.shape}.")


def chronological_subject_split(
    data: LoadedData,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Fallback chronological split when the paper split is not requested."""

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for subject in pd.unique(data.subjects):
        subject_idx = np.flatnonzero(data.subjects == subject)
        ordered = subject_idx[np.argsort(data.sample_order[subject_idx], kind="stable")]
        n_train = int(np.ceil((1.0 - test_size) * len(ordered)))
        n_train = min(max(n_train, 1), len(ordered) - 1)
        train_parts.append(ordered[:n_train])
        test_parts.append(ordered[n_train:])

    return (
        np.concatenate(train_parts).astype(np.int32),
        np.concatenate(test_parts).astype(np.int32),
        "chronological_by_sample_order",
    )


def prepare_views(
    data: LoadedData,
    split_mode: str = "paper_reference_test",
    test_size: float = 0.30,
) -> PreparedViews:
    """Create train/test views without random shuffle."""

    if split_mode == "paper_reference_test":
        train_idx = np.flatnonzero(data.partitions == "reference").astype(np.int32)
        test_idx = np.flatnonzero(data.partitions == "test").astype(np.int32)
        split_strategy = "paper_reference_test"
    elif split_mode == "chronological_70_30":
        train_idx, test_idx, split_strategy = chronological_subject_split(data, test_size)
    else:
        raise ValueError(f"Unknown split_mode {split_mode!r}.")

    if len(np.unique(data.y[train_idx])) != 2 or len(np.unique(data.y[test_idx])) != 2:
        raise ValueError("Train and test splits must both contain genuine and imposters.")

    X_train = data.X[train_idx]
    X_test = data.X[test_idx]

    X_train_flat_raw = flatten_samples(X_train)
    X_test_flat_raw = flatten_samples(X_test)

    scaler = StandardScaler()
    X_train_flat_scaled = scaler.fit_transform(X_train_flat_raw).astype(np.float32)
    X_test_flat_scaled = scaler.transform(X_test_flat_raw).astype(np.float32)

    sequence_shape = data.X.shape[1:]
    X_train_seq_scaled = X_train_flat_scaled.reshape(
        X_train_flat_scaled.shape[0], *sequence_shape
    ).astype(np.float32)
    X_test_seq_scaled = X_test_flat_scaled.reshape(
        X_test_flat_scaled.shape[0], *sequence_shape
    ).astype(np.float32)

    return PreparedViews(
        train_indices=train_idx,
        test_indices=test_idx,
        X_train_flat_raw=X_train_flat_raw,
        X_test_flat_raw=X_test_flat_raw,
        X_train_seq_scaled=X_train_seq_scaled,
        X_test_seq_scaled=X_test_seq_scaled,
        y_train=data.y[train_idx].astype(np.int32),
        y_test=data.y[test_idx].astype(np.int32),
        subjects_train=data.subjects[train_idx],
        subjects_test=data.subjects[test_idx],
        order_train=data.sample_order[train_idx],
        order_test=data.sample_order[test_idx],
        partitions_train=data.partitions[train_idx],
        partitions_test=data.partitions[test_idx],
        sequence_shape=sequence_shape,
        split_strategy=split_strategy,
    )


def get_view(views: PreparedViews, split: str, view_name: str) -> np.ndarray:
    attr = f"X_{split}_{view_name}"
    if not hasattr(views, attr):
        raise ValueError(f"Unknown view {view_name!r}; expected {attr!r}.")
    return getattr(views, attr)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AuthModel:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AuthModel":
        raise NotImplementedError

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return scores where larger means more likely genuine."""

        raise NotImplementedError


class ScaledManhattanDistanceModel(AuthModel):
    """Ultra-light one-class baseline using genuine-user timing template."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScaledManhattanDistanceModel":
        genuine = X[y == 1]
        if genuine.size == 0:
            raise ValueError("Scaled Manhattan model needs at least one genuine sample.")
        self.center_ = genuine.mean(axis=0)
        self.scale_ = genuine.std(axis=0)
        self.scale_ = np.where(self.scale_ < self.epsilon, self.epsilon, self.scale_)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        distances = np.abs((X - self.center_) / self.scale_).sum(axis=1)
        return -distances.astype(np.float32)


class LightGBMAuthModel(AuthModel):
    """Lightweight tree ensemble via native LightGBM Booster."""

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.random_state = random_state
        self.model: Optional[lgb.Booster] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMAuthModel":
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int32)
        positive = int((y_np == 1).sum())
        negative = int((y_np == 0).sum())

        train_set = lgb.Dataset(X_np, label=y_np, free_raw_data=True)
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 10,
            "bagging_fraction": 0.90,
            "bagging_freq": 1,
            "feature_fraction": 0.90,
            "lambda_l2": 1.0,
            "scale_pos_weight": negative / max(positive, 1),
            "verbosity": -1,
            "seed": self.random_state,
            "feature_pre_filter": False,
        }
        self.model = lgb.train(params, train_set, num_boost_round=150)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(np.asarray(X, dtype=np.float32)).astype(np.float32)


class CNN1DAuthModel(AuthModel):
    """Small 1D-CNN over timing-transition sequences."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        epochs: int = 25,
        batch_size: int = 128,
        random_state: int = RANDOM_STATE,
        verbose: int = 0,
    ) -> None:
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.model: Optional[keras.Model] = None

    def _build(self) -> keras.Model:
        tf.keras.utils.set_random_seed(self.random_state)

        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv1D(24, kernel_size=5, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(48, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.20)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs, outputs, name="mmc_browser_timing_1d_cnn")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
        )
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNN1DAuthModel":
        self.model = self._build()
        classes = np.array([0, 1], dtype=np.int32)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="loss",
                mode="min",
                patience=5,
                restore_best_weights=True,
            )
        ]

        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=self.verbose,
        )
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(X, batch_size=self.batch_size, verbose=0).ravel().astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics and evaluation
# ---------------------------------------------------------------------------


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def compute_security_metrics(y_true: np.ndarray, scores: np.ndarray) -> SecurityMetrics:
    """Report FAR/FRR at the same threshold that defines test-set EER."""

    eer, threshold = compute_eer(y_true, scores)
    y_pred = (scores >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    auc = roc_auc_score(y_true, scores)
    accuracy = accuracy_score(y_true, y_pred)
    return SecurityMetrics(
        far=float(far),
        frr=float(frr),
        eer=float(eer),
        accuracy=float(accuracy),
        auc=float(auc),
        threshold=float(threshold),
    )


def measure_inference(
    predict_scores: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    warmup_size: int = 32,
) -> Tuple[np.ndarray, EfficiencyMetrics]:
    if len(X_test) > 0:
        predict_scores(X_test[: min(warmup_size, len(X_test))])

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()
    scores = predict_scores(X_test)
    elapsed = time.perf_counter() - start_time
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    blocking_time_ms = elapsed * 1000.0
    return scores, EfficiencyMetrics(
        inference_time_ms=float(blocking_time_ms / max(len(X_test), 1)),
        ui_blocking_time_ms=float(blocking_time_ms),
        peak_memory_mb=float(peak_bytes / (1024.0 * 1024.0)),
    )


def train_and_evaluate_model(
    spec: ModelSpec,
    views: PreparedViews,
) -> Tuple[AuthModel, SecurityMetrics, EfficiencyMetrics, Dict[str, np.ndarray]]:
    X_train = get_view(views, "train", spec.input_view)
    X_test = get_view(views, "test", spec.input_view)

    model = spec.factory()
    model.fit(X_train, views.y_train)
    test_scores, efficiency = measure_inference(model.predict_scores, X_test)
    security = compute_security_metrics(views.y_test, test_scores)
    fpr, tpr, _ = roc_curve(views.y_test, test_scores, pos_label=1)
    return model, security, efficiency, {"fpr": fpr, "tpr": tpr, "scores": test_scores}


def make_training_sizes(y_train: np.ndarray, requested_sizes: Iterable[int]) -> List[int]:
    max_genuine = int((y_train == 1).sum())
    sizes = [int(size) for size in requested_sizes if 0 < int(size) <= max_genuine]
    sizes.append(max_genuine)
    return sorted(set(sizes))


def evaluate_data_efficiency(
    spec: ModelSpec,
    views: PreparedViews,
    training_sizes: Iterable[int],
    target_eer: float = EER_TARGET,
) -> Tuple[pd.DataFrame, Optional[int]]:
    """Use sequential genuine prefixes: first 1, first 2, first 5, etc."""

    X_train_full = get_view(views, "train", spec.input_view)
    X_test = get_view(views, "test", spec.input_view)
    genuine_idx = np.flatnonzero(views.y_train == 1)
    imposter_idx = np.flatnonzero(views.y_train == 0)
    genuine_idx = genuine_idx[np.argsort(views.order_train[genuine_idx], kind="stable")]
    imposter_idx = imposter_idx[np.argsort(views.order_train[imposter_idx], kind="stable")]

    rows = []
    min_requirement: Optional[int] = None
    for size in training_sizes:
        chosen_genuine = genuine_idx[:size]
        chosen_idx = np.concatenate([chosen_genuine, imposter_idx])

        model = spec.factory()
        model.fit(X_train_full[chosen_idx], views.y_train[chosen_idx])
        scores = model.predict_scores(X_test)
        eer, _ = compute_eer(views.y_test, scores)

        rows.append(
            {
                "Model": spec.name,
                "Genuine Training Samples": int(size),
                "EER": float(eer),
            }
        )
        if min_requirement is None and eer < target_eer:
            min_requirement = int(size)

        if isinstance(model, CNN1DAuthModel):
            keras.backend.clear_session()
        del model
        gc.collect()

    return pd.DataFrame(rows), min_requirement


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_roc_curves(
    curve_data_by_model: Dict[str, Dict[str, np.ndarray]],
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    for model_name, curve_data in curve_data_by_model.items():
        auc = float(summary_df.loc[summary_df["Model"] == model_name, "AUC"].iloc[0])
        plt.plot(
            curve_data["fpr"],
            curve_data["tpr"],
            linewidth=2,
            label=f"{model_name} (AUC={auc:.3f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Acceptance Rate (FAR / FPR)")
    plt.ylabel("True Acceptance Rate (TPR)")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_efficiency_bars(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(
        data=summary_df,
        x="Model",
        y="Inference Time (ms/sample)",
        hue="Model",
        ax=axes[0],
        palette="Set2",
        legend=False,
    )
    axes[0].set_title("Inference Time")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=summary_df,
        x="Model",
        y="Peak Memory During Inference (MB)",
        hue="Model",
        ax=axes[1],
        palette="Set2",
        legend=False,
    )
    axes[1].set_title("Peak Memory")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_eer_vs_training_size(data_efficiency_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=data_efficiency_df,
        x="Genuine Training Samples",
        y="EER",
        hue="Model",
        marker="o",
        linewidth=2,
    )
    plt.axhline(EER_TARGET, linestyle="--", color="red", linewidth=1, label="10% EER target")
    plt.xlabel("Genuine Training Samples")
    plt.ylabel("Equal Error Rate (EER)")
    plt.title("Data Efficiency")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_model_specs(
    sequence_shape: Tuple[int, ...],
    cnn_epochs: int,
    cnn_batch_size: int,
) -> List[ModelSpec]:
    return [
        ModelSpec(
            name="Scaled Manhattan Distance",
            input_view="flat_raw",
            factory=lambda: ScaledManhattanDistanceModel(),
        ),
        ModelSpec(
            name="LightGBM Classifier",
            input_view="flat_raw",
            factory=lambda: LightGBMAuthModel(random_state=RANDOM_STATE),
        ),
        ModelSpec(
            name="1D-CNN",
            input_view="seq_scaled",
            factory=lambda: CNN1DAuthModel(
                input_shape=sequence_shape,
                epochs=cnn_epochs,
                batch_size=cnn_batch_size,
                random_state=RANDOM_STATE,
                verbose=0,
            ),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MMC .npy benchmark using browser-compatible timing features only."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Folder containing mmc1.npy..mmc8.npy.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for CSV and plot outputs.")
    parser.add_argument(
        "--genuine-subject",
        default="auto",
        help="Subject ID to treat as genuine. Use 'auto' to select the max-sample subject.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["paper_reference_test", "chronological_70_30"],
        default="paper_reference_test",
        help="Use paper reference/test split by default; no random shuffle is used.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Only used by chronological_70_30 fallback split.",
    )
    parser.add_argument("--cnn-epochs", type=int, default=25, help="Maximum CNN training epochs.")
    parser.add_argument("--cnn-batch-size", type=int, default=128, help="CNN batch size.")
    parser.add_argument(
        "--training-sizes",
        type=int,
        nargs="*",
        default=[1, 2, 5, 10],
        help="Sequential genuine enrollment sizes. MAX is always added automatically.",
    )
    parser.add_argument(
        "--temporal-feature-indices",
        type=int,
        nargs="*",
        default=None,
        help="Override timing channel indices. Default keeps MMC channels 14-19.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=DEFAULT_TIME_SCALE,
        help="Divide raw timing channels by this value. Default converts ns to ms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    sns.set_theme(style="whitegrid")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_npy_data(
        data_dir=args.data_dir,
        temporal_feature_indices=args.temporal_feature_indices,
        genuine_subject=args.genuine_subject,
        time_scale=args.time_scale,
    )
    views = prepare_views(data, split_mode=args.split_mode, test_size=args.test_size)
    model_specs = build_model_specs(
        sequence_shape=views.sequence_shape,
        cnn_epochs=args.cnn_epochs,
        cnn_batch_size=args.cnn_batch_size,
    )
    training_sizes = make_training_sizes(views.y_train, args.training_sizes)

    print("\nMMC dataset summary")
    print(f"  Samples: {len(data.y):,}")
    print(f"  Sequence shape: {data.X.shape[1:]}")
    print(f"  Flattened features: {views.X_train_flat_raw.shape[1]:,}")
    print(f"  Browser timing indices: {list(data.temporal_feature_indices)}")
    print(f"  Timing feature names: {data.feature_names}")
    print(f"  Genuine subject: {data.genuine_subject}")
    print(f"  Genuine samples: {int(data.y.sum()):,}")
    print(f"  Split strategy: {views.split_strategy}")
    print(f"  Train/Test: {len(views.y_train):,}/{len(views.y_test):,}")
    print(f"  Train genuine/test genuine: {int(views.y_train.sum())}/{int(views.y_test.sum())}")
    print(f"  Data efficiency sizes: {training_sizes}")
    print("\nFeature/label pairs")
    print(data.pair_manifest.to_string(index=False))
    print()

    summary_rows = []
    curve_data_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    data_efficiency_frames = []

    for spec in model_specs:
        print(f"Training and evaluating: {spec.name}")
        model, security, efficiency, curve_data = train_and_evaluate_model(spec, views)
        curve_data_by_model[spec.name] = curve_data

        print(f"  Data efficiency sweep: {spec.name}")
        data_eff_df, min_requirement = evaluate_data_efficiency(
            spec,
            views,
            training_sizes=training_sizes,
            target_eer=EER_TARGET,
        )
        data_efficiency_frames.append(data_eff_df)

        summary_rows.append(
            {
                "Model": spec.name,
                "FAR": security.far,
                "FRR": security.frr,
                "EER": security.eer,
                "Accuracy": security.accuracy,
                "AUC": security.auc,
                "Inference Time (ms/sample)": efficiency.inference_time_ms,
                "UI Blocking Time (ms/test batch)": efficiency.ui_blocking_time_ms,
                "Peak Memory During Inference (MB)": efficiency.peak_memory_mb,
                "Min Genuine Samples for EER < 10%": (
                    min_requirement if min_requirement is not None else "Not reached"
                ),
            }
        )

        if isinstance(model, CNN1DAuthModel):
            keras.backend.clear_session()
        del model
        gc.collect()

    summary_df = pd.DataFrame(summary_rows)
    data_efficiency_df = pd.concat(data_efficiency_frames, ignore_index=True)

    summary_csv = output_dir / "summary_metrics.csv"
    data_efficiency_csv = output_dir / "data_efficiency_eer.csv"
    pair_manifest_csv = output_dir / "pair_manifest.csv"
    summary_df.to_csv(summary_csv, index=False)
    data_efficiency_df.to_csv(data_efficiency_csv, index=False)
    data.pair_manifest.to_csv(pair_manifest_csv, index=False)

    plot_roc_curves(curve_data_by_model, summary_df, output_dir / "roc_curves.png")
    plot_efficiency_bars(summary_df, output_dir / "efficiency_bars.png")
    plot_eer_vs_training_size(data_efficiency_df, output_dir / "eer_vs_training_size.png")

    print("\nSummary metrics")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        180,
        "display.float_format",
        "{:.4f}".format,
    ):
        print(summary_df)

    print("\nData efficiency EER values")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        120,
        "display.float_format",
        "{:.4f}".format,
    ):
        print(data_efficiency_df)

    print("\nSaved outputs")
    print(f"  {summary_csv}")
    print(f"  {data_efficiency_csv}")
    print(f"  {pair_manifest_csv}")
    print(f"  {output_dir / 'roc_curves.png'}")
    print(f"  {output_dir / 'efficiency_bars.png'}")
    print(f"  {output_dir / 'eer_vs_training_size.png'}")


if __name__ == "__main__":
    main()
