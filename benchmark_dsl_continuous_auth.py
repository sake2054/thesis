#!/usr/bin/env python3
"""
Benchmark continuous biometric authentication models on the
DSL-StrongPasswordData.csv keystroke dynamics dataset.

The script is intentionally modular:
  1. load_data() returns X, y, and metadata only.
  2. prepare_views() creates the flat and sequence views used by models.
  3. models expose fit() and predict_scores() with "higher score = genuine".
  4. the same metric/evaluation code can be reused for future .npy loaders.

Target task:
  Genuine user: s002 -> label 1
  Imposters: all other subjects -> label 0

Dependencies:
  pip install pandas numpy scikit-learn lightgbm tensorflow matplotlib seaborn
"""

from __future__ import annotations

import argparse
import gc
import os
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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
        "  pip install pandas numpy scikit-learn lightgbm tensorflow matplotlib seaborn"
    ) from exc


RANDOM_STATE = 42
GENUINE_SUBJECT = "s002"
EER_TARGET = 0.10


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedData:
    """Raw data returned by load_data().

    X may be either:
      - 2D: (samples, features), as in DSL-StrongPasswordData.csv
      - 3D: (samples, timesteps, channels/features), as in future .npy data

    y is always binary: 1 = genuine, 0 = imposter.
    """

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    feature_names: List[str]
    sample_order: Optional[np.ndarray] = None
    sessions: Optional[np.ndarray] = None
    repetitions: Optional[np.ndarray] = None


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
    sessions_train: Optional[np.ndarray]
    sessions_test: Optional[np.ndarray]
    sequence_shape: Tuple[int, ...]
    flat_feature_names: List[str]
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


# ---------------------------------------------------------------------------
# Data loading and adaptation
# ---------------------------------------------------------------------------


def load_data(
    csv_path: str | Path = "DSL-StrongPasswordData.csv",
    genuine_subject: str = GENUINE_SUBJECT,
) -> LoadedData:
    """Load DSL-StrongPasswordData.csv and build the binary auth task.

    To adapt this function for future .npy experiments, keep the same return
    contract:
      X: np.ndarray, either 2D (n_samples, n_features) or 3D
         (n_samples, timesteps, channels)
      y: np.ndarray with 1 for the target/genuine user and 0 for imposters
      subjects: optional subject/user ids for traceability
      feature_names: names for flattened features if available

    Example .npy sketch:

        X = np.load("keystrokes_X.npy")        # shape: (N, T, C)
        subjects = np.load("subjects.npy")    # shape: (N,)
        y = (subjects == genuine_subject).astype(int)
        feature_names = [f"t{t}_c{c}" for t in range(X.shape[1])
                         for c in range(X.shape[2])]
        return LoadedData(X=X.astype(np.float32), y=y, subjects=subjects,
                          feature_names=feature_names,
                          sample_order=np.arange(len(y), dtype=np.int32))

    No model or metric code needs to change if the returned shapes follow this
    interface.
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    feature_names = [
        col for col in df.columns if col.startswith(("H", "UD", "DD"))
    ]
    if not feature_names:
        raise ValueError("No feature columns found. Expected columns starting with H, UD, or DD.")
    if "subject" not in df.columns:
        raise ValueError("Expected a 'subject' column in the DSL CSV file.")

    X = df[feature_names].astype(np.float32).to_numpy()
    subjects = df["subject"].astype(str).to_numpy()
    y = (subjects == genuine_subject).astype(np.int32)
    sample_order = np.arange(len(df), dtype=np.int32)
    sessions = (
        df["sessionIndex"].astype(np.int32).to_numpy()
        if "sessionIndex" in df.columns
        else None
    )
    repetitions = (
        df["rep"].astype(np.int32).to_numpy()
        if "rep" in df.columns
        else None
    )

    if y.sum() == 0:
        raise ValueError(f"No samples found for genuine subject {genuine_subject!r}.")

    return LoadedData(
        X=X,
        y=y,
        subjects=subjects,
        feature_names=feature_names,
        sample_order=sample_order,
        sessions=sessions,
        repetitions=repetitions,
    )


def flatten_samples(X: np.ndarray) -> np.ndarray:
    """Flatten each sample while preserving the sample dimension."""

    if X.ndim == 2:
        return X
    if X.ndim == 3:
        return X.reshape(X.shape[0], -1)
    raise ValueError(f"Expected X to be 2D or 3D, got shape {X.shape}.")


def sequence_view_from_scaled_flat(
    X_scaled_flat: np.ndarray,
    original_sample_shape: Tuple[int, ...],
) -> np.ndarray:
    """Convert scaled flat features into Conv1D-ready sequence tensors."""

    if len(original_sample_shape) == 1:
        # DSL tabular data: treat the feature vector as a 1D temporal sequence.
        return X_scaled_flat.reshape(X_scaled_flat.shape[0], original_sample_shape[0], 1)
    if len(original_sample_shape) == 2:
        # Future .npy data: restore (timesteps, channels).
        return X_scaled_flat.reshape(X_scaled_flat.shape[0], *original_sample_shape)
    raise ValueError(f"Expected sample shape length 1 or 2, got {original_sample_shape}.")


def chronological_subject_split(
    data: LoadedData,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Split each subject by time, never by random shuffle.

    If session labels are available, complete sessions are kept intact. For the
    DSL dataset this means the default 30% test target becomes sessions 1-6 for
    training and sessions 7-8 for testing. That slight 75/25 split is preferable
    to cutting through a session and better matches a real enrollment-then-use
    continuous-authentication workflow.
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

    sample_order = (
        data.sample_order
        if data.sample_order is not None
        else np.arange(len(data.y), dtype=np.int32)
    )
    train_indices: List[np.ndarray] = []
    test_indices: List[np.ndarray] = []
    used_session_split = data.sessions is not None

    for subject in pd.unique(data.subjects):
        subject_idx = np.flatnonzero(data.subjects == subject)
        if len(subject_idx) < 2:
            raise ValueError(f"Subject {subject!r} has fewer than two samples.")

        if data.sessions is not None:
            repetition_key = (
                data.repetitions[subject_idx]
                if data.repetitions is not None
                else sample_order[subject_idx]
            )
            local_order = np.lexsort(
                (sample_order[subject_idx], repetition_key, data.sessions[subject_idx])
            )
        else:
            local_order = np.argsort(sample_order[subject_idx], kind="stable")

        ordered_idx = subject_idx[local_order]

        if data.sessions is not None and len(np.unique(data.sessions[ordered_idx])) > 1:
            ordered_sessions = pd.unique(data.sessions[ordered_idx])
            n_train_sessions = int(np.ceil((1.0 - test_size) * len(ordered_sessions)))
            n_train_sessions = min(max(n_train_sessions, 1), len(ordered_sessions) - 1)
            train_session_values = set(ordered_sessions[:n_train_sessions].tolist())
            is_train = np.array(
                [session in train_session_values for session in data.sessions[ordered_idx]],
                dtype=bool,
            )
            subject_train = ordered_idx[is_train]
            subject_test = ordered_idx[~is_train]
        else:
            used_session_split = False
            n_train = int(np.ceil((1.0 - test_size) * len(ordered_idx)))
            n_train = min(max(n_train, 1), len(ordered_idx) - 1)
            subject_train = ordered_idx[:n_train]
            subject_test = ordered_idx[n_train:]

        train_indices.append(subject_train)
        test_indices.append(subject_test)

    train_idx = np.concatenate(train_indices).astype(np.int32)
    test_idx = np.concatenate(test_indices).astype(np.int32)

    if len(np.unique(data.y[train_idx])) != 2 or len(np.unique(data.y[test_idx])) != 2:
        raise ValueError(
            "Chronological split must contain both genuine and imposter classes "
            "in train and test sets."
        )

    split_strategy = (
        "chronological_by_complete_session"
        if used_session_split
        else "chronological_by_sample_order"
    )
    return train_idx, test_idx, split_strategy


def prepare_views(
    data: LoadedData,
    test_size: float = 0.30,
    random_state: int = RANDOM_STATE,
) -> PreparedViews:
    """Create chronological train/test splits and model-specific views.

    Flat raw features are used by Manhattan Distance and LightGBM.
    Scaled sequence features are used by the 1D-CNN.
    """

    # random_state is kept in the signature for CLI/API compatibility, but the
    # split itself is deterministic and chronological to avoid temporal leakage.
    _ = random_state
    train_idx, test_idx, split_strategy = chronological_subject_split(data, test_size)

    X_train = data.X[train_idx]
    X_test = data.X[test_idx]
    y_train = data.y[train_idx]
    y_test = data.y[test_idx]
    subjects_train = data.subjects[train_idx]
    subjects_test = data.subjects[test_idx]
    sample_order = (
        data.sample_order
        if data.sample_order is not None
        else np.arange(len(data.y), dtype=np.int32)
    )
    order_train = sample_order[train_idx]
    order_test = sample_order[test_idx]
    sessions_train = data.sessions[train_idx] if data.sessions is not None else None
    sessions_test = data.sessions[test_idx] if data.sessions is not None else None

    X_train_flat_raw = flatten_samples(X_train).astype(np.float32)
    X_test_flat_raw = flatten_samples(X_test).astype(np.float32)

    scaler = StandardScaler()
    X_train_flat_scaled = scaler.fit_transform(X_train_flat_raw).astype(np.float32)
    X_test_flat_scaled = scaler.transform(X_test_flat_raw).astype(np.float32)

    original_sample_shape = data.X.shape[1:]
    X_train_seq_scaled = sequence_view_from_scaled_flat(
        X_train_flat_scaled, original_sample_shape
    ).astype(np.float32)
    X_test_seq_scaled = sequence_view_from_scaled_flat(
        X_test_flat_scaled, original_sample_shape
    ).astype(np.float32)

    return PreparedViews(
        train_indices=train_idx,
        test_indices=test_idx,
        X_train_flat_raw=X_train_flat_raw,
        X_test_flat_raw=X_test_flat_raw,
        X_train_seq_scaled=X_train_seq_scaled,
        X_test_seq_scaled=X_test_seq_scaled,
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
        subjects_train=subjects_train,
        subjects_test=subjects_test,
        order_train=order_train,
        order_test=order_test,
        sessions_train=sessions_train,
        sessions_test=sessions_test,
        sequence_shape=X_train_seq_scaled.shape[1:],
        flat_feature_names=data.feature_names,
        split_strategy=split_strategy,
    )


def get_view(views: PreparedViews, split: str, view_name: str) -> np.ndarray:
    """Fetch a named model input view."""

    attr = f"X_{split}_{view_name}"
    if not hasattr(views, attr):
        raise ValueError(f"Unknown view {view_name!r}; expected attribute {attr!r}.")
    return getattr(views, attr)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AuthModel:
    """Minimal model protocol used by the benchmark."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AuthModel":
        raise NotImplementedError

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return continuous scores where larger values mean more genuine."""

        raise NotImplementedError


class ScaledManhattanDistanceModel(AuthModel):
    """Ultra-light template model using scaled Manhattan distance.

    The model stores only a genuine-user centroid and feature scales. It ignores
    imposter samples during fitting, which matches one-class enrollment settings.
    """

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
    """Lightweight tree ensemble classifier."""

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
            "min_data_in_leaf": 20,
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
        X_np = np.asarray(X, dtype=np.float32)
        return self.model.predict(X_np).astype(np.float32)


class CNN1DAuthModel(AuthModel):
    """Small Conv1D model for sequence-shaped biometric signals."""

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
        x = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.20)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs, outputs, name="browser_lightweight_1d_cnn")
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
# Metrics
# ---------------------------------------------------------------------------


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate and its approximate threshold."""

    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def threshold_at_eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose the threshold nearest the EER point."""

    _, threshold = compute_eer(y_true, scores)
    return threshold


def compute_security_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> SecurityMetrics:
    """Compute security metrics at the test-set EER threshold.

    FAR/FRR are reported at the same threshold that defines EER. This avoids
    the misleading default-probability-threshold behavior that can make a
    heavily imbalanced authentication model look like it rejects half of the
    genuine attempts while still having a low EER.
    """

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
    """Measure synchronous batch inference time and peak memory increment."""

    # Warm up TensorFlow graph tracing / LightGBM internals outside measurement.
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
    inference_time_ms = blocking_time_ms / max(len(X_test), 1)
    peak_memory_mb = peak_bytes / (1024.0 * 1024.0)

    return scores, EfficiencyMetrics(
        inference_time_ms=float(inference_time_ms),
        ui_blocking_time_ms=float(blocking_time_ms),
        peak_memory_mb=float(peak_memory_mb),
    )


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------


def train_and_evaluate_model(
    spec: ModelSpec,
    views: PreparedViews,
) -> Tuple[AuthModel, SecurityMetrics, EfficiencyMetrics, Dict[str, np.ndarray]]:
    """Train one model and evaluate security + efficiency metrics."""

    X_train = get_view(views, "train", spec.input_view)
    X_test = get_view(views, "test", spec.input_view)

    model = spec.factory()
    model.fit(X_train, views.y_train)

    test_scores, efficiency = measure_inference(model.predict_scores, X_test)
    security = compute_security_metrics(views.y_test, test_scores)

    fpr, tpr, _ = roc_curve(views.y_test, test_scores, pos_label=1)
    curve_data = {
        "fpr": fpr,
        "tpr": tpr,
        "scores": test_scores,
    }

    return model, security, efficiency, curve_data


def make_training_sizes(y_train: np.ndarray, requested_sizes: Iterable[int]) -> List[int]:
    """Build ordered genuine-user enrollment sizes including MAX."""

    max_genuine = int((y_train == 1).sum())
    sizes = [size for size in requested_sizes if 0 < size <= max_genuine]
    sizes.append(max_genuine)
    return sorted(set(sizes))


def evaluate_data_efficiency(
    spec: ModelSpec,
    views: PreparedViews,
    training_sizes: Iterable[int],
    target_eer: float = EER_TARGET,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Optional[int]]:
    """Train with sequential genuine enrollment prefixes and report EER.

    For supervised models, all available imposter training samples are kept
    fixed while the number of genuine samples varies. For the Manhattan model,
    imposter samples are passed through the same interface but ignored by fit().
    Genuine samples are not randomly sampled: size=50 means the first 50
    chronological genuine training attempts.
    """

    # random_state is kept for API compatibility; the cold-start sweep is now
    # deterministic and chronological.
    _ = random_state
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

        # Release TensorFlow graphs between repeated CNN fits.
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
    """Save ROC curves for all models in one figure."""

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
    """Save bar charts for inference time and memory usage."""

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
    """Save EER vs genuine enrollment size line chart."""

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
    """Create all benchmark model specifications."""

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
        description="Continuous authentication benchmark for DSL-StrongPasswordData.csv"
    )
    parser.add_argument("--csv", default="DSL-StrongPasswordData.csv", help="Path to DSL CSV file.")
    parser.add_argument("--genuine-subject", default=GENUINE_SUBJECT, help="Target genuine subject id.")
    parser.add_argument("--output-dir", default="benchmark_outputs", help="Directory for plots and CSV outputs.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help=(
            "Target chronological test fraction. With sessionIndex data, "
            "complete future sessions are held out."
        ),
    )
    parser.add_argument("--cnn-epochs", type=int, default=25, help="Maximum CNN training epochs.")
    parser.add_argument("--cnn-batch-size", type=int, default=128, help="CNN batch size.")
    parser.add_argument(
        "--training-sizes",
        type=int,
        nargs="*",
        default=[50, 100, 200],
        help="Genuine-user training subset sizes. MAX is always added automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    sns.set_theme(style="whitegrid")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.csv, genuine_subject=args.genuine_subject)
    views = prepare_views(data, test_size=args.test_size, random_state=RANDOM_STATE)
    model_specs = build_model_specs(
        sequence_shape=views.sequence_shape,
        cnn_epochs=args.cnn_epochs,
        cnn_batch_size=args.cnn_batch_size,
    )
    training_sizes = make_training_sizes(views.y_train, args.training_sizes)

    print("\nDataset summary")
    print(f"  Samples: {len(data.y):,}")
    print(f"  Features: {views.X_train_flat_raw.shape[1]:,}")
    print(f"  Genuine subject: {args.genuine_subject}")
    print(f"  Genuine samples: {int(data.y.sum()):,}")
    print(f"  Split strategy: {views.split_strategy}")
    if views.sessions_train is not None and views.sessions_test is not None:
        print(f"  Train sessions: {sorted(pd.unique(views.sessions_train).tolist())}")
        print(f"  Test sessions: {sorted(pd.unique(views.sessions_test).tolist())}")
    print(f"  Train/Test: {len(views.y_train):,}/{len(views.y_test):,}")
    print(f"  Data efficiency sizes: {training_sizes}\n")

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
            random_state=RANDOM_STATE,
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
    summary_df.to_csv(summary_csv, index=False)
    data_efficiency_df.to_csv(data_efficiency_csv, index=False)

    plot_roc_curves(
        curve_data_by_model,
        summary_df,
        output_dir / "roc_curves.png",
    )
    plot_efficiency_bars(
        summary_df,
        output_dir / "efficiency_bars.png",
    )
    plot_eer_vs_training_size(
        data_efficiency_df,
        output_dir / "eer_vs_training_size.png",
    )

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
    print(f"  {output_dir / 'roc_curves.png'}")
    print(f"  {output_dir / 'efficiency_bars.png'}")
    print(f"  {output_dir / 'eer_vs_training_size.png'}")


if __name__ == "__main__":
    main()
