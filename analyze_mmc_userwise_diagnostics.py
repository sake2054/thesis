#!/usr/bin/env python3
"""
User-wise MMC diagnostics for browser-compatible keystroke timing features.

This script builds on benchmark_mmc_web_temporal_auth.py and repeats the
paper_reference_test evaluation with every subject treated once as the genuine
user. It writes user-wise metrics, LightGBM feature importance diagnostics, and
genuine/impostor score distribution CSV/plots.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
import types
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/codex-cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name!r}. Install the benchmark stack with:\n"
        "  pip install numpy pandas scikit-learn lightgbm tensorflow matplotlib seaborn"
    ) from exc

try:
    import seaborn as sns  # type: ignore
except ModuleNotFoundError:
    sns = None
    seaborn_shim = types.ModuleType("seaborn")
    seaborn_shim.set_theme = lambda *args, **kwargs: None
    sys.modules.setdefault("seaborn", seaborn_shim)

try:
    import benchmark_mmc_web_temporal_auth as base
    from benchmark_mmc_web_temporal_auth import (
        CNN1DAuthModel,
        DEFAULT_DATA_DIR,
        DEFAULT_TIME_SCALE,
        LightGBMAuthModel,
        ModelSpec,
        ScaledManhattanDistanceModel,
        get_view,
        load_npy_data,
        measure_inference,
        prepare_views,
    )
except SystemExit as exc:
    raise SystemExit(
        "Could not import benchmark_mmc_web_temporal_auth.py. Make sure it is in "
        "the same folder and that its dependencies are installed.\n"
        f"Original error:\n{exc}"
    ) from exc
except ModuleNotFoundError as exc:
    if exc.name == "benchmark_mmc_web_temporal_auth":
        raise SystemExit(
            "Could not find benchmark_mmc_web_temporal_auth.py. Place this script "
            "in the same folder as the existing MMC benchmark script."
        ) from exc
    raise SystemExit(
        f"Missing dependency while importing the base benchmark: {exc.name!r}.\n"
        "Install the benchmark stack with:\n"
        "  pip install numpy pandas scikit-learn lightgbm tensorflow matplotlib seaborn"
    ) from exc
except Exception as exc:
    raise SystemExit(
        "Could not import benchmark_mmc_web_temporal_auth.py. Place this script "
        "next to the base benchmark and check that the file imports cleanly.\n"
        f"Original error: {exc}"
    ) from exc


RANDOM_STATE = 42
DEFAULT_OUTPUT_DIR = "benchmark_outputs_mmc_diagnostics"
LIGHTGBM_NAME = "LightGBM Classifier"

METRIC_COLUMNS = [
    "subject",
    "model",
    "train_genuine_count",
    "test_genuine_count",
    "train_impostor_count",
    "test_impostor_count",
    "FAR",
    "FRR",
    "EER",
    "Accuracy",
    "AUC",
    "threshold",
    "inference_time_ms_per_sample",
    "peak_memory_mb",
]

SUMMARY_COLUMNS = [
    "model",
    "mean_EER",
    "std_EER",
    "median_EER",
    "min_EER",
    "max_EER",
    "mean_AUC",
    "std_AUC",
    "mean_FAR",
    "mean_FRR",
    "num_subjects",
]

SCORE_COLUMNS = [
    "genuine_subject",
    "model",
    "sample_index",
    "true_label",
    "subject_of_sample",
    "score",
    "score_z",
    "score_minmax",
    "threshold",
    "predicted_label",
    "is_error",
    "error_type",
]

FAILED_COLUMNS = ["subject", "model", "stage", "error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all-subject MMC diagnostics with timing-only browser features."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Folder containing mmc1.npy..mmc8.npy.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["paper_reference_test", "chronological_70_30"],
        default="paper_reference_test",
        help="Use the paper reference/test split by default; no random shuffle is used.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Only used by chronological_70_30 fallback split.",
    )
    parser.add_argument(
        "--include-cnn",
        action="store_true",
        help="Also evaluate 1D-CNN for every subject. This can be slow.",
    )
    parser.add_argument(
        "--cnn-epochs",
        type=int,
        default=25,
        help="Maximum CNN training epochs when --include-cnn is used.",
    )
    parser.add_argument(
        "--cnn-batch-size",
        type=int,
        default=128,
        help="CNN batch size when --include-cnn is used.",
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


def subject_sort_key(subject: object) -> Tuple[int, object]:
    text = str(subject)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def build_diagnostic_model_specs(
    sequence_shape: Tuple[int, ...],
    include_cnn: bool,
    cnn_epochs: int,
    cnn_batch_size: int,
) -> List[ModelSpec]:
    specs = [
        ModelSpec(
            name="Scaled Manhattan Distance",
            input_view="flat_raw",
            factory=lambda: ScaledManhattanDistanceModel(),
        ),
        ModelSpec(
            name=LIGHTGBM_NAME,
            input_view="flat_raw",
            factory=lambda: LightGBMAuthModel(random_state=RANDOM_STATE),
        ),
    ]
    if include_cnn:
        specs.append(
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
            )
        )
    return specs


def flattened_feature_names(
    sequence_shape: Tuple[int, ...],
    channel_names: Sequence[str],
) -> List[str]:
    if len(sequence_shape) != 2:
        raise ValueError(f"Expected sequence_shape=(positions, channels), got {sequence_shape}.")
    positions, channels = int(sequence_shape[0]), int(sequence_shape[1])
    if channels != len(channel_names):
        raise ValueError(
            f"sequence_shape has {channels} channels but {len(channel_names)} names were provided."
        )

    names: List[str] = []
    for pos in range(positions):
        for channel_name in channel_names:
            names.append(f"pos_{pos:03d}_{channel_name}")
    return names


def channel_from_flat_feature(feature: str) -> str:
    parts = feature.split("_", 2)
    if len(parts) == 3 and parts[0] == "pos":
        return parts[2]
    return feature


def finite_series(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.isfinite(pd.to_numeric(series, errors="coerce")),
        index=series.index,
    )


def safe_security_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    finite_scores = np.isfinite(scores)

    metrics = {
        "FAR": np.nan,
        "FRR": np.nan,
        "EER": np.nan,
        "Accuracy": np.nan,
        "AUC": np.nan,
        "threshold": np.nan,
    }
    if len(y_true) == 0 or len(np.unique(y_true)) < 2 or not np.all(finite_scores):
        return metrics

    try:
        metrics["AUC"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        metrics["AUC"] = np.nan

    try:
        fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
        fnr = 1.0 - tpr
        diffs = np.abs(fpr - fnr)
        finite = np.isfinite(diffs) & np.isfinite(thresholds)
        if not np.any(finite):
            return metrics
        candidate_indices = np.flatnonzero(finite)
        idx = int(candidate_indices[np.argmin(diffs[finite])])
        threshold = float(thresholds[idx])
        if not np.isfinite(threshold):
            return metrics
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
    except ValueError:
        return metrics

    y_pred = (scores >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) else np.nan
    frr = fn / (fn + tp) if (fn + tp) else np.nan

    metrics.update(
        {
            "FAR": float(far),
            "FRR": float(frr),
            "EER": float(eer),
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "threshold": threshold,
        }
    )
    return metrics


def normalize_scores(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    if std > 0.0 and np.isfinite(std):
        z_scores = (scores - mean) / std
    else:
        z_scores = np.zeros_like(scores, dtype=np.float64)

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    score_range = max_score - min_score
    if score_range > 0.0 and np.isfinite(score_range):
        minmax = (scores - min_score) / score_range
    else:
        minmax = np.zeros_like(scores, dtype=np.float64)
    return z_scores.astype(np.float32), minmax.astype(np.float32)


def score_distribution_frame(
    subject: str,
    model_name: str,
    test_indices: np.ndarray,
    y_true: np.ndarray,
    subjects_test: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=np.float64)
    score_z, score_minmax = normalize_scores(scores)

    if np.isfinite(threshold):
        predicted = (scores >= threshold).astype(np.int32)
        error_types = np.full(len(y_true), "correct", dtype=object)
        error_types[(y_true == 0) & (predicted == 1)] = "false_accept"
        error_types[(y_true == 1) & (predicted == 0)] = "false_reject"
        is_error = error_types != "correct"
    else:
        predicted = np.full(len(y_true), -1, dtype=np.int32)
        error_types = np.full(len(y_true), "threshold_unavailable", dtype=object)
        is_error = np.full(len(y_true), True, dtype=bool)

    return pd.DataFrame(
        {
            "genuine_subject": str(subject),
            "model": model_name,
            "sample_index": test_indices.astype(np.int64),
            "true_label": y_true.astype(np.int32),
            "subject_of_sample": subjects_test.astype(str),
            "score": scores.astype(np.float32),
            "score_z": score_z,
            "score_minmax": score_minmax,
            "threshold": float(threshold) if np.isfinite(threshold) else np.nan,
            "predicted_label": predicted,
            "is_error": is_error,
            "error_type": error_types,
        },
        columns=SCORE_COLUMNS,
    )


def lightgbm_importance_frame(
    subject: str,
    model: LightGBMAuthModel,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if model.model is None:
        raise RuntimeError("LightGBM booster is not available after fitting.")

    split_importance = model.model.feature_importance(importance_type="split")
    gain_importance = model.model.feature_importance(importance_type="gain")
    if len(split_importance) != len(feature_names):
        raise ValueError(
            "LightGBM importance length does not match flattened feature names: "
            f"{len(split_importance)} != {len(feature_names)}"
        )

    return pd.DataFrame(
        {
            "subject": str(subject),
            "feature": list(feature_names),
            "split_importance": split_importance.astype(np.float64),
            "gain_importance": gain_importance.astype(np.float64),
        }
    )


def summarize_userwise_metrics(
    metrics_df: pd.DataFrame,
    model_order: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for model_name in model_order:
        model_df = metrics_df.loc[metrics_df["model"] == model_name]
        if model_df.empty:
            continue
        rows.append(
            {
                "model": model_name,
                "mean_EER": float(model_df["EER"].mean()),
                "std_EER": float(model_df["EER"].std(ddof=1)),
                "median_EER": float(model_df["EER"].median()),
                "min_EER": float(model_df["EER"].min()),
                "max_EER": float(model_df["EER"].max()),
                "mean_AUC": float(model_df["AUC"].mean()),
                "std_AUC": float(model_df["AUC"].std(ddof=1)),
                "mean_FAR": float(model_df["FAR"].mean()),
                "mean_FRR": float(model_df["FRR"].mean()),
                "num_subjects": int(model_df["subject"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).fillna(
        {
            "std_EER": 0.0,
            "std_AUC": 0.0,
        }
    )


def summarize_feature_importance(
    importance_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if importance_df.empty:
        mean_df = pd.DataFrame(
            columns=[
                "feature",
                "mean_split_importance",
                "std_split_importance",
                "mean_gain_importance",
                "std_gain_importance",
            ]
        )
        channel_df = pd.DataFrame(
            columns=[
                "channel",
                "mean_split_importance",
                "mean_gain_importance",
                "percentage_gain_importance",
            ]
        )
        return mean_df, channel_df

    mean_df = (
        importance_df.groupby("feature", sort=False)
        .agg(
            mean_split_importance=("split_importance", "mean"),
            std_split_importance=("split_importance", "std"),
            mean_gain_importance=("gain_importance", "mean"),
            std_gain_importance=("gain_importance", "std"),
        )
        .reset_index()
    )
    mean_df[["std_split_importance", "std_gain_importance"]] = mean_df[
        ["std_split_importance", "std_gain_importance"]
    ].fillna(0.0)
    mean_df = mean_df.sort_values(
        ["mean_gain_importance", "feature"], ascending=[False, True], kind="stable"
    )

    channel_source = importance_df.copy()
    channel_source["channel"] = channel_source["feature"].map(channel_from_flat_feature)
    subject_channel = (
        channel_source.groupby(["subject", "channel"], sort=False)
        .agg(
            split_importance=("split_importance", "sum"),
            gain_importance=("gain_importance", "sum"),
        )
        .reset_index()
    )
    channel_df = (
        subject_channel.groupby("channel", sort=False)
        .agg(
            mean_split_importance=("split_importance", "mean"),
            mean_gain_importance=("gain_importance", "mean"),
        )
        .reset_index()
    )
    total_gain = float(channel_df["mean_gain_importance"].sum())
    if total_gain > 0.0:
        channel_df["percentage_gain_importance"] = (
            channel_df["mean_gain_importance"] / total_gain * 100.0
        )
    else:
        channel_df["percentage_gain_importance"] = 0.0
    channel_df = channel_df.sort_values(
        ["mean_gain_importance", "channel"], ascending=[False, True], kind="stable"
    )
    return mean_df, channel_df


def save_no_data_plot(path: Path, message: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def set_plot_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid")
        return
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_userwise_eer_boxplot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = metrics_df.loc[finite_series(metrics_df["EER"])].copy()
    if plot_df.empty:
        save_no_data_plot(output_path, "No finite EER values available")
        return

    models = list(pd.unique(plot_df["model"]))
    values_by_model = [
        plot_df.loc[plot_df["model"] == model, "EER"].astype(float).to_numpy()
        for model in models
    ]
    fig, ax = plt.subplots(figsize=(max(7, 2.4 * len(models)), 5))
    try:
        box = ax.boxplot(
            values_by_model,
            tick_labels=models,
            patch_artist=True,
            showfliers=False,
        )
    except TypeError:
        box = ax.boxplot(
            values_by_model,
            labels=models,
            patch_artist=True,
            showfliers=False,
        )
    for patch in box["boxes"]:
        patch.set_facecolor("#4C78A8")
        patch.set_alpha(0.45)
    for idx, values in enumerate(values_by_model, start=1):
        offsets = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(values), idx) + offsets,
            values,
            color="black",
            alpha=0.45,
            s=12,
            zorder=3,
        )
    ax.set_xlabel("Model")
    ax.set_ylabel("EER")
    ax.set_title("User-wise EER Distribution")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_userwise_eer_barplot(
    metrics_df: pd.DataFrame,
    subject_order: Sequence[str],
    model_order: Sequence[str],
    output_path: Path,
) -> None:
    plot_df = metrics_df.loc[finite_series(metrics_df["EER"])].copy()
    if plot_df.empty:
        save_no_data_plot(output_path, "No finite EER values available")
        return

    width = min(max(12.0, 0.45 * max(len(subject_order), 1)), 42.0)
    fig, ax = plt.subplots(figsize=(width, 6))
    pivot = plot_df.pivot_table(
        index="subject",
        columns="model",
        values="EER",
        aggfunc="first",
    ).reindex(list(subject_order))
    models = [model for model in model_order if model in pivot.columns]
    x = np.arange(len(pivot.index))
    bar_width = 0.82 / max(len(models), 1)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for model_idx, model_name in enumerate(models):
        offset = (model_idx - (len(models) - 1) / 2.0) * bar_width
        ax.bar(
            x + offset,
            pivot[model_name].astype(float).to_numpy(),
            width=bar_width,
            label=model_name,
            color=colors[model_idx % len(colors)],
        )
    ax.set_xlabel("Subject")
    ax.set_ylabel("EER")
    ax.set_title("User-wise EER by Subject")
    ax.set_xticks(x)
    ax.set_xticklabels(list(pivot.index))
    rotation = 90 if len(subject_order) > 20 else 45
    ax.tick_params(axis="x", rotation=rotation)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(title="Model", loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_top_features_gain(mean_importance_df: pd.DataFrame, output_path: Path) -> None:
    if mean_importance_df.empty:
        save_no_data_plot(output_path, "No LightGBM feature importance available")
        return

    top_df = mean_importance_df.head(30).copy()
    top_df = top_df.sort_values("mean_gain_importance", ascending=True, kind="stable")
    fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(top_df))))
    ax.barh(
        top_df["feature"].astype(str),
        top_df["mean_gain_importance"].astype(float),
        color="#4C78A8",
    )
    ax.set_xlabel("Mean Gain Importance")
    ax.set_ylabel("Feature")
    ax.set_title("LightGBM Top Timing Features by Gain")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_importance_by_channel(channel_df: pd.DataFrame, output_path: Path) -> None:
    if channel_df.empty:
        save_no_data_plot(output_path, "No LightGBM channel importance available")
        return

    plot_df = channel_df.sort_values(
        "percentage_gain_importance", ascending=False, kind="stable"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        plot_df["channel"].astype(str),
        plot_df["percentage_gain_importance"].astype(float),
        color="#59A14F",
    )
    ax.set_xlabel("Timing Channel")
    ax.set_ylabel("Gain Importance (%)")
    ax.set_title("LightGBM Gain Importance by Timing Channel")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def select_lightgbm_representatives(metrics_df: pd.DataFrame) -> Dict[str, Optional[str]]:
    lgbm_df = metrics_df.loc[
        (metrics_df["model"] == LIGHTGBM_NAME) & finite_series(metrics_df["EER"])
    ].copy()
    if lgbm_df.empty:
        return {"lowest": None, "median": None, "highest": None}

    lgbm_df = lgbm_df.sort_values(["EER", "subject"], kind="stable")
    median_eer = float(lgbm_df["EER"].median())
    lowest = str(lgbm_df.iloc[0]["subject"])
    highest = str(lgbm_df.iloc[-1]["subject"])
    median_subject = str(
        lgbm_df.assign(distance=(lgbm_df["EER"] - median_eer).abs())
        .sort_values(["distance", "EER", "subject"], kind="stable")
        .iloc[0]["subject"]
    )
    return {"lowest": lowest, "median": median_subject, "highest": highest}


def ordered_unique(values: Iterable[Optional[str]]) -> List[str]:
    output: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text not in output:
            output.append(text)
    return output


def plot_score_distribution_examples(
    score_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    representatives: Dict[str, Optional[str]],
    output_path: Path,
) -> None:
    selected_subjects = ordered_unique(
        [
            representatives.get("lowest"),
            representatives.get("median"),
            representatives.get("highest"),
        ]
    )
    if not selected_subjects:
        save_no_data_plot(output_path, "No LightGBM score distributions available")
        return

    ncols = len(selected_subjects)
    fig, axes = plt.subplots(1, ncols, figsize=(5.3 * ncols, 4.2), squeeze=False)
    for ax, subject in zip(axes[0], selected_subjects):
        subset = score_df.loc[
            (score_df["model"] == LIGHTGBM_NAME)
            & (score_df["genuine_subject"] == subject)
        ].copy()
        if subset.empty:
            ax.text(0.5, 0.5, f"Subject {subject}\nNo scores", ha="center", va="center")
            ax.axis("off")
            continue

        genuine_scores = subset.loc[subset["true_label"] == 1, "score"].astype(float)
        impostor_scores = subset.loc[subset["true_label"] == 0, "score"].astype(float)
        ax.hist(
            impostor_scores,
            bins=35,
            density=True,
            histtype="stepfilled",
            alpha=0.25,
            color="#F58518",
            label="impostor",
        )
        ax.hist(
            genuine_scores,
            bins=35,
            density=True,
            histtype="step",
            linewidth=2.0,
            color="#4C78A8",
            label="genuine",
        )
        threshold = subset["threshold"].dropna()
        if not threshold.empty:
            ax.axvline(
                float(threshold.iloc[0]),
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="EER threshold",
            )
        eer_values = metrics_df.loc[
            (metrics_df["model"] == LIGHTGBM_NAME) & (metrics_df["subject"] == subject),
            "EER",
        ]
        eer_text = f"{float(eer_values.iloc[0]):.3f}" if not eer_values.empty else "NA"
        ax.set_title(f"Subject {subject} (EER={eer_text})")
        ax.set_xlabel("LightGBM Score")
        ax.set_ylabel("Density")
        ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_lightgbm_all_subject_scores(score_df: pd.DataFrame, output_path: Path) -> None:
    subset = score_df.loc[score_df["model"] == LIGHTGBM_NAME].copy()
    if subset.empty:
        save_no_data_plot(output_path, "No LightGBM score distributions available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, column, title, xlabel in [
        (axes[0], "score", "Raw LightGBM Scores", "Raw score"),
        (
            axes[1],
            "score_z",
            "Subject-wise Z-normalized LightGBM Scores",
            "Z-normalized score",
        ),
    ]:
        genuine_scores = subset.loc[subset["true_label"] == 1, column].astype(float)
        impostor_scores = subset.loc[subset["true_label"] == 0, column].astype(float)
        ax.hist(
            impostor_scores,
            bins=60,
            density=True,
            histtype="stepfilled",
            alpha=0.25,
            color="#F58518",
            label="impostor",
        )
        ax.hist(
            genuine_scores,
            bins=60,
            density=True,
            histtype="step",
            linewidth=2.0,
            color="#4C78A8",
            label="genuine",
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(loc="best")
    axes[0].set_title("Raw LightGBM Scores")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_outputs(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    mean_importance_df: pd.DataFrame,
    channel_importance_df: pd.DataFrame,
    score_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    subject_order: Sequence[str],
    model_order: Sequence[str],
    representatives: Dict[str, Optional[str]],
) -> List[Path]:
    paths = [
        output_dir / "userwise_metrics.csv",
        output_dir / "userwise_summary_by_model.csv",
        output_dir / "userwise_eer_boxplot.png",
        output_dir / "userwise_eer_barplot.png",
        output_dir / "lightgbm_feature_importance_userwise.csv",
        output_dir / "lightgbm_feature_importance_mean.csv",
        output_dir / "lightgbm_feature_importance_by_channel.csv",
        output_dir / "lightgbm_top_features_gain.png",
        output_dir / "lightgbm_importance_by_channel.png",
        output_dir / "score_distributions.csv",
        output_dir / "score_distribution_examples.png",
        output_dir / "score_distribution_lightgbm_all_subjects.png",
        output_dir / "failed_subjects.csv",
    ]

    metrics_df.to_csv(paths[0], index=False)
    summary_df.to_csv(paths[1], index=False)
    plot_userwise_eer_boxplot(metrics_df, paths[2])
    plot_userwise_eer_barplot(metrics_df, subject_order, model_order, paths[3])

    importance_df.to_csv(paths[4], index=False)
    mean_importance_df.to_csv(paths[5], index=False)
    channel_importance_df.to_csv(paths[6], index=False)
    plot_top_features_gain(mean_importance_df, paths[7])
    plot_importance_by_channel(channel_importance_df, paths[8])

    score_df.to_csv(paths[9], index=False)
    plot_score_distribution_examples(score_df, metrics_df, representatives, paths[10])
    plot_lightgbm_all_subject_scores(score_df, paths[11])

    failed_df.to_csv(paths[12], index=False)
    return paths


def print_final_report(
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    channel_importance_df: pd.DataFrame,
    representatives: Dict[str, Optional[str]],
    output_paths: Sequence[Path],
) -> None:
    print("\nModel summary by average EER/AUC")
    if summary_df.empty:
        print("  No successful model evaluations.")
    else:
        with pd.option_context(
            "display.max_columns",
            None,
            "display.width",
            160,
            "display.float_format",
            "{:.4f}".format,
        ):
            print(summary_df.to_string(index=False))

    print("\nLightGBM representative subjects by EER")
    lgbm_metrics = metrics_df.loc[metrics_df["model"] == LIGHTGBM_NAME]
    for label, subject in representatives.items():
        if subject is None:
            print(f"  {label}: unavailable")
            continue
        row = lgbm_metrics.loc[lgbm_metrics["subject"] == subject]
        if row.empty:
            print(f"  {label}: subject {subject}, EER unavailable")
        else:
            print(
                f"  {label}: subject {subject}, "
                f"EER={float(row.iloc[0]['EER']):.4f}, "
                f"AUC={float(row.iloc[0]['AUC']):.4f}"
            )

    print("\nLightGBM channel importance ranking")
    if channel_importance_df.empty:
        print("  No LightGBM feature importance available.")
    else:
        for _, row in channel_importance_df.iterrows():
            print(
                f"  {row['channel']}: "
                f"{float(row['percentage_gain_importance']):.2f}% gain "
                f"(mean_gain={float(row['mean_gain_importance']):.4f})"
            )

    print("\nGenerated files")
    for path in output_paths:
        print(f"  {path}")


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_STATE)
    base.tf.keras.utils.set_random_seed(RANDOM_STATE)
    set_plot_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MMC timing data")
    data = load_npy_data(
        data_dir=args.data_dir,
        temporal_feature_indices=args.temporal_feature_indices,
        genuine_subject="auto",
        time_scale=args.time_scale,
    )

    subject_order = sorted(map(str, pd.unique(data.subjects)), key=subject_sort_key)
    if not subject_order:
        raise SystemExit("No subjects found in the MMC labels.")

    print("Preparing train/test views")
    base_views = prepare_views(data, split_mode=args.split_mode, test_size=args.test_size)
    model_specs = build_diagnostic_model_specs(
        sequence_shape=base_views.sequence_shape,
        include_cnn=args.include_cnn,
        cnn_epochs=args.cnn_epochs,
        cnn_batch_size=args.cnn_batch_size,
    )
    model_order = [spec.name for spec in model_specs]
    feature_names = flattened_feature_names(base_views.sequence_shape, data.feature_names)

    print("\nMMC diagnostic setup")
    print(f"  Samples: {len(data.y):,}")
    print(f"  Subjects: {len(subject_order):,}")
    print(f"  Sequence shape: {base_views.sequence_shape}")
    print(f"  Flattened features: {len(feature_names):,}")
    print(f"  Timing indices: {list(data.temporal_feature_indices)}")
    print(f"  Timing feature names: {data.feature_names}")
    print(f"  Split strategy: {base_views.split_strategy}")
    print(f"  Train/Test samples: {len(base_views.y_train):,}/{len(base_views.y_test):,}")
    print(f"  Models: {', '.join(model_order)}")

    metric_rows: List[Dict[str, object]] = []
    score_frames: List[pd.DataFrame] = []
    importance_frames: List[pd.DataFrame] = []
    failed_rows: List[Dict[str, str]] = []

    for subject_index, subject in enumerate(subject_order, start=1):
        y_all = (data.subjects == subject).astype(np.int32)
        y_train = y_all[base_views.train_indices].astype(np.int32)
        y_test = y_all[base_views.test_indices].astype(np.int32)

        train_genuine_count = int(y_train.sum())
        test_genuine_count = int(y_test.sum())
        train_impostor_count = int(len(y_train) - train_genuine_count)
        test_impostor_count = int(len(y_test) - test_genuine_count)

        print(
            f"\nSubject {subject_index}/{len(subject_order)}: {subject} "
            f"(train genuine={train_genuine_count}, test genuine={test_genuine_count})"
        )

        if (
            train_genuine_count == 0
            or test_genuine_count == 0
            or train_impostor_count == 0
            or test_impostor_count == 0
        ):
            message = (
                "Split does not contain both genuine and impostor samples in train/test."
            )
            print(f"  Skipping subject: {message}")
            failed_rows.append(
                {
                    "subject": subject,
                    "model": "all",
                    "stage": "split",
                    "error": message,
                }
            )
            gc.collect()
            continue

        views = replace(base_views, y_train=y_train, y_test=y_test)

        abort_subject = False
        for spec in model_specs:
            print(f"  Training {spec.name}")
            model = None
            try:
                X_train = get_view(views, "train", spec.input_view)
                X_test = get_view(views, "test", spec.input_view)
                model = spec.factory()
                model.fit(X_train, views.y_train)
                scores, efficiency = measure_inference(model.predict_scores, X_test)
                security = safe_security_metrics(views.y_test, scores)

                metric_rows.append(
                    {
                        "subject": subject,
                        "model": spec.name,
                        "train_genuine_count": train_genuine_count,
                        "test_genuine_count": test_genuine_count,
                        "train_impostor_count": train_impostor_count,
                        "test_impostor_count": test_impostor_count,
                        "FAR": security["FAR"],
                        "FRR": security["FRR"],
                        "EER": security["EER"],
                        "Accuracy": security["Accuracy"],
                        "AUC": security["AUC"],
                        "threshold": security["threshold"],
                        "inference_time_ms_per_sample": efficiency.inference_time_ms,
                        "peak_memory_mb": efficiency.peak_memory_mb,
                    }
                )

                score_frames.append(
                    score_distribution_frame(
                        subject=subject,
                        model_name=spec.name,
                        test_indices=views.test_indices,
                        y_true=views.y_test,
                        subjects_test=views.subjects_test,
                        scores=scores,
                        threshold=security["threshold"],
                    )
                )

                if spec.name == LIGHTGBM_NAME and isinstance(model, LightGBMAuthModel):
                    importance_frames.append(
                        lightgbm_importance_frame(subject, model, feature_names)
                    )

                print(
                    f"    EER={security['EER']:.4f}, AUC={security['AUC']:.4f}, "
                    f"threshold={security['threshold']:.6g}"
                )
            except Exception as exc:
                error_message = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                print(f"    FAILED: {error_message}", file=sys.stderr)
                failed_rows.append(
                    {
                        "subject": subject,
                        "model": spec.name,
                        "stage": "train_evaluate",
                        "error": error_message,
                    }
                )
                if spec.name == LIGHTGBM_NAME:
                    abort_subject = True
            finally:
                if spec.name == "1D-CNN":
                    base.keras.backend.clear_session()
                del model
                gc.collect()
            if abort_subject:
                print("    Moving to next subject after LightGBM failure.")
                break

        gc.collect()

    metrics_df = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    summary_df = summarize_userwise_metrics(metrics_df, model_order)

    if importance_frames:
        importance_df = pd.concat(importance_frames, ignore_index=True)
    else:
        importance_df = pd.DataFrame(
            columns=["subject", "feature", "split_importance", "gain_importance"]
        )
    mean_importance_df, channel_importance_df = summarize_feature_importance(importance_df)

    if score_frames:
        score_df = pd.concat(score_frames, ignore_index=True)
    else:
        score_df = pd.DataFrame(columns=SCORE_COLUMNS)
    failed_df = pd.DataFrame(failed_rows, columns=FAILED_COLUMNS)

    representatives = select_lightgbm_representatives(metrics_df)
    output_paths = write_outputs(
        output_dir=output_dir,
        metrics_df=metrics_df,
        summary_df=summary_df,
        importance_df=importance_df,
        mean_importance_df=mean_importance_df,
        channel_importance_df=channel_importance_df,
        score_df=score_df,
        failed_df=failed_df,
        subject_order=subject_order,
        model_order=model_order,
        representatives=representatives,
    )
    print_final_report(
        summary_df=summary_df,
        metrics_df=metrics_df,
        channel_importance_df=channel_importance_df,
        representatives=representatives,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()
