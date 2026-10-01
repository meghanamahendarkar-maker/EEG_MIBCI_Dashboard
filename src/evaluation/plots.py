"""Plotting utilities to replicate publication figures from Hwaidi & Ghanem (2026).
- Figure 6: Global & 4-MI task accuracies across subjects (S1-S10)
- Figure 7: Mean confusion matrices
- Figure 8: ROC curves for all classes
- Figure 9: Training and validation loss & accuracy curves across epochs
- Computational Benchmark summary charts
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt

# Professional publication styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def plot_subject_accuracies(
    results_by_subject: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Replicates Figure 6: Global average accuracy and 4-MI task accuracy across subjects."""
    subjects = list(results_by_subject.keys())
    n_subjs = len(subjects)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (a): MiniRocket Global Accuracy
    mr_globals = [results_by_subject[s]["minirocket"]["accuracy"] * 100 for s in subjects]
    axes[0, 0].bar(subjects, mr_globals, color="#1f77b4", edgecolor="black", alpha=0.85)
    axes[0, 0].set_title("(a) Global Average Accuracy - MiniRocket Pipeline", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0, 0].set_ylim(80, 102)
    for i, v in enumerate(mr_globals):
        axes[0, 0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)

    # Panel (b): MiniRocket 4-MI Class Accuracy per Subject
    x = np.arange(n_subjs)
    width = 0.2
    class_labels = ["T1 (Left Fist)", "T2 (Right Fist)", "T3 (Both Fists)", "T4 (Both Feet)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for c in range(4):
        c_accs = [results_by_subject[s]["minirocket"]["per_class_accuracy"][c] * 100 for s in subjects]
        axes[0, 1].bar(x + c * width, c_accs, width, label=class_labels[c], color=colors[c], alpha=0.85)

    axes[0, 1].set_title("(b) 4-MI Class Accuracy - MiniRocket Pipeline", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticks(x + 1.5 * width)
    axes[0, 1].set_xticklabels(subjects)
    axes[0, 1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0, 1].set_ylim(70, 102)
    axes[0, 1].legend(loc="lower right", fontsize=9)

    # Panel (c): CNN-LSTM Global Accuracy
    dl_globals = [results_by_subject[s]["cnn_lstm"]["accuracy"] * 100 for s in subjects]
    axes[1, 0].bar(subjects, dl_globals, color="#2ca02c", edgecolor="black", alpha=0.85)
    axes[1, 0].set_title("(c) Global Average Accuracy - Hybrid CNN-LSTM Model", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1, 0].set_ylim(80, 102)
    for i, v in enumerate(dl_globals):
        axes[1, 0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)

    # Panel (d): CNN-LSTM 4-MI Class Accuracy per Subject
    for c in range(4):
        c_accs = [results_by_subject[s]["cnn_lstm"]["per_class_accuracy"][c] * 100 for s in subjects]
        axes[1, 1].bar(x + c * width, c_accs, width, label=class_labels[c], color=colors[c], alpha=0.85)

    axes[1, 1].set_title("(d) 4-MI Class Accuracy - Hybrid CNN-LSTM Model", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks(x + 1.5 * width)
    axes[1, 1].set_xticklabels(subjects)
    axes[1, 1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1, 1].set_ylim(70, 102)
    axes[1, 1].legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrices(
    mr_conf_mat: np.ndarray,
    dl_conf_mat: np.ndarray,
    class_names: List[str] = ["Left Fist", "Right Fist", "Both Fists", "Both Feet"],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Replicates Figure 7: Mean confusion matrices for all subjects."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Normalize if not already normalized
    mr_norm = mr_conf_mat.astype(np.float64) / mr_conf_mat.sum(axis=1, keepdims=True)
    dl_norm = dl_conf_mat.astype(np.float64) / dl_conf_mat.sum(axis=1, keepdims=True)

    for idx, (mat, title) in enumerate([
        (mr_norm, "(a) MiniRocket + Ridge Mean Confusion Matrix"),
        (dl_norm, "(b) Hybrid CNN-LSTM Mean Confusion Matrix"),
    ]):
        ax = axes[idx]
        im = ax.imshow(mat, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = mat[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_roc_curves(
    mr_metrics: Dict[str, Any],
    dl_metrics: Dict[str, Any],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Replicates Figure 8: ROC curves for MiniRocket and CNN-LSTM."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    class_names = mr_metrics.get("class_names", ["Class 0", "Class 1", "Class 2", "Class 3"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (metrics, title) in enumerate([
        (mr_metrics, f"(a) MiniRocket ROC Curves (Macro AUC = {mr_metrics['macro_auc']:.3f})"),
        (dl_metrics, f"(b) Hybrid CNN-LSTM ROC Curves (Macro AUC = {dl_metrics['macro_auc']:.3f})"),
    ]):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Chance Level (AUC = 0.5)")

        for c_idx in range(len(class_names)):
            fpr = metrics["fpr"][c_idx]
            tpr = metrics["tpr"][c_idx]
            c_auc = metrics["roc_auc"][c_idx]
            ax.plot(fpr, tpr, color=colors[c_idx], lw=2.0, label=f"{class_names[c_idx]} (AUC = {c_auc:.3f})")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=10, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Replicates Figure 9: Epoch versus accuracy and loss plots for training and validation."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    if "val_loss" in history and history["val_loss"]:
        axes[0].plot(epochs, history["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2, linestyle="--")
    axes[0].set_title("Epoch vs Loss Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=10)
    axes[0].legend(fontsize=10)

    # Accuracy plot
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#1f77b4", lw=2)
    if "val_acc" in history and history["val_acc"]:
        axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation Accuracy", color="#2ca02c", lw=2, linestyle="--")
    axes[1].set_title("Epoch vs Accuracy Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_computational_benchmark(
    benchmarks: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Bar charts summarizing Table 2 & 3 computational benchmarks."""
    methods = list(benchmarks.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. Trainable Parameters
    params = [benchmarks[m].get("trainable_params", 0) for m in methods]
    axes[0, 0].bar(methods, params, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"][:len(methods)], edgecolor="black")
    axes[0, 0].set_title("Trainable Parameters Count", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Number of Parameters", fontsize=10)
    for i, p in enumerate(params):
        axes[0, 0].text(i, p + max(params)*0.02, f"{p:,}", ha="center", fontsize=9)

    # 2. Training Time (min)
    train_times = [benchmarks[m].get("train_time_min", 0.0) for m in methods]
    axes[0, 1].bar(methods, train_times, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"][:len(methods)], edgecolor="black")
    axes[0, 1].set_title("Total CPU Training Time (minutes)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Minutes", fontsize=10)
    for i, t in enumerate(train_times):
        axes[0, 1].text(i, t + max(train_times)*0.02, f"{t:.1f}m", ha="center", fontsize=9)

    # 3. Inference Latency per Sample (ms)
    latencies = [benchmarks[m].get("avg_latency_ms", 0.0) for m in methods]
    axes[1, 0].bar(methods, latencies, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"][:len(methods)], edgecolor="black")
    axes[1, 0].set_title("Average CPU Latency per Sample (ms)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Latency (ms)", fontsize=10)
    for i, l in enumerate(latencies):
        axes[1, 0].text(i, l + max(latencies)*0.02, f"{l:.2f} ms", ha="center", fontsize=9)

    # 4. Latency per 4s Trial (ms)
    trial_lat = [benchmarks[m].get("latency_per_trial_ms", 0.0) for m in methods]
    axes[1, 1].bar(methods, trial_lat, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"][:len(methods)], edgecolor="black")
    axes[1, 1].set_title("Latency per 4-Second Trial (9 samples, ms)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Latency (ms)", fontsize=10)
    for i, tl in enumerate(trial_lat):
        axes[1, 1].text(i, tl + max(trial_lat)*0.02, f"{tl:.1f} ms", ha="center", fontsize=9)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
