"""
Publication-quality figure generation.

Every figure in the report should be produced by this module,
not manually screenshotted from a notebook.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import CHANNELS, FIGURES_DIR

# Set global style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


class PlotGenerator:
    """Generates all figures for the project."""

    def __init__(self, output_dir: Path = FIGURES_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training diagnostics
    # ------------------------------------------------------------------

    def plot_training_loss(self, losses: list[float], model_name: str) -> Path:
        """Training loss curve."""
        fig, ax = plt.subplots()
        ax.plot(losses, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss — {model_name}")
        ax.set_yscale("log")
        path = self.output_dir / f"training_loss_{model_name}.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Anomaly scores
    # ------------------------------------------------------------------

    def plot_anomaly_scores(
        self,
        scores: np.ndarray,
        meta: pd.DataFrame,
        threshold: float,
        model_name: str,
        title_suffix: str = "",
    ) -> Path:
        """Reconstruction error per lap with threshold line and label coloring."""
        fig, ax = plt.subplots(figsize=(14, 5))

        colors = {
            "normal": "#2196F3",
            "pit": "#FF9800",
            "sc": "#9E9E9E",
            "pre_failure": "#F44336",
            "formation": "#795548",
        }
        if "Label" in meta.columns:
            for label, color in colors.items():
                mask = meta["Label"] == label
                if mask.any():
                    ax.scatter(
                        np.where(mask)[0], scores[mask],
                        c=color, s=15, label=label, alpha=0.7, zorder=3,
                    )
        else:
            ax.scatter(range(len(scores)), scores, s=10, alpha=0.5)

        ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label="threshold")
        ax.set_xlabel("Lap index")
        ax.set_ylabel("Anomaly Score")
        ax.set_title(f"Anomaly Scores — {model_name} {title_suffix}")
        ax.legend(loc="upper right", framealpha=0.9)
        path = self.output_dir / f"anomaly_scores_{model_name}{title_suffix.replace(' ', '_')}.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Telemetry overlay
    # ------------------------------------------------------------------

    def plot_telemetry_comparison(
        self,
        X: np.ndarray,
        meta: pd.DataFrame,
        normal_idx: int,
        anomaly_idx: int,
        channels: list[str] | None = None,
        model_name: str = "",
    ) -> Path:
        """
        Overlay normal vs anomalous lap telemetry traces.
        X shape: (n, resample_points, n_channels)
        """
        channels = channels or CHANNELS
        n_ch = len(channels)
        fig, axes = plt.subplots(n_ch, 1, figsize=(12, 3 * n_ch), sharex=True)
        if n_ch == 1:
            axes = [axes]

        normal_meta = meta.iloc[normal_idx]
        anom_meta = meta.iloc[anomaly_idx]

        for c, (ax, ch) in enumerate(zip(axes, channels)):
            ax.plot(X[normal_idx, :, c], label=f"Normal (Lap {normal_meta.get('LapNumber', '?')})",
                    linewidth=1, alpha=0.8)
            ax.plot(X[anomaly_idx, :, c], label=f"Anomalous (Lap {anom_meta.get('LapNumber', '?')})",
                    linewidth=1, alpha=0.8)
            ax.set_ylabel(ch)
            ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Sample point")
        fig.suptitle(f"Telemetry Comparison — {model_name}", y=1.01)
        path = self.output_dir / f"telemetry_comparison_{model_name}.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Channel error heatmap
    # ------------------------------------------------------------------

    def plot_channel_error_heatmap(
        self,
        channel_errors: np.ndarray,
        meta: pd.DataFrame,
        channels: list[str] | None = None,
        model_name: str = "",
    ) -> Path:
        """Heatmap of per-channel reconstruction error across all laps."""
        channels = channels or CHANNELS
        fig, ax = plt.subplots(figsize=(8, max(6, len(channel_errors) * 0.05)))
        sns.heatmap(
            channel_errors,
            xticklabels=channels,
            yticklabels=False,
            cmap="YlOrRd",
            ax=ax,
        )
        ax.set_xlabel("Channel")
        ax.set_ylabel("Lap index")
        ax.set_title(f"Per-channel Error — {model_name}")
        path = self.output_dir / f"channel_heatmap_{model_name}.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> Path:
        """Grouped bar chart comparing models on key metrics."""
        metrics = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
        available = [m for m in metrics if m in comparison_df.columns]
        df_plot = comparison_df.set_index("Model")[available]

        fig, ax = plt.subplots(figsize=(10, 5))
        df_plot.plot(kind="bar", ax=ax, rot=15)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right")
        path = self.output_dir / "model_comparison.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # ROC and PR curves
    # ------------------------------------------------------------------

    def plot_roc_curves(self, eval_results: list[dict]) -> Path:
        """Overlay ROC curves for multiple models."""
        fig, ax = plt.subplots()
        for r in eval_results:
            if r.get("roc_curve"):
                ax.plot(r["roc_curve"]["fpr"], r["roc_curve"]["tpr"],
                        label=f"{r['model']} (AUC={r['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        path = self.output_dir / "roc_curves.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_pr_curves(self, eval_results: list[dict]) -> Path:
        """Overlay precision-recall curves for multiple models."""
        fig, ax = plt.subplots()
        for r in eval_results:
            if r.get("pr_curve"):
                ax.plot(r["pr_curve"]["recall"], r["pr_curve"]["precision"],
                        label=f"{r['model']} (AP={r['pr_auc']:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend()
        path = self.output_dir / "pr_curves.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(self, cm: list[list[int]], model_name: str) -> Path:
        """Confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            np.array(cm), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_name}")
        path = self.output_dir / f"confusion_matrix_{model_name}.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Threshold sensitivity
    # ------------------------------------------------------------------

    def plot_threshold_sensitivity(self, sensitivity_df: pd.DataFrame, model_name: str) -> Path:
        """Plot precision, recall, F1 vs threshold multiplier k."""
        fig, ax = plt.subplots()
        ax.plot(sensitivity_df["k"], sensitivity_df["precision"], "o-", label="Precision")
        ax.plot(sensitivity_df["k"], sensitivity_df["recall"], "s-", label="Recall")
        ax.plot(sensitivity_df["k"], sensitivity_df["f1"], "^-", label="F1")
        ax.set_xlabel("Threshold multiplier (k)")
        ax.set_ylabel("Score")
        ax.set_title(f"Threshold Sensitivity — {model_name}")
        ax.legend()
        path = self.output_dir / f"threshold_sensitivity_{model_name}.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Hyperparameter search
    # ------------------------------------------------------------------

    def plot_hp_search(self, log_df: pd.DataFrame, param_name: str, model_name: str) -> Path:
        """Scatter: hyperparameter value vs F1 score."""
        fig, ax = plt.subplots()
        ax.scatter(log_df[param_name], log_df["f1"], alpha=0.6, s=30)
        ax.set_xlabel(param_name)
        ax.set_ylabel("F1 Score")
        ax.set_title(f"HP Search: {param_name} — {model_name}")
        path = self.output_dir / f"hp_{param_name}_{model_name}.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Circuit generalization
    # ------------------------------------------------------------------

    def plot_circuit_performance(self, circuit_results: pd.DataFrame) -> Path:
        """Bar chart of F1 score per circuit."""
        fig, ax = plt.subplots(figsize=(14, 5))
        circuit_results = circuit_results.sort_values("f1", ascending=True)
        ax.barh(circuit_results["circuit"], circuit_results["f1"])
        ax.set_xlabel("F1 Score")
        ax.set_title("Per-circuit Generalization Performance")
        ax.set_xlim(0, 1)
        path = self.output_dir / "circuit_generalization.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Failure mode distribution
    # ------------------------------------------------------------------

    def plot_failure_modes(self, summary_df: pd.DataFrame) -> Path:
        """Horizontal bar chart of failure mode counts."""
        fig, ax = plt.subplots(figsize=(10, 5))
        summary_df = summary_df.sort_values("count", ascending=True)
        ax.barh(summary_df["failure_mode"], summary_df["count"])
        ax.set_xlabel("Count")
        ax.set_title("Failure Mode Distribution")
        path = self.output_dir / "failure_modes.png"
        fig.savefig(path)
        plt.close(fig)
        return path
