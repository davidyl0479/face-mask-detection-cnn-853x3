"""
Face Mask Detection Visualization Module

This module implements a comprehensive visualization suite for face mask detection
training analysis, model evaluation, and dataset exploration. Generates training curves,
confusion matrices, class performance metrics, sample predictions, and dataset analysis plots.
"""

import os  # noqa: F401
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import functional as TF
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import sparse

from .config import VISUALIZATION_CONFIG, DATASET_CONFIG, FIGURES_ROOT, IMAGE_CONFIG

# Set up matplotlib and seaborn styling
plt.style.use(VISUALIZATION_CONFIG.get("style", "default"))
sns.set_palette(VISUALIZATION_CONFIG.get("color_palette", "viridis"))


class FaceMaskPlotter:
    """
    Main plotting class for face mask detection visualizations.

    Provides comprehensive plotting capabilities for training analysis,
    model evaluation, and dataset exploration.
    """

    def __init__(
        self,
        save_plots: bool = True,
        show_plots: bool = True,
        figure_size: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the plotter.

        Args:
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots
            figure_size: Default figure size
            dpi: DPI for saved figures
            output_dir: Directory to save plots
        """
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.figure_size = figure_size
        self.dpi = dpi
        self.output_dir = output_dir or FIGURES_ROOT
        self.output_dir.mkdir(exist_ok=True)

        # Class information
        self.class_names = DATASET_CONFIG["classes"]
        self.num_classes = DATASET_CONFIG["num_classes"]

        # Color schemes
        self.class_colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, self.num_classes))
        self.primary_color = "#2E86AB"
        self.secondary_color = "#A23B72"
        self.accent_color = "#F18F01"

        print(f"Plotter initialized. Plots will be saved to: {self.output_dir}")

    def _save_figure(self, fig, filename: str, tight_layout: bool = True):
        """
        Save figure to file and optionally display.

        Args:
            fig: Matplotlib figure
            filename: Filename to save
            tight_layout: Whether to apply tight layout
        """
        if tight_layout:
            fig.tight_layout()

        if self.save_plots:
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved: {filepath}")

        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def plot_training_history(
        self,
        train_history: List[Dict[str, float]],
        val_history: List[Dict[str, float]],
        metrics: Optional[List[str]] = None,
        filename: str = "training_history.png",
    ):
        """
        Plot training and validation metrics over epochs.

        Args:
            train_history: List of training metrics per epoch
            val_history: List of validation metrics per epoch
            metrics: List of metrics to plot
            filename: Output filename
        """
        if metrics is None:
            metrics = ["loss", "accuracy", "f1_score"]

        # Filter metrics that exist in the history
        available_metrics = [m for m in metrics if m in train_history[0]]

        if not available_metrics:
            print("No available metrics to plot")
            return

        # Create subplots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        epochs = range(1, len(train_history) + 1)

        for i, metric in enumerate(available_metrics):
            ax = axes[i]

            # Extract metric values
            train_values = [epoch_data[metric] for epoch_data in train_history]
            val_values = [epoch_data[metric] for epoch_data in val_history]

            # Plot lines
            ax.plot(
                epochs,
                train_values,
                "o-",
                label=f"Train {metric.title()}",
                color=self.primary_color,
                linewidth=2,
                markersize=4,
            )
            ax.plot(
                epochs,
                val_values,
                "s-",
                label=f"Val {metric.title()}",
                color=self.secondary_color,
                linewidth=2,
                markersize=4,
            )

            # Customize plot
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.title())
            ax.set_title(f"{metric.title()} Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add best value annotation
            if "loss" in metric.lower():
                best_val_idx = np.argmin(val_values)
                best_val = val_values[best_val_idx]
                ax.annotate(
                    f"Best: {best_val:.4f}",
                    xy=(best_val_idx + 1, best_val),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )
            else:
                best_val_idx = np.argmax(val_values)
                best_val = val_values[best_val_idx]
                ax.annotate(
                    f"Best: {best_val:.4f}",
                    xy=(best_val_idx + 1, best_val),
                    xytext=(10, -10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

        plt.suptitle("Training History - Face Mask Detection", fontsize=16, y=1.02)
        self._save_figure(fig, filename)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True,
        filename: str = "confusion_matrix.png",
    ):
        """
        Plot confusion matrix for face mask detection.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the matrix
            filename: Output filename
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2%"
            title = "Normalized Confusion Matrix"
        else:
            fmt = "d"
            title = "Confusion Matrix"

        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=[name.replace("_", " ").title() for name in self.class_names],
            yticklabels=[name.replace("_", " ").title() for name in self.class_names],
            ax=ax,
            square=True,
            cbar_kws={"label": "Proportion" if normalize else "Count"},
        )

        ax.set_title(f"{title} - Face Mask Detection", fontsize=14, pad=20)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)

        # Rotate labels for better readability
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        self._save_figure(fig, filename)

    def plot_class_performance(
        self, metrics_dict: Dict[str, Any], filename: str = "class_performance.png"
    ):
        """
        Plot per-class performance metrics.

        Args:
            metrics_dict: Dictionary containing per-class metrics
            filename: Output filename
        """
        # Extract per-class metrics
        per_class_metrics = metrics_dict.get("per_class_metrics", {})

        if not per_class_metrics:
            print("No per-class metrics available")
            return

        # Prepare data
        classes = list(per_class_metrics.keys())
        metrics = ["precision", "recall", "f1_score"]

        data = {metric: [] for metric in metrics}
        data["support"] = []

        for class_name in classes:
            class_data = per_class_metrics[class_name]
            for metric in metrics:
                data[metric].append(class_data.get(metric, 0))
            data["support"].append(class_data.get("support", 0))

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Bar width and positions
        x = np.arange(len(classes))
        width = 0.25

        # Plot 1: Precision, Recall, F1-Score
        ax1.bar(
            x - width,
            data["precision"],
            width,
            label="Precision",
            color=self.primary_color,
            alpha=0.8,
        )
        ax1.bar(x, data["recall"], width, label="Recall", color=self.secondary_color, alpha=0.8)
        ax1.bar(
            x + width,
            data["f1_score"],
            width,
            label="F1-Score",
            color=self.accent_color,
            alpha=0.8,
        )

        ax1.set_xlabel("Class")
        ax1.set_ylabel("Score")
        ax1.set_title("Per-Class Performance Metrics")
        ax1.set_xticks(x)
        ax1.set_xticklabels([name.replace("_", " ").title() for name in classes], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)

        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(data["precision"], data["recall"], data["f1_score"])):
            ax1.text(i - width, p + 0.01, f"{p:.3f}", ha="center", va="bottom", fontsize=8)
            ax1.text(i, r + 0.01, f"{r:.3f}", ha="center", va="bottom", fontsize=8)
            ax1.text(i + width, f + 0.01, f"{f:.3f}", ha="center", va="bottom", fontsize=8)

        # Plot 2: Support (number of samples per class)
        bars = ax2.bar(classes, data["support"], color=self.class_colors[: len(classes)])
        ax2.set_xlabel("Class")
        ax2.set_ylabel("Number of Samples")
        ax2.set_title("Class Distribution (Support)")
        ax2.set_xticklabels([name.replace("_", " ").title() for name in classes], rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, data["support"]):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(data["support"]) * 0.01,
                f"{value}",
                ha="center",
                va="bottom",
            )

        # Plot 3: Radar chart for metrics
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        ax3 = fig.add_subplot(2, 2, 3, projection="polar")

        for i, class_name in enumerate(classes):
            values = [data[metric][i] for metric in metrics]
            values += values[:1]  # Complete the circle

            ax3.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=class_name.replace("_", " ").title(),
                color=self.class_colors[i],
            )
            ax3.fill(angles, values, alpha=0.25, color=self.class_colors[i])

        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels([metric.title() for metric in metrics])
        ax3.set_ylim(0, 1)
        ax3.set_title("Performance Radar Chart")
        ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        # Plot 4: Heatmap of metrics
        metrics_matrix = np.array(
            [[data[metric][i] for metric in metrics] for i in range(len(classes))]
        )

        sns.heatmap(
            metrics_matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=[metric.title() for metric in metrics],
            yticklabels=[name.replace("_", " ").title() for name in classes],
            ax=ax4,
            square=True,
        )
        ax4.set_title("Metrics Heatmap")
        ax4.set_xlabel("Metrics")
        ax4.set_ylabel("Classes")

        plt.suptitle("Face Mask Detection - Class Performance Analysis", fontsize=16, y=0.98)
        self._save_figure(fig, filename)

    def plot_sample_predictions(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        predicted_labels: torch.Tensor,
        probabilities: torch.Tensor,
        num_samples: int = 16,
        filename: str = "sample_predictions.png",
    ):
        """
        Plot sample predictions with images and probabilities.

        Args:
            images: Batch of images
            true_labels: True labels
            predicted_labels: Predicted labels
            probabilities: Class probabilities
            num_samples: Number of samples to display
            filename: Output filename
        """
        # Limit to available samples
        num_samples = min(num_samples, len(images))

        # Calculate grid size
        cols = 4
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 5))

        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        # Denormalize images for display
        mean = torch.tensor(IMAGE_CONFIG["mean"]).view(3, 1, 1)
        std = torch.tensor(IMAGE_CONFIG["std"]).view(3, 1, 1)

        for i in range(num_samples):
            row = i // cols
            col = i % cols
            ax = axes[row, col]

            # Denormalize image
            image = images[i] * std + mean
            image = torch.clamp(image, 0, 1)

            # Convert to PIL format for display
            image_pil = TF.to_pil_image(image)

            # Display image
            ax.imshow(image_pil)
            ax.axis("off")

            # Get labels and probabilities
            true_label = int(true_labels[i].item())
            pred_label = int(predicted_labels[i].item())
            probs = probabilities[i]

            true_class = self.class_names[true_label]
            pred_class = self.class_names[pred_label]
            confidence = probs[pred_label].item()

            # Determine color based on correctness
            color = "green" if true_label == pred_label else "red"

            # Create title with prediction info
            title = f"True: {true_class.replace('_', ' ').title()}\n"
            title += f"Pred: {pred_class.replace('_', ' ').title()}\n"
            title += f"Conf: {confidence:.3f}"

            ax.set_title(title, fontsize=10, color=color, weight="bold")

            # Add probability bar
            prob_text = ""
            for j, class_name in enumerate(self.class_names):
                prob_text += f"{class_name.replace('_', ' ').title()}: {probs[j]:.3f}\n"

            # Add text box with probabilities
            ax.text(
                0.02,
                0.98,
                prob_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # Hide unused subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis("off")

        plt.suptitle("Sample Predictions - Face Mask Detection", fontsize=16, y=0.98)
        self._save_figure(fig, filename)

    def plot_data_augmentation_examples(
        self,
        original_images: torch.Tensor,
        augmented_images: torch.Tensor,
        num_examples: int = 4,
        filename: str = "data_augmentation_examples.png",
    ):
        """
        Plot examples of data augmentation.

        Args:
            original_images: Original images
            augmented_images: Augmented versions
            num_examples: Number of examples to show
            filename: Output filename
        """
        num_examples = min(num_examples, len(original_images))

        fig, axes = plt.subplots(2, num_examples, figsize=(num_examples * 4, 8))

        if num_examples == 1:
            axes = axes.reshape(2, 1)

        # Denormalize images for display
        mean = torch.tensor(IMAGE_CONFIG["mean"]).view(3, 1, 1)
        std = torch.tensor(IMAGE_CONFIG["std"]).view(3, 1, 1)

        for i in range(num_examples):
            # Original image
            orig_img = original_images[i] * std + mean
            orig_img = torch.clamp(orig_img, 0, 1)
            axes[0, i].imshow(TF.to_pil_image(orig_img))
            axes[0, i].set_title("Original", fontsize=12)
            axes[0, i].axis("off")

            # Augmented image
            aug_img = augmented_images[i] * std + mean
            aug_img = torch.clamp(aug_img, 0, 1)
            axes[1, i].imshow(TF.to_pil_image(aug_img))
            axes[1, i].set_title("Augmented", fontsize=12)
            axes[1, i].axis("off")

        plt.suptitle("Data Augmentation Examples", fontsize=16, y=0.95)
        self._save_figure(fig, filename)

    def plot_dataset_distribution(
        self, class_counts: Dict[str, int], filename: str = "dataset_distribution.png"
    ):
        """
        Plot dataset class distribution.

        Args:
            class_counts: Dictionary of class counts
            filename: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = self.class_colors[: len(classes)]

        # Bar plot
        bars = ax1.bar(classes, counts, color=colors, alpha=0.8)
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Number of Samples")
        ax1.set_title("Class Distribution")
        ax1.set_xticklabels([name.replace("_", " ").title() for name in classes], rotation=45)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(counts) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Pie chart
        pie_colors = [colors[i] for i in range(len(classes))]
        wedges, texts, autotexts = ax2.pie(
            counts,
            labels=[name.replace("_", " ").title() for name in classes],
            autopct="%1.1f%%",
            colors=pie_colors,
            startangle=90,
        )
        ax2.set_title("Class Distribution (Proportions)")

        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_fontweight("bold")
            autotext.set_color("white")

        plt.suptitle(f"Face Mask Dataset Distribution (Total: {sum(counts)} samples)", fontsize=16)
        self._save_figure(fig, filename)

    def plot_roc_curves(
        self, y_true: np.ndarray, y_probabilities: np.ndarray, filename: str = "roc_curves.png"
    ):
        """
        Plot ROC curves for multi-class classification.

        Args:
            y_true: True labels
            y_probabilities: Class probabilities
            filename: Output filename
        """
        # Binarize the output
        y_bin_raw = label_binarize(y_true, classes=range(self.num_classes))
        
        # Convert to numpy array regardless of input type
        try:
            # Try to convert from sparse matrix
            y_bin = np.array(y_bin_raw.toarray())  # type: ignore
        except AttributeError:
            # This is already a numpy array or can be converted to one
            y_bin = np.array(y_bin_raw)
        
        # Handle case where label_binarize returns 1D array (when only one class is present)
        if self.num_classes == 2 and y_bin.ndim == 1:
            y_bin = np.column_stack([1 - y_bin, y_bin])
        elif y_bin.ndim == 1:
            # If we have more than 2 classes but only one is present, expand to 2D
            temp = np.zeros((len(y_bin), self.num_classes))
            temp[:, 0] = y_bin
            y_bin = temp

        fig, ax = plt.subplots(figsize=self.figure_size)

        # Calculate ROC curve for each class
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curves
        for i, class_name in enumerate(self.class_names):
            ax.plot(
                fpr[i],
                tpr[i],
                color=self.class_colors[i],
                lw=2,
                label=f"{class_name.replace('_', ' ').title()} (AUC = {roc_auc[i]:.3f})",
            )

        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", alpha=0.5)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves - Face Mask Detection")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        self._save_figure(fig, filename)

    def plot_learning_curves(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        metric_name: str = "Accuracy",
        filename: str = "learning_curves.png",
    ):
        """
        Plot learning curves showing performance vs training set size.

        Args:
            train_sizes: Array of training set sizes
            train_scores: Training scores for each size
            val_scores: Validation scores for each size
            metric_name: Name of the metric being plotted
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot learning curves
        ax.plot(
            train_sizes,
            train_mean,
            "o-",
            color=self.primary_color,
            label=f"Training {metric_name}",
            linewidth=2,
            markersize=6,
        )
        ax.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1,
            color=self.primary_color,
        )

        ax.plot(
            train_sizes,
            val_mean,
            "s-",
            color=self.secondary_color,
            label=f"Validation {metric_name}",
            linewidth=2,
            markersize=6,
        )
        ax.fill_between(
            train_sizes,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.1,
            color=self.secondary_color,
        )

        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Learning Curves - {metric_name}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        self._save_figure(fig, filename)

    def create_training_dashboard(
        self,
        train_history: List[Dict[str, float]],
        val_history: List[Dict[str, float]],
        confusion_matrix_data: Tuple[np.ndarray, np.ndarray],
        class_metrics: Dict[str, Any],
        filename: str = "training_dashboard.png",
    ):
        """
        Create a comprehensive training dashboard.

        Args:
            train_history: Training history
            val_history: Validation history
            confusion_matrix_data: Tuple of (y_true, y_pred)
            class_metrics: Per-class metrics
            filename: Output filename
        """
        fig = plt.figure(figsize=(20, 15))

        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Training curves
        ax1 = fig.add_subplot(gs[0, :2])
        epochs = range(1, len(train_history) + 1)

        # Plot loss and accuracy
        ax1_twin = ax1.twinx()  # type: ignore

        train_loss = [h["loss"] for h in train_history]
        val_loss = [h["loss"] for h in val_history]
        train_acc = [h.get("accuracy", 0) for h in train_history]
        val_acc = [h.get("accuracy", 0) for h in val_history]

        line1 = ax1.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
        line2 = ax1.plot(epochs, val_loss, "r-", label="Val Loss", linewidth=2)
        line3 = ax1_twin.plot(epochs, train_acc, "b--", label="Train Acc", linewidth=2)  # type: ignore
        line4 = ax1_twin.plot(epochs, val_acc, "r--", label="Val Acc", linewidth=2)  # type: ignore

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="black")
        ax1_twin.set_ylabel("Accuracy", color="gray")
        ax1.set_title("Training Progress")

        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="center right")

        # 2. Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 2])
        y_true, y_pred = confusion_matrix_data
        cm = confusion_matrix(y_true, y_pred, normalize="true")

        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=[name.replace("_", "\n").title() for name in self.class_names],
            yticklabels=[name.replace("_", "\n").title() for name in self.class_names],
            ax=ax2,
            square=True,
        )
        ax2.set_title("Confusion Matrix")

        # 3. Per-class performance
        ax3 = fig.add_subplot(gs[1, :])
        if "per_class_metrics" in class_metrics:
            per_class = class_metrics["per_class_metrics"]
            classes = list(per_class.keys())

            x = np.arange(len(classes))
            width = 0.25

            precision = [per_class[c]["precision"] for c in classes]
            recall = [per_class[c]["recall"] for c in classes]
            f1 = [per_class[c]["f1_score"] for c in classes]

            ax3.bar(x - width, precision, width, label="Precision", alpha=0.8)
            ax3.bar(x, recall, width, label="Recall", alpha=0.8)
            ax3.bar(x + width, f1, width, label="F1-Score", alpha=0.8)

            ax3.set_xlabel("Class")
            ax3.set_ylabel("Score")
            ax3.set_title("Per-Class Performance")
            ax3.set_xticks(x)
            ax3.set_xticklabels([name.replace("_", " ").title() for name in classes])
            ax3.legend()
            ax3.set_ylim(0, 1.1)

        # 4. Final metrics summary
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis("off")

        # Create metrics summary text
        if class_metrics:
            metrics_text = f"""
            Final Model Performance Summary:
            
            Overall Accuracy: {class_metrics.get("accuracy", 0):.4f}
            Macro F1-Score: {class_metrics.get("macro_f1", 0):.4f}
            Macro Precision: {class_metrics.get("macro_precision", 0):.4f}
            Macro Recall: {class_metrics.get("macro_recall", 0):.4f}
            
            Training completed after {len(train_history)} epochs
            Best validation loss: {min(val_loss):.4f}
            """

            ax4.text(
                0.1,
                0.5,
                metrics_text,
                fontsize=12,
                verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            )

        plt.suptitle("Face Mask Detection - Training Dashboard", fontsize=20, y=0.98)
        self._save_figure(fig, filename)


# Convenience functions
def plot_training_results(
    train_history: List[Dict[str, float]],
    val_history: List[Dict[str, float]],
    output_dir: Optional[Path] = None,
):
    """
    Create standard training result plots.

    Args:
        train_history: Training history
        val_history: Validation history
        output_dir: Output directory for plots
    """
    plotter = FaceMaskPlotter(output_dir=output_dir)

    # Plot training history
    plotter.plot_training_history(train_history, val_history)

    print("Training result plots created!")


def plot_evaluation_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probabilities: Optional[np.ndarray] = None,
    metrics_dict: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
):
    """
    Create standard evaluation result plots.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probabilities: Class probabilities (optional)
        metrics_dict: Metrics dictionary (optional)
        output_dir: Output directory for plots
    """
    plotter = FaceMaskPlotter(output_dir=output_dir)

    # Plot confusion matrix
    plotter.plot_confusion_matrix(y_true, y_pred)

    # Plot class performance if metrics available
    if metrics_dict:
        plotter.plot_class_performance(metrics_dict)

    # Plot ROC curves if probabilities available
    if y_probabilities is not None:
        plotter.plot_roc_curves(y_true, y_probabilities)

    print("Evaluation result plots created!")


if __name__ == "__main__":
    # Example usage
    print("Face Mask Detection Plotting Module")
    print("=" * 50)

    # Create sample data for testing
    np.random.seed(42)

    # Sample training history
    train_history = [
        {"loss": 1.2 - i * 0.1, "accuracy": 0.4 + i * 0.05, "f1_score": 0.3 + i * 0.06}
        for i in range(10)
    ]
    val_history = [
        {"loss": 1.1 - i * 0.08, "accuracy": 0.45 + i * 0.04, "f1_score": 0.35 + i * 0.05}
        for i in range(10)
    ]

    # Create plotter and test
    plotter = FaceMaskPlotter()
    plotter.plot_training_history(train_history, val_history)

    # Sample class distribution
    class_counts = {"with_mask": 300, "without_mask": 280, "mask_weared_incorrect": 273}
    plotter.plot_dataset_distribution(class_counts)

    print("Sample plots created successfully!")
