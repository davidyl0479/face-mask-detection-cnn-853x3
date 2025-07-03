"""
Face Mask Detection CNN Feature Analysis Module

This module implements a comprehensive CNN interpretability toolkit that extracts
intermediate layer representations, performs statistical analysis, dimensionality
reduction, clustering, and creates visualizations to understand what different
CNN architectures learn from face mask detection patterns.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .config import (
    DATASET_CONFIG,
    FEATURE_ANALYSIS_CONFIG,
    FIGURES_ROOT,
    VISUALIZATION_CONFIG,
    set_random_seeds,
)
from .dataset import get_dataloaders
from .modeling.model import ModelFactory

warnings.filterwarnings("ignore")


class FeatureExtractor:
    """
    Extract features from intermediate layers of CNN models.

    Provides hooks to capture activations from specified layers
    during forward pass for analysis and visualization.
    """

    def __init__(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        """
        Initialize feature extractor.

        Args:
            model: PyTorch model to extract features from
            layer_names: List of layer names to extract features from
        """
        self.model = model
        self.layer_names = layer_names or FEATURE_ANALYSIS_CONFIG["extract_layers"]
        self.features = {}
        self.hooks = []

        # Register hooks for feature extraction
        self._register_hooks()

        print(f"Feature extractor initialized for layers: {self.layer_names}")

    def _register_hooks(self):
        """Register forward hooks to capture layer activations."""

        def get_activation(name):
            def hook(module, input, output):
                # Store activation with layer name
                if isinstance(output, torch.Tensor):
                    self.features[name] = output.detach().cpu()
                else:
                    # Handle cases where output is a tuple/list
                    self.features[name] = (
                        output[0].detach().cpu() if isinstance(output, (tuple, list)) else output
                    )

            return hook

        # Find and register hooks for specified layers
        for name, module in self.model.named_modules():
            if any(layer_name in name for layer_name in self.layer_names):
                handle = module.register_forward_hook(get_activation(name))
                self.hooks.append(handle)
                print(f"Registered hook for layer: {name}")

    def extract_features(
        self, data_loader: DataLoader, max_samples: Optional[int] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract features from the model for given data.

        Args:
            data_loader: DataLoader containing input data
            max_samples: Maximum number of samples to process

        Returns:
            Dictionary mapping layer names to extracted features and labels
        """
        self.model.eval()

        all_features = {layer: [] for layer in self.layer_names}
        all_labels = []
        sample_count = 0

        print("Extracting features...")

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(tqdm(data_loader)):
                if max_samples and sample_count >= max_samples:
                    break

                # Forward pass to trigger hooks
                _ = self.model(data)

                # Store features from each layer
                for layer_name in self.layer_names:
                    if layer_name in self.features:
                        features = self.features[layer_name]

                        # Flatten spatial dimensions if needed
                        if len(features.shape) > 2:
                            features = features.view(features.size(0), -1)

                        all_features[layer_name].append(features.numpy())

                # Store labels
                all_labels.append(labels.numpy())
                sample_count += len(data)

                # Clear features for next batch
                self.features.clear()

        # Concatenate all features
        feature_dict = {}
        all_labels = np.concatenate(all_labels)

        for layer_name in self.layer_names:
            if all_features[layer_name]:
                features = np.concatenate(all_features[layer_name], axis=0)
                feature_dict[layer_name] = {
                    "features": features,
                    "labels": all_labels[: len(features)],
                }
                print(f"Extracted features from {layer_name}: {features.shape}")

        return feature_dict

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("All hooks removed")


class FeatureAnalyzer:
    """
    Analyze extracted CNN features for face mask detection interpretability.

    Provides statistical analysis, dimensionality reduction, clustering,
    and visualization capabilities for understanding learned representations.
    """

    def __init__(
        self, feature_dict: Dict[str, Dict[str, np.ndarray]], class_names: Optional[List[str]] = None
    ):
        """
        Initialize feature analyzer.

        Args:
            feature_dict: Dictionary of extracted features per layer
            class_names: Names of the classes
        """
        self.feature_dict = feature_dict
        self.class_names = class_names or DATASET_CONFIG["classes"]
        self.num_classes = len(self.class_names)

        # Analysis results storage
        self.statistics = {}
        self.reduced_features = {}
        self.cluster_results = {}

        print(f"Feature analyzer initialized with {len(feature_dict)} layers")

    def compute_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistical measures for extracted features.

        Returns:
            Dictionary of statistics per layer
        """
        print("Computing feature statistics...")

        for layer_name, layer_data in self.feature_dict.items():
            features = layer_data["features"]
            labels = layer_data["labels"]

            stats = {}

            # Basic statistics
            stats["mean"] = np.mean(features, axis=0)
            stats["std"] = np.std(features, axis=0)
            stats["min"] = np.min(features, axis=0)
            stats["max"] = np.max(features, axis=0)

            # Per-class statistics
            per_class_stats = {}
            for class_idx, class_name in enumerate(self.class_names):
                class_mask = labels == class_idx
                if np.any(class_mask):
                    class_features = features[class_mask]
                    per_class_stats[class_name] = {
                        "mean": np.mean(class_features, axis=0),
                        "std": np.std(class_features, axis=0),
                        "count": np.sum(class_mask),
                    }

            stats["per_class"] = per_class_stats

            # Feature activation statistics
            stats["sparsity"] = np.mean(features == 0)  # Proportion of zero activations
            stats["average_activation"] = np.mean(np.abs(features))
            stats["feature_variance"] = np.var(features, axis=0)
            stats["feature_importance"] = np.var(features, axis=0) / np.sum(
                np.var(features, axis=0)
            )

            # Inter-class separability
            if len(per_class_stats) > 1:
                class_means = np.array(
                    [
                        per_class_stats[cn]["mean"]
                        for cn in self.class_names
                        if cn in per_class_stats
                    ]
                )
                if len(class_means) > 1:
                    # Between-class variance
                    between_class_var = np.var(class_means, axis=0)

                    # Within-class variance (average)
                    within_class_vars = []
                    for class_name in self.class_names:
                        if class_name in per_class_stats:
                            within_class_vars.append(per_class_stats[class_name]["std"] ** 2)

                    if within_class_vars:
                        within_class_var = np.mean(within_class_vars, axis=0)

                        # Fisher's criterion (separability measure)
                        fisher_ratio = between_class_var / (within_class_var + 1e-8)
                        stats["fisher_ratio"] = fisher_ratio
                        stats["separability_score"] = np.mean(fisher_ratio)

            self.statistics[layer_name] = stats
            print(f"Statistics computed for {layer_name}")

        return self.statistics

    def reduce_dimensionality(
        self, method: str = "tsne", n_components: int = 2, **kwargs
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Perform dimensionality reduction on extracted features.

        Args:
            method: Reduction method ('tsne', 'pca')
            n_components: Number of components to reduce to
            **kwargs: Additional parameters for the reduction method

        Returns:
            Dictionary of reduced features per layer
        """
        print(f"Performing {method.upper()} dimensionality reduction...")

        # Set random seed for reproducible results
        set_random_seeds()

        for layer_name, layer_data in self.feature_dict.items():
            features = layer_data["features"]
            labels = layer_data["labels"]

            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Apply dimensionality reduction
            if method.lower() == "tsne":
                reducer = TSNE(
                    n_components=n_components,
                    random_state=42,
                    perplexity=kwargs.get("perplexity", 30),
                    n_iter=kwargs.get("n_iter", 1000),
                    verbose=1,
                )
            elif method.lower() == "pca":
                reducer = PCA(n_components=n_components, random_state=42)
            else:
                raise ValueError(f"Unsupported method: {method}")

            # Fit and transform
            try:
                reduced_features = reducer.fit_transform(features_scaled)

                self.reduced_features[layer_name] = {
                    "features": reduced_features,
                    "labels": labels,
                    "method": method,
                    "scaler": scaler,
                    "reducer": reducer,
                }

                # Add explained variance for PCA
                if method.lower() == "pca" and isinstance(reducer, PCA):
                    self.reduced_features[layer_name]["explained_variance_ratio"] = (
                        reducer.explained_variance_ratio_
                    )
                    self.reduced_features[layer_name]["cumulative_variance"] = np.cumsum(
                        reducer.explained_variance_ratio_
                    )

                print(
                    f"Reduced {layer_name} from {features.shape[1]} to {n_components} dimensions"
                )

            except Exception as e:
                print(f"Error reducing dimensionality for {layer_name}: {e}")
                continue

        return self.reduced_features

    def perform_clustering(
        self, method: str = "kmeans", use_reduced: bool = True, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform clustering analysis on features.

        Args:
            method: Clustering method ('kmeans', 'dbscan')
            use_reduced: Whether to use dimensionality-reduced features
            **kwargs: Additional parameters for clustering

        Returns:
            Dictionary of clustering results per layer
        """
        print(f"Performing {method.upper()} clustering...")

        # Choose feature source
        feature_source = self.reduced_features if use_reduced else self.feature_dict

        if not feature_source:
            if use_reduced:
                print("No reduced features available. Run reduce_dimensionality() first.")
                return {}
            else:
                print("No features available for clustering.")
                return {}

        for layer_name, layer_data in feature_source.items():
            features = layer_data["features"]
            true_labels = layer_data["labels"]

            # Apply clustering
            if method.lower() == "kmeans":
                n_clusters = kwargs.get("n_clusters", self.num_classes)
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            elif method.lower() == "dbscan":
                clusterer = DBSCAN(
                    eps=kwargs.get("eps", 0.5), min_samples=kwargs.get("min_samples", 5)
                )
            else:
                raise ValueError(f"Unsupported clustering method: {method}")

            try:
                cluster_labels = clusterer.fit_predict(features)

                # Calculate clustering metrics
                if len(np.unique(cluster_labels)) > 1 and -1 not in cluster_labels:
                    silhouette = silhouette_score(features, cluster_labels)
                    ari = adjusted_rand_score(true_labels, cluster_labels)
                else:
                    silhouette = -1
                    ari = -1

                self.cluster_results[layer_name] = {
                    "cluster_labels": cluster_labels,
                    "true_labels": true_labels,
                    "clusterer": clusterer,
                    "method": method,
                    "n_clusters": len(np.unique(cluster_labels[cluster_labels != -1])),
                    "silhouette_score": silhouette,
                    "adjusted_rand_index": ari,
                    "features": features,
                }

                print(
                    f"Clustering for {layer_name}: {len(np.unique(cluster_labels))} clusters, "
                    f"Silhouette={silhouette:.3f}, ARI={ari:.3f}"
                )

            except Exception as e:
                print(f"Error clustering {layer_name}: {e}")
                continue

        return self.cluster_results

    def analyze_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Analyze which features are most important for class separation.

        Returns:
            Dictionary of feature importance scores per layer
        """
        print("Analyzing feature importance...")

        importance_scores = {}

        for layer_name, stats in self.statistics.items():
            if "fisher_ratio" in stats:
                # Use Fisher ratio as importance measure
                importance = stats["fisher_ratio"]

                # Normalize importance scores
                importance_normalized = importance / np.sum(importance)
                importance_scores[layer_name] = importance_normalized

                print(f"Feature importance computed for {layer_name}")

        return importance_scores

    def compare_layers(self) -> pd.DataFrame:
        """
        Compare different layers based on various metrics.

        Returns:
            DataFrame comparing layer characteristics
        """
        print("Comparing layers...")

        comparison_data = []

        for layer_name, stats in self.statistics.items():
            layer_info = {
                "layer_name": layer_name,
                "feature_dimensions": len(stats["mean"]),
                "sparsity": stats.get("sparsity", 0),
                "average_activation": stats.get("average_activation", 0),
                "separability_score": stats.get("separability_score", 0),
            }

            # Add clustering metrics if available
            if layer_name in self.cluster_results:
                cluster_data = self.cluster_results[layer_name]
                layer_info["silhouette_score"] = cluster_data.get("silhouette_score", -1)
                layer_info["adjusted_rand_index"] = cluster_data.get("adjusted_rand_index", -1)
                layer_info["n_clusters_found"] = cluster_data.get("n_clusters", 0)

            comparison_data.append(layer_info)

        return pd.DataFrame(comparison_data)


class FeatureVisualizer:
    """
    Visualize extracted features and analysis results.

    Creates comprehensive visualizations for understanding CNN learned
    representations in face mask detection.
    """

    def __init__(
        self, analyzer: FeatureAnalyzer, output_dir: Optional[Path] = None, save_plots: bool = True
    ):
        """
        Initialize feature visualizer.

        Args:
            analyzer: FeatureAnalyzer instance with computed results
            output_dir: Directory to save plots
            save_plots: Whether to save plots to files
        """
        self.analyzer = analyzer
        self.output_dir = output_dir or FIGURES_ROOT / "feature_analysis"
        self.output_dir.mkdir(exist_ok=True)
        self.save_plots = save_plots

        # Visualization settings
        self.class_colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, self.analyzer.num_classes))
        self.figure_size = VISUALIZATION_CONFIG.get("figure_size", (12, 8))
        self.dpi = VISUALIZATION_CONFIG.get("dpi", 300)

        print(f"Feature visualizer initialized. Plots will be saved to: {self.output_dir}")

    def _save_figure(self, fig, filename: str):
        """Save figure to file."""
        if self.save_plots:
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved: {filepath}")
        plt.show()

    def plot_feature_distributions(self, layer_name: str, max_features: int = 20):
        """
        Plot feature activation distributions for a specific layer.

        Args:
            layer_name: Name of layer to visualize
            max_features: Maximum number of features to plot
        """
        if layer_name not in self.analyzer.feature_dict:
            print(f"Layer {layer_name} not found in feature dictionary")
            return

        layer_data = self.analyzer.feature_dict[layer_name]
        features = layer_data["features"]
        labels = layer_data["labels"]

        # Select subset of features to plot
        n_features = min(max_features, features.shape[1])
        feature_indices = np.linspace(0, features.shape[1] - 1, n_features, dtype=int)

        # Create subplots
        cols = 4
        rows = (n_features + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)

        for i, feature_idx in enumerate(feature_indices):
            row = i // cols
            col = i % cols
            ax = axes[row, col]

            # Plot distribution for each class
            for class_idx, class_name in enumerate(self.analyzer.class_names):
                class_mask = labels == class_idx
                if np.any(class_mask):
                    class_features = features[class_mask, feature_idx]
                    ax.hist(
                        class_features,
                        alpha=0.6,
                        label=class_name.replace("_", " ").title(),
                        color=self.class_colors[class_idx],
                        bins=20,
                    )

            ax.set_title(f"Feature {feature_idx}")
            ax.set_xlabel("Activation Value")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_features, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis("off")

        plt.suptitle(f"Feature Distributions - {layer_name}", fontsize=16)
        plt.tight_layout()
        self._save_figure(fig, f"feature_distributions_{layer_name}.png")

    def plot_dimensionality_reduction(self, layer_name: str):
        """
        Plot dimensionality reduction results.

        Args:
            layer_name: Name of layer to visualize
        """
        if layer_name not in self.analyzer.reduced_features:
            print(f"No reduced features found for {layer_name}")
            return

        layer_data = self.analyzer.reduced_features[layer_name]
        features = layer_data["features"]
        labels = layer_data["labels"]
        method = layer_data["method"]

        fig, ax = plt.subplots(figsize=self.figure_size)

        # Plot each class with different colors
        for class_idx, class_name in enumerate(self.analyzer.class_names):
            class_mask = labels == class_idx
            if np.any(class_mask):
                ax.scatter(
                    features[class_mask, 0],
                    features[class_mask, 1],
                    c=[self.class_colors[class_idx]],
                    label=class_name.replace("_", " ").title(),
                    alpha=0.7,
                    s=30,
                )

        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_title(f"{method.upper()} Visualization - {layer_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add explained variance for PCA
        if method.lower() == "pca" and "explained_variance_ratio" in layer_data:
            variance_ratio = layer_data["explained_variance_ratio"]
            ax.set_xlabel(f"PC1 ({variance_ratio[0]:.1%} variance)")
            ax.set_ylabel(f"PC2 ({variance_ratio[1]:.1%} variance)")

        self._save_figure(fig, f"{method}_visualization_{layer_name}.png")

    def plot_clustering_results(self, layer_name: str):
        """
        Plot clustering results.

        Args:
            layer_name: Name of layer to visualize
        """
        if layer_name not in self.analyzer.cluster_results:
            print(f"No clustering results found for {layer_name}")
            return

        cluster_data = self.analyzer.cluster_results[layer_name]
        features = cluster_data["features"]
        cluster_labels = cluster_data["cluster_labels"]
        true_labels = cluster_data["true_labels"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Clustering results
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster_id in enumerate(unique_clusters):
            if cluster_id == -1:  # Noise points (for DBSCAN)
                color = "black"
                label = "Noise"
            else:
                color = colors[i]
                label = f"Cluster {cluster_id}"

            cluster_mask = cluster_labels == cluster_id
            if features.shape[1] >= 2:
                ax1.scatter(
                    features[cluster_mask, 0],
                    features[cluster_mask, 1],
                    c=[color],
                    label=label,
                    alpha=0.7,
                    s=30,
                )

        ax1.set_title(f"Clustering Results - {layer_name}")
        ax1.set_xlabel("Feature 1" if features.shape[1] >= 2 else "Feature Index")
        ax1.set_ylabel("Feature 2" if features.shape[1] >= 2 else "Feature Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: True class labels for comparison
        for class_idx, class_name in enumerate(self.analyzer.class_names):
            class_mask = true_labels == class_idx
            if np.any(class_mask) and features.shape[1] >= 2:
                ax2.scatter(
                    features[class_mask, 0],
                    features[class_mask, 1],
                    c=[self.class_colors[class_idx]],
                    label=class_name.replace("_", " ").title(),
                    alpha=0.7,
                    s=30,
                )

        ax2.set_title(f"True Class Labels - {layer_name}")
        ax2.set_xlabel("Feature 1" if features.shape[1] >= 2 else "Feature Index")
        ax2.set_ylabel("Feature 2" if features.shape[1] >= 2 else "Feature Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add metrics to title
        silhouette = cluster_data.get("silhouette_score", -1)
        ari = cluster_data.get("adjusted_rand_index", -1)
        plt.suptitle(f"Clustering Analysis - Silhouette: {silhouette:.3f}, ARI: {ari:.3f}")

        self._save_figure(fig, f"clustering_results_{layer_name}.png")

    def plot_layer_comparison(self):
        """Plot comparison of different layers."""
        comparison_df = self.analyzer.compare_layers()

        if comparison_df.empty:
            print("No layer comparison data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Separability scores
        axes[0, 0].bar(comparison_df["layer_name"], comparison_df["separability_score"])
        axes[0, 0].set_title("Layer Separability Scores")
        axes[0, 0].set_ylabel("Separability Score")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Plot 2: Feature dimensions
        axes[0, 1].bar(comparison_df["layer_name"], comparison_df["feature_dimensions"])
        axes[0, 1].set_title("Feature Dimensions per Layer")
        axes[0, 1].set_ylabel("Number of Features")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Plot 3: Sparsity
        axes[1, 0].bar(comparison_df["layer_name"], comparison_df["sparsity"])
        axes[1, 0].set_title("Feature Sparsity")
        axes[1, 0].set_ylabel("Sparsity (Proportion of Zeros)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 4: Average activation
        axes[1, 1].bar(comparison_df["layer_name"], comparison_df["average_activation"])
        axes[1, 1].set_title("Average Feature Activation")
        axes[1, 1].set_ylabel("Average Activation Magnitude")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        self._save_figure(fig, "layer_comparison.png")

    def create_feature_analysis_dashboard(self):
        """Create comprehensive feature analysis dashboard."""
        # Get a representative layer for detailed analysis
        if not self.analyzer.reduced_features:
            print("No reduced features available for dashboard")
            return

        layer_name = list(self.analyzer.reduced_features.keys())[0]

        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. t-SNE visualization
        if layer_name in self.analyzer.reduced_features:
            ax1 = fig.add_subplot(gs[0, 0])
            layer_data = self.analyzer.reduced_features[layer_name]
            features = layer_data["features"]
            labels = layer_data["labels"]

            for class_idx, class_name in enumerate(self.analyzer.class_names):
                class_mask = labels == class_idx
                if np.any(class_mask):
                    ax1.scatter(
                        features[class_mask, 0],
                        features[class_mask, 1],
                        c=[self.class_colors[class_idx]],
                        label=class_name.replace("_", " ").title(),
                        alpha=0.7,
                        s=20,
                    )

            ax1.set_title(f"t-SNE Visualization\n{layer_name}")
            ax1.legend(fontsize=8)

        # 2. Layer comparison
        ax2 = fig.add_subplot(gs[0, 1:])
        comparison_df = self.analyzer.compare_layers()
        if not comparison_df.empty:
            x = np.arange(len(comparison_df))
            width = 0.35

            ax2.bar(
                x - width / 2,
                comparison_df["separability_score"],
                width,
                label="Separability",
                alpha=0.8,
            )
            ax2.bar(x + width / 2, comparison_df["sparsity"], width, label="Sparsity", alpha=0.8)

            ax2.set_xlabel("Layer")
            ax2.set_ylabel("Score")
            ax2.set_title("Layer Characteristics Comparison")
            ax2.set_xticks(x)
            ax2.set_xticklabels(comparison_df["layer_name"], rotation=45)
            ax2.legend()

        # 3. Feature importance (if available)
        importance_scores = self.analyzer.analyze_feature_importance()
        if importance_scores:
            ax3 = fig.add_subplot(gs[1, :])
            layer_name_imp = list(importance_scores.keys())[0]
            importance = importance_scores[layer_name_imp]

            # Show top 20 most important features
            top_indices = np.argsort(importance)[-20:]
            ax3.bar(range(len(top_indices)), importance[top_indices])
            ax3.set_title(f"Top 20 Most Important Features - {layer_name_imp}")
            ax3.set_xlabel("Feature Index")
            ax3.set_ylabel("Importance Score")

        # 4. Clustering quality metrics
        ax4 = fig.add_subplot(gs[2, :])
        if self.analyzer.cluster_results:
            layers = list(self.analyzer.cluster_results.keys())
            silhouettes = [self.analyzer.cluster_results[layer]["silhouette_score"] for layer in layers]
            aris = [self.analyzer.cluster_results[layer]["adjusted_rand_index"] for layer in layers]

            x = np.arange(len(layers))
            width = 0.35

            ax4.bar(x - width / 2, silhouettes, width, label="Silhouette Score", alpha=0.8)
            ax4.bar(x + width / 2, aris, width, label="Adjusted Rand Index", alpha=0.8)

            ax4.set_xlabel("Layer")
            ax4.set_ylabel("Score")
            ax4.set_title("Clustering Quality Metrics")
            ax4.set_xticks(x)
            ax4.set_xticklabels(layers, rotation=45)
            ax4.legend()

        plt.suptitle("CNN Feature Analysis Dashboard - Face Mask Detection", fontsize=20)
        self._save_figure(fig, "feature_analysis_dashboard.png")


# Convenience functions
def analyze_model_features(
    model_path: str,
    layer_names: Optional[List[str]] = None,
    max_samples: int = 1000,
    output_dir: Optional[Path] = None,
) -> FeatureAnalyzer:
    """
    Complete feature analysis pipeline for a trained model.

    Args:
        model_path: Path to saved model
        layer_names: List of layer names to analyze
        max_samples: Maximum samples to analyze
        output_dir: Output directory for plots

    Returns:
        FeatureAnalyzer with computed results
    """
    print("Starting complete feature analysis pipeline...")

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    model = ModelFactory.create_model(
        checkpoint["model_name"], **checkpoint.get("model_params", {})
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get data loader
    _, _, test_loader = get_dataloaders()

    # Extract features
    extractor = FeatureExtractor(model, layer_names)
    feature_dict = extractor.extract_features(test_loader, max_samples)
    extractor.remove_hooks()

    # Analyze features
    analyzer = FeatureAnalyzer(feature_dict)
    analyzer.compute_statistics()
    analyzer.reduce_dimensionality(method="tsne")
    analyzer.perform_clustering()

    # Create visualizations
    visualizer = FeatureVisualizer(analyzer, output_dir)

    # Create plots for each layer
    for layer_name in feature_dict.keys():
        visualizer.plot_dimensionality_reduction(layer_name)
        visualizer.plot_clustering_results(layer_name)

    visualizer.plot_layer_comparison()
    visualizer.create_feature_analysis_dashboard()

    print("Feature analysis pipeline completed!")
    return analyzer


if __name__ == "__main__":
    # Example usage
    print("CNN Feature Analysis Module")
    print("=" * 50)

    # Note: This requires a trained model
    from .config import MODELS_ROOT

    model_path = MODELS_ROOT / "best_model.pth"

    if model_path.exists():
        analyzer = analyze_model_features(str(model_path), max_samples=500)
        print("Feature analysis completed!")
    else:
        print(f"No trained model found at {model_path}")
        print("Please train a model first using train.py")
