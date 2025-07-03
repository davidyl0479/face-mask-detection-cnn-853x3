"""
Face Mask Detection Prediction Module

This module handles face mask detection model inference and evaluation,
including loading trained models, making batch predictions on test images,
calculating comprehensive performance metrics, and saving prediction results.
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import (
    DATASET_CONFIG,
    DEVICE_CONFIG,
    EVALUATION_CONFIG,
    FIGURES_ROOT,
    MODELS_ROOT,
    PREDICTION_CONFIG,
    set_random_seeds,
)
from ..dataset import get_dataloaders
from .model import ModelFactory


class FaceMaskPredictor:
    """
    Predictor class for face mask detection models.

    Handles model loading, inference, evaluation, and results analysis.
    """

    def __init__(
        self, model_path: str, device: Optional[str] = None, batch_size: Optional[int] = None
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the saved model checkpoint
            device: Device to use for inference
            batch_size: Batch size for inference
        """
        # Set random seeds for reproducible predictions
        set_random_seeds()

        # Set device
        self.device = device or DEVICE_CONFIG["device"]
        print(f"Using device: {self.device}")

        # Set batch size
        self.batch_size = batch_size or PREDICTION_CONFIG["batch_size"]

        # Load model
        self.model_path = model_path
        self.model: Optional[nn.Module] = None
        self.model_info = {}
        self._load_model()

        # Initialize prediction storage
        self.predictions = []
        self.probabilities = []
        self.true_labels = []
        self.prediction_df = None

        # Class information
        self.class_names = DATASET_CONFIG["classes"]
        self.num_classes = DATASET_CONFIG["num_classes"]

        print(f"Predictor initialized with model: {self.model_path}")

    @property
    def loaded_model(self) -> nn.Module:
        """
        Get the loaded model, ensuring it's not None.
        
        Returns:
            The loaded model
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please check model path and initialization.")
        return self.model

    def _load_model(self):
        """Load the trained model from checkpoint."""
        print(f"Loading model from: {self.model_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Extract model information
            self.model_info = {
                "model_name": checkpoint.get("model_name", "unknown"),
                "model_params": checkpoint.get("model_params", {}),
                "epoch": checkpoint.get("epoch", 0),
                "metrics": checkpoint.get("metrics", {}),
            }

            # Create model
            self.model = ModelFactory.create_model(
                self.model_info["model_name"], **self.model_info["model_params"]
            )

            # Load weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()

            print("Model loaded successfully!")
            print(f"Model: {self.model_info['model_name']}")
            print(f"Epoch: {self.model_info['epoch']}")

            if self.model_info["metrics"]:
                print(f"Best metrics: {self.model_info['metrics']}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict_batch(
        self, data_loader: DataLoader, return_probabilities: bool = True, use_tta: bool = False
    ) -> Tuple[List[int], List[List[float]], List[int]]:
        """
        Make predictions on a batch of data.

        Args:
            data_loader: DataLoader for the data
            return_probabilities: Whether to return class probabilities
            use_tta: Whether to use Test Time Augmentation

        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Check model path and initialization.")
        
        self.model.eval()
        predictions = []
        probabilities = []
        true_labels = []

        with torch.no_grad():
            for data, targets in tqdm(data_loader, desc="Making predictions"):
                data = data.to(self.device)

                if use_tta and PREDICTION_CONFIG["tta"]:
                    # Test Time Augmentation
                    outputs = self._predict_with_tta(data)
                else:
                    # Standard prediction
                    outputs = self.model(data)  # Model is guaranteed to be not None due to check above

                # Get probabilities
                probs = F.softmax(outputs, dim=1)

                # Get predictions
                preds = outputs.argmax(dim=1)

                # Store results
                predictions.extend(preds.cpu().numpy().tolist())
                if return_probabilities:
                    probabilities.extend(probs.cpu().numpy().tolist())
                true_labels.extend(targets.numpy().tolist())

        return predictions, probabilities, true_labels

    def _predict_with_tta(self, data: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using Test Time Augmentation.

        Args:
            data: Input data tensor

        Returns:
            Averaged predictions from multiple augmentations
        """
        # TODO: Implement TTA with different augmentations
        # For now, just return standard prediction
        if self.model is None:
            raise RuntimeError("Model not loaded. Check model path and initialization.")
        return self.model(data)

    def predict_dataset(
        self, data_loader: Optional[DataLoader] = None, dataset_split: str = "test"
    ) -> pd.DataFrame:
        """
        Make predictions on a complete dataset.

        Args:
            data_loader: DataLoader for the dataset
            dataset_split: Which dataset split to use ("train", "val", "test")

        Returns:
            DataFrame with predictions and metadata
        """
        # Get data loader if not provided
        if data_loader is None:
            train_loader, val_loader, test_loader = get_dataloaders()

            if dataset_split == "train":
                data_loader = train_loader
            elif dataset_split == "val":
                data_loader = val_loader
            else:
                data_loader = test_loader

        print(f"Making predictions on {dataset_split} dataset...")

        # Make predictions
        predictions, probabilities, true_labels = self.predict_batch(
            data_loader, return_probabilities=True, use_tta=PREDICTION_CONFIG["tta"]
        )

        # Store results
        self.predictions = predictions
        self.probabilities = probabilities
        self.true_labels = true_labels

        # Create DataFrame with results
        self.prediction_df = self._create_prediction_dataframe()

        print(f"Predictions completed! Total samples: {len(predictions)}")

        return self.prediction_df

    def _create_prediction_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with prediction results."""
        data = []

        for i, (pred, true_label) in enumerate(zip(self.predictions, self.true_labels)):
            row = {
                "sample_id": i,
                "true_label": true_label,
                "true_class": self.class_names[true_label],
                "predicted_label": pred,
                "predicted_class": self.class_names[pred],
                "correct": pred == true_label,
                "confidence": max(self.probabilities[i]) if self.probabilities else 0.0,
            }

            # Add class probabilities
            if self.probabilities:
                for j, class_name in enumerate(self.class_names):
                    row[f"prob_{class_name}"] = self.probabilities[i][j]

            data.append(row)

        return pd.DataFrame(data)

    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.

        Returns:
            Dictionary of computed metrics
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run predict_dataset() first.")

        predictions = np.array(self.predictions)
        true_labels = np.array(self.true_labels)

        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)

        # Per-class and macro metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        # Ensure per-class metrics are arrays
        precision = np.array(precision) if not isinstance(precision, np.ndarray) else precision
        recall = np.array(recall) if not isinstance(recall, np.ndarray) else recall
        f1 = np.array(f1) if not isinstance(f1, np.ndarray) else f1
        support = np.array(support) if not isinstance(support, np.ndarray) else support

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="macro", zero_division=0
        )

        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="weighted", zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)

        # Classification report
        class_report = classification_report(
            true_labels,
            predictions,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        # ROC AUC for multi-class (if probabilities available)
        roc_auc_scores = {}
        if self.probabilities:
            try:
                # Binarize labels for multi-class ROC
                y_bin_raw = label_binarize(true_labels, classes=range(self.num_classes))
                
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
                
                y_score = np.array(self.probabilities)

                # Calculate ROC AUC for each class
                for i, class_name in enumerate(self.class_names):
                    if np.sum(y_bin[:, i]) > 0:  # Check if class exists in true labels
                        roc_auc_scores[f"roc_auc_{class_name}"] = roc_auc_score(
                            y_bin[:, i], y_score[:, i]
                        )

                # Macro and weighted average ROC AUC
                roc_auc_scores["roc_auc_macro"] = roc_auc_score(
                    y_bin, y_score, multi_class="ovr", average="macro"
                )
                roc_auc_scores["roc_auc_weighted"] = roc_auc_score(
                    y_bin, y_score, multi_class="ovr", average="weighted"
                )

            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC scores: {e}")

        # Compile all metrics
        metrics = {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
            "per_class_metrics": {},
        }

        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics["per_class_metrics"][class_name] = {
                "precision": float(precision[i]) if i < len(precision) else 0.0,
                "recall": float(recall[i]) if i < len(recall) else 0.0,
                "f1_score": float(f1[i]) if i < len(f1) else 0.0,
                "support": int(support[i]) if i < len(support) else 0,
            }

        # Add ROC AUC scores
        metrics.update(roc_auc_scores)

        return metrics

    def print_metrics_summary(self, metrics: Optional[Dict[str, Any]] = None):
        """
        Print a formatted summary of evaluation metrics.

        Args:
            metrics: Metrics dictionary (calculated if not provided)
        """
        if metrics is None:
            metrics = self.calculate_metrics()

        print("\nFace Mask Detection - Model Evaluation Summary")
        print("=" * 60)

        # Overall metrics
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")

        if "roc_auc_macro" in metrics:
            print(f"Macro ROC AUC: {metrics['roc_auc_macro']:.4f}")

        print("\nPer-Class Performance:")
        print("-" * 40)

        # Per-class metrics
        for class_name, class_metrics in metrics["per_class_metrics"].items():
            print(f"\n{class_name.replace('_', ' ').title()}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall: {class_metrics['recall']:.4f}")
            print(f"  F1-Score: {class_metrics['f1_score']:.4f}")
            print(f"  Support: {class_metrics['support']}")

            if f"roc_auc_{class_name}" in metrics:
                print(f"  ROC AUC: {metrics[f'roc_auc_{class_name}']:.4f}")

        print("\nConfusion Matrix:")
        print("-" * 20)
        cm = np.array(metrics["confusion_matrix"])

        # Print confusion matrix with class names
        print("Predicted ->")
        print("True ↓   ", end="")
        for name in self.class_names:
            print(f"{name[:8]:>8}", end=" ")
        print()

        for i, true_class in enumerate(self.class_names):
            print(f"{true_class[:8]:>8} ", end="")
            for j in range(len(self.class_names)):
                print(f"{cm[i, j]:>8}", end=" ")
            print()

    def save_predictions(
        self, output_path: Optional[str] = None, include_probabilities: bool = True
    ) -> str:
        """
        Save predictions to CSV file.

        Args:
            output_path: Path to save predictions
            include_probabilities: Whether to include class probabilities

        Returns:
            Path to saved file
        """
        if self.prediction_df is None:
            raise ValueError("No predictions available. Run predict_dataset() first.")

        if output_path is None:
            output_path = str(FIGURES_ROOT / PREDICTION_CONFIG["predictions_filename"])

        # Select columns to save
        columns_to_save = [
            "sample_id",
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "correct",
            "confidence",
        ]

        if include_probabilities and self.probabilities:
            prob_columns = [f"prob_{class_name}" for class_name in self.class_names]
            columns_to_save.extend(prob_columns)

        # Save to CSV
        df_to_save = self.prediction_df[columns_to_save]
        df_to_save.to_csv(output_path, index=False)

        print(f"Predictions saved to: {output_path}")
        return str(output_path)

    def save_metrics(self, metrics: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Save metrics to JSON file.

        Args:
            metrics: Metrics dictionary
            output_path: Path to save metrics

        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = str(FIGURES_ROOT / "evaluation_metrics.json")

        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            else:
                metrics_serializable[key] = value

        # Add model information
        metrics_serializable["model_info"] = self.model_info
        metrics_serializable["evaluation_timestamp"] = time.time()

        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(metrics_serializable, f, indent=2)

        print(f"Metrics saved to: {output_path}")
        return str(output_path)

    def get_misclassified_samples(self, top_k: int = 10) -> pd.DataFrame:
        """
        Get the most confidently misclassified samples.

        Args:
            top_k: Number of top misclassified samples to return

        Returns:
            DataFrame with misclassified samples
        """
        if self.prediction_df is None:
            raise ValueError("No predictions available. Run predict_dataset() first.")

        # Filter misclassified samples
        misclassified = self.prediction_df[~self.prediction_df["correct"]].copy()

        if len(misclassified) == 0:
            print("No misclassified samples found!")
            return pd.DataFrame()

        # Sort by confidence (most confident misclassifications first)
        misclassified = misclassified.sort_values("confidence", ascending=False)

        return misclassified.head(top_k)

    def analyze_class_performance(self) -> Dict[str, Any]:
        """
        Analyze performance for each class in detail.

        Returns:
            Dictionary with detailed class analysis
        """
        if self.prediction_df is None:
            raise ValueError("No predictions available. Run predict_dataset() first.")

        analysis = {}

        for i, class_name in enumerate(self.class_names):
            class_true = self.prediction_df[self.prediction_df["true_label"] == i]
            class_pred = self.prediction_df[self.prediction_df["predicted_label"] == i]

            analysis[class_name] = {
                "total_true_samples": len(class_true),
                "total_predicted_samples": len(class_pred),
                "correctly_classified": len(class_true[class_true["correct"]]),
                "misclassified_as": class_true[~class_true["correct"]]["predicted_class"]
                .value_counts()
                .to_dict(),
                "confused_with": class_pred[~class_pred["correct"]]["true_class"]
                .value_counts()
                .to_dict(),
                "average_confidence_correct": class_true[class_true["correct"]][
                    "confidence"
                ].mean()
                if len(class_true[class_true["correct"]]) > 0
                else 0.0,
                "average_confidence_incorrect": class_true[~class_true["correct"]][
                    "confidence"
                ].mean()
                if len(class_true[~class_true["correct"]]) > 0
                else 0.0,
            }

        return analysis


# Convenience functions
def evaluate_model(
    model_path: str, dataset_split: str = "test", save_results: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a trained model on a dataset.

    Args:
        model_path: Path to the saved model
        dataset_split: Dataset split to evaluate on
        save_results: Whether to save results to files

    Returns:
        Dictionary of evaluation metrics
    """
    # Create predictor
    predictor = FaceMaskPredictor(model_path)

    # Make predictions
    prediction_df = predictor.predict_dataset(dataset_split=dataset_split)

    # Calculate metrics
    metrics = predictor.calculate_metrics()

    # Print summary
    predictor.print_metrics_summary(metrics)

    # Save results if requested
    if save_results:
        predictor.save_predictions()
        predictor.save_metrics(metrics)

    return metrics


def compare_models(
    model_paths: List[str], model_names: Optional[List[str]] = None, dataset_split: str = "test"
) -> pd.DataFrame:
    """
    Compare multiple trained models.

    Args:
        model_paths: List of paths to saved models
        model_names: List of model names (optional)
        dataset_split: Dataset split to evaluate on

    Returns:
        DataFrame comparing model performance
    """
    if model_names is None:
        model_names = [f"Model_{i + 1}" for i in range(len(model_paths))]

    comparison_results = []

    for model_path, model_name in zip(model_paths, model_names):
        print(f"\nEvaluating {model_name}...")

        try:
            predictor = FaceMaskPredictor(model_path)
            predictor.predict_dataset(dataset_split=dataset_split)
            metrics = predictor.calculate_metrics()

            result = {
                "model_name": model_name,
                "model_path": model_path,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "weighted_f1": metrics["weighted_f1"],
            }

            # Add per-class F1 scores
            for class_name, class_metrics in metrics["per_class_metrics"].items():
                result[f"f1_{class_name}"] = class_metrics["f1_score"]

            comparison_results.append(result)

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

    return pd.DataFrame(comparison_results)


if __name__ == "__main__":
    # Example usage
    print("Face Mask Detection Model Evaluation")
    print("=" * 50)

    # Note: This requires a trained model file
    model_path = MODELS_ROOT / "best_model.pth"

    if model_path.exists():
        # Evaluate the model
        metrics = evaluate_model(str(model_path))
        print("\nEvaluation completed!")
    else:
        print(f"No trained model found at {model_path}")
        print("Please train a model first using train.py")
