"""
Face Mask Detection Training Module

This module implements the complete face mask detection model training pipeline
with epoch-based training loops, validation monitoring, learning rate scheduling,
early stopping, checkpoint saving, and comprehensive logging.
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from ..config import (
    DEVICE_CONFIG,
    EVALUATION_CONFIG,
    LOGGING_CONFIG,
    LOGS_ROOT,
    LOSS_CONFIG,
    MODEL_SAVE_CONFIG,
    MODELS_ROOT,
    TRAINING_CONFIG,
    set_random_seeds,
)
from ..dataset import get_dataloaders
from .losses import get_loss_function
from .model import ModelFactory, get_model_summary


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop early.

        Args:
            val_loss: Current validation loss
            model: Model being trained

        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights:
                    if self.verbose:
                        print("Restoring best weights...")
                    model.load_state_dict(self.best_weights)

        return self.early_stop

    def save_checkpoint(self, model: nn.Module):
        """Save the best model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


class MetricsTracker:
    """Track and compute training metrics."""

    def __init__(self, num_classes: int = 3):
        """
        Initialize metrics tracker.

        Args:
            num_classes: Number of classes for classification
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        """
        Update metrics with batch results.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            loss: Batch loss
        """
        # Convert to numpy for metric calculation
        preds = predictions.detach().cpu().numpy()
        targs = targets.detach().cpu().numpy()

        self.predictions.extend(preds.argmax(axis=1))
        self.targets.extend(targs)
        self.losses.append(loss)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of computed metrics
        """
        if not self.predictions:
            return {}

        # Convert to numpy arrays
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        # Compute metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average="macro", zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )

        # Ensure per-class metrics are arrays
        precision_per_class = np.atleast_1d(precision_per_class)
        recall_per_class = np.atleast_1d(recall_per_class)
        f1_per_class = np.atleast_1d(f1_per_class)

        # Average loss
        avg_loss = np.mean(self.losses)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        # Add per-class metrics
        class_names = ["with_mask", "without_mask", "mask_weared_incorrect"]
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                metrics[f"precision_{class_name}"] = float(precision_per_class[i])
                metrics[f"recall_{class_name}"] = float(recall_per_class[i])
                metrics[f"f1_{class_name}"] = float(f1_per_class[i])

        return metrics


class FaceMaskTrainer:
    """
    Main trainer class for face mask detection models.

    Handles complete training pipeline including data loading, model training,
    validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model_name: str = "basic_cnn",
        model_params: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        use_tensorboard: bool = True,
        experiment_name: Optional[str] = None,
        model_prefix: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model_name: Name of the model to train
            model_params: Additional model parameters
            device: Device to use for training
            use_tensorboard: Whether to use TensorBoard logging
            experiment_name: Name for the experiment
            model_prefix: Prefix for saved model files (e.g., "basic_cnn", "tl_resnet18")
        """
        # Set random seeds for reproducibility
        set_random_seeds()

        # Set device
        self.device = device or DEVICE_CONFIG["device"]
        print(f"Using device: {self.device}")

        # Create model
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = self._create_model()

        # Setup experiment
        self.experiment_name = experiment_name or f"{model_name}_{int(time.time())}"
        self.experiment_dir = LOGS_ROOT / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        # Setup model prefix for file naming
        self.model_prefix = model_prefix or model_name

        # Setup logging
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.experiment_dir / "tensorboard")

        # Initialize training components
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[
            Union[
                optim.lr_scheduler.StepLR,
                optim.lr_scheduler.ReduceLROnPlateau,
                optim.lr_scheduler.CosineAnnealingWarmRestarts,
            ]
        ] = None
        self.criterion: Optional[nn.Module] = None
        self.early_stopping: Optional[EarlyStopping] = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_history = []
        self.val_history = []

        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()

        print(f"Trainer initialized for experiment: {self.experiment_name}")

    def _create_model(self) -> nn.Module:
        """Create and initialize the model."""
        model = ModelFactory.create_model(self.model_name, **self.model_params)
        model = model.to(self.device)

        # Print model summary
        print(f"\nModel: {self.model_name}")
        print("=" * 50)
        summary = get_model_summary(model)
        for key, value in summary.items():
            if key == "parameter_size_mb":
                print(f"{key}: {value:.2f} MB")
            elif isinstance(value, int):
                print(f"{key}: {value:,}")
            elif isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("=" * 50)

        return model

    def setup_training(self):
        """Setup optimizer, loss function, scheduler, and early stopping."""
        # Setup optimizer
        if TRAINING_CONFIG["optimizer"].lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=TRAINING_CONFIG["learning_rate"],
                weight_decay=TRAINING_CONFIG["weight_decay"],
            )
        elif TRAINING_CONFIG["optimizer"].lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=TRAINING_CONFIG["learning_rate"],
                weight_decay=TRAINING_CONFIG["weight_decay"],
                betas=TRAINING_CONFIG.get("betas", (0.9, 0.999)),
                eps=TRAINING_CONFIG.get("eps", 1e-8),
            )
        elif TRAINING_CONFIG["optimizer"].lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=TRAINING_CONFIG["learning_rate"],
                momentum=TRAINING_CONFIG["momentum"],
                weight_decay=TRAINING_CONFIG["weight_decay"],
                nesterov=TRAINING_CONFIG.get("nesterov", False),
            )
        elif TRAINING_CONFIG["optimizer"].lower() == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=TRAINING_CONFIG["learning_rate"],
                alpha=TRAINING_CONFIG.get("alpha", 0.99),
                weight_decay=TRAINING_CONFIG["weight_decay"],
                momentum=TRAINING_CONFIG.get("momentum", 0),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {TRAINING_CONFIG['optimizer']}")

        # Setup loss function
        self.criterion = get_loss_function(LOSS_CONFIG["primary_loss"])

        # Setup scheduler
        if TRAINING_CONFIG["scheduler"] == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=TRAINING_CONFIG["step_size"],
                gamma=TRAINING_CONFIG["gamma"],
            )
        elif TRAINING_CONFIG["scheduler"] == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=TRAINING_CONFIG["gamma"],
                patience=TRAINING_CONFIG["step_size"],
            )
        else:
            self.scheduler = None

        # Setup early stopping
        early_stop_config = TRAINING_CONFIG["early_stopping"]
        self.early_stopping = EarlyStopping(
            patience=early_stop_config["patience"],
            min_delta=early_stop_config["min_delta"],
            verbose=True,
        )

        print("Training setup completed!")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} - Training")

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            optimizer = self.optimizer
            if optimizer is None:
                raise RuntimeError("Optimizer not initialized. Call setup_training() first.")
            optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            criterion = self.criterion
            if criterion is None:
                raise RuntimeError("Criterion not initialized. Call setup_training() first.")
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            self.train_metrics.update(output, target, loss.item())

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        # Compute epoch metrics
        train_metrics = self.train_metrics.compute_metrics()

        return train_metrics

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} - Validation")

            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)
                criterion = self.criterion
                if criterion is None:
                    raise RuntimeError("Criterion not initialized. Call setup_training() first.")
                loss = criterion(output, target)

                # Update metrics
                self.val_metrics.update(output, target, loss.item())

        # Compute epoch metrics
        val_metrics = self.val_metrics.compute_metrics()

        return val_metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_name": self.model_name,
            "model_params": self.model_params,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }

        # Save regular checkpoint
        if (
            MODEL_SAVE_CONFIG["save_checkpoints"]
            and (self.current_epoch + 1) % MODEL_SAVE_CONFIG["checkpoint_frequency"] == 0
        ):
            checkpoint_filename = (
                f"{self.model_prefix}_checkpoint_epoch_{self.current_epoch + 1:03d}.pth"
            )
            checkpoint_path = self.experiment_dir / checkpoint_filename
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if is_best and MODEL_SAVE_CONFIG["save_best_only"]:
            best_filename = f"{self.model_prefix}_best_model.pth"
            best_path = self.experiment_dir / best_filename
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

        # Save last model
        if MODEL_SAVE_CONFIG["save_last"]:
            last_filename = f"{self.model_prefix}_last_model.pth"
            last_path = self.experiment_dir / last_filename
            torch.save(checkpoint, last_path)

    def log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """
        Log metrics to various outputs.

        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Log to TensorBoard
        if self.use_tensorboard:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"Train/{key}", value, self.current_epoch)

            for key, value in val_metrics.items():
                self.writer.add_scalar(f"Val/{key}", value, self.current_epoch)

            # Log learning rate
            if self.optimizer is not None:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("Learning_Rate", current_lr, self.current_epoch)

        # Print metrics
        print(f"\nEpoch {self.current_epoch + 1} Results:")
        print(
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}"
        )
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1_score']:.4f}")

        # Save metrics history
        self.train_history.append(train_metrics)
        self.val_history.append(val_metrics)

        # Save metrics to JSON
        metrics_file = self.experiment_dir / "metrics_history.json"
        with open(metrics_file, "w") as f:
            json.dump(
                {"train_history": self.train_history, "val_history": self.val_history}, f, indent=2
            )

    def train(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
    ):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
        """
        # Setup training if not already done
        if self.optimizer is None:
            self.setup_training()

        # Get data loaders if not provided
        if train_loader is None or val_loader is None:
            train_loader, val_loader, _ = get_dataloaders()

        # Set number of epochs
        num_epochs = num_epochs or TRAINING_CONFIG["num_epochs"]

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate_epoch(val_loader)

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Log metrics
            self.log_metrics(train_metrics, val_metrics)

            # Check for best model
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]

            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best)

            # Early stopping check
            if self.early_stopping is not None and self.early_stopping(
                val_metrics["loss"], self.model
            ):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Close TensorBoard writer
        if self.use_tensorboard:
            self.writer.close()

        return self.train_history, self.val_history

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.train_history = checkpoint.get("train_history", [])
        self.val_history = checkpoint.get("val_history", [])

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch + 1}")




# Convenience functions
def train_model(
    model_name: str = "basic_cnn",
    model_params: Optional[Dict[str, Any]] = None,
    experiment_name: Optional[str] = None,
    num_epochs: Optional[int] = None,
    model_prefix: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Train a face mask detection model.

    Args:
        model_name: Name of the model to train
        model_params: Additional model parameters
        experiment_name: Name for the experiment
        num_epochs: Number of epochs to train
        model_prefix: Prefix for saved model files

    Returns:
        Tuple of (train_history, val_history)
    """
    trainer = FaceMaskTrainer(
        model_name=model_name,
        model_params=model_params,
        experiment_name=experiment_name,
        model_prefix=model_prefix,
    )

    return trainer.train(num_epochs=num_epochs)


if __name__ == "__main__":
    # Example usage
    print("Face Mask Detection Training")
    print("=" * 50)

    # Train Basic CNN
    print("\nTraining Basic CNN...")
    train_history, val_history = train_model(
        model_name="basic_cnn",
        experiment_name="basic_cnn_experiment",
        num_epochs=5,  # Reduced for testing
    )

    print("Training completed!")
