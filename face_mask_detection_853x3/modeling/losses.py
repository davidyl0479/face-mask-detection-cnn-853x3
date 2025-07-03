"""
Face Mask Detection Custom Loss Functions

This module contains custom loss function implementations including Weighted Focal Loss
for handling potential class imbalance in face mask detection data, with configurable
alpha/gamma parameters, class weight integration, and enhanced training stability.
"""

from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..config import DATASET_CONFIG, LOSS_CONFIG


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for addressing class imbalance.

    Focal Loss is designed to address class imbalance by down-weighting
    easy examples and focusing learning on hard negatives.

    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV, 2017.
    """

    def __init__(
        self,
        alpha: Union[float, List[float], Tensor] = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for class balance. Can be:
                   - float: Same weight for all classes
                   - list: Different weight for each class
                   - tensor: Pre-computed class weights
            gamma: Focusing parameter. Higher gamma puts more focus on hard examples
            reduction: Reduction method ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss calculation
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        # Convert alpha to tensor if needed
        if isinstance(alpha, (list, tuple)):
            self.alpha: Tensor = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, (int, float)):
            self.alpha: Tensor = torch.tensor([alpha], dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha: Tensor = alpha.float()
        else:
            raise ValueError(f"Unsupported alpha type: {type(alpha)}")

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Focal Loss.

        Args:
            inputs: Model predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)

        Returns:
            Computed focal loss
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", ignore_index=self.ignore_index
        )

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Handle alpha weighting
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        if len(self.alpha) == 1:
            # Single alpha value for all classes
            alpha_t = self.alpha[0]
        else:
            # Different alpha for each class
            alpha_t = self.alpha.gather(0, targets)

        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Enhanced Focal Loss with automatic class weight calculation.

    Combines Focal Loss with automatic class weight computation based on
    class frequencies in the dataset.
    """

    def __init__(
        self,
        alpha: Optional[Union[str, List[float], Tensor]] = "auto",
        gamma: float = 2.0,
        reduction: str = "mean",
        class_weights: Optional[Tensor] = None,
        smooth_factor: float = 0.1,
    ):
        """
        Initialize Weighted Focal Loss.

        Args:
            alpha: Class weighting strategy:
                   - 'auto': Automatically compute from class frequencies
                   - 'balanced': Use sklearn-style balanced weights
                   - list/tensor: Manual class weights
            gamma: Focusing parameter
            reduction: Reduction method
            class_weights: Pre-computed class weights
            smooth_factor: Smoothing factor for automatic weight computation
        """
        super(WeightedFocalLoss, self).__init__()
        self.alpha_strategy = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        self.smooth_factor = smooth_factor

        # Initialize focal loss with placeholder alpha
        self.focal_loss = FocalLoss(alpha=1.0, gamma=gamma, reduction=reduction)

        # Track class frequencies for auto weight computation
        self.class_counts = None
        self.total_samples = 0

    def update_class_weights(self, targets: Tensor):
        """
        Update class weights based on observed class frequencies.

        Args:
            targets: Ground truth labels
        """
        if self.alpha_strategy == "auto":
            # Count class frequencies
            unique, counts = torch.unique(targets, return_counts=True)

            if self.class_counts is None:
                self.class_counts = torch.zeros(
                    DATASET_CONFIG["num_classes"], device=targets.device
                )

            # Update counts
            for class_idx, count in zip(unique, counts):
                if 0 <= class_idx < len(self.class_counts):
                    self.class_counts[class_idx] += count

            self.total_samples += len(targets)

            # Compute balanced weights with smoothing
            if self.total_samples > 0:
                class_frequencies = self.class_counts / self.total_samples
                # Add smoothing to prevent division by zero
                class_frequencies = class_frequencies + self.smooth_factor

                # Compute inverse frequency weights
                weights = 1.0 / class_frequencies
                # Normalize weights
                weights = weights / weights.sum() * len(weights)

                # Update focal loss alpha
                self.focal_loss.alpha = weights

        elif self.alpha_strategy == "balanced":
            # Use sklearn-style balanced weights
            unique, counts = torch.unique(targets, return_counts=True)
            n_classes = DATASET_CONFIG["num_classes"]
            n_samples = len(targets)

            # Compute balanced weights: n_samples / (n_classes * bincount(y))
            weights = torch.zeros(n_classes, device=targets.device)
            for class_idx, count in zip(unique, counts):
                if 0 <= class_idx < n_classes:
                    weights[class_idx] = n_samples / (n_classes * count)

            self.focal_loss.alpha = weights

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Weighted Focal Loss.

        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Computed weighted focal loss
        """
        # Update class weights if using automatic weighting
        if self.alpha_strategy in ["auto", "balanced"]:
            self.update_class_weights(targets)

        # Apply focal loss
        return self.focal_loss(inputs, targets)


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.

    Label smoothing helps prevent overfitting and overconfident predictions
    by smoothing the target distribution.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean", ignore_index: int = -100):
        """
        Initialize Label Smoothing Cross Entropy Loss.

        Args:
            smoothing: Label smoothing factor (0.0 to 1.0)
            reduction: Reduction method
            ignore_index: Index to ignore in loss calculation
        """
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Label Smoothing Cross Entropy Loss.

        Args:
            inputs: Model predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)

        Returns:
            Computed label smoothing cross entropy loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        num_classes = inputs.size(1)

        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

            # Handle ignore_index
            if self.ignore_index >= 0:
                mask = targets != self.ignore_index
                true_dist = true_dist * mask.unsqueeze(1).float()

        # Compute loss
        loss = -true_dist * log_probs
        loss = loss.sum(dim=1)

        # Handle ignore_index in reduction
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            if self.reduction == "mean":
                return loss[mask].mean() if mask.any() else torch.tensor(0.0, device=inputs.device)
            elif self.reduction == "sum":
                return loss[mask].sum()

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions with configurable weights.

    Allows combining different loss functions (e.g., Cross Entropy + Focal Loss)
    for enhanced training performance.
    """

    def __init__(self, losses: Dict[str, nn.Module], weights: Optional[Dict[str, float]] = None):
        """
        Initialize Combined Loss.

        Args:
            losses: Dictionary of loss functions
            weights: Dictionary of loss weights (default: equal weights)
        """
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleDict(losses)

        # Set default weights if not provided
        if weights is None:
            weights = {name: 1.0 for name in losses.keys()}
        self.weights = weights

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Combined Loss.

        Args:
            inputs: Model predictions
            targets: Ground truth labels

        Returns:
            Weighted combination of losses
        """
        total_loss = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)

        for name, loss_fn in self.losses.items():
            weight = self.weights.get(name, 1.0)
            loss_value = loss_fn(inputs, targets)
            total_loss = total_loss + weight * loss_value

        return total_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.

    Addresses class imbalance by re-weighting loss based on effective
    number of samples per class.

    Reference: Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019).
    Class-balanced loss based on effective number of samples. CVPR, 2019.
    """

    def __init__(
        self,
        samples_per_class: Union[List[int], Tensor],
        beta: float = 0.9999,
        loss_type: str = "focal",
        gamma: float = 2.0,
    ):
        """
        Initialize Class-Balanced Loss.

        Args:
            samples_per_class: Number of samples per class
            beta: Hyperparameter for re-weighting
            loss_type: Base loss type ('focal', 'cross_entropy')
            gamma: Gamma parameter for focal loss
        """
        super(ClassBalancedLoss, self).__init__()
        self.samples_per_class = torch.tensor(samples_per_class, dtype=torch.float32)
        self.beta = beta
        self.loss_type = loss_type
        self.gamma = gamma

        # Compute effective number of samples
        effective_num = 1.0 - torch.pow(beta, self.samples_per_class)
        weights = (1.0 - beta) / effective_num
        self.weights = weights / weights.sum() * len(weights)

        # Initialize base loss
        if loss_type == "focal":
            self.base_loss = FocalLoss(alpha=self.weights, gamma=gamma)
        else:
            self.base_loss = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Class-Balanced Loss.

        Args:
            inputs: Model predictions
            targets: Ground truth labels

        Returns:
            Class-balanced loss
        """
        # Move weights to correct device
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)
            if hasattr(self.base_loss, "weight"):
                self.base_loss.weight = self.weights
            elif hasattr(self.base_loss, "alpha"):
                self.base_loss.alpha = self.weights

        return self.base_loss(inputs, targets)


# Loss factory function
def get_loss_function(
    loss_name: str, class_weights: Optional[Tensor] = None, **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_name: Name of the loss function
        class_weights: Optional class weights
        **kwargs: Additional parameters for loss function

    Returns:
        Initialized loss function
    """
    loss_config = LOSS_CONFIG.copy()
    loss_config.update(kwargs)

    if loss_name.lower() == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=loss_config.get("label_smoothing", 0.0)
        )

    elif loss_name.lower() == "focal":
        return FocalLoss(
            alpha=loss_config.get("focal_loss", {}).get("alpha", 1.0),
            gamma=loss_config.get("focal_loss", {}).get("gamma", 2.0),
            reduction=loss_config.get("focal_loss", {}).get("reduction", "mean"),
        )

    elif loss_name.lower() == "weighted_focal":
        return WeightedFocalLoss(
            alpha="auto",
            gamma=loss_config.get("focal_loss", {}).get("gamma", 2.0),
            reduction=loss_config.get("focal_loss", {}).get("reduction", "mean"),
        )

    elif loss_name.lower() == "label_smoothing":
        return LabelSmoothingCrossEntropyLoss(smoothing=loss_config.get("label_smoothing", 0.1))

    elif loss_name.lower() == "class_balanced":
        # Need to provide samples_per_class
        samples_per_class = kwargs.get("samples_per_class", [1, 1, 1])  # Default equal
        return ClassBalancedLoss(
            samples_per_class=samples_per_class,
            beta=kwargs.get("beta", 0.9999),
            loss_type=kwargs.get("base_loss_type", "focal"),
        )

    elif loss_name.lower() == "combined":
        # Create combination of losses
        losses = {}

        # Add cross entropy
        losses["cross_entropy"] = nn.CrossEntropyLoss(weight=class_weights)

        # Add focal loss
        losses["focal"] = FocalLoss(
            alpha=loss_config.get("focal_loss", {}).get("alpha", 1.0),
            gamma=loss_config.get("focal_loss", {}).get("gamma", 2.0),
        )

        # Default weights
        weights = kwargs.get("loss_weights", {"cross_entropy": 0.5, "focal": 0.5})

        return CombinedLoss(losses, weights)

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# Utility functions
def compute_class_weights(
    targets: Union[List[int], np.ndarray, Tensor], method: str = "balanced"
) -> Tensor:
    """
    Compute class weights for handling class imbalance.

    Args:
        targets: Target labels
        method: Weighting method ('balanced', 'inverse_freq')

    Returns:
        Computed class weights
    """
    if isinstance(targets, (list, np.ndarray)):
        targets = torch.tensor(targets)

    unique, counts = torch.unique(targets, return_counts=True)
    num_classes = len(unique)
    total_samples = len(targets)

    if method == "balanced":
        # sklearn-style balanced weights
        weights = torch.zeros(num_classes)
        for i, (class_idx, count) in enumerate(zip(unique, counts)):
            weights[class_idx] = total_samples / (num_classes * count)

    elif method == "inverse_freq":
        # Inverse frequency weights
        weights = torch.zeros(num_classes)
        for i, (class_idx, count) in enumerate(zip(unique, counts)):
            weights[class_idx] = 1.0 / count
        # Normalize
        weights = weights / weights.sum() * num_classes

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return weights


def test_loss_functions():
    """Test all implemented loss functions with dummy data."""
    print("Testing loss functions...")

    # Create dummy data
    batch_size = 16
    num_classes = 3
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Test Cross Entropy
    ce_loss = nn.CrossEntropyLoss()
    ce_value = ce_loss(inputs, targets)
    print(f"Cross Entropy Loss: {ce_value:.4f}")

    # Test Focal Loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    focal_value = focal_loss(inputs, targets)
    print(f"Focal Loss: {focal_value:.4f}")

    # Test Weighted Focal Loss
    weighted_focal = WeightedFocalLoss(alpha="auto", gamma=2.0)
    wf_value = weighted_focal(inputs, targets)
    print(f"Weighted Focal Loss: {wf_value:.4f}")

    # Test Label Smoothing
    label_smooth = LabelSmoothingCrossEntropyLoss(smoothing=0.1)
    ls_value = label_smooth(inputs, targets)
    print(f"Label Smoothing Loss: {ls_value:.4f}")

    # Test factory function
    factory_loss = get_loss_function("focal")
    factory_value = factory_loss(inputs, targets)
    print(f"Factory Focal Loss: {factory_value:.4f}")

    print("All loss functions tested successfully!")


def get_advanced_loss_function(model_name: str = "advanced_transfer_learning") -> nn.Module:
    """
    Get the appropriate loss function for advanced models based on config.

    Args:
        model_name: Name of the model to get loss function for

    Returns:
        Configured loss function
    """
    # Get loss type from model mapping
    loss_type = LOSS_CONFIG.get("model_loss_mapping", {}).get(model_name, "cross_entropy")

    if loss_type == "weighted_focal_loss":
        focal_config = LOSS_CONFIG["weighted_focal_loss"]
        return WeightedFocalLoss(
            alpha=focal_config["alpha"],
            gamma=focal_config["gamma"],
            reduction=focal_config["reduction"],
        )
    else:
        return get_loss_function(loss_type)


def get_class_weights_from_config() -> torch.Tensor:
    """Get class weights from advanced configuration."""
    weights = LOSS_CONFIG["weighted_focal_loss"]["alpha"]
    return torch.tensor(weights, dtype=torch.float32)


# Update the existing get_loss_function to handle model-specific routing
def get_loss_function_for_model(model_name: str, **kwargs) -> nn.Module:
    """Get loss function specifically configured for a model type."""
    return get_advanced_loss_function(model_name)


if __name__ == "__main__":
    # Test the loss functions
    test_loss_functions()

    # Demonstrate class weight computation
    print("\nTesting class weight computation...")
    dummy_targets = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]  # Imbalanced classes
    weights = compute_class_weights(dummy_targets, method="balanced")
    print(f"Balanced weights: {weights}")

    weights_inv = compute_class_weights(dummy_targets, method="inverse_freq")
    print(f"Inverse frequency weights: {weights_inv}")

    print("Loss module testing completed!")
