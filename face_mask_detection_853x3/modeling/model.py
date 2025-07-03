"""
Face Mask Detection CNN Models

This module defines CNN architectures specifically optimized for face mask detection
from RGB images. Includes Basic CNN and Transfer Learning models with factory pattern
for model creation, proper weight initialization, and detailed architecture documentation.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..config import DATASET_CONFIG, IMAGE_CONFIG, MODEL_CONFIGS


class BasicCNN(nn.Module):
    """
    Basic CNN architecture for face mask detection.

    Designed specifically for facial feature recognition and mask pattern detection.
    Uses progressive feature extraction with batch normalization and dropout.
    """

    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 3,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
        activation: str = "relu",
    ):
        """
        Initialize Basic CNN.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB)
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'elu')
        """
        super(BasicCNN, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Set activation function
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: Initial feature extraction
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(32) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(32) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.5),  # Lighter dropout for early layers
            # Block 2: Enhanced feature detection
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.6),
            # Block 3: Complex pattern recognition
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.7),
            # Block 4: High-level feature extraction
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(256) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(256) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.AdaptiveAvgPool2d((7, 7)),  # Adaptive pooling for flexible input sizes
        )

        # ================================================================================================
        # FEATURE EXTRACTION BLOCKS
        # ================================================================================================

        # ↓ CONV BLOCK 1 - Initial Feature Extraction (edges, lines)
        # ↓ conv1 (3→32, kernel=3, pad=1) + bn + relu
        # (batch, 32, 224, 224)                   # 32 edge detectors, same spatial size
        # ↓ conv2 (32→32, kernel=3, pad=1) + bn + relu
        # (batch, 32, 224, 224)                   # Refined edge features
        # ↓ maxpool(2x2) + dropout2d(0.25)
        # (batch, 32, 112, 112)                   # Downsampled by 2x, spatial reduced

        # ↓ CONV BLOCK 2 - Enhanced Feature Detection (textures, corners)
        # ↓ conv1 (32→64, kernel=3, pad=1) + bn + relu
        # (batch, 64, 112, 112)                   # 64 texture detectors
        # ↓ conv2 (64→64, kernel=3, pad=1) + bn + relu
        # (batch, 64, 112, 112)                   # Enhanced texture features
        # ↓ maxpool(2x2) + dropout2d(0.3)
        # (batch, 64, 56, 56)                     # Downsampled again, more abstract

        # ↓ CONV BLOCK 3 - Complex Pattern Recognition (face parts, mask shapes)
        # ↓ conv1 (64→128, kernel=3, pad=1) + bn + relu
        # (batch, 128, 56, 56)                    # 128 object part detectors
        # ↓ conv2 (128→128, kernel=3, pad=1) + bn + relu
        # (batch, 128, 56, 56)                    # Refined object parts
        # ↓ maxpool(2x2) + dropout2d(0.35)
        # (batch, 128, 28, 28)                    # Higher-level features

        # ↓ CONV BLOCK 4 - High-Level Feature Extraction (complete concepts)
        # ↓ conv1 (128→256, kernel=3, pad=1) + bn + relu
        # (batch, 256, 28, 28)                    # 256 high-level concept detectors
        # ↓ conv2 (256→256, kernel=3, pad=1) + bn + relu
        # (batch, 256, 28, 28)                    # Complex pattern recognition
        # ↓ adaptive_avg_pool2d(7x7)
        # (batch, 256, 7, 7)                      # Fixed size for classification head

        # Calculate the size of flattened features
        # After adaptive pooling: 256 * 7 * 7 = 12544
        self.feature_size = 256 * 7 * 7

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.Dropout(p=dropout_rate * 0.5),  # Lighter dropout before final layer
            nn.Linear(128, num_classes),
        )

        # ================================================================================================
        # CLASSIFICATION HEAD
        # ================================================================================================

        # ↓ FLATTEN
        # (batch, 12544)                          # 256 * 7 * 7 = 12,544 features

        # ↓ FULLY CONNECTED LAYER 1 - Main Processing
        # ↓ dropout(0.5)
        # ↓ fc1 (12544→512) + bn + relu
        # (batch, 512)                            # Compressed high-level features

        # ↓ FULLY CONNECTED LAYER 2 - Feature Refinement
        # ↓ dropout(0.5)
        # ↓ fc2 (512→128) + bn + relu
        # (batch, 128)                            # Further refined features

        # ↓ FULLY CONNECTED LAYER 3 - Final Classification
        # ↓ dropout(0.25)                         # Lighter dropout before output
        # ↓ fc3 (128→3)
        # (batch, 3)                              # Final class logits
        #                                         # [with_mask, without_mask, mask_weared_incorrect]

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using appropriate initialization strategies."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = self.features(x)

        # Flatten for classification
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x

    def predict_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make prediction on a single sample by temporarily setting model to eval mode.

        Args:
            x: Input tensor (can be batch size 1)

        Returns:
            Output predictions
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            output = self.forward(x)

        if was_training:
            self.train()

        return output

    def get_feature_maps(self, x: torch.Tensor, layer_name: Optional[str] = None) -> torch.Tensor:
        """
        Extract feature maps from intermediate layers.

        Args:
            x: Input tensor
            layer_name: Name of layer to extract features from

        Returns:
            Feature maps from specified layer
        """
        features = self.features(x)
        return features


class TransferLearningCNN(nn.Module):
    """
    Transfer Learning CNN using pre-trained backbones for face mask detection.

    Leverages pre-trained models (ResNet, MobileNet) and fine-tunes them
    for face mask classification with custom classifier heads.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        fine_tune_layers: int = 2,
        activation: str = "relu",
    ):
        """
        Initialize Transfer Learning CNN.

        Args:
            backbone: Backbone architecture ('resnet18', 'resnet34', 'mobilenet_v2')
            num_classes: Number of output classes
            pretrained: Whether to use pre-trained weights
            freeze_backbone: Whether to freeze backbone parameters
            dropout_rate: Dropout probability
            fine_tune_layers: Number of layers to fine-tune from the end
            activation: Activation function ('relu', 'silu', 'gelu', 'leaky_relu')
        """
        super(TransferLearningCNN, self).__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone_layers = freeze_backbone
        self.dropout_rate = dropout_rate
        self.fine_tune_layers = fine_tune_layers
        self.activation = activation

        # Load backbone
        self.backbone, self.feature_size = self._create_backbone()

        # Create custom classifier
        self.classifier = self._create_classifier()

        # Apply fine-tuning strategy
        self._apply_fine_tuning()

    def _get_activation_function(self) -> nn.Module:
        """Get the specified activation function."""
        if self.activation == "relu":
            return nn.ReLU(inplace=True)
        elif self.activation == "silu":
            return nn.SiLU(inplace=True)
        elif self.activation == "gelu":
            return nn.GELU()
        elif self.activation == "leaky_relu":
            return nn.LeakyReLU(0.01, inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def _create_backbone(self) -> tuple:
        """Create and configure the backbone network."""
        if self.backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=self.pretrained)
            feature_size = backbone.fc.in_features
            # Remove the final classification layer
            backbone = nn.Sequential(*list(backbone.children())[:-1])

        elif self.backbone_name == "resnet34":
            backbone = models.resnet34(pretrained=self.pretrained)
            feature_size = backbone.fc.in_features
            backbone = nn.Sequential(*list(backbone.children())[:-1])

        elif self.backbone_name == "mobilenet_v2":
            backbone = models.mobilenet_v2(pretrained=self.pretrained)
            feature_size = backbone.classifier[1].in_features
            # Remove the final classification layer
            backbone = backbone.features

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        return backbone, feature_size

    def _create_classifier(self) -> nn.Module:
        """Create custom classifier head."""
        if self.backbone_name == "mobilenet_v2":
            # MobileNet needs adaptive pooling
            classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(self.feature_size, 256),
                self._get_activation_function(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(256, 128),
                self._get_activation_function(),
                nn.Dropout(p=self.dropout_rate * 0.5),
                nn.Linear(128, self.num_classes),
            )
        else:
            # ResNet already has global average pooling
            classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(self.feature_size, 256),
                self._get_activation_function(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(256, 128),
                self._get_activation_function(),
                nn.Dropout(p=self.dropout_rate * 0.5),
                nn.Linear(128, self.num_classes),
            )

        return classifier

    def _apply_fine_tuning(self):
        """Apply fine-tuning strategy to the backbone."""
        if self.freeze_backbone_layers:
            # Freeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Fine-tune only the last few layers
            if self.fine_tune_layers > 0:
                # Freeze early layers
                backbone_children = list(self.backbone.children())
                num_children = len(backbone_children)

                for i, child in enumerate(backbone_children):
                    if i < num_children - self.fine_tune_layers:
                        for param in child.parameters():
                            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Extract features using backbone
        features = self.backbone(x)

        # Classify using custom head
        output = self.classifier(features)

        return output

    def predict_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make prediction on a single sample by temporarily setting model to eval mode.

        Args:
            x: Input tensor (can be batch size 1)

        Returns:
            Output predictions
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            output = self.forward(x)

        if was_training:
            self.train()

        return output

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps from the backbone."""
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False




class ModelFactory:
    """Factory class for creating face mask detection models."""

    @staticmethod
    def create_model(model_name: str, **kwargs) -> nn.Module:
        """
        Create a model by name.

        Args:
            model_name: Name of the model to create
            **kwargs: Additional model parameters

        Returns:
            Instantiated model
        """
        model_name = model_name.lower()

        if model_name == "basic_cnn":
            config = MODEL_CONFIGS["basic_cnn"].copy()
            config.update(kwargs)
            # Remove 'name' key as it's not a parameter for BasicCNN
            config.pop("name", None)
            return BasicCNN(**config)

        elif model_name == "transfer_learning":
            config = MODEL_CONFIGS["transfer_learning"].copy()
            config.update(kwargs)
            # Remove 'name' key as it's not a parameter for TransferLearningCNN
            config.pop("name", None)
            return TransferLearningCNN(**config)


        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model names."""
        return list(MODEL_CONFIGS.keys())

    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        return MODEL_CONFIGS[model_name].copy()


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Count the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Get a summary of the model including parameter counts and layer information.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model summary information
    """
    trainable_params = count_parameters(model)
    total_params = count_total_parameters(model)

    summary = {
        "model_name": model.__class__.__name__,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "frozen_parameters": total_params - trainable_params,
        "parameter_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
    }

    return summary


def print_model_summary(model: nn.Module):
    """Print a formatted model summary."""
    summary = get_model_summary(model)

    print(f"Model: {summary['model_name']}")
    print("-" * 50)
    print(f"Total parameters: {summary['total_parameters']:,}")
    print(f"Trainable parameters: {summary['trainable_parameters']:,}")
    print(f"Frozen parameters: {summary['frozen_parameters']:,}")
    print(f"Trainable ratio: {summary['trainable_ratio']:.2%}")
    print(f"Model size: {summary['parameter_size_mb']:.2f} MB")
    print("-" * 50)


# Convenience functions
def create_basic_cnn(**kwargs) -> BasicCNN:
    """Create a Basic CNN model with default configuration."""
    config = MODEL_CONFIGS["basic_cnn"].copy()
    config.update(kwargs)
    # Remove 'name' key as it's not a parameter for BasicCNN
    config.pop("name", None)
    return BasicCNN(**config)


def create_transfer_learning_model(**kwargs) -> TransferLearningCNN:
    """Create a Transfer Learning model with default configuration."""
    config = MODEL_CONFIGS["transfer_learning"].copy()
    config.update(kwargs)
    # Remove 'name' key as it's not a parameter for TransferLearningCNN
    config.pop("name", None)
    return TransferLearningCNN(**config)




def create_model_from_config(config_name: str, **kwargs) -> nn.Module:
    """Create a model from configuration name."""
    return ModelFactory.create_model(config_name, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    print("Face Mask Detection CNN Models")
    print("=" * 50)

    # Test Basic CNN
    print("\n1. Basic CNN Model:")
    basic_model = create_basic_cnn()
    print_model_summary(basic_model)

    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = basic_model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test Transfer Learning Model
    print("\n2. Transfer Learning Model (ResNet18):")
    transfer_model = create_transfer_learning_model(backbone="resnet18")
    print_model_summary(transfer_model)

    output = transfer_model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test MobileNet
    print("\n3. Transfer Learning Model (MobileNetV2):")
    mobile_model = create_transfer_learning_model(backbone="mobilenet_v2")
    print_model_summary(mobile_model)

    output = mobile_model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test Factory
    print("\n4. Available Models:")
    print(ModelFactory.get_available_models())
