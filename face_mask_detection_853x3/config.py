"""
Face Mask Detection CNN Project Configuration

This module contains all project configuration settings including file paths,
Kaggle dataset specifications, model hyperparameters, training settings,
hardware configurations, and reproducibility settings.
"""

import os
from pathlib import Path
import random

import numpy as np
import torch

# Set random seeds for reproducibility
RANDOM_SEED = 42


def set_random_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducible results across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Initialize seeds
set_random_seeds()

# Project paths
PROJECT_ROOT = Path(
    __file__
).parent.parent  # Go up one level from face_mask_detection_853x3/ to project root
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
FIGURES_ROOT = PROJECT_ROOT / "reports" / "figures"  # Use existing reports/figures structure
LOGS_ROOT = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for path in [DATA_ROOT, MODELS_ROOT, FIGURES_ROOT, LOGS_ROOT]:
    path.mkdir(exist_ok=True)

# Data paths
DATA_PATHS = {
    "raw": DATA_ROOT / "raw",
    "processed": DATA_ROOT / "processed",
    "interim": DATA_ROOT / "interim",
    "external": DATA_ROOT / "external",
}

# Create data subdirectories
for path in DATA_PATHS.values():
    path.mkdir(exist_ok=True)

# Kaggle dataset configuration
KAGGLE_CONFIG = {
    "dataset_name": "andrewmvd/face-mask-detection",
    "download_path": DATA_PATHS["raw"],
    "extract_path": DATA_PATHS["raw"],  # Data is directly in raw folder
}

# Dataset configuration
DATASET_CONFIG = {
    "classes": ["with_mask", "without_mask", "mask_weared_incorrect"],
    "num_classes": 3,
    "image_extensions": [".jpg", ".jpeg", ".png"],
    "annotation_extension": ".xml",
    "total_images": 853,
}

# Image preprocessing configuration
IMAGE_CONFIG = {
    "input_size": (224, 224),
    "channels": 3,
    "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
    "std": [0.229, 0.224, 0.225],
    "resize_method": "bilinear",
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    "rotation_degrees": 15,
    "horizontal_flip_prob": 0.5,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.1,
    "gaussian_blur_prob": 0.1,
    "gaussian_blur_kernel": (5, 5),
    "random_erasing_prob": 0.1,
    "random_erasing_scale": (0.02, 0.33),
    "random_erasing_ratio": (0.3, 3.3),
    # MINORITY CLASS SPECIFIC AUGMENTATION
    "minority_class_augmentation": {
        "enable": True,
        "augmentation_factor": 5,  # Apply 5x more augmentation to minority classes
        "without_mask": {
            "rotation_degrees": 30,
            "perspective_distortion": 0.3,
            "color_jitter": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.2},
            "random_erasing_prob": 0.3,
            "gaussian_blur_prob": 0.2,
            "elastic_transform_prob": 0.3,
            "cutout_prob": 0.2,
            "cutout_size": 0.1,
        },
        "mask_weared_incorrect": {
            "rotation_degrees": 35,
            "perspective_distortion": 0.4,
            "color_jitter": {"brightness": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.3},
            "random_erasing_prob": 0.4,
            "gaussian_blur_prob": 0.3,
            "elastic_transform_prob": 0.4,
            "cutout_prob": 0.3,
            "cutout_size": 0.15,
        },
    },
}

# Data splitting configuration
SPLIT_CONFIG = {
    "train_ratio": 0.7,
    "val_ratio": 0.2,
    "test_ratio": 0.1,
    "stratify": True,
    "shuffle": True,
}

# Model configurations
MODEL_CONFIGS = {
    "basic_cnn": {
        "name": "BasicCNN",
        "input_channels": 3,
        "num_classes": 3,
        "dropout_rate": 0.5,
        "use_batch_norm": True,
        "activation": "relu",
    },
    "transfer_learning": {
        "name": "TransferLearning",
        "backbone": "resnet18",  # or "mobilenet_v2"
        "pretrained": True,
        "freeze_backbone": False,
        "num_classes": 3,
        "dropout_rate": 0.5,
        "fine_tune_layers": 2,  # Number of layers to fine-tune from the end
    },
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "momentum": 0.9,  # For SGD
    "scheduler": "step",
    "step_size": 10,
    "gamma": 0.1,
    "early_stopping": {"patience": 10, "min_delta": 0.001, "monitor": "val_loss", "mode": "min"},
    "class_balanced_sampling": True,
}

# Loss function configuration
LOSS_CONFIG = {
    "primary_loss": "cross_entropy",
    "focal_loss": {"alpha": 1.0, "gamma": 2.0, "reduction": "mean"},
    "class_weights": [1.0, 853 / (3 * 119), 853 / (3 * 36)],  # Use your calculated weights
    "label_smoothing": 0.1,
    "model_loss_mapping": {
        "basic_cnn": "cross_entropy",
        "transfer_learning": "cross_entropy",
    },
}

# Hardware configuration
DEVICE_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "pin_memory": True if torch.cuda.is_available() else False,
    "persistent_workers": True,
}

# Logging configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_file": LOGS_ROOT / "training.log",
    "log_format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    "tensorboard_log_dir": LOGS_ROOT / "tensorboard",
    "wandb_project": "face-mask-detection",
    "wandb_entity": None,
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1_score"],
    "average": "macro",  # For multi-class metrics
    "save_predictions": True,
    "save_confusion_matrix": True,
    "class_names": DATASET_CONFIG["classes"],
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "viridis",
    "save_format": "png",
    "show_plots": True,
    "save_plots": True,
}

# Feature analysis configuration
FEATURE_ANALYSIS_CONFIG = {
    "extract_layers": ["conv1", "conv2", "conv3", "fc1"],
    "dimensionality_reduction": {
        "method": "tsne",  # or "pca"
        "n_components": 2,
        "perplexity": 30,
        "n_iter": 1000,
    },
    "clustering": {"method": "kmeans", "n_clusters": 3, "random_state": RANDOM_SEED},
}

# Model saving configuration
MODEL_SAVE_CONFIG = {
    "save_best_only": True,
    "save_last": True,
    "save_checkpoints": True,
    "checkpoint_frequency": 5,  # Save every N epochs
    "model_filename": "best_model.pth",
    "checkpoint_pattern": "checkpoint_epoch_{epoch:03d}.pth",
}

# Prediction configuration
PREDICTION_CONFIG = {
    "confidence_threshold": 0.5,
    "save_predictions_csv": True,
    "predictions_filename": "predictions.csv",
    "batch_size": 16,
    "tta": False,  # Test Time Augmentation
    "tta_transforms": 5,
}


# Export convenience function
def get_config(config_name):
    """Get configuration by name."""
    config_map = {
        "kaggle": KAGGLE_CONFIG,
        "dataset": DATASET_CONFIG,
        "image": IMAGE_CONFIG,
        "augmentation": AUGMENTATION_CONFIG,
        "split": SPLIT_CONFIG,
        "model": MODEL_CONFIGS,
        "training": TRAINING_CONFIG,
        "loss": LOSS_CONFIG,
        "device": DEVICE_CONFIG,
        "logging": LOGGING_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "feature_analysis": FEATURE_ANALYSIS_CONFIG,
        "model_save": MODEL_SAVE_CONFIG,
        "prediction": PREDICTION_CONFIG,
    }
    return config_map.get(config_name, None)


# Print configuration summary
def print_config_summary():
    """Print a summary of key configuration settings."""
    print("Face Mask Detection CNN - Configuration Summary")
    print("=" * 50)
    print(f"Device: {DEVICE_CONFIG['device']}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Dataset: {KAGGLE_CONFIG['dataset_name']}")
    print(f"Classes: {DATASET_CONFIG['classes']}")
    print(f"Image Size: {IMAGE_CONFIG['input_size']}")
    print(f"Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Epochs: {TRAINING_CONFIG['num_epochs']}")
    print("=" * 50)


# HELPER FUNCTIONS
def get_model_specific_config(model_name: str, config_type: str):
    """Get model-specific configuration."""
    configs = {
        "loss": LOSS_CONFIG,
        "training": TRAINING_CONFIG,
        "augmentation": AUGMENTATION_CONFIG,
    }

    if config_type not in configs:
        return None

    config = configs[config_type].copy()

    # Apply model-specific settings (none currently)

    return config




def get_loss_for_model(model_name: str):
    """Get appropriate loss function for model."""
    mapping = LOSS_CONFIG.get("model_loss_mapping", {})
    return mapping.get(model_name, "cross_entropy")


# Print updated configuration summary


if __name__ == "__main__":
    print_config_summary()
