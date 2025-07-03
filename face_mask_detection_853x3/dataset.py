"""
Face Mask Detection Dataset Module

This module implements a complete data pipeline for the Face Mask Detection dataset,
including automated Kaggle downloading, XML annotation parsing, train/validation/test
splitting, data augmentation, preprocessing, and efficient DataLoader creation.
"""

import json
import os
from pathlib import Path
import random
import shutil
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import zipfile

import kaggle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import (
    AUGMENTATION_CONFIG,
    DATASET_CONFIG,
    DEVICE_CONFIG,
    IMAGE_CONFIG,
    KAGGLE_CONFIG,
    SPLIT_CONFIG,
    TRAINING_CONFIG,
    set_random_seeds,
)


class FaceMaskDataset(Dataset):
    """
    Custom PyTorch Dataset for Face Mask Detection.

    Supports loading images with XML annotations, applying transforms,
    and returning preprocessed data for training/validation/testing.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_df: DataFrame with columns ['image_path', 'class_name', 'class_id']
            transform: Image transformations to apply
            target_transform: Target transformations to apply
        """
        self.data_df = data_df.reset_index(drop=True)
        self.transform = transform
        self.target_transform = target_transform
        self.classes = DATASET_CONFIG["classes"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, Image.Image], int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image_tensor_or_pil, class_id)
        """
        row = self.data_df.iloc[idx]
        image_path = row["image_path"]
        class_id = int(row["class_id"])  # Ensure class_id is int

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", IMAGE_CONFIG["input_size"], (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Apply target transforms
        if self.target_transform:
            class_id = self.target_transform(class_id)

        return image, class_id

    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        return self.data_df["class_name"].value_counts().to_dict()

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling class imbalance."""
        weights = compute_class_weight(
            "balanced",
            classes=np.arange(len(self.classes)),
            y=np.array(self.data_df["class_id"].values),  # Convert to numpy array
        )
        return torch.FloatTensor(weights)


class FaceMaskDataManager:
    """
    Manager class for handling all data operations including download,
    preprocessing, splitting, and DataLoader creation.
    """

    def __init__(self):
        """Initialize the data manager."""
        self.kaggle_config = KAGGLE_CONFIG
        self.dataset_config = DATASET_CONFIG
        self.image_config = IMAGE_CONFIG
        self.augmentation_config = AUGMENTATION_CONFIG
        self.split_config = SPLIT_CONFIG

        # Initialize transforms
        self.train_transform = self._create_train_transforms()
        self.val_transform = self._create_val_transforms()
        self.test_transform = self._create_test_transforms()

        # Data storage
        self.data_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def download_dataset(self, force_download: bool = False) -> None:
        """
        Download the Kaggle dataset.

        Args:
            force_download: If True, re-download even if dataset exists
        """
        dataset_path = self.kaggle_config["extract_path"]

        if dataset_path.exists() and not force_download:
            print(f"Dataset already exists at {dataset_path}")
            return

        print(f"Downloading dataset: {self.kaggle_config['dataset_name']}")

        try:
            # Download dataset using Kaggle API
            kaggle.api.dataset_download_files(
                self.kaggle_config["dataset_name"],
                path=self.kaggle_config["download_path"],
                unzip=True,
            )
            print("Dataset downloaded successfully!")

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure you have:")
            print("1. Kaggle API installed: pip install kaggle")
            print("2. Kaggle API token configured (~/.kaggle/kaggle.json)")
            print("3. Internet connection")
            raise

    def parse_annotations(self, images_dir: Path, annotations_dir: Path) -> pd.DataFrame:
        """
        Parse XML annotations and create a DataFrame with image paths and labels.

        Args:
            images_dir: Directory containing images
            annotations_dir: Directory containing XML annotations

        Returns:
            DataFrame with columns ['image_path', 'class_name', 'class_id']
        """
        data_list = []

        # Get all XML files
        xml_files = list(annotations_dir.glob("*.xml"))

        if not xml_files:
            print(f"No XML files found in {annotations_dir}")
            return pd.DataFrame()

        print(f"Processing {len(xml_files)} annotation files...")

        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Get image filename
                filename_element = root.find("filename")
                if filename_element is None or filename_element.text is None:
                    print(f"No filename found in {xml_file}")
                    continue

                filename = filename_element.text
                image_path = images_dir / filename

                # Check if image exists
                if not image_path.exists():
                    print(f"Image not found: {image_path}")
                    continue

                # Get class information from objects
                objects = root.findall("object")

                if not objects:
                    print(f"No objects found in {xml_file}")
                    continue

                # For face mask detection, we typically have one class per image
                # Take the first object's class
                name_element = objects[0].find("name")
                if name_element is None or name_element.text is None:
                    print(f"No class name found in first object of {xml_file}")
                    continue

                class_name = name_element.text

                # Map class name to our standard classes
                class_name = self._normalize_class_name(class_name)

                if class_name in self.dataset_config["classes"]:
                    class_id = self.dataset_config["classes"].index(class_name)

                    data_list.append(
                        {
                            "image_path": str(image_path),
                            "class_name": class_name,
                            "class_id": class_id,
                            "xml_file": str(xml_file),
                        }
                    )
                else:
                    print(f"Unknown class: {class_name} in {xml_file}")

            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
                continue

        df = pd.DataFrame(data_list)
        print(f"Successfully parsed {len(df)} samples")
        print(f"Class distribution: {df['class_name'].value_counts().to_dict()}")

        return df

    def _normalize_class_name(self, class_name: str) -> str:
        """
        Normalize class names to match our standard classes.

        Args:
            class_name: Original class name from annotation

        Returns:
            Normalized class name
        """
        class_name = class_name.lower().strip()

        # Map variations to standard names
        class_mapping = {
            "with_mask": "with_mask",
            "mask": "with_mask",
            "face_with_mask": "with_mask",
            "without_mask": "without_mask",
            "no_mask": "without_mask",
            "face_no_mask": "without_mask",
            "mask_weared_incorrect": "mask_weared_incorrect",
            "incorrect_mask": "mask_weared_incorrect",
            "face_with_mask_incorrect": "mask_weared_incorrect",
        }

        return class_mapping.get(class_name, class_name)

    def load_dataset(self) -> pd.DataFrame:
        """
        Load and parse the complete dataset.

        Returns:
            DataFrame with all dataset samples
        """
        dataset_path = self.kaggle_config["extract_path"]

        if not dataset_path.exists():
            print("Dataset not found. Downloading...")
            self.download_dataset()

        # Look for images and annotations directories
        images_dir = None
        annotations_dir = None

        # Common directory names
        possible_dirs = list(dataset_path.iterdir())

        for dir_path in possible_dirs:
            if dir_path.is_dir():
                dir_name = dir_path.name.lower()
                if "image" in dir_name:
                    images_dir = dir_path
                elif "annotation" in dir_name or "xml" in dir_name:
                    annotations_dir = dir_path

        # If not found, assume images and annotations are in the same directory
        if images_dir is None:
            images_dir = dataset_path
        if annotations_dir is None:
            annotations_dir = dataset_path

        print(f"Images directory: {images_dir}")
        print(f"Annotations directory: {annotations_dir}")

        # Parse annotations
        self.data_df = self.parse_annotations(images_dir, annotations_dir)

        if self.data_df.empty:
            raise ValueError("No valid data found in the dataset")

        return self.data_df

    def split_dataset(
        self, stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            stratify: Whether to stratify the split by class

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.data_df is None:
            self.load_dataset()

        # Check if data was loaded successfully
        if self.data_df is None or self.data_df.empty:
            raise ValueError("No data available for splitting. Please load dataset first.")

        # Set random seed for reproducible splits
        set_random_seeds()

        # Prepare stratification variable
        stratify_var = np.array(self.data_df["class_id"].values) if stratify else None

        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            self.data_df,
            test_size=self.split_config["test_ratio"],
            random_state=42,
            stratify=stratify_var,
            shuffle=self.split_config["shuffle"],
        )

        # Second split: train vs val
        val_ratio_adjusted = self.split_config["val_ratio"] / (1 - self.split_config["test_ratio"])
        stratify_var_train_val = np.array(train_val_df["class_id"].values) if stratify else None

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=42,
            stratify=stratify_var_train_val,
            shuffle=self.split_config["shuffle"],
        )

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        print("Dataset split completed:")
        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")

        # Print class distribution for each split
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            dist = split_df["class_name"].value_counts().to_dict()
            print(f"{split_name} class distribution: {dist}")

        return train_df, val_df, test_df

    def _create_train_transforms(self) -> transforms.Compose:
        """Create transforms for training data with augmentation."""
        return transforms.Compose(
            [
                transforms.Resize(IMAGE_CONFIG["input_size"]),
                transforms.RandomRotation(self.augmentation_config["rotation_degrees"]),
                transforms.RandomHorizontalFlip(
                    p=self.augmentation_config["horizontal_flip_prob"]
                ),
                transforms.ColorJitter(
                    brightness=self.augmentation_config["brightness"],
                    contrast=self.augmentation_config["contrast"],
                    saturation=self.augmentation_config["saturation"],
                    hue=self.augmentation_config["hue"],
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_CONFIG["mean"], std=IMAGE_CONFIG["std"]),
                transforms.RandomErasing(
                    p=self.augmentation_config["random_erasing_prob"],
                    scale=self.augmentation_config["random_erasing_scale"],
                    ratio=self.augmentation_config["random_erasing_ratio"],
                ),
            ]
        )

    def _create_val_transforms(self) -> transforms.Compose:
        """Create transforms for validation data without augmentation."""
        return transforms.Compose(
            [
                transforms.Resize(IMAGE_CONFIG["input_size"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_CONFIG["mean"], std=IMAGE_CONFIG["std"]),
            ]
        )

    def _create_test_transforms(self) -> transforms.Compose:
        """Create transforms for test data without augmentation."""
        return self._create_val_transforms()

    def create_datasets(self) -> Tuple[FaceMaskDataset, FaceMaskDataset, FaceMaskDataset]:
        """
        Create PyTorch datasets for train, validation, and test.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if any(df is None for df in [self.train_df, self.val_df, self.test_df]):
            self.split_dataset()

        # Ensure all DataFrames are available after splitting
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise ValueError("Failed to create train/val/test splits. Please check your data.")

        train_dataset = FaceMaskDataset(self.train_df, transform=self.train_transform)
        val_dataset = FaceMaskDataset(self.val_df, transform=self.val_transform)
        test_dataset = FaceMaskDataset(self.test_df, transform=self.test_transform)

        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(
        self, use_weighted_sampler: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train, validation, and test.

        Args:
            use_weighted_sampler: Whether to use weighted sampling for training

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_dataset, val_dataset, test_dataset = self.create_datasets()

        # Create weighted sampler for training if requested
        train_sampler = None
        if use_weighted_sampler:
            class_weights = train_dataset.get_class_weights()
            class_indices = torch.from_numpy(np.array(train_dataset.data_df["class_id"].values))
            sample_weights = class_weights[class_indices]
            train_sampler = WeightedRandomSampler(
                weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True
            )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=DEVICE_CONFIG["num_workers"],
            pin_memory=DEVICE_CONFIG["pin_memory"],
            persistent_workers=DEVICE_CONFIG["persistent_workers"],
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
            num_workers=DEVICE_CONFIG["num_workers"],
            pin_memory=DEVICE_CONFIG["pin_memory"],
            persistent_workers=DEVICE_CONFIG["persistent_workers"],
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
            num_workers=DEVICE_CONFIG["num_workers"],
            pin_memory=DEVICE_CONFIG["pin_memory"],
            persistent_workers=DEVICE_CONFIG["persistent_workers"],
        )

        return train_loader, val_loader, test_loader

    def save_splits(self, save_dir: Path) -> None:
        """
        Save the train/val/test splits to CSV files.

        Args:
            save_dir: Directory to save the split files
        """
        save_dir.mkdir(exist_ok=True)

        if self.train_df is not None:
            self.train_df.to_csv(save_dir / "train_split.csv", index=False)
        if self.val_df is not None:
            self.val_df.to_csv(save_dir / "val_split.csv", index=False)
        if self.test_df is not None:
            self.test_df.to_csv(save_dir / "test_split.csv", index=False)

        print(f"Splits saved to {save_dir}")

    def load_splits(self, save_dir: Path) -> None:
        """
        Load train/val/test splits from CSV files.

        Args:
            save_dir: Directory containing the split files
        """
        train_file = save_dir / "train_split.csv"
        val_file = save_dir / "val_split.csv"
        test_file = save_dir / "test_split.csv"

        if train_file.exists():
            self.train_df = pd.read_csv(train_file)
        if val_file.exists():
            self.val_df = pd.read_csv(val_file)
        if test_file.exists():
            self.test_df = pd.read_csv(test_file)

        print(f"Splits loaded from {save_dir}")

    def create_minority_augmentation_transforms(self, class_name: str) -> transforms.Compose:
        """
        Create enhanced augmentation transforms for minority classes.

        Args:
            class_name: Name of the class to create transforms for

        Returns:
            Composed transforms with minority-specific augmentation
        """
        minority_config = AUGMENTATION_CONFIG.get("minority_class_augmentation", {})

        if not minority_config.get("enable", False) or class_name not in minority_config:
            # Return standard training transforms
            return self._create_train_transforms()

        class_config = minority_config[class_name]

        # Build enhanced transform pipeline
        transform_list = [
            transforms.Resize(IMAGE_CONFIG["input_size"]),
            transforms.RandomRotation(class_config.get("rotation_degrees", 15)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

        # Add perspective distortion if configured
        if "perspective_distortion" in class_config:
            transform_list.append(
                transforms.RandomPerspective(
                    distortion_scale=class_config["perspective_distortion"], p=0.5
                )
            )

        # Add color jitter
        if "color_jitter" in class_config:
            cj_config = class_config["color_jitter"]
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj_config.get("brightness", 0.2),
                    contrast=cj_config.get("contrast", 0.2),
                    saturation=cj_config.get("saturation", 0.2),
                    hue=cj_config.get("hue", 0.1),
                )
            )

        # Convert to tensor and normalize
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_CONFIG["mean"], std=IMAGE_CONFIG["std"]),
            ]
        )

        # Add post-tensor transforms
        if class_config.get("random_erasing_prob", 0) > 0:
            transform_list.append(
                transforms.RandomErasing(
                    p=class_config["random_erasing_prob"], scale=(0.02, 0.33), ratio=(0.3, 3.3)
                )
            )

        return transforms.Compose(transform_list)

    def create_class_balanced_sampler(self, dataset) -> Optional[WeightedRandomSampler]:
        """
        Create a weighted sampler that balances classes according to config.

        Args:
            dataset: Dataset to create sampler for

        Returns:
            WeightedRandomSampler for class balancing or None if dataset invalid
        """
        if not hasattr(dataset, "data_df"):
            return None

        # Calculate class weights from config
        minority_factor = TRAINING_CONFIG.get("minority_oversampling_factor", 5)

        # Create weights: majority class = 1.0, minority classes get higher weights
        sample_weights = []
        for _, row in dataset.data_df.iterrows():
            class_id = row["class_id"]
            if class_id == 0:  # with_mask (majority)
                weight = 1.0
            else:  # minority classes
                weight = minority_factor
            sample_weights.append(weight)

        return WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

    def create_enhanced_datasets(self) -> Tuple[FaceMaskDataset, FaceMaskDataset, FaceMaskDataset]:
        """
        Create datasets with minority class specific augmentation.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset) with enhanced augmentation
        """
        if any(df is None for df in [self.train_df, self.val_df, self.test_df]):
            self.split_dataset()

        # Ensure all DataFrames are available after splitting
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise ValueError("Failed to create train/val/test splits. Please check your data.")

        # Create enhanced training dataset with class-specific augmentation
        enhanced_train_dataset = EnhancedFaceMaskDataset(
            self.train_df, base_transform=self.train_transform, minority_augmentation=True
        )

        # Regular validation and test datasets
        val_dataset = FaceMaskDataset(self.val_df, transform=self.val_transform)
        test_dataset = FaceMaskDataset(self.test_df, transform=self.test_transform)

        return enhanced_train_dataset, val_dataset, test_dataset


# Add new enhanced dataset class
class EnhancedFaceMaskDataset(FaceMaskDataset):
    """Enhanced dataset with minority class specific augmentation."""

    def __init__(self, data_df, base_transform=None, minority_augmentation=True):
        super().__init__(data_df, base_transform)
        self.minority_augmentation = minority_augmentation
        self.data_manager = FaceMaskDataManager()

        # Pre-create minority transforms
        self.minority_transforms = {}
        if minority_augmentation:
            for class_name in ["without_mask", "mask_weared_incorrect"]:
                self.minority_transforms[class_name] = (
                    self.data_manager.create_minority_augmentation_transforms(class_name)
                )

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        image_path = row["image_path"]
        class_id = int(row["class_id"])
        class_name = self.classes[class_id]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new("RGB", IMAGE_CONFIG["input_size"], (0, 0, 0))

        # Apply appropriate transform based on class
        if (
            self.minority_augmentation
            and class_name in self.minority_transforms
            and random.random() < 0.8
        ):  # 80% chance to use minority augmentation
            image = self.minority_transforms[class_name](image)
        else:
            # Use standard transform
            if self.transform:
                image = self.transform(image)

        return image, class_id


# Enhanced data loader creation function
def get_enhanced_dataloaders(
    use_weighted_sampler: bool = True, use_minority_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get enhanced DataLoaders with minority class handling.

    Args:
        use_weighted_sampler: Whether to use class-balanced sampling
        use_minority_augmentation: Whether to use enhanced minority augmentation

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_manager = get_data_manager()

    if use_minority_augmentation:
        train_dataset, val_dataset, test_dataset = data_manager.create_enhanced_datasets()
    else:
        train_dataset, val_dataset, test_dataset = data_manager.create_datasets()

    # Create enhanced sampler for training
    train_sampler = None
    if use_weighted_sampler:
        train_sampler = data_manager.create_class_balanced_sampler(train_dataset)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=DEVICE_CONFIG["num_workers"],
        pin_memory=DEVICE_CONFIG["pin_memory"],
        persistent_workers=DEVICE_CONFIG["persistent_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=DEVICE_CONFIG["num_workers"],
        pin_memory=DEVICE_CONFIG["pin_memory"],
        persistent_workers=DEVICE_CONFIG["persistent_workers"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=DEVICE_CONFIG["num_workers"],
        pin_memory=DEVICE_CONFIG["pin_memory"],
        persistent_workers=DEVICE_CONFIG["persistent_workers"],
    )

    return train_loader, val_loader, test_loader


# Convenience functions
def get_data_manager() -> FaceMaskDataManager:
    """Get a configured data manager instance."""
    return FaceMaskDataManager()


def get_dataloaders(
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get DataLoaders for the face mask detection dataset.

    Args:
        use_weighted_sampler: Whether to use weighted sampling for training

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_manager = get_data_manager()
    return data_manager.create_dataloaders(use_weighted_sampler)


if __name__ == "__main__":
    # Example usage
    data_manager = get_data_manager()

    # Load and split dataset
    data_manager.load_dataset()
    data_manager.split_dataset()

    # Create DataLoaders
    train_loader, val_loader, test_loader = data_manager.create_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test loading a batch
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")
