"""
grid_search.py

Automated Grid Search Module for Face Mask Detection Model Combinations
Save this file as: face_mask_detection_853x3/grid_search.py
"""

from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

# Import your existing modules
from .config import *
from .dataset import get_dataloaders
from .modeling.predict import FaceMaskPredictor
from .modeling.train import FaceMaskTrainer


class GridSearchExperiment:
    """
    Automated grid search for hyperparameter combinations in face mask detection.

    This class systematically tests different combinations of:
    - Loss functions (LabelSmoothingCE, ClassBalanced, Combined)
    - Optimizers (AdamW, SGD, RMSprop)
    - Activation functions (Swish/SiLU, GELU, LeakyReLU)

    Example:
        grid_search = GridSearchExperiment(base_output_dir="my_experiments")
        results = grid_search.run_grid_search(max_experiments=6)
        grid_search.print_summary()
    """

    def __init__(self, base_output_dir: str = "grid_search_results", reduced_epochs: int = 25):
        """
        Initialize the grid search experiment.

        Args:
            base_output_dir: Directory to save all experiment results
            reduced_epochs: Number of epochs per experiment (reduced for faster grid search)
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.reduced_epochs = reduced_epochs

        # Results tracking
        self.results: List[Dict[str, Any]] = []
        self.results_file = self.base_output_dir / "grid_search_results.csv"

        # Store original configs to restore after each experiment
        self.original_configs = {
            "loss": LOSS_CONFIG.copy(),
            "training": TRAINING_CONFIG.copy(),
            "model": MODEL_CONFIGS.copy(),
        }

        print("🔬 Grid Search initialized")
        print(f"📁 Results directory: {self.base_output_dir}")
        print(f"⏱️  Epochs per experiment: {reduced_epochs}")
        print(f"📊 Results will be saved to: {self.results_file}")

    def define_combinations(self, include_all: bool = False) -> List[Dict[str, Any]]:
        """
        Define combinations for testing.

        Args:
            include_all: If True, returns all 27 combinations. If False, returns top 6.

        Returns:
            List of combination dictionaries
        """

        # Top 6 Priority Combinations (original)
        top_6_combinations = [
            {
                "id": "A1",
                "name": "LabelSmoothingCE_AdamW_Swish",
                "description": "Label smoothing with AdamW and Swish activation",
                "loss_type": "label_smoothing",
                "optimizer": "adamw",
                "activation": "silu",
                "priority": 1,
                "expected_accuracy": "85-88%",
                "tier": "Top Priority",
            },
            {
                "id": "A2",
                "name": "LabelSmoothingCE_AdamW_GELU",
                "description": "Label smoothing with AdamW and GELU activation",
                "loss_type": "label_smoothing",
                "optimizer": "adamw",
                "activation": "gelu",
                "priority": 2,
                "expected_accuracy": "84-87%",
                "tier": "Top Priority",
            },
            {
                "id": "B1",
                "name": "ClassBalanced_AdamW_Swish",
                "description": "Class-balanced loss with AdamW and Swish activation",
                "loss_type": "class_balanced",
                "optimizer": "adamw",
                "activation": "silu",
                "priority": 3,
                "expected_accuracy": "84-87%",
                "tier": "Top Priority",
            },
            {
                "id": "A3",
                "name": "LabelSmoothingCE_SGD_Swish",
                "description": "Label smoothing with SGD momentum and Swish activation",
                "loss_type": "label_smoothing",
                "optimizer": "sgd",
                "activation": "silu",
                "priority": 4,
                "expected_accuracy": "83-86%",
                "tier": "Top Priority",
            },
            {
                "id": "B2",
                "name": "ClassBalanced_SGD_Swish",
                "description": "Class-balanced loss with SGD momentum and Swish activation",
                "loss_type": "class_balanced",
                "optimizer": "sgd",
                "activation": "silu",
                "priority": 5,
                "expected_accuracy": "83-86%",
                "tier": "Top Priority",
            },
            {
                "id": "C1",
                "name": "CombinedLoss_AdamW_Swish",
                "description": "Combined loss with AdamW and Swish activation",
                "loss_type": "combined",
                "optimizer": "adamw",
                "activation": "silu",
                "priority": 6,
                "expected_accuracy": "83-86%",
                "tier": "Top Priority",
            },
        ]

        # Additional High Priority Combinations (7-12)
        high_priority_combinations = [
            {
                "id": "A4",
                "name": "LabelSmoothingCE_AdamW_LeakyReLU",
                "description": "Label smoothing with AdamW and LeakyReLU activation",
                "loss_type": "label_smoothing",
                "optimizer": "adamw",
                "activation": "leaky_relu",
                "priority": 7,
                "expected_accuracy": "82-85%",
                "tier": "High Priority",
            },
            {
                "id": "A5",
                "name": "LabelSmoothingCE_SGD_GELU",
                "description": "Label smoothing with SGD momentum and GELU activation",
                "loss_type": "label_smoothing",
                "optimizer": "sgd",
                "activation": "gelu",
                "priority": 8,
                "expected_accuracy": "82-84%",
                "tier": "High Priority",
            },
            {
                "id": "B3",
                "name": "ClassBalanced_AdamW_GELU",
                "description": "Class-balanced loss with AdamW and GELU activation",
                "loss_type": "class_balanced",
                "optimizer": "adamw",
                "activation": "gelu",
                "priority": 9,
                "expected_accuracy": "83-85%",
                "tier": "High Priority",
            },
            {
                "id": "B4",
                "name": "ClassBalanced_AdamW_LeakyReLU",
                "description": "Class-balanced loss with AdamW and LeakyReLU activation",
                "loss_type": "class_balanced",
                "optimizer": "adamw",
                "activation": "leaky_relu",
                "priority": 10,
                "expected_accuracy": "82-84%",
                "tier": "High Priority",
            },
            {
                "id": "C2",
                "name": "CombinedLoss_SGD_Swish",
                "description": "Combined loss with SGD momentum and Swish activation",
                "loss_type": "combined",
                "optimizer": "sgd",
                "activation": "silu",
                "priority": 11,
                "expected_accuracy": "82-85%",
                "tier": "High Priority",
            },
            {
                "id": "C3",
                "name": "CombinedLoss_AdamW_GELU",
                "description": "Combined loss with AdamW and GELU activation",
                "loss_type": "combined",
                "optimizer": "adamw",
                "activation": "gelu",
                "priority": 12,
                "expected_accuracy": "82-84%",
                "tier": "High Priority",
            },
        ]

        # Medium Priority with RMSprop (13-21)
        medium_priority_combinations = [
            {
                "id": "A6",
                "name": "LabelSmoothingCE_RMSprop_Swish",
                "description": "Label smoothing with RMSprop and Swish activation",
                "loss_type": "label_smoothing",
                "optimizer": "rmsprop",
                "activation": "silu",
                "priority": 13,
                "expected_accuracy": "81-84%",
                "tier": "Medium Priority",
            },
            {
                "id": "A7",
                "name": "LabelSmoothingCE_SGD_LeakyReLU",
                "description": "Label smoothing with SGD momentum and LeakyReLU activation",
                "loss_type": "label_smoothing",
                "optimizer": "sgd",
                "activation": "leaky_relu",
                "priority": 14,
                "expected_accuracy": "81-83%",
                "tier": "Medium Priority",
            },
            {
                "id": "A8",
                "name": "LabelSmoothingCE_RMSprop_GELU",
                "description": "Label smoothing with RMSprop and GELU activation",
                "loss_type": "label_smoothing",
                "optimizer": "rmsprop",
                "activation": "gelu",
                "priority": 15,
                "expected_accuracy": "80-83%",
                "tier": "Medium Priority",
            },
            {
                "id": "A9",
                "name": "LabelSmoothingCE_RMSprop_LeakyReLU",
                "description": "Label smoothing with RMSprop and LeakyReLU activation",
                "loss_type": "label_smoothing",
                "optimizer": "rmsprop",
                "activation": "leaky_relu",
                "priority": 16,
                "expected_accuracy": "79-82%",
                "tier": "Medium Priority",
            },
            {
                "id": "B5",
                "name": "ClassBalanced_SGD_GELU",
                "description": "Class-balanced loss with SGD momentum and GELU activation",
                "loss_type": "class_balanced",
                "optimizer": "sgd",
                "activation": "gelu",
                "priority": 17,
                "expected_accuracy": "81-84%",
                "tier": "Medium Priority",
            },
            {
                "id": "B6",
                "name": "ClassBalanced_RMSprop_Swish",
                "description": "Class-balanced loss with RMSprop and Swish activation",
                "loss_type": "class_balanced",
                "optimizer": "rmsprop",
                "activation": "silu",
                "priority": 18,
                "expected_accuracy": "81-83%",
                "tier": "Medium Priority",
            },
            {
                "id": "B7",
                "name": "ClassBalanced_SGD_LeakyReLU",
                "description": "Class-balanced loss with SGD momentum and LeakyReLU activation",
                "loss_type": "class_balanced",
                "optimizer": "sgd",
                "activation": "leaky_relu",
                "priority": 19,
                "expected_accuracy": "80-83%",
                "tier": "Medium Priority",
            },
            {
                "id": "B8",
                "name": "ClassBalanced_RMSprop_GELU",
                "description": "Class-balanced loss with RMSprop and GELU activation",
                "loss_type": "class_balanced",
                "optimizer": "rmsprop",
                "activation": "gelu",
                "priority": 20,
                "expected_accuracy": "80-82%",
                "tier": "Medium Priority",
            },
            {
                "id": "B9",
                "name": "ClassBalanced_RMSprop_LeakyReLU",
                "description": "Class-balanced loss with RMSprop and LeakyReLU activation",
                "loss_type": "class_balanced",
                "optimizer": "rmsprop",
                "activation": "leaky_relu",
                "priority": 21,
                "expected_accuracy": "79-81%",
                "tier": "Medium Priority",
            },
        ]

        # Lower Priority Combined Loss variations (22-27)
        lower_priority_combinations = [
            {
                "id": "C4",
                "name": "CombinedLoss_AdamW_LeakyReLU",
                "description": "Combined loss with AdamW and LeakyReLU activation",
                "loss_type": "combined",
                "optimizer": "adamw",
                "activation": "leaky_relu",
                "priority": 22,
                "expected_accuracy": "81-84%",
                "tier": "Lower Priority",
            },
            {
                "id": "C5",
                "name": "CombinedLoss_SGD_GELU",
                "description": "Combined loss with SGD momentum and GELU activation",
                "loss_type": "combined",
                "optimizer": "sgd",
                "activation": "gelu",
                "priority": 23,
                "expected_accuracy": "81-83%",
                "tier": "Lower Priority",
            },
            {
                "id": "C6",
                "name": "CombinedLoss_RMSprop_Swish",
                "description": "Combined loss with RMSprop and Swish activation",
                "loss_type": "combined",
                "optimizer": "rmsprop",
                "activation": "silu",
                "priority": 24,
                "expected_accuracy": "80-83%",
                "tier": "Lower Priority",
            },
            {
                "id": "C7",
                "name": "CombinedLoss_SGD_LeakyReLU",
                "description": "Combined loss with SGD momentum and LeakyReLU activation",
                "loss_type": "combined",
                "optimizer": "sgd",
                "activation": "leaky_relu",
                "priority": 25,
                "expected_accuracy": "80-82%",
                "tier": "Lower Priority",
            },
            {
                "id": "C8",
                "name": "CombinedLoss_RMSprop_GELU",
                "description": "Combined loss with RMSprop and GELU activation",
                "loss_type": "combined",
                "optimizer": "rmsprop",
                "activation": "gelu",
                "priority": 26,
                "expected_accuracy": "79-82%",
                "tier": "Lower Priority",
            },
            {
                "id": "C9",
                "name": "CombinedLoss_RMSprop_LeakyReLU",
                "description": "Combined loss with RMSprop and LeakyReLU activation",
                "loss_type": "combined",
                "optimizer": "rmsprop",
                "activation": "leaky_relu",
                "priority": 27,
                "expected_accuracy": "78-81%",
                "tier": "Lower Priority",
            },
        ]

        # Return appropriate combinations based on include_all parameter
        if include_all:
            all_combinations = (
                top_6_combinations
                + high_priority_combinations
                + medium_priority_combinations
                + lower_priority_combinations
            )
            return all_combinations
        else:
            return top_6_combinations

    def create_config_for_combination(
        self, combination: Dict[str, Any]
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Create configuration modifications for a specific combination.

        Args:
            combination: Dictionary containing combination parameters

        Returns:
            Tuple of (loss_config, training_config, model_config)
        """

        # Loss configuration
        loss_config = self.original_configs["loss"].copy()

        if combination["loss_type"] == "label_smoothing":
            loss_config.update(
                {
                    "primary_loss": "cross_entropy",
                    "class_weights": [
                        1.0,
                        853 / (3 * 119),
                        853 / (3 * 36),
                    ],  # Your calculated weights
                    "label_smoothing": 0.15,  # Higher smoothing for small dataset
                    "model_loss_mapping": {
                        "basic_cnn": "cross_entropy",
                        "transfer_learning": "label_smoothing",
                    },
                }
            )

        elif combination["loss_type"] == "class_balanced":
            loss_config.update(
                {
                    "primary_loss": "class_balanced",
                    "samples_per_class": [698, 119, 36],  # Your actual class distribution
                    "beta": 0.999,  # Effective number parameter
                    "base_loss_type": "cross_entropy",  # Use stable CE as base
                    "model_loss_mapping": {
                        "basic_cnn": "cross_entropy",
                        "transfer_learning": "class_balanced",
                    },
                }
            )

        elif combination["loss_type"] == "combined":
            loss_config.update(
                {
                    "primary_loss": "combined",
                    "combined_losses": {
                        "cross_entropy": {"weight": 0.7, "class_weights": [1.0, 2.39, 7.89]},
                        "focal": {"weight": 0.3, "alpha": 1.0, "gamma": 1.0},  # Conservative focal
                    },
                    "model_loss_mapping": {
                        "basic_cnn": "cross_entropy",
                        "transfer_learning": "combined",
                    },
                }
            )

        # Training configuration
        training_config = self.original_configs["training"].copy()

        if combination["optimizer"] == "adamw":
            training_config.update(
                {
                    "optimizer": "adamw",
                    "learning_rate": 3e-4,  # Conservative learning rate
                    "weight_decay": 1e-3,  # Higher weight decay for small dataset
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                }
            )
        elif combination["optimizer"] == "sgd":
            training_config.update(
                {
                    "optimizer": "sgd",
                    "learning_rate": 5e-4,  # Slightly higher for SGD
                    "momentum": 0.9,
                    "weight_decay": 5e-4,
                    "nesterov": True,  # Nesterov momentum
                }
            )
        elif combination["optimizer"] == "rmsprop":
            training_config.update(
                {
                    "optimizer": "rmsprop",
                    "learning_rate": 1e-4,  # Lower for RMSprop
                    "alpha": 0.99,  # Smoothing constant
                    "weight_decay": 1e-3,
                    "momentum": 0.9,
                }
            )

        # Model configuration
        model_config = self.original_configs["model"]["transfer_learning"].copy()
        model_config.update(
            {
                "activation": combination["activation"],  # Now supported!
                "dropout_schedule": [0.4, 0.3, 0.2],  # Conservative decreasing dropout
                "hidden_dims": [256, 128, 64],  # Simplified architecture
                "use_residual_classifier": False,  # Remove complexity for grid search
                "enhanced_classifier": True,  # Keep enhanced features
            }
        )

        return loss_config, training_config, model_config

    def run_single_experiment(self, combination: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment with the given combination.

        Args:
            combination: Dictionary containing combination parameters

        Returns:
            Dictionary containing experiment results
        """

        print(f"\n{'=' * 70}")
        print(f"🚀 EXPERIMENT {combination['id']}: {combination['name']}")
        print(f"📝 Description: {combination['description']}")
        print(f"🎯 Expected: {combination['expected_accuracy']}")
        print(f"{'=' * 70}")

        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{combination['id']}_{combination['name']}_{timestamp}"
        exp_dir = self.base_output_dir / exp_name
        exp_dir.mkdir(exist_ok=True)

        try:
            # Get configurations for this combination
            loss_config, training_config, model_config = self.create_config_for_combination(
                combination
            )

            # Save experiment configuration
            config_save = {
                "combination": combination,
                "loss_config": loss_config,
                "training_config": training_config,
                "model_config": model_config,
                "timestamp": datetime.now().isoformat(),
                "epochs": self.reduced_epochs,
            }

            config_file = exp_dir / "experiment_config.json"
            with open(config_file, "w") as f:
                json.dump(config_save, f, indent=2, default=str)

            print(f"📋 Configuration saved to: {config_file}")

            # Temporarily modify global configs
            self._apply_configs(loss_config, training_config, model_config)

            # Create trainer with experiment-specific settings
            trainer = FaceMaskTrainer(
                model_name="transfer_learning",
                experiment_name=exp_name,
                model_prefix=f"{combination['id']}_tl",
                use_tensorboard=True,
            )

            # Run training
            print(f"🏃 Starting training for {self.reduced_epochs} epochs...")
            start_time = time.time()

            train_history, val_history = trainer.train(num_epochs=self.reduced_epochs)

            training_time = time.time() - start_time

            # Extract best metrics from training
            best_val_loss = min([h["loss"] for h in val_history])
            best_val_acc = max([h["accuracy"] for h in val_history])
            best_val_f1 = max([h["f1_score"] for h in val_history])
            best_epoch = (
                val_history.index(next(h for h in val_history if h["accuracy"] == best_val_acc))
                + 1
            )

            # Evaluate on test set
            test_accuracy, test_f1 = self._evaluate_test_set(trainer, combination)

            # Compile results
            result = {
                "experiment_id": combination["id"],
                "experiment_name": combination["name"],
                "description": combination["description"],
                "loss_type": combination["loss_type"],
                "optimizer": combination["optimizer"],
                "activation": combination["activation"],
                "priority": combination["priority"],
                "expected_accuracy": combination["expected_accuracy"],
                # Training results
                "best_val_loss": round(best_val_loss, 4),
                "best_val_accuracy": round(best_val_acc, 4),
                "best_val_f1": round(best_val_f1, 4),
                "best_epoch": best_epoch,
                # Test results
                "test_accuracy": round(test_accuracy, 4),
                "test_f1": round(test_f1, 4),
                # Meta information
                "training_time_minutes": round(training_time / 60, 1),
                "total_epochs": len(val_history),
                "model_path": str(
                    trainer.experiment_dir / f"{combination['id']}_tl_best_model.pth"
                ),
                "experiment_dir": str(exp_dir),
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
            }

            print(f"\n✅ EXPERIMENT {combination['id']} COMPLETED!")
            print(f"📊 Best Val Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
            print(f"📊 Best Val F1: {best_val_f1:.4f}")
            print(f"🧪 Test Accuracy: {test_accuracy:.4f}")
            print(f"⏱️  Training Time: {training_time / 60:.1f} minutes")

            return result

        except Exception as e:
            print(f"\n❌ EXPERIMENT {combination['id']} FAILED!")
            print(f"💥 Error: {str(e)}")

            # Return error result
            result = {
                "experiment_id": combination["id"],
                "experiment_name": combination["name"],
                "description": combination["description"],
                "loss_type": combination["loss_type"],
                "optimizer": combination["optimizer"],
                "activation": combination["activation"],
                "priority": combination["priority"],
                "expected_accuracy": combination["expected_accuracy"],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
            }

            return result

        finally:
            # Always restore original configurations
            self._restore_configs()

    def _apply_configs(self, loss_config: Dict, training_config: Dict, model_config: Dict):
        """Apply temporary configurations."""
        global LOSS_CONFIG, TRAINING_CONFIG, MODEL_CONFIGS

        LOSS_CONFIG.clear()
        LOSS_CONFIG.update(loss_config)

        TRAINING_CONFIG.clear()
        TRAINING_CONFIG.update(training_config)

        MODEL_CONFIGS["transfer_learning"].clear()
        MODEL_CONFIGS["transfer_learning"].update(model_config)

    def _restore_configs(self):
        """Restore original configurations."""
        global LOSS_CONFIG, TRAINING_CONFIG, MODEL_CONFIGS

        LOSS_CONFIG.clear()
        LOSS_CONFIG.update(self.original_configs["loss"])

        TRAINING_CONFIG.clear()
        TRAINING_CONFIG.update(self.original_configs["training"])

        MODEL_CONFIGS.clear()
        MODEL_CONFIGS.update(self.original_configs["model"])

    def _evaluate_test_set(self, trainer, combination: Dict[str, Any]) -> Tuple[float, float]:
        """Evaluate the trained model on test set."""
        try:
            best_model_path = (
                trainer.experiment_dir / f"{combination['id']}_tl_best_model.pth"
            )

            if best_model_path.exists():
                predictor = FaceMaskPredictor(str(best_model_path))
                predictor.predict_dataset(dataset_split="test")
                test_metrics = predictor.calculate_metrics()
                return test_metrics["accuracy"], test_metrics["macro_f1"]
            else:
                print(f"⚠️  Model file not found: {best_model_path}")
                return 0.0, 0.0

        except Exception as e:
            print(f"⚠️  Test evaluation failed: {e}")
            return 0.0, 0.0

    def run_grid_search(
        self, max_experiments: int = 6, early_stop_threshold: float = 0.87
    ) -> List[Dict[str, Any]]:
        """
        Run the complete grid search with all combinations.

        Args:
            max_experiments: Maximum number of experiments to run
            early_stop_threshold: Stop if validation accuracy exceeds this threshold

        Returns:
            List of experiment results
        """

        combinations = self.define_combinations()[:max_experiments]

        print("\n🎯 STARTING AUTOMATED GRID SEARCH")
        print(f"🧪 Number of combinations: {len(combinations)}")
        print(f"⏱️  Estimated time: {len(combinations) * (self.reduced_epochs * 2)} minutes")
        print(f"🎚️  Early stop threshold: {early_stop_threshold:.1%}")
        print(f"📁 Results directory: {self.base_output_dir}")

        total_start_time = time.time()

        for i, combination in enumerate(combinations, 1):
            print(f"\n[{i}/{len(combinations)}] 🧪 Running combination {combination['id']}")

            result = self.run_single_experiment(combination)
            self.results.append(result)

            # Save results incrementally
            self.save_results()

            # Print progress
            elapsed = (time.time() - total_start_time) / 60
            remaining = len(combinations) - i
            eta = (elapsed / i) * remaining if i > 0 else 0

            print(f"\n📈 Progress: {i}/{len(combinations)} completed")
            print(f"⏱️  Elapsed: {elapsed:.1f}min | ETA: {eta:.1f}min")

            # Early stopping check
            if (
                result.get("status") == "completed"
                and result.get("best_val_accuracy", 0) > early_stop_threshold
            ):
                print("\n🎉 EARLY STOPPING TRIGGERED!")
                print(
                    f"🏆 Excellent results found: {result['best_val_accuracy']:.4f} > {early_stop_threshold:.4f}"
                )
                print(
                    "🤔 You can continue testing remaining combinations for comparison if desired."
                )
                break

        total_time = (time.time() - total_start_time) / 60
        print("\n🏁 GRID SEARCH COMPLETED!")
        print(f"⏱️  Total time: {total_time:.1f} minutes")
        print(
            f"✅ Completed experiments: {len([r for r in self.results if r.get('status') == 'completed'])}"
        )

        # Generate final summary
        self.print_summary()

        return self.results

    def save_results(self):
        """Save current results to CSV file."""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.results_file, index=False)
            print(f"💾 Results saved to: {self.results_file}")

    def load_results(self) -> pd.DataFrame:
        """Load existing results from CSV file."""
        if self.results_file.exists():
            return pd.read_csv(self.results_file)
        return pd.DataFrame()

    def print_summary(self):
        """Print a comprehensive summary of all results."""
        if not self.results:
            print("❌ No results to summarize.")
            return

        print(f"\n{'=' * 80}")
        print("🏆 GRID SEARCH RESULTS SUMMARY")
        print(f"{'=' * 80}")

        df = pd.DataFrame(self.results)

        # Completed experiments
        completed = df[df["status"] == "completed"].copy()
        failed = df[df["status"] == "failed"]

        print(f"📊 Total Experiments: {len(df)}")
        print(f"✅ Completed: {len(completed)}")
        print(f"❌ Failed: {len(failed)}")

        if not completed.empty:
            # Sort by validation accuracy
            completed = completed.sort_values("best_val_accuracy", ascending=False)

            print("\n🥇 TOP RESULTS (by Validation Accuracy):")
            print("-" * 80)

            for i, (_, row) in enumerate(completed.head(3).iterrows(), 1):
                medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"{i}."
                print(f"{medal} {row['experiment_id']}: {row['experiment_name']}")
                print(
                    f"   📈 Val Acc: {row['best_val_accuracy']:.4f} | Test Acc: {row['test_accuracy']:.4f}"
                )
                print(f"   📊 Val F1:  {row['best_val_f1']:.4f} | Test F1:  {row['test_f1']:.4f}")
                print(
                    f"   ⏱️  Time: {row['training_time_minutes']:.1f}min | Epoch: {row.get('best_epoch', 'N/A')}"
                )
                print(f"   📁 Model: {Path(row['model_path']).name}")
                print()

            # Performance statistics
            print("📈 PERFORMANCE STATISTICS:")
            print(f"   Best Val Accuracy: {completed['best_val_accuracy'].max():.4f}")
            print(f"   Average Val Accuracy: {completed['best_val_accuracy'].mean():.4f}")
            print(f"   Best Test Accuracy: {completed['test_accuracy'].max():.4f}")
            print(
                f"   Average Training Time: {completed['training_time_minutes'].mean():.1f} minutes"
            )

        # Failed experiments
        if not failed.empty:
            print("\n❌ FAILED EXPERIMENTS:")
            for _, row in failed.iterrows():
                print(f"   {row['experiment_id']}: {row.get('error', 'Unknown error')}")

        print(f"\n📁 Full results saved to: {self.results_file}")
        print(f"📂 Individual experiment details in: {self.base_output_dir}")

    def get_best_result(self) -> Dict[str, Any]:
        """Get the best performing experiment result."""
        if not self.results:
            return {}

        completed_results = [r for r in self.results if r.get("status") == "completed"]
        if not completed_results:
            return {}

        return max(completed_results, key=lambda x: x.get("best_val_accuracy", 0))


# Convenience functions for easy use in Jupyter Notebook
def run_quick_grid_search(
    output_dir: str = "face_mask_grid_search", max_experiments: int = 6, epochs: int = 25
) -> Tuple[List[Dict], GridSearchExperiment]:
    """
    Quick function to run grid search with default settings.

    Args:
        output_dir: Directory to save results
        max_experiments: Number of experiments to run
        epochs: Epochs per experiment

    Returns:
        Tuple of (results_list, grid_search_instance)
    """
    grid_search = GridSearchExperiment(base_output_dir=output_dir, reduced_epochs=epochs)

    results = grid_search.run_grid_search(max_experiments=max_experiments)

    return results, grid_search


def load_and_analyze_results(results_file: str) -> pd.DataFrame:
    """
    Load and analyze existing grid search results.

    Args:
        results_file: Path to the CSV results file

    Returns:
        DataFrame with results analysis
    """
    df = pd.read_csv(results_file)

    # Add analysis columns
    completed = df[df["status"] == "completed"].copy()

    if not completed.empty:
        completed["rank"] = completed["best_val_accuracy"].rank(ascending=False)
        completed["improvement_over_baseline"] = (
            completed["best_val_accuracy"] - 0.81
        )  # Your current accuracy

        print("📊 Results Analysis:")
        print(f"   Best Accuracy: {completed['best_val_accuracy'].max():.4f}")
        print(
            f"   Improvement over baseline (0.81): {completed['improvement_over_baseline'].max():.4f}"
        )
        print(
            f"   Number of experiments > 0.85: {len(completed[completed['best_val_accuracy'] > 0.85])}"
        )

    return df


# Usage examples for different experiment scales:


def run_extended_grid_search(
    max_experiments: int = 12, output_dir: str = "extended_grid_search", epochs: int = 25
):
    """
    Run extended grid search with more combinations.

    Recommended experiment counts:
    - 6: Top priority only (original)
    - 12: Top + High priority
    - 21: Top + High + Medium priority
    - 27: All combinations (full grid search)
    """

    # Modify the GridSearchExperiment class to use extended combinations
    class ExtendedGridSearchExperiment(GridSearchExperiment):
        def define_combinations(self, include_all: bool = True):
            return super().define_combinations(include_all=include_all)

    # Create extended grid search
    grid_search = ExtendedGridSearchExperiment(output_dir, epochs)

    # Run with specified number of experiments
    results = grid_search.run_grid_search(max_experiments=max_experiments)

    return results, grid_search
