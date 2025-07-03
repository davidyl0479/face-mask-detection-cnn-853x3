# Face Mask Detection CNN Project

A PyTorch-based CNN project for multi-class face mask detection using transfer learning techniques. This project demonstrates the effectiveness of simple, well-implemented solutions over complex approaches when working with small datasets.

## Project Overview

This project implements a CNN-based face mask detection system using PyTorch for multi-class image classification:

- **Dataset**: Kaggle's "Face Mask Detection" dataset by Andrew MV
- **Dataset Size**: 853 images total
- **Classes**: 3 classes
  - `with_mask` (698 images)
  - `without_mask` (119 images) 
  - `mask_weared_incorrect` (36 images)
- **Task**: Multi-class image classification for face mask compliance
- **Framework**: PyTorch with transfer learning using pre-trained models

## Development Approach

This project follows an iterative development approach, progressing from simple to complex techniques:

### 1. Started with Basic CNN
Built a custom CNN from scratch as a baseline to understand the problem and establish initial performance metrics.

### 2. Moved to Transfer Learning
Implemented transfer learning with pre-trained models (ResNet18, MobileNetV2) which became the main approach due to superior performance.

### 3. Experimentation Phase
Extensively tested various advanced techniques attempting to improve upon the transfer learning baseline, including:
- Advanced loss functions (Focal Loss, Class-Balanced Loss)
- Complex optimizers and learning rate schedules
- Enhanced architectures with residual connections
- Progressive training strategies
- Grid search optimization

## Final Results

The **Transfer Learning model** achieved the best overall performance and became the final solution:

### Training Performance
| Metric | Value |
|--------|--------|
| Training Accuracy | 99.33% |
| Training F1-Score | 99.31% |
| Final Training Loss | 0.3311 |

### Validation Performance
| Metric | Value |
|--------|--------|
| Validation Accuracy | **87.13%** |
| Validation F1-Score | 70.36% |
| Best Validation Loss | 0.5088 |

### Test Performance
| Metric | Value |
|--------|--------|
| Test Accuracy | **79.07%** |
| Overall Macro F1-Score | 51.85% |

#### Per-Class F1-Scores
| Class | F1-Score |
|-------|----------|
| With Mask | 88.28% |
| Without Mask | 27.27% |
| Mask Weared Incorrect | 40.00% |

## What I Tried But Didn't Work

Despite extensive experimentation with advanced techniques, **none improved upon the simple transfer learning baseline**. Here's a detailed account of failed experiments:

### Advanced Loss Functions Tested

#### 1. Weighted Focal Loss
- **Configuration**: α=[1.0, 2.39, 7.89], γ=2.0
- **Purpose**: Address class imbalance by focusing on hard examples
- **Result**: Worse performance than standard cross-entropy
- **Class weights calculated**: `[1.0, 853/(3*119), 853/(3*36)]`

#### 2. Class-Balanced Loss
- **Configuration**: β=0.999, effective number weighting
- **Purpose**: Re-weight loss based on effective number of samples
- **Formula**: `EN = (1-β^n)/(1-β)` where n is samples per class
- **Result**: Degraded performance, likely over-corrected for imbalance

#### 3. Combined Loss Functions
- **Configuration**: 70% Cross-Entropy + 30% Focal Loss
- **Purpose**: Balance standard classification with hard example focus
- **Result**: No improvement over pure cross-entropy

#### 4. Label Smoothing Cross-Entropy
- **Configuration**: ε=0.1-0.15 smoothing factor
- **Purpose**: Prevent overconfident predictions and improve generalization
- **Result**: Marginal degradation in performance

### Grid Search Experiments

Conducted automated grid search across **6 top combinations** testing:

#### Optimizer Variations
- **AdamW**: lr=3e-4, weight_decay=1e-3, β=(0.9,0.999)
- **SGD with Momentum**: lr=5e-4, momentum=0.9, Nesterov=True
- **RMSprop**: lr=1e-4, α=0.99, momentum=0.9

#### Activation Functions
- **ReLU**: Standard baseline activation
- **Swish/SiLU**: Smooth, non-monotonic activation
- **GELU**: Gaussian Error Linear Unit
- **LeakyReLU**: Prevents dead neurons with α=0.01

#### Grid Search Results
| Combination | Validation Accuracy | vs Baseline |
|-------------|-------------------|-------------|
| LabelSmoothing + AdamW + Swish | 55.6% | -31.5% |
| ClassBalanced + SGD + Swish | 52.3% | -34.8% |
| Combined + AdamW + GELU | 48.9% | -38.2% |
| **Transfer Learning Baseline** | **87.13%** | **--** |

**Key Finding**: All advanced combinations performed significantly worse (best: 55.6% vs 87.13% baseline).

### Architecture Experiments

#### 1. Enhanced Classifier Heads
- **Progressive Dropout**: [0.7, 0.6, 0.5, 0.3] schedule
- **Hidden Dimensions**: [512, 512, 256, 128] with batch normalization
- **Result**: Overfitting on small dataset

#### 2. Residual Connections in Classifier
- **Implementation**: Skip connections between linear layers
- **Advanced Initialization**: He/Xavier initialization
- **Result**: Increased complexity without performance gains

#### 3. Progressive Unfreezing Strategies
- **Phase 1**: Train only classifier (10 epochs)
- **Phase 2**: Unfreeze last 2 ResNet blocks (15 epochs)  
- **Phase 3**: Full fine-tuning (20 epochs)
- **Result**: No improvement over simple end-to-end training

### Key Learnings

1. **Dataset Size Matters**: 853 samples are insufficient for complex techniques
2. **Class Imbalance Handling**: Advanced techniques made performance worse
3. **Occam's Razor**: Simple transfer learning outperformed all complex approaches
4. **Over-Engineering**: "More complex ≠ better" for small datasets
5. **Generalization**: Complex models overfit severely on limited data

## Technical Implementation

### Project Structure

```
face_mask_detection_853x3/
├── config.py              # Centralized configuration management
├── dataset.py             # Data pipeline and augmentation
├── features.py            # CNN interpretability and analysis
├── plots.py              # Comprehensive visualization suite
├── grid_search.py         # Automated hyperparameter optimization
└── modeling/
    ├── model.py           # CNN architectures (Basic + Transfer Learning)
    ├── train.py           # Training pipeline with early stopping
    ├── predict.py         # Inference and evaluation
    └── losses.py          # Custom loss functions
```

### Model Configurations

#### Transfer Learning Model (Final Solution)
```python
MODEL_CONFIG = {
    "backbone": "resnet18",
    "pretrained": True,
    "freeze_backbone": False,
    "num_classes": 3,
    "dropout_rate": 0.5,
    "fine_tune_layers": 2,
    "activation": "relu"
}
```

#### Training Configuration
```python
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "scheduler": "step",
    "step_size": 10,
    "gamma": 0.1,
    "early_stopping": {
        "patience": 10, 
        "min_delta": 0.001
    }
}
```

#### Data Processing
```python
IMAGE_CONFIG = {
    "input_size": (224, 224),
    "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
    "std": [0.229, 0.224, 0.225],
}

AUGMENTATION_CONFIG = {
    "rotation_degrees": 15,
    "horizontal_flip_prob": 0.5,
    "brightness": 0.2,
    "contrast": 0.2,
    "gaussian_blur_prob": 0.1,
    "random_erasing_prob": 0.1
}
```

### Advanced Loss Functions Implemented

While these didn't improve performance, they demonstrate comprehensive experimentation:

#### Focal Loss
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        # Implementation for hard example mining
```

#### Class-Balanced Loss  
```python
class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.999):
        # Effective number based re-weighting
```

#### Combined Loss Functions
```python
def get_combined_loss(loss_weights):
    # Mixture of multiple loss functions
```

## Dataset Challenges

Several dataset characteristics influenced the results and explain why simple approaches worked best:

### Size Limitations
- **Total Images**: 853 (very small by modern standards)
- **Train/Val/Test Split**: 596/171/86 samples
- **Training Challenge**: Limited data for complex model learning

### Severe Class Imbalance
- **Class Distribution**: 698/119/36 (19.5:1 ratio)
- **Impact**: Advanced balancing techniques proved counterproductive
- **Test Set Limitation**: Only 4 "mask incorrect" samples

### Data Quality
- **Source**: Kaggle community dataset with PASCAL VOC annotations
- **Variability**: Mixed lighting, angles, and mask types
- **Preprocessing**: Standard ImageNet normalization and augmentation

## Development Environment

### Core Technologies
- **Framework**: PyTorch 2.2.2
- **Development**: Jupyter Notebook for experimentation
- **Models**: Pre-trained ResNet18, MobileNetV2
- **Optimization**: Adam, AdamW, SGD, RMSprop optimizers

### Hardware Requirements
- **GPU**: CUDA-compatible (automatic CPU fallback)
- **Memory**: 8GB+ RAM recommended
- **Storage**: ~50MB for dataset

### Installation
```bash
# Create conda environment
conda env create -f environment.yml
conda activate face-mask-detection

# Or install manually
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn
pip install jupyter notebook
```

### Reproducibility
```python
RANDOM_SEED = 42  # Set across all frameworks
torch.backends.cudnn.deterministic = True
```

## Usage

### Training
```python
from face_mask_detection_853x3.modeling.train import train_model

# Train transfer learning model
train_history, val_history = train_model(
    model_name="transfer_learning",
    experiment_name="transfer_learning_experiment",
    num_epochs=50
)
```

### Evaluation
```python
from face_mask_detection_853x3.modeling.predict import FaceMaskPredictor

# Load trained model and evaluate
predictor = FaceMaskPredictor("models/transfer_learning_trained_model.pth")
metrics = predictor.calculate_metrics()
```

### Grid Search (Experimental)
```python
from face_mask_detection_853x3.grid_search import GridSearchExperiment

# Run automated hyperparameter search
grid_search = GridSearchExperiment(base_output_dir="grid_search_results")
results = grid_search.run_grid_search(max_experiments=6)
```

## Results Analysis

### Why Simple Transfer Learning Won

1. **Appropriate Complexity**: Matched model capacity to dataset size
2. **Pre-trained Features**: Leveraged ImageNet knowledge for face/object detection
3. **Stable Training**: Avoided overfitting common with complex approaches
4. **Robust Generalization**: Better validation→test performance transfer

### Class-wise Performance Analysis

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|----------|---------|
| With Mask | 85.7% | 91.4% | 88.5% | 70 |
| Without Mask | 50.0% | 16.7% | 25.0% | 12 |
| Mask Incorrect | 25.0% | 100.0% | 40.0% | 4 |

**Observations**:
- Excellent performance on majority class (with_mask)
- Poor recall on minority classes due to limited training examples
- Test set too small for reliable minority class evaluation

## Conclusion

This project demonstrates several key machine learning principles:

### Primary Findings
1. **Simple Solutions Often Win**: Transfer learning with ResNet18 achieved 87% validation accuracy
2. **Advanced Techniques Failed**: All complex approaches (advanced losses, architectures) performed worse
3. **Dataset Size Constraints**: 853 samples insufficient for sophisticated techniques
4. **Class Imbalance Reality**: Advanced balancing methods can harm more than help

### Engineering Insights
- **Appropriate Model Complexity**: Match technique sophistication to data availability
- **Validation Importance**: Advanced techniques showed large validation-test gaps
- **Baseline Strength**: Well-implemented simple solutions are hard to beat
- **Experimentation Value**: Failed experiments provide valuable negative results

### Future Improvements
Given more data (10K+ samples), advanced techniques might prove beneficial:
- Advanced augmentation strategies
- Self-supervised pre-training
- Ensemble methods
- Data synthesis techniques

### Portfolio Value
This project showcases:
- **Systematic Experimentation**: Comprehensive testing of multiple approaches
- **Critical Analysis**: Understanding why techniques fail
- **Engineering Judgment**: Choosing appropriate solutions for constraints
- **Reproducible Research**: Detailed documentation and configuration management

---

**Note**: All experimental configurations, hyperparameters, and detailed results are preserved in the codebase. The modular structure allows for easy reproduction and extension of any experiment.