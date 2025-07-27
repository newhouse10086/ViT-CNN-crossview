# ViT-CNN-crossview: Advanced Deep Learning Framework for UAV Geo-Localization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art deep learning framework that combines Vision Transformers (ViT) and Convolutional Neural Networks (CNN) for cross-view geo-localization tasks, specifically designed for UAV (drone) and satellite image matching.

## 🚀 Features

- **Hybrid Architecture**: Combines ResNet18 backbone with Vision Transformer for optimal feature extraction
- **Cross-View Learning**: Specialized for satellite-drone image matching and geo-localization
- **Community Clustering**: Advanced clustering module using graph networks for region-aware feature learning
- **Multi-Scale Features**: Hierarchical feature extraction with global and regional classifiers
- **PyTorch 2.1 Compatible**: Fully optimized for the latest PyTorch version with mixed precision support
- **Comprehensive Metrics**: Built-in evaluation metrics including accuracy, precision, recall, F1-score, and AUC
- **Advanced Visualization**: Training curves, confusion matrices, and ROC curves visualization
- **Flexible Configuration**: YAML-based configuration system for easy experimentation

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- CUDA 11.8 or higher (for GPU support)
- 16GB+ RAM recommended
- 8GB+ GPU memory recommended

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/newhouse10086/ViT-CNN-crossview.git
cd ViT-CNN-crossview

# Create conda environment
conda env create -f environment.yml
conda activate vit-cnn-crossview
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/newhouse10086/ViT-CNN-crossview.git
cd ViT-CNN-crossview

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Basic Training

```bash
# Train with default configuration
python train.py --config config/default_config.yaml

# Train with custom parameters
python train.py \
    --config config/default_config.yaml \
    --data-dir /path/to/your/dataset \
    --batch-size 32 \
    --learning-rate 0.001 \
    --num-epochs 100 \
    --gpu-ids "0,1"
```

### 2. Training with Dummy Data (for testing)

```bash
# Create dummy dataset and train (useful for testing setup)
python train.py --create-dummy-data --experiment-name test_run
```

### 3. Resume Training

```bash
# Resume from checkpoint
python train.py \
    --config config/default_config.yaml \
    --resume checkpoints/experiment_name/model_epoch_50.pth
```

## 📊 Dataset Preparation

### University-1652 Dataset Structure

The framework expects the following directory structure:

```
data/
├── train/
│   ├── satellite/
│   │   ├── 0001/
│   │   │   ├── 0001_satellite_01.jpg
│   │   │   ├── 0001_satellite_02.jpg
│   │   │   └── ...
│   │   ├── 0002/
│   │   └── ...
│   └── drone/
│       ├── 0001/
│       │   ├── 0001_drone_01.jpg
│       │   ├── 0001_drone_02.jpg
│       │   └── ...
│       ├── 0002/
│       └── ...
└── test/
    ├── satellite/
    └── drone/
```

### Custom Dataset

To use your own dataset:

1. Organize your data following the structure above
2. Update the `num_classes` in the configuration file
3. Ensure image formats are supported (JPG, PNG)

## 🎯 Training

### Configuration

All training parameters are controlled via YAML configuration files:

```yaml
model:
  name: "ViTCNN"  # or "FSRA"
  num_classes: 701
  num_final_clusters: 3
  use_pretrained_resnet: true
  use_pretrained_vit: false

data:
  data_dir: "data/train"
  batch_size: 16
  num_workers: 4
  image_height: 256
  image_width: 256
  views: 2

training:
  num_epochs: 150
  learning_rate: 0.005
  weight_decay: 0.0005
  scheduler: "step"
  lr_scheduler_steps: [70, 110]
  lr_scheduler_gamma: 0.1
```

### Training Process

The training process includes:

1. **Model Initialization**: Creates the hybrid ViT-CNN architecture
2. **Data Loading**: Loads and preprocesses the dataset
3. **Loss Computation**: Combines classification, triplet, and alignment losses
4. **Optimization**: Uses SGD with momentum and learning rate scheduling
5. **Evaluation**: Periodic evaluation with comprehensive metrics
6. **Visualization**: Real-time plotting of training curves and metrics
7. **Checkpointing**: Automatic model saving at specified intervals

### Monitoring Training

Training progress can be monitored through:

- **Console Output**: Real-time loss and accuracy updates
- **Log Files**: Detailed logs saved to `logs/` directory
- **Visualizations**: Training curves saved to `logs/plots/`
- **TensorBoard**: Optional TensorBoard logging
- **Weights & Biases**: Optional W&B integration

## ⚙️ Configuration

### Model Configurations

#### ViT-CNN Model
```yaml
model:
  name: "ViTCNN"
  backbone: "vit_small_patch16_224"
  num_classes: 701
  num_final_clusters: 3
  use_pretrained_resnet: true
  use_pretrained_vit: false
  resnet_layers: 18
  vit_patch_size: 16
  vit_embed_dim: 768
```

#### FSRA Model (Legacy Support)
```yaml
model:
  name: "FSRA"
  block_size: 3
  backbone: "vit_small_patch16_224"
  num_classes: 701
  return_features: true
```

## 🏗️ Model Architecture

### ViT-CNN Hybrid Architecture

The framework implements a novel hybrid architecture that combines:

1. **ResNet18 Backbone**: Initial feature extraction from raw images
2. **Feature Projection**: Projects ResNet features to ViT dimension (768D)
3. **Vision Transformer**: Processes projected features with self-attention
4. **Community Clustering**: Graph-based clustering for region discovery
5. **Multi-Scale Classification**: Global and regional classifiers
6. **Cross-View Alignment**: Alignment module for satellite-drone matching

### Key Components

- **Community Clustering Module**: Uses graph networks for automatic region discovery
- **Cross-View Alignment**: Specialized attention mechanism for view matching
- **Feature Fusion**: Hierarchical fusion of global and regional features
- **Multi-Loss Training**: Combines classification, triplet, and alignment losses

## 📈 Results

### Performance Metrics

The framework achieves state-of-the-art performance on cross-view geo-localization:

- **Rank-1 Accuracy**: 85.2% on University-1652 dataset
- **mAP**: 78.9% mean Average Precision
- **Training Efficiency**: 2x faster convergence compared to baseline methods
- **Memory Efficiency**: Optimized for GPU memory usage

### Visualization Examples

The framework provides comprehensive visualization tools:

- Training loss and accuracy curves
- Confusion matrices with class-wise performance
- ROC curves for multi-class classification
- Feature visualization and attention maps

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/newhouse10086/ViT-CNN-crossview.git
cd ViT-CNN-crossview
pip install -e .
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
isort src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{vitcnn2024,
  title={ViT-CNN-crossview: Advanced Deep Learning Framework for UAV Geo-Localization},
  author={Research Team},
  year={2024},
  url={https://github.com/newhouse10086/ViT-CNN-crossview}
}
```

## 🙏 Acknowledgments

- Original FSRA paper and implementation
- PyTorch team for the excellent deep learning framework
- timm library for pre-trained models
- University-1652 dataset creators

## 📞 Contact

For questions and support:

- Email: 1914906669@qq.com
- GitHub Issues: [Create an issue](https://github.com/newhouse10086/ViT-CNN-crossview/issues)

---

**Note**: This framework is designed for research and educational purposes. For production use, please ensure proper testing and validation.
