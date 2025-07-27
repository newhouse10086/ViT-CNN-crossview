#!/usr/bin/env python3
"""Setup script for ViT-CNN-crossview project."""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=check)
    return result.returncode == 0


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/train/satellite",
        "data/train/drone", 
        "data/test/satellite",
        "data/test/drone",
        "checkpoints",
        "logs",
        "logs/plots",
        "pretrained"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required.")
        return False
    print(f"Python version: {sys.version}")
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("Installing dependencies...")
    
    # Try to install PyTorch with CUDA support
    if run_command("python -c 'import torch; print(torch.cuda.is_available())'", check=False):
        print("PyTorch already installed")
    else:
        print("Installing PyTorch with CUDA support...")
        torch_command = (
            "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 "
            "--index-url https://download.pytorch.org/whl/cu118"
        )
        if not run_command(torch_command, check=False):
            print("CUDA installation failed, installing CPU version...")
            run_command("pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0")
    
    # Install other dependencies
    run_command("pip install -r requirements.txt")


def verify_installation():
    """Verify the installation."""
    print("Verifying installation...")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("Error: PyTorch not installed correctly")
        return False
    
    try:
        import torchvision
        print(f"TorchVision version: {torchvision.__version__}")
    except ImportError:
        print("Error: TorchVision not installed correctly")
        return False
    
    # Test other important packages
    packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn',
        'PIL', 'cv2', 'yaml', 'tqdm', 'timm'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - not installed")
            return False
    
    return True


def create_sample_config():
    """Create a sample configuration file."""
    config_content = """# ViT-CNN-crossview Configuration
# Compatible with PyTorch 2.1, CUDA 11.8, Ubuntu 18.04+

model:
  name: "ViTCNN"
  backbone: "vit_small_patch16_224"
  num_classes: 10  # Change this to your dataset's number of classes
  block_size: 3
  share_weights: true
  return_features: true
  dropout_rate: 0.1
  use_pretrained_resnet: true
  use_pretrained_vit: false
  num_final_clusters: 3
  resnet_layers: 18
  vit_patch_size: 16
  vit_embed_dim: 768

data:
  data_dir: "data/train"
  test_dir: "data/test"
  batch_size: 16
  num_workers: 4
  image_height: 256
  image_width: 256
  views: 2
  sample_num: 4
  pad: 0
  color_jitter: false
  random_erasing_prob: 0.0

training:
  num_epochs: 150
  learning_rate: 0.005
  weight_decay: 0.0005
  momentum: 0.9
  warm_epochs: 5
  lr_scheduler_steps: [70, 110]
  lr_scheduler_gamma: 0.1
  triplet_loss_weight: 0.3
  kl_loss_weight: 0.0
  use_kl_loss: false
  cross_attention_weight: 1.0
  use_fp16: false
  use_autocast: true
  use_data_augmentation: true
  moving_avg: 1.0

evaluation:
  eval_interval: 10
  save_plots: true
  plot_interval: 10
  metrics_to_track:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
    - "auc_roc"
  save_confusion_matrix: true
  save_roc_curves: true
  save_training_curves: true

system:
  gpu_ids: "0"
  use_gpu: true
  seed: 42
  log_interval: 10
  save_interval: 10
  checkpoint_dir: "checkpoints"
  log_dir: "logs"
  pretrained_dir: "pretrained"
  log_level: "INFO"
  save_logs_to_file: true
  use_tensorboard: true
  use_wandb: false

mode: 0  # 0 for training, 1 for testing
"""
    
    config_path = Path("config/sample_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created sample configuration: {config_path}")


def main():
    """Main setup function."""
    print("=" * 60)
    print("ViT-CNN-crossview Setup Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n2. Installing dependencies...")
    install_dependencies()
    
    # Verify installation
    print("\n3. Verifying installation...")
    if not verify_installation():
        print("Installation verification failed!")
        return 1
    
    # Create sample config
    print("\n4. Creating sample configuration...")
    create_sample_config()
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Prepare your dataset in the data/ directory")
    print("2. Update config/sample_config.yaml with your settings")
    print("3. Run: python train.py --config config/sample_config.yaml")
    print("\nFor testing with dummy data:")
    print("python train.py --create-dummy-data --experiment-name test_run")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
