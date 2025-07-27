#!/usr/bin/env python3
"""Simple test script to check basic functionality."""

import sys
import os
from pathlib import Path

print("Python version:", sys.version)
print("Current directory:", os.getcwd())

# Add src to path
src_path = str(Path(__file__).parent / "src")
print("Adding to path:", src_path)
sys.path.insert(0, src_path)

print("Python path:", sys.path[:3])

# Test basic imports
try:
    import torch
    print("✓ PyTorch version:", torch.__version__)
except ImportError as e:
    print("✗ PyTorch import failed:", e)
    sys.exit(1)

try:
    import torchvision
    print("✓ TorchVision version:", torchvision.__version__)
except ImportError as e:
    print("✗ TorchVision import failed:", e)

# Test project imports step by step
print("\nTesting project imports...")

try:
    print("Testing utils...")
    from src.utils.config_utils import load_config
    print("✓ config_utils import successful")
except ImportError as e:
    print("✗ config_utils import failed:", e)

try:
    print("Testing models...")
    from src.models.model_factory import create_model
    print("✓ model_factory import successful")
except ImportError as e:
    print("✗ model_factory import failed:", e)

try:
    print("Testing datasets...")
    from src.datasets.dataset_factory import make_dataloader
    print("✓ dataset_factory import successful")
except ImportError as e:
    print("✗ dataset_factory import failed:", e)

try:
    print("Testing losses...")
    from src.losses.combined_loss import CombinedLoss
    print("✓ combined_loss import successful")
except ImportError as e:
    print("✗ combined_loss import failed:", e)

# Test basic functionality
print("\nTesting basic functionality...")

try:
    # Test config
    test_config = {
        'model': {'name': 'ViTCNN', 'num_classes': 10},
        'data': {'batch_size': 4, 'views': 2},
        'training': {'num_epochs': 1}
    }
    
    # Test model creation
    from src.models.model_factory import create_model
    model = create_model(test_config)
    print("✓ Model creation successful")
    
    # Test loss function
    from src.losses.combined_loss import CombinedLoss
    criterion = CombinedLoss(num_classes=10)
    print("✓ Loss function creation successful")
    
    print("\n🎉 All basic tests passed!")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nProject structure check:")
for item in Path(".").iterdir():
    if item.is_dir() and not item.name.startswith('.'):
        print(f"📁 {item.name}/")
    elif item.is_file() and item.suffix in ['.py', '.yaml', '.yml', '.txt', '.md']:
        print(f"📄 {item.name}")
