#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test all important imports."""
    print("Testing imports...")
    
    try:
        print("Testing basic Python imports...")
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import yaml
        print("✓ Basic imports successful")
    except ImportError as e:
        print(f"✗ Basic import failed: {e}")
        return False
    
    try:
        print("Testing src.utils imports...")
        from src.utils import load_config, save_config, merge_configs
        from src.utils import setup_logger, get_logger
        from src.utils import MetricsCalculator, AverageMeter
        from src.utils import TrainingVisualizer
        print("✓ Utils imports successful")
    except ImportError as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    try:
        print("Testing src.models imports...")
        from src.models import create_model
        print("✓ Models imports successful")
    except ImportError as e:
        print(f"✗ Models import failed: {e}")
        return False
    
    try:
        print("Testing src.datasets imports...")
        from src.datasets import make_dataloader
        print("✓ Datasets imports successful")
    except ImportError as e:
        print(f"✗ Datasets import failed: {e}")
        return False
    
    try:
        print("Testing src.losses imports...")
        from src.losses import CombinedLoss, TripletLoss
        print("✓ Losses imports successful")
    except ImportError as e:
        print(f"✗ Losses import failed: {e}")
        return False
    
    try:
        print("Testing src.optimizers imports...")
        from src.optimizers import create_optimizer, create_scheduler
        print("✓ Optimizers imports successful")
    except ImportError as e:
        print(f"✗ Optimizers import failed: {e}")
        return False
    
    print("\n🎉 All imports successful!")
    return True

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        # Test config loading
        from src.utils import load_config, save_config
        
        test_config = {
            'model': {'name': 'ViTCNN', 'num_classes': 10},
            'data': {'batch_size': 16},
            'training': {'num_epochs': 100}
        }
        
        # Save and load config
        save_config(test_config, 'test_config.yaml')
        loaded_config = load_config('test_config.yaml')
        
        # Cleanup
        import os
        os.remove('test_config.yaml')
        
        print("✓ Config save/load works")
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False
    
    try:
        # Test model creation
        from src.models import create_model
        
        config = {
            'model': {
                'name': 'ViTCNN',
                'num_classes': 10,
                'use_pretrained_resnet': False,
                'use_pretrained_vit': False
            },
            'data': {'views': 2}
        }
        
        model = create_model(config)
        print("✓ Model creation works")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    print("✓ Basic functionality test passed")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Import and Functionality Test")
    print("=" * 60)
    
    success = True
    
    if not test_imports():
        success = False
    
    if not test_basic_functionality():
        success = False
    
    if success:
        print("\n✅ All tests passed! The project is ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)
