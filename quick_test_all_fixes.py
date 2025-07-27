#!/usr/bin/env python3
"""Quick test for all fixes."""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test critical imports."""
    print("Testing imports...")
    
    try:
        from src.utils import RankingMetricsCalculator, plot_roc_curves, log_system_info, validate_config
        from src.optimizers import create_optimizer_with_config
        print("✓ All critical imports OK")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_weights_init():
    """Test weights initialization."""
    print("Testing weights initialization...")
    
    try:
        from src.models.components import weights_init_classifier, weights_init_kaiming
        
        # Test with Linear layer without bias
        linear_no_bias = nn.Linear(10, 5, bias=False)
        weights_init_classifier(linear_no_bias)
        weights_init_kaiming(linear_no_bias)
        print("✓ Weights initialization OK")
        return True
    except Exception as e:
        print(f"✗ Weights init failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("Testing model creation...")
    
    try:
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
        print("✓ Model creation OK")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def main():
    """Main test."""
    print("=" * 50)
    print("Quick Test for All Fixes")
    print("=" * 50)
    
    all_good = True
    
    if not test_imports():
        all_good = False
    
    if not test_weights_init():
        all_good = False
    
    if not test_model_creation():
        all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 ALL FIXES WORKING!")
        print("You can now run train.py successfully.")
    else:
        print("❌ Some issues remain.")
    print("=" * 50)

if __name__ == "__main__":
    main()
