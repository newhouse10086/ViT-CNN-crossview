#!/usr/bin/env python3
"""Test weights initialization functions."""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_weights_init():
    """Test weights initialization functions."""
    print("Testing weights initialization functions...")
    
    try:
        from src.models.components import weights_init_classifier, weights_init_kaiming
        
        # Test with Linear layer with bias
        print("1. Testing Linear layer with bias...")
        linear_with_bias = nn.Linear(10, 5, bias=True)
        weights_init_classifier(linear_with_bias)
        weights_init_kaiming(linear_with_bias)
        print("✓ Linear with bias - OK")
        
        # Test with Linear layer without bias
        print("2. Testing Linear layer without bias...")
        linear_no_bias = nn.Linear(10, 5, bias=False)
        weights_init_classifier(linear_no_bias)
        weights_init_kaiming(linear_no_bias)
        print("✓ Linear without bias - OK")
        
        # Test with Conv layer
        print("3. Testing Conv layer...")
        conv_layer = nn.Conv2d(3, 64, 3, bias=True)
        weights_init_kaiming(conv_layer)
        print("✓ Conv layer - OK")
        
        # Test with BatchNorm layer
        print("4. Testing BatchNorm layer...")
        bn_layer = nn.BatchNorm2d(64)
        weights_init_kaiming(bn_layer)
        print("✓ BatchNorm layer - OK")
        
        return True
        
    except Exception as e:
        print(f"✗ Weights initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test full model creation."""
    print("\nTesting full model creation...")
    
    try:
        from src.models import create_model
        
        config = {
            'model': {
                'name': 'ViTCNN',
                'num_classes': 10,
                'use_pretrained_resnet': False,
                'use_pretrained_vit': False,
                'num_final_clusters': 3,
                'resnet_layers': 18,
                'vit_patch_size': 16,
                'vit_embed_dim': 768
            },
            'data': {
                'views': 2
            }
        }
        
        model = create_model(config)
        print("✓ Full model creation successful")
        print(f"Model type: {type(model).__name__}")
        
        # Test forward pass with dummy data
        print("Testing forward pass...")
        batch_size = 2
        sat_images = torch.randn(batch_size, 3, 256, 256)
        drone_images = torch.randn(batch_size, 3, 256, 256)
        
        with torch.no_grad():
            outputs = model(sat_images, drone_images)
        
        print("✓ Forward pass successful")
        print(f"Output keys: {list(outputs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation/forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_components():
    """Test training components."""
    print("\nTesting training components...")
    
    try:
        from src.models import create_model
        from src.losses import CombinedLoss
        from src.optimizers import create_optimizer_with_config
        
        config = {
            'model': {
                'name': 'ViTCNN',
                'num_classes': 10,
                'use_pretrained_resnet': False,
                'use_pretrained_vit': False
            },
            'data': {
                'views': 2
            },
            'training': {
                'learning_rate': 0.01,
                'weight_decay': 0.0001,
                'optimizer': 'sgd',
                'scheduler': 'step'
            }
        }
        
        # Create model
        model = create_model(config)
        print("✓ Model created")
        
        # Create loss
        criterion = CombinedLoss(num_classes=10)
        print("✓ Loss function created")
        
        # Create optimizer
        optimizer, scheduler = create_optimizer_with_config(model, config)
        print("✓ Optimizer and scheduler created")
        
        # Test a mini training step
        print("Testing mini training step...")
        batch_size = 2
        sat_images = torch.randn(batch_size, 3, 256, 256)
        drone_images = torch.randn(batch_size, 3, 256, 256)
        labels = torch.randint(0, 10, (batch_size,))
        
        # Forward pass
        outputs = model(sat_images, drone_images)
        
        # Compute loss
        losses = criterion(outputs, labels)
        total_loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print("✓ Mini training step successful")
        print(f"Loss: {total_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Weights Initialization and Model Test")
    print("=" * 60)
    
    success = True
    
    if not test_weights_init():
        success = False
    
    if not test_model_creation():
        success = False
    
    if not test_training_components():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The weights initialization issue has been resolved.")
        print("You can now run:")
        print("  python train.py --config config/default_config.yaml")
        print("  python train.py --create-dummy-data --experiment-name test")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above.")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
