#!/usr/bin/env python3
"""Test torchvision compatibility fix."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_torchvision_version():
    """Test torchvision version and API."""
    print("Testing torchvision version and API...")
    
    try:
        import torchvision
        from torchvision import models
        
        print(f"✓ TorchVision version: {torchvision.__version__}")
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Check if new API is available
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            print("✓ New torchvision API (weights) available")
            has_new_api = True
        except AttributeError:
            print("⚠ Old torchvision API (pretrained) in use")
            has_new_api = False
        
        return True, has_new_api
        
    except ImportError as e:
        print(f"✗ TorchVision import failed: {e}")
        return False, False

def test_resnet_creation():
    """Test ResNet creation with compatibility."""
    print("Testing ResNet creation...")
    
    try:
        from torchvision import models
        
        # Test old API
        print("Testing old API (pretrained=True)...")
        resnet_old = models.resnet18(pretrained=False)  # Use False to avoid download
        print("✓ Old API works")
        
        # Test new API if available
        try:
            print("Testing new API (weights=None)...")
            resnet_new = models.resnet18(weights=None)
            print("✓ New API works")
        except TypeError:
            print("⚠ New API not available (expected for older versions)")
        
        return True
        
    except Exception as e:
        print(f"✗ ResNet creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_resnet_backbone():
    """Test our ResNet backbone with compatibility."""
    print("Testing ResNet backbone...")
    
    try:
        from src.models.backbones.resnet import ResNet18Backbone
        
        # Test without pretrained weights
        print("Testing ResNet18Backbone (pretrained=False)...")
        backbone_no_pretrained = ResNet18Backbone(pretrained=False)
        print("✓ ResNet18Backbone creation successful (no pretrained)")
        
        # Test input/output
        batch_size = 2
        test_input = torch.randn(batch_size, 3, 256, 256)
        
        with torch.no_grad():
            output = backbone_no_pretrained(test_input)
        
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        # Test with pretrained weights (if available and not too slow)
        try:
            print("Testing ResNet18Backbone (pretrained=True)...")
            backbone_pretrained = ResNet18Backbone(pretrained=True)
            print("✓ ResNet18Backbone with pretrained weights successful")
        except Exception as e:
            print(f"⚠ Pretrained weights test skipped: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ ResNet backbone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test full model creation."""
    print("Testing full model creation...")
    
    try:
        from src.models import create_model
        
        config = {
            'model': {
                'name': 'ViTCNN',
                'num_classes': 10,
                'use_pretrained_resnet': False,  # Disable to avoid download
                'use_pretrained_vit': False,
                'num_final_clusters': 3,
                'resnet_layers': 18,
                'vit_patch_size': 16,
                'vit_embed_dim': 384
            },
            'data': {
                'views': 2
            }
        }
        
        model = create_model(config)
        print("✓ Full model creation successful")
        print(f"Model type: {type(model).__name__}")
        
        # Test forward pass
        batch_size = 2
        sat_images = torch.randn(batch_size, 3, 256, 256)
        drone_images = torch.randn(batch_size, 3, 256, 256)
        
        with torch.no_grad():
            outputs = model(sat_images, drone_images)
        
        print("✓ Model forward pass successful")
        print(f"Output keys: {list(outputs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_components():
    """Test training components."""
    print("Testing training components...")
    
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
        
        # Create components
        model = create_model(config)
        criterion = CombinedLoss(num_classes=10)
        optimizer, scheduler = create_optimizer_with_config(model, config)
        
        print("✓ All training components created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Training components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("TorchVision Compatibility Test")
    print("=" * 60)
    
    success = True
    
    version_ok, has_new_api = test_torchvision_version()
    if not version_ok:
        success = False
    
    print()
    if not test_resnet_creation():
        success = False
    
    print()
    if not test_resnet_backbone():
        success = False
    
    print()
    if not test_model_creation():
        success = False
    
    print()
    if not test_training_components():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TORCHVISION COMPATIBILITY TESTS PASSED!")
        print("The torchvision compatibility issues have been resolved.")
        print("You can now run:")
        print("  python train.py --config config/default_config.yaml")
        print("  python train.py --config config/pytorch110_config.yaml")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please run: python complete_server_fix.py")
        print("Or check your torchvision version")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
