#!/usr/bin/env python3
"""Test model creation after fixing patch_size conflict."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_vit_creation():
    """Test ViT model creation with different patch sizes."""
    print("Testing ViT model creation...")
    
    try:
        from src.models.backbones.vit_pytorch import vit_small_patch16_224
        
        # Test default patch size
        print("1. Testing default patch_size=16...")
        model1 = vit_small_patch16_224()
        print(f"✓ Default model created, patch_size: {model1.patch_embed.patch_size}")
        
        # Test custom patch size
        print("2. Testing custom patch_size=1...")
        model2 = vit_small_patch16_224(
            patch_size=1,
            img_size=(16, 16),
            in_chans=768,
            embed_dim=768
        )
        print(f"✓ Custom model created, patch_size: {model2.patch_embed.patch_size}")
        
        return True
        
    except Exception as e:
        print(f"✗ ViT creation failed: {e}")
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
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_setup():
    """Test complete training setup."""
    print("\nTesting complete training setup...")
    
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
        
        return True
        
    except Exception as e:
        print(f"✗ Training setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Model Creation Test (After patch_size Fix)")
    print("=" * 60)
    
    success = True
    
    if not test_vit_creation():
        success = False
    
    if not test_model_creation():
        success = False
    
    if not test_training_setup():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The patch_size conflict has been resolved.")
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
