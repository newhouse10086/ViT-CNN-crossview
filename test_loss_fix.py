#!/usr/bin/env python3
"""Test loss function fix."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_loss_computation():
    """Test loss computation with fixed CombinedLoss."""
    print("Testing loss computation...")
    
    try:
        from src.models import create_model
        from src.losses import CombinedLoss
        
        # Create test config
        config = {
            'model': {
                'name': 'ViTCNN',
                'num_classes': 10,
                'use_pretrained_resnet': False,
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
        
        # Create model and loss
        model = create_model(config)
        criterion = CombinedLoss(num_classes=10)
        
        model.eval()
        
        print(f"✓ Model and loss created")
        
        # Create test inputs
        batch_size = 2
        sat_images = torch.randn(batch_size, 3, 256, 256)
        drone_images = torch.randn(batch_size, 3, 256, 256)
        labels = torch.randint(0, 10, (batch_size,))
        
        print(f"Input shapes:")
        print(f"  Satellite images: {sat_images.shape}")
        print(f"  Drone images: {drone_images.shape}")
        print(f"  Labels: {labels.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(sat_images, drone_images)
        
        print(f"\nModel outputs structure:")
        for key, value in outputs.items():
            print(f"  {key}:")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        print(f"    {sub_key}: list of {len(sub_value)} items")
                        for i, item in enumerate(sub_value):
                            if isinstance(item, torch.Tensor):
                                print(f"      [{i}]: tensor {item.shape}")
                            else:
                                print(f"      [{i}]: {type(item)}")
                    elif isinstance(sub_value, torch.Tensor):
                        print(f"    {sub_key}: tensor {sub_value.shape}")
                    else:
                        print(f"    {sub_key}: {type(sub_value)}")
        
        # Test loss computation
        print(f"\nTesting loss computation...")
        losses = criterion(outputs, labels)
        
        print(f"✓ Loss computation successful!")
        print(f"Loss values:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test a complete training step."""
    print("\nTesting complete training step...")
    
    try:
        from src.models import create_model
        from src.losses import CombinedLoss
        from src.optimizers import create_optimizer_with_config
        
        # Create test config
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
        
        model.train()
        
        # Create test batch
        batch_size = 2
        sat_images = torch.randn(batch_size, 3, 256, 256)
        drone_images = torch.randn(batch_size, 3, 256, 256)
        labels = torch.randint(0, 10, (batch_size,))
        
        # Training step
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sat_images, drone_images)
        
        # Compute loss
        losses = criterion(outputs, labels)
        total_loss = losses['total']
        
        print(f"Loss before backward: {total_loss.item():.4f}")
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful!")
        print(f"Total loss: {total_loss.item():.4f}")
        
        # Test another step to make sure gradients are working
        optimizer.zero_grad()
        outputs2 = model(sat_images, drone_images)
        losses2 = criterion(outputs2, labels)
        total_loss2 = losses2['total']
        total_loss2.backward()
        optimizer.step()
        
        print(f"✓ Second training step successful!")
        print(f"Second step loss: {total_loss2.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Loss Function Fix Test")
    print("=" * 60)
    
    success = True
    
    if not test_loss_computation():
        success = False
    
    if not test_training_step():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The loss function issues have been fixed.")
        print("You can now run training:")
        print("  python train.py --config config/default_config.yaml --data-dir data --batch-size 8 --learning-rate 0.001 --num-epochs 10 --gpu-ids \"0\"")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
