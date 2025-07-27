#!/usr/bin/env python3
"""Test model fix for shape issues."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_model_forward():
    """Test model forward pass."""
    print("Testing model forward pass...")
    
    try:
        from src.models import create_model
        from src.losses import CombinedLoss
        
        # Create test config
        config = {
            'model': {
                'name': 'ViTCNN',
                'num_classes': 10,  # Use smaller number for testing
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
        
        # Create model
        model = create_model(config)
        model.eval()
        
        print(f"✓ Model created: {type(model).__name__}")
        
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
        
        print(f"\nModel outputs:")
        print(f"  Output type: {type(outputs)}")
        print(f"  Output keys: {list(outputs.keys())}")
        
        # Check satellite outputs
        if 'satellite' in outputs:
            sat_out = outputs['satellite']
            if 'predictions' in sat_out:
                sat_preds = sat_out['predictions']
                print(f"\nSatellite predictions:")
                print(f"  Type: {type(sat_preds)}")
                if isinstance(sat_preds, list):
                    print(f"  List length: {len(sat_preds)}")
                    for i, pred in enumerate(sat_preds):
                        if isinstance(pred, torch.Tensor):
                            print(f"    [{i}]: {pred.shape}")
                        else:
                            print(f"    [{i}]: {type(pred)}")
                elif isinstance(sat_preds, torch.Tensor):
                    print(f"  Tensor shape: {sat_preds.shape}")
        
        # Test loss computation
        print(f"\nTesting loss computation...")
        criterion = CombinedLoss(num_classes=10)
        
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
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful!")
        print(f"Total loss: {total_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Model Fix Test")
    print("=" * 60)
    
    success = True
    
    if not test_model_forward():
        success = False
    
    if not test_training_step():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The model shape issues have been fixed.")
        print("You can now run training:")
        print("  python train.py --config config/default_config.yaml --data-dir data --batch-size 8 --learning-rate 0.001 --num-epochs 10 --gpu-ids \"0\"")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
