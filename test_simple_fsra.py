#!/usr/bin/env python3
"""Test Simple FSRA model."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_simple_fsra():
    """Test Simple FSRA model creation and forward pass."""
    print("Testing Simple FSRA model...")
    
    try:
        from src.models import create_model
        from src.losses import CombinedLoss
        from src.utils import load_config
        
        # Load Simple FSRA config
        config = load_config('config/simple_fsra_config.yaml')
        
        print(f"Model config:")
        print(f"  Name: {config['model']['name']}")
        print(f"  Classes: {config['model']['num_classes']}")
        print(f"  Regions: {config['model']['num_regions']}")
        print(f"  Views: {config['data']['views']}")
        
        # Create model
        model = create_model(config)
        model.eval()
        
        print(f"✓ Simple FSRA model created successfully")
        print(f"  Model type: {type(model).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Test forward pass
        batch_size = 4
        sat_images = torch.randn(batch_size, 3, 256, 256)
        drone_images = torch.randn(batch_size, 3, 256, 256)
        labels = torch.randint(0, config['model']['num_classes'], (batch_size,))
        
        print(f"\nTesting forward pass...")
        print(f"  Input shapes: sat={sat_images.shape}, drone={drone_images.shape}")
        
        with torch.no_grad():
            outputs = model(sat_images, drone_images)
        
        print(f"✓ Forward pass successful")
        print(f"  Output keys: {list(outputs.keys())}")
        
        # Check output structure in detail
        for view in ['satellite', 'drone']:
            if view in outputs and outputs[view] is not None:
                view_data = outputs[view]
                print(f"\n  {view.capitalize()} view:")
                
                if 'predictions' in view_data:
                    preds = view_data['predictions']
                    print(f"    Predictions: {type(preds)}")
                    if isinstance(preds, list):
                        for i, pred in enumerate(preds):
                            if isinstance(pred, torch.Tensor):
                                print(f"      [{i}]: shape={pred.shape}, ndim={pred.ndim}")
                                if pred.ndim == 1:
                                    print(f"           ❌ FOUND 1D TENSOR!")
                                elif pred.ndim == 2:
                                    print(f"           ✓ 2D tensor OK")
                
                if 'features' in view_data:
                    feats = view_data['features']
                    print(f"    Features: {type(feats)}")
                    if isinstance(feats, list):
                        for i, feat in enumerate(feats):
                            if isinstance(feat, torch.Tensor):
                                print(f"      [{i}]: shape={feat.shape}")
        
        # Test loss computation
        print(f"\nTesting loss computation...")
        criterion = CombinedLoss(num_classes=config['model']['num_classes'])
        
        try:
            losses = criterion(outputs, labels)
            print(f"✓ Loss computation successful!")
            
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.item():.4f}")
        except Exception as loss_error:
            print(f"❌ Loss computation failed: {loss_error}")
            
            # Debug the specific issue
            print("\nDebugging loss computation...")
            
            # Check satellite predictions
            if 'satellite' in outputs and outputs['satellite'] is not None:
                sat_preds = outputs['satellite']['predictions']
                print(f"Satellite predictions type: {type(sat_preds)}")
                
                if isinstance(sat_preds, list):
                    print(f"Satellite predictions list:")
                    for i, pred in enumerate(sat_preds):
                        print(f"  [{i}]: {pred.shape if isinstance(pred, torch.Tensor) else type(pred)}")
                        
                        # Check if any prediction is 1D
                        if isinstance(pred, torch.Tensor) and pred.ndim == 1:
                            print(f"    ⚠️  Found 1D tensor! This is the problem.")
                            print(f"    Expected: (batch_size, num_classes)")
                            print(f"    Got: {pred.shape}")
            
            return False
        
        # Test training step
        print(f"\nTesting training step...")
        from src.optimizers import create_optimizer_with_config
        
        train_config = {
            'training': {
                'learning_rate': 0.01,
                'optimizer': 'sgd'
            }
        }
        
        optimizer, _ = create_optimizer_with_config(model, train_config)
        
        model.train()
        optimizer.zero_grad()
        
        outputs = model(sat_images, drone_images)
        losses = criterion(outputs, labels)
        total_loss = losses['total']
        
        total_loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful!")
        print(f"  Loss: {total_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_view():
    """Test single view model."""
    print("\nTesting single view Simple FSRA...")
    
    try:
        from src.models.simple_fsra import SimpleFSRAModel
        
        # Create single view model
        model = SimpleFSRAModel(num_classes=10, use_pretrained=False)
        model.eval()
        
        # Test input
        x = torch.randn(2, 3, 256, 256)
        
        with torch.no_grad():
            predictions, features = model(x)
        
        print(f"✓ Single view model works!")
        print(f"  Predictions: {len(predictions)} items")
        print(f"  Features: {len(features)} items")
        
        for i, pred in enumerate(predictions):
            print(f"    Prediction [{i}]: {pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Single view test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Simple FSRA Model Test")
    print("=" * 60)
    
    success = True
    
    if not test_single_view():
        success = False
    
    if not test_simple_fsra():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("Simple FSRA model is working correctly!")
        print("\nYou can now run training with:")
        print("  python train.py --config config/simple_fsra_config.yaml --data-dir data --batch-size 8 --learning-rate 0.001 --num-epochs 10 --gpu-ids \"0\"")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
