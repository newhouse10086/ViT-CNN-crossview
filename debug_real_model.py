#!/usr/bin/env python3
"""Debug real model with actual config."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_real_model():
    """Debug with the actual config used in training."""
    print("Debugging real model with actual config...")
    
    try:
        from src.models import create_model
        from src.losses import CombinedLoss
        from src.utils import load_config
        
        # Load the actual config
        config = load_config('config/default_config.yaml')
        
        print(f"Model config:")
        print(f"  Name: {config['model']['name']}")
        print(f"  Classes: {config['model']['num_classes']}")
        print(f"  Use pretrained ResNet: {config['model']['use_pretrained_resnet']}")
        print(f"  Use pretrained ViT: {config['model']['use_pretrained_vit']}")
        
        # Create model
        model = create_model(config)
        model.eval()
        
        print(f"✓ Model created successfully")
        
        # Create test inputs with the same size as training
        batch_size = 2  # Small batch for testing
        sat_images = torch.randn(batch_size, 3, 256, 256)
        drone_images = torch.randn(batch_size, 3, 256, 256)
        labels = torch.randint(0, config['model']['num_classes'], (batch_size,))
        
        print(f"Input shapes:")
        print(f"  Satellite: {sat_images.shape}")
        print(f"  Drone: {drone_images.shape}")
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
                            elif isinstance(item, list):
                                print(f"      [{i}]: list of {len(item)} items")
                                for j, subitem in enumerate(item):
                                    if isinstance(subitem, torch.Tensor):
                                        print(f"        [{j}]: tensor {subitem.shape}")
                            else:
                                print(f"      [{i}]: {type(item)}")
                    elif isinstance(sub_value, torch.Tensor):
                        print(f"    {sub_key}: tensor {sub_value.shape}")
                    else:
                        print(f"    {sub_key}: {type(sub_value)}")
        
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
            if 'satellite' in outputs and 'predictions' in outputs['satellite']:
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
                
                # Try manual CrossEntropyLoss
                try:
                    ce_loss = torch.nn.CrossEntropyLoss()
                    if isinstance(sat_preds, list) and len(sat_preds) > 0:
                        test_pred = sat_preds[0]
                        if isinstance(test_pred, torch.Tensor):
                            print(f"Testing first prediction: {test_pred.shape}")
                            if test_pred.ndim == 2:
                                manual_loss = ce_loss(test_pred, labels)
                                print(f"✓ Manual CE loss works: {manual_loss.item():.4f}")
                            else:
                                print(f"❌ Prediction has wrong dimensions: {test_pred.ndim}D")
                except Exception as ce_error:
                    print(f"❌ Manual CE loss failed: {ce_error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    print("=" * 60)
    print("Real Model Debug")
    print("=" * 60)
    
    success = debug_real_model()
    
    print("\n" + "=" * 60)
    if success:
        print("Debug completed. Check the output above for issues.")
    else:
        print("Debug failed. Check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
