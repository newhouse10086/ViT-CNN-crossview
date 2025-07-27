#!/usr/bin/env python3
"""Debug prediction shapes."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_predictions():
    """Debug prediction shapes in detail."""
    print("Debugging prediction shapes...")
    
    try:
        from src.models import create_model
        from src.utils import load_config
        
        # Load config
        config = load_config('config/default_config.yaml')
        
        # Create model
        model = create_model(config)
        model.eval()
        
        print(f"✓ Model created")
        
        # Create test inputs
        batch_size = 4
        sat_images = torch.randn(batch_size, 3, 256, 256)
        drone_images = torch.randn(batch_size, 3, 256, 256)
        
        print(f"Input shapes: sat={sat_images.shape}, drone={drone_images.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(sat_images, drone_images)
        
        print(f"\nDetailed output analysis:")
        
        # Check satellite predictions
        if 'satellite' in outputs and outputs['satellite'] is not None:
            sat_preds = outputs['satellite']['predictions']
            print(f"\nSatellite predictions:")
            print(f"  Type: {type(sat_preds)}")
            
            if isinstance(sat_preds, list):
                print(f"  List length: {len(sat_preds)}")
                for i, pred in enumerate(sat_preds):
                    if isinstance(pred, torch.Tensor):
                        print(f"    [{i}]: shape={pred.shape}, ndim={pred.ndim}")
                        if pred.ndim == 1:
                            print(f"         ❌ FOUND 1D TENSOR! This is the problem!")
                        elif pred.ndim == 2:
                            print(f"         ✓ 2D tensor OK")
                        else:
                            print(f"         ⚠️  Unexpected dimensions: {pred.ndim}")
                    else:
                        print(f"    [{i}]: type={type(pred)}")
            else:
                print(f"  Not a list: {type(sat_preds)}")
        
        # Check drone predictions
        if 'drone' in outputs and outputs['drone'] is not None:
            drone_preds = outputs['drone']['predictions']
            print(f"\nDrone predictions:")
            print(f"  Type: {type(drone_preds)}")
            
            if isinstance(drone_preds, list):
                print(f"  List length: {len(drone_preds)}")
                for i, pred in enumerate(drone_preds):
                    if isinstance(pred, torch.Tensor):
                        print(f"    [{i}]: shape={pred.shape}, ndim={pred.ndim}")
                        if pred.ndim == 1:
                            print(f"         ❌ FOUND 1D TENSOR! This is the problem!")
                        elif pred.ndim == 2:
                            print(f"         ✓ 2D tensor OK")
                        else:
                            print(f"         ⚠️  Unexpected dimensions: {pred.ndim}")
                    else:
                        print(f"    [{i}]: type={type(pred)}")
            else:
                print(f"  Not a list: {type(drone_preds)}")
        
        # Test loss computation to see where exactly it fails
        print(f"\nTesting loss computation...")
        from src.losses import CombinedLoss
        
        criterion = CombinedLoss(num_classes=config['model']['num_classes'])
        labels = torch.randint(0, config['model']['num_classes'], (batch_size,))
        
        try:
            losses = criterion(outputs, labels)
            print(f"✓ Loss computation successful!")
        except Exception as loss_error:
            print(f"❌ Loss computation failed: {loss_error}")
            
            # Try to isolate the problem
            print(f"\nTrying to isolate the problem...")
            
            # Test classification loss manually
            try:
                ce_loss = torch.nn.CrossEntropyLoss()
                
                # Test satellite predictions one by one
                if 'satellite' in outputs and outputs['satellite'] is not None:
                    sat_preds = outputs['satellite']['predictions']
                    if isinstance(sat_preds, list):
                        for i, pred in enumerate(sat_preds):
                            try:
                                if isinstance(pred, torch.Tensor):
                                    print(f"Testing sat pred [{i}]: shape={pred.shape}")
                                    test_loss = ce_loss(pred, labels)
                                    print(f"  ✓ Works: {test_loss.item():.4f}")
                            except Exception as pred_error:
                                print(f"  ❌ Failed: {pred_error}")
                                print(f"     Pred shape: {pred.shape if isinstance(pred, torch.Tensor) else type(pred)}")
                                print(f"     Labels shape: {labels.shape}")
                
            except Exception as manual_error:
                print(f"Manual test failed: {manual_error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    print("=" * 60)
    print("Prediction Shape Debug")
    print("=" * 60)
    
    success = debug_predictions()
    
    print("\n" + "=" * 60)
    if success:
        print("Debug completed. Check the output above for 1D tensors.")
    else:
        print("Debug failed. Check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
