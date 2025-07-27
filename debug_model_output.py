#!/usr/bin/env python3
"""Debug model output shapes."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_model_output():
    """Debug model output shapes."""
    print("Debugging model output shapes...")
    
    try:
        from src.models import create_model
        from src.losses import CombinedLoss
        
        # Create test config
        config = {
            'model': {
                'name': 'ViTCNN',
                'num_classes': 701,
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
        
        # Create model
        model = create_model(config)
        model.eval()
        
        print(f"✓ Model created: {type(model).__name__}")
        
        # Create test inputs
        batch_size = 2
        sat_images = torch.randn(batch_size, 3, 256, 256)
        drone_images = torch.randn(batch_size, 3, 256, 256)
        labels = torch.randint(0, 701, (batch_size,))
        
        print(f"Input shapes:")
        print(f"  Satellite images: {sat_images.shape}")
        print(f"  Drone images: {drone_images.shape}")
        print(f"  Labels: {labels.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(sat_images, drone_images)
        
        print(f"\nModel outputs:")
        print(f"  Output type: {type(outputs)}")
        print(f"  Output keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'Not a dict'}")
        
        # Check each output
        for key, value in outputs.items():
            print(f"\n  {key}:")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        print(f"    {sub_key}: {sub_value.shape}")
                    elif isinstance(sub_value, list):
                        print(f"    {sub_key}: list of {len(sub_value)} tensors")
                        for i, tensor in enumerate(sub_value):
                            if isinstance(tensor, torch.Tensor):
                                print(f"      [{i}]: {tensor.shape}")
                    else:
                        print(f"    {sub_key}: {type(sub_value)}")
            elif isinstance(value, torch.Tensor):
                print(f"    tensor: {value.shape}")
            else:
                print(f"    type: {type(value)}")
        
        # Test loss computation
        print(f"\nTesting loss computation...")
        criterion = CombinedLoss(num_classes=701)
        
        try:
            losses = criterion(outputs, labels)
            print(f"✓ Loss computation successful!")
            print(f"Loss keys: {list(losses.keys())}")
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.item():.4f}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"❌ Loss computation failed: {e}")
            
            # Debug the specific tensors causing issues
            print("\nDebugging loss computation...")
            
            if 'satellite' in outputs and outputs['satellite'] is not None:
                sat_preds = outputs['satellite']['predictions']
                print(f"Satellite predictions type: {type(sat_preds)}")
                if isinstance(sat_preds, list):
                    print(f"Satellite predictions list length: {len(sat_preds)}")
                    for i, pred in enumerate(sat_preds):
                        print(f"  Prediction {i}: {pred.shape if isinstance(pred, torch.Tensor) else type(pred)}")
                elif isinstance(sat_preds, torch.Tensor):
                    print(f"Satellite predictions shape: {sat_preds.shape}")
                
                # Try manual loss computation
                try:
                    if isinstance(sat_preds, list) and len(sat_preds) > 0:
                        test_pred = sat_preds[0]
                    else:
                        test_pred = sat_preds
                    
                    print(f"Testing CrossEntropyLoss with:")
                    print(f"  Prediction shape: {test_pred.shape}")
                    print(f"  Labels shape: {labels.shape}")
                    print(f"  Labels dtype: {labels.dtype}")
                    print(f"  Prediction dtype: {test_pred.dtype}")
                    
                    ce_loss = torch.nn.CrossEntropyLoss()
                    loss_value = ce_loss(test_pred, labels)
                    print(f"✓ Manual CrossEntropyLoss works: {loss_value.item():.4f}")
                    
                except Exception as ce_error:
                    print(f"❌ Manual CrossEntropyLoss failed: {ce_error}")
                    
                    # Check if we need to reshape
                    if isinstance(test_pred, torch.Tensor):
                        print(f"Prediction tensor details:")
                        print(f"  Shape: {test_pred.shape}")
                        print(f"  Ndim: {test_pred.ndim}")
                        print(f"  Min value: {test_pred.min().item():.4f}")
                        print(f"  Max value: {test_pred.max().item():.4f}")
                        
                        # Try reshaping if needed
                        if test_pred.ndim == 1:
                            print("Prediction is 1D, this is the problem!")
                            print("CrossEntropyLoss expects 2D input (batch_size, num_classes)")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    print("=" * 60)
    print("Model Output Shape Debug")
    print("=" * 60)
    
    success = debug_model_output()
    
    print("\n" + "=" * 60)
    if success:
        print("Debug completed. Check the output above for shape issues.")
    else:
        print("Debug failed. Check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
