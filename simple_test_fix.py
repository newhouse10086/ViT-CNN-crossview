#!/usr/bin/env python3
"""Simple test for the fix."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing model fix...")

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
    print("✓ Model created")
    
    # Test forward
    sat = torch.randn(1, 3, 256, 256)
    drone = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        outputs = model(sat, drone)
    
    print("✓ Forward pass successful")
    print(f"Output keys: {list(outputs.keys())}")
    
    if 'satellite' in outputs and 'predictions' in outputs['satellite']:
        preds = outputs['satellite']['predictions']
        print(f"Predictions type: {type(preds)}")
        if isinstance(preds, list):
            print(f"Predictions list length: {len(preds)}")
            for i, p in enumerate(preds):
                print(f"  [{i}]: {p.shape if isinstance(p, torch.Tensor) else type(p)}")
    
    print("🎉 Test passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
