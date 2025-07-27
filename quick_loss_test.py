#!/usr/bin/env python3
"""Quick loss test."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Quick loss test...")

try:
    from src.losses import CombinedLoss
    
    # Create simple test data
    criterion = CombinedLoss(num_classes=10)
    
    # Create mock outputs that match expected structure
    outputs = {
        'satellite': {
            'predictions': [torch.randn(2, 10), torch.randn(2, 10)],  # List of prediction tensors
            'features': [torch.randn(2, 512), torch.randn(2, 512)]
        },
        'drone': {
            'predictions': [torch.randn(2, 10), torch.randn(2, 10)],
            'features': [torch.randn(2, 512), torch.randn(2, 512)]
        }
    }
    
    labels = torch.randint(0, 10, (2,))
    
    print("Testing loss computation...")
    losses = criterion(outputs, labels)
    
    print("✓ Loss computation successful!")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
    
    print("🎉 Test passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
