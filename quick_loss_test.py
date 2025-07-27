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
    batch_size = 4  # Use larger batch size for triplet loss
    outputs = {
        'satellite': {
            'predictions': [torch.randn(batch_size, 10), torch.randn(batch_size, 10)],  # List of prediction tensors
            'features': [torch.randn(batch_size, 512), torch.randn(batch_size, 512)]
        },
        'drone': {
            'predictions': [torch.randn(batch_size, 10), torch.randn(batch_size, 10)],
            'features': [torch.randn(batch_size, 512), torch.randn(batch_size, 512)]
        }
    }

    # Ensure we have different labels for triplet loss
    labels = torch.tensor([0, 1, 0, 2])  # Mixed labels
    
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
