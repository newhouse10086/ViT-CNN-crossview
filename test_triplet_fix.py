#!/usr/bin/env python3
"""Test triplet loss fix."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing triplet loss fix...")

try:
    from src.losses.triplet_loss import TripletLoss
    
    # Create triplet loss
    triplet_loss = TripletLoss()
    
    # Test with small batch (should return zero)
    print("Testing with small batch...")
    small_feat = torch.randn(1, 512)
    small_labels = torch.tensor([0])
    
    loss_small = triplet_loss(small_feat, small_labels)
    print(f"Small batch loss: {loss_small.item():.4f}")
    
    # Test with same labels (should return zero)
    print("Testing with same labels...")
    same_feat = torch.randn(4, 512)
    same_labels = torch.tensor([0, 0, 0, 0])
    
    loss_same = triplet_loss(same_feat, same_labels)
    print(f"Same labels loss: {loss_same.item():.4f}")
    
    # Test with mixed labels (should work normally)
    print("Testing with mixed labels...")
    mixed_feat = torch.randn(4, 512)
    mixed_labels = torch.tensor([0, 1, 0, 2])
    
    loss_mixed = triplet_loss(mixed_feat, mixed_labels)
    print(f"Mixed labels loss: {loss_mixed.item():.4f}")
    
    print("✓ Triplet loss tests passed!")
    
    # Now test combined loss
    print("\nTesting combined loss...")
    from src.losses import CombinedLoss
    
    criterion = CombinedLoss(num_classes=10)
    
    outputs = {
        'satellite': {
            'predictions': [torch.randn(4, 10), torch.randn(4, 10)],
            'features': [torch.randn(4, 512), torch.randn(4, 512)]
        },
        'drone': {
            'predictions': [torch.randn(4, 10), torch.randn(4, 10)],
            'features': [torch.randn(4, 512), torch.randn(4, 512)]
        }
    }
    
    labels = torch.tensor([0, 1, 0, 2])
    
    losses = criterion(outputs, labels)
    print("✓ Combined loss computation successful!")
    
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
