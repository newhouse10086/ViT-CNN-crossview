#!/usr/bin/env python3
"""Final test for all fixes."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Final test for all fixes...")

try:
    # Test triplet loss directly
    print("1. Testing triplet loss...")
    from src.losses.triplet_loss import TripletLoss
    
    triplet_loss = TripletLoss()
    
    # Test with proper batch
    feat = torch.randn(4, 512)
    labels = torch.tensor([0, 1, 0, 2])
    
    loss = triplet_loss(feat, labels)
    print(f"   Triplet loss: {loss.item():.4f}")
    
    # Test combined loss
    print("2. Testing combined loss...")
    from src.losses import CombinedLoss
    
    criterion = CombinedLoss(num_classes=10)
    
    outputs = {
        'satellite': {
            'predictions': [torch.randn(4, 10)],
            'features': [torch.randn(4, 512)]
        },
        'drone': {
            'predictions': [torch.randn(4, 10)],
            'features': [torch.randn(4, 512)]
        }
    }
    
    labels = torch.tensor([0, 1, 0, 2])
    
    losses = criterion(outputs, labels)
    print(f"   Combined loss: {losses['total'].item():.4f}")
    
    # Test model creation
    print("3. Testing model creation...")
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
    print("   Model created successfully")
    
    # Test forward pass
    print("4. Testing forward pass...")
    sat = torch.randn(2, 3, 224, 224)
    drone = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(sat, drone)
    
    print("   Forward pass successful")
    
    # Test training step
    print("5. Testing training step...")
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
    
    outputs = model(sat, drone)
    test_labels = torch.tensor([0, 1])
    losses = criterion(outputs, test_labels)
    
    losses['total'].backward()
    optimizer.step()
    
    print(f"   Training step successful, loss: {losses['total'].item():.4f}")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("The system is ready for training!")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
