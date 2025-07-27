#!/usr/bin/env python3
"""Test ResNet fix."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing ResNet fix...")

try:
    from src.models.backbones.resnet import ResNet18Backbone
    
    # Test without pretrained
    print("Testing ResNet18Backbone (pretrained=False)...")
    backbone = ResNet18Backbone(pretrained=False)
    print("✓ ResNet18Backbone created successfully")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = backbone(test_input)
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    
    print("🎉 ResNet fix successful!")
    
except Exception as e:
    print(f"❌ ResNet test failed: {e}")
    import traceback
    traceback.print_exc()
