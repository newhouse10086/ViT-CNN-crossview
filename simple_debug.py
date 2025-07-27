#!/usr/bin/env python3
"""Simple debug for 1D tensor issue."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Simple debug for 1D tensor issue...")

try:
    # Test ClassBlock directly
    from src.models.components import ClassBlock
    
    print("Testing ClassBlock...")
    
    # Create ClassBlock
    class_block = ClassBlock(input_dim=512, class_num=10, return_f=True)
    
    # Test input
    test_input = torch.randn(4, 512)
    
    output = class_block(test_input)
    print(f"ClassBlock output type: {type(output)}")
    
    if isinstance(output, list):
        print(f"Output list length: {len(output)}")
        for i, item in enumerate(output):
            if isinstance(item, torch.Tensor):
                print(f"  [{i}]: shape={item.shape}, ndim={item.ndim}")
            else:
                print(f"  [{i}]: type={type(item)}")
    else:
        print(f"Output shape: {output.shape}")
    
    print("✓ ClassBlock test completed")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
