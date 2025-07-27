#!/usr/bin/env python3
"""
Quick test to verify ViT module fix.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_vit_module():
    """Test the fixed ViT module."""
    print("🧪 Testing Fixed ViT Module")
    print("="*40)
    
    try:
        from src.models.vit_module import VisionTransformer
        
        # Create ViT with correct parameters
        vit = VisionTransformer(
            img_size=256,
            patch_size=10,
            embed_dim=768,
            depth=6,
            output_dim=100
        )
        
        print(f"✅ ViT created successfully!")
        print(f"   Patch size: {vit.patch_embed.patch_size}")
        print(f"   Num patches: {vit.patch_embed.num_patches}")
        print(f"   Expected: 256//10 = 25, so 25×25 = 625 patches")
        
        # Test forward pass
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        
        print(f"\n🔍 Testing forward pass...")
        print(f"   Input shape: {input_tensor.shape}")
        
        with torch.no_grad():
            output = vit(input_tensor)
        
        print(f"   Output shape: {output.shape}")
        print(f"   Expected: ({batch_size}, 100, 8, 8)")
        
        if output.shape == (batch_size, 100, 8, 8):
            print(f"✅ ViT forward pass successful!")
            return True
        else:
            print(f"❌ Output shape mismatch!")
            return False
            
    except Exception as e:
        print(f"❌ ViT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fsra_vit_model():
    """Test the complete FSRA ViT model."""
    print("\n🧪 Testing Complete FSRA ViT Model")
    print("="*50)
    
    try:
        from src.utils import load_config
        from src.models import create_model
        
        # Load config
        config = load_config("config/fsra_vit_improved_config.yaml")
        
        # Create model
        model = create_model(config)
        model.eval()
        
        print(f"✅ FSRA ViT model created successfully!")
        
        # Test forward pass
        batch_size = 2
        sat_images = torch.randn(batch_size, 3, 256, 256)
        drone_images = torch.randn(batch_size, 3, 256, 256)
        
        print(f"\n🔍 Testing model forward pass...")
        print(f"   Satellite input: {sat_images.shape}")
        print(f"   Drone input: {drone_images.shape}")
        
        with torch.no_grad():
            outputs = model(sat_images, drone_images)
        
        print(f"✅ Model forward pass successful!")
        
        # Check outputs
        if 'satellite' in outputs:
            sat_outputs = outputs['satellite']
            if 'predictions' in sat_outputs:
                predictions = sat_outputs['predictions']
                print(f"   Predictions: {len(predictions)} levels")
                for i, pred in enumerate(predictions):
                    if isinstance(pred, torch.Tensor):
                        print(f"     Level {i}: {pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ FSRA ViT model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🎯 ViT Module Fix Verification")
    print("="*50)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: ViT Module
    if test_vit_module():
        success_count += 1
    
    # Test 2: Complete Model
    if test_fsra_vit_model():
        success_count += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"🎊 TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print(f"🎉 All tests passed! ViT fix is working!")
        print(f"🚀 You can now run training without the shape error!")
    else:
        print(f"⚠️  Some tests failed. Please check the issues above.")
    
    return success_count == total_tests


if __name__ == "__main__":
    main()
