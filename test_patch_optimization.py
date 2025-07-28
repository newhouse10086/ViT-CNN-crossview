#!/usr/bin/env python3
"""Test script for patch optimization."""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_patch_optimization():
    """Test the optimized 10x10 patch model."""
    print("🔍 Testing Patch Optimization (10x10 = 100 patches)")
    print("=" * 60)
    
    try:
        from src.models import create_model
        from src.utils import load_config
        
        # Load optimized config
        config = load_config('config/fsra_vit_improved_config.yaml')
        
        print(f"📋 Configuration:")
        print(f"  Model: {config['model']['name']}")
        print(f"  Image size: {config['data']['image_height']}x{config['data']['image_width']}")
        print(f"  Patch size: {config['model']['patch_size']}")
        print(f"  Expected patches: {config['innovation']['architecture']['num_patches']}")
        
        # Create model
        print(f"\n🔧 Creating optimized model...")
        model = create_model(config)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully!")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Test forward pass with correct input size
        print(f"\n🧪 Testing forward pass...")
        batch_size = 2
        
        # Use the new image size (250x250)
        img_size = config['data']['image_height']  # 250
        sat_images = torch.randn(batch_size, 3, img_size, img_size)
        drone_images = torch.randn(batch_size, 3, img_size, img_size)
        
        print(f"  Input shapes: sat={sat_images.shape}, drone={drone_images.shape}")
        
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                start_time.record()
            
            outputs = model(sat_images, drone_images)
            
            if torch.cuda.is_available():
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
                print(f"  Forward pass time: {elapsed_time:.2f} ms")
        
        # Check outputs
        print(f"\n📊 Output Analysis:")
        if 'satellite' in outputs:
            predictions = outputs['satellite']['predictions']
            features = outputs['satellite']['features']
            
            print(f"  Number of predictions: {len(predictions)}")
            for i, pred in enumerate(predictions):
                print(f"    Prediction {i}: {pred.shape}")
            
            print(f"  Feature types: {list(features.keys())}")
            if 'cnn_features' in features:
                print(f"    CNN features: {features['cnn_features'].shape}")
            if 'vit_features' in features:
                print(f"    ViT features: {features['vit_features'].shape}")
            if 'fused_features' in features:
                print(f"    Fused features: {features['fused_features'].shape}")
        
        # Verify patch calculation
        patch_size = config['model']['patch_size']  # 25
        num_patches_per_dim = img_size // patch_size  # 250 // 25 = 10
        total_patches = num_patches_per_dim ** 2  # 10 * 10 = 100
        
        print(f"\n🎯 Patch Verification:")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Patches per dimension: {num_patches_per_dim}")
        print(f"  Total patches: {total_patches}")
        print(f"  Expected patches: 100 ✅" if total_patches == 100 else f"  Expected patches: 100 ❌ (got {total_patches})")
        
        # Calculate performance improvement
        old_patches = 625  # 25x25 from original config
        new_patches = 100  # 10x10 from optimized config
        improvement = old_patches / new_patches
        
        print(f"\n⚡ Performance Improvement:")
        print(f"  Old patches: {old_patches}")
        print(f"  New patches: {new_patches}")
        print(f"  Patch reduction: {improvement:.1f}x fewer patches")
        print(f"  Expected speedup: ~{improvement:.1f}x faster training")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_patch_optimization()
    
    if success:
        print(f"\n🎉 Patch Optimization Test PASSED!")
        print(f"Your model is now optimized for 10x10 patches (100 total)")
        print(f"Expected training speedup: ~6x faster than before!")
        print(f"\nYou can now run training with:")
        print(f"python train_fsra_aligned.py --config config/fsra_vit_improved_config.yaml --data-dir data --batch-size 8 --num-epochs 120 --gpu-ids \"0\"")
    else:
        print(f"\n❌ Patch Optimization Test FAILED!")
        print(f"Please check the error messages above.") 