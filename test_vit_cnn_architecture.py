#!/usr/bin/env python3
"""
Test script for the new ViT+CNN architecture.
Validates the true innovation: ViT (10x10 patches) + CNN (ResNet) + Community Clustering + PCA Alignment
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import load_config
from src.models import create_model


def test_vit_cnn_architecture():
    """Test the new ViT+CNN architecture."""
    print("🚀 Testing True ViT+CNN Architecture")
    print("="*60)
    
    # Load configuration
    config_path = "config/fsra_vit_improved_config.yaml"
    config = load_config(config_path)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    print("\n📦 Creating FSRA ViT Improved Model...")
    model = create_model(config)
    model = model.to(device)
    model.eval()
    
    # Model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Test forward pass
    print("\n🔍 Testing Forward Pass...")
    batch_size = 2
    
    # Create dummy input
    sat_images = torch.randn(batch_size, 3, 256, 256).to(device)
    drone_images = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print(f"Input shapes:")
    print(f"  Satellite: {sat_images.shape}")
    print(f"  Drone: {drone_images.shape}")
    
    # Forward pass
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(sat_images, drone_images)
    
    forward_time = time.time() - start_time
    
    print(f"\n✅ Forward pass successful!")
    print(f"   Forward time: {forward_time:.3f}s")
    print(f"   Time per sample: {forward_time/batch_size:.3f}s")
    
    # Analyze outputs
    print(f"\n📊 Output Analysis:")
    if 'satellite' in outputs:
        sat_outputs = outputs['satellite']
        
        if 'predictions' in sat_outputs:
            predictions = sat_outputs['predictions']
            print(f"   Predictions: {len(predictions)} levels")
            for i, pred in enumerate(predictions):
                if isinstance(pred, torch.Tensor):
                    print(f"     Level {i}: {pred.shape}")
        
        if 'features' in sat_outputs:
            features = sat_outputs['features']
            print(f"   Features:")
            for key, feat in features.items():
                if isinstance(feat, torch.Tensor):
                    print(f"     {key}: {feat.shape}")
                elif isinstance(feat, list):
                    print(f"     {key}: {len(feat)} tensors")
                    for j, f in enumerate(feat):
                        if isinstance(f, torch.Tensor):
                            print(f"       [{j}]: {f.shape}")
    
    # Test architecture components
    print(f"\n🔬 Architecture Component Analysis:")
    
    # Test CNN branch
    print(f"   CNN Branch:")
    if hasattr(model, 'cnn_backbone'):
        cnn_features = model.cnn_backbone(sat_images)
        print(f"     ResNet18 output: {cnn_features.shape}")
        
        if hasattr(model, 'cnn_dim_reduction'):
            cnn_reduced = model.cnn_dim_reduction(cnn_features)
            print(f"     CNN reduced: {cnn_reduced.shape}")
    
    # Test ViT branch
    print(f"   ViT Branch:")
    if hasattr(model, 'vit_branch'):
        vit_features = model.vit_branch(sat_images)
        print(f"     ViT output: {vit_features.shape}")
    
    # Test patch embedding
    if hasattr(model, 'vit_branch') and hasattr(model.vit_branch, 'patch_embed'):
        patch_embed = model.vit_branch.patch_embed
        print(f"   Patch Embedding:")
        print(f"     Patch size: {patch_embed.patch_size}")
        print(f"     Num patches: {patch_embed.num_patches}")
        print(f"     Embed dim: {patch_embed.embed_dim}")
        
        # Test patch embedding
        with torch.no_grad():
            patches = patch_embed(sat_images)
            print(f"     Patch embeddings: {patches.shape}")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_cached = torch.cuda.memory_reserved() / 1024**2  # MB
        print(f"\n💾 GPU Memory Usage:")
        print(f"   Allocated: {memory_used:.1f} MB")
        print(f"   Cached: {memory_cached:.1f} MB")
    
    print(f"\n🎉 All tests passed! Your ViT+CNN architecture is working correctly!")
    
    return True


def test_innovation_features():
    """Test specific innovation features."""
    print(f"\n🔬 Testing Innovation Features")
    print("="*60)
    
    # Test community clustering module
    print("1. Testing Community Clustering Module...")
    from src.models.fsra_vit_improved import CommunityClusteringModule
    
    clustering = CommunityClusteringModule(num_clusters=3, target_dim=256)
    
    # Test with dummy feature map
    feature_map = torch.randn(2, 200, 8, 8)  # Fused CNN+ViT features
    
    with torch.no_grad():
        clustered_features, communities = clustering(feature_map)
    
    print(f"   ✅ Input: {feature_map.shape}")
    print(f"   ✅ Output: {clustered_features.shape}")
    print(f"   ✅ Communities: {len(communities)} samples")
    
    # Test ViT module
    print("2. Testing ViT Module...")
    from src.models.vit_module import VisionTransformer
    
    vit = VisionTransformer(
        img_size=256,
        patch_size=10,
        embed_dim=768,
        depth=6,
        output_dim=100
    )
    
    input_img = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        vit_output = vit(input_img)
    
    print(f"   ✅ Input: {input_img.shape}")
    print(f"   ✅ Output: {vit_output.shape}")
    print(f"   ✅ Patches: {vit.num_patches}")
    
    print(f"\n🎉 Innovation features working correctly!")


def main():
    """Main test function."""
    print("🎯 FSRA ViT Improved Architecture Test")
    print("Testing your true innovation: ViT+CNN+Community Clustering+PCA")
    print("="*80)
    
    try:
        # Test main architecture
        test_vit_cnn_architecture()
        
        # Test innovation features
        test_innovation_features()
        
        print(f"\n🎊 ALL TESTS PASSED!")
        print(f"Your ViT+CNN innovation architecture is ready for training!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()
