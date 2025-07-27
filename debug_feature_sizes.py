#!/usr/bin/env python3
"""
Debug the actual feature sizes in FSRA_IMPROVED to understand complexity.
"""

import torch
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import load_config
from src.models import create_model


def debug_feature_sizes():
    """Debug the actual feature sizes."""
    print("🔍 Debugging FSRA_IMPROVED Feature Sizes")
    
    # Load config
    config = load_config('config/your_innovation_config.yaml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    model.eval()
    
    # Create dummy data
    batch_size = 8
    sat_images = torch.randn(batch_size, 3, 256, 256).to(device)
    drone_images = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print(f"Input images shape: {sat_images.shape}")
    
    # Hook to capture feature map sizes
    feature_sizes = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                feature_sizes[name] = output.shape
                print(f"{name}: {output.shape}")
        return hook
    
    # Register hooks
    if hasattr(model, 'backbone'):
        model.backbone.register_forward_hook(hook_fn("backbone_output"))
    
    if hasattr(model, 'feature_projection'):
        model.feature_projection.register_forward_hook(hook_fn("feature_projection"))
    
    if hasattr(model, 'community_clustering'):
        # Add debug prints to community clustering
        original_forward = model.community_clustering.forward
        
        def debug_forward(feature_map):
            B, C, H, W = feature_map.shape
            print(f"\n🔬 Community Clustering Input:")
            print(f"  Feature map shape: {feature_map.shape}")
            print(f"  Spatial resolution: {H}×{W} = {H*W} points per sample")
            print(f"  Total points for batch: {B} × {H*W} = {B * H * W}")
            
            # Time the clustering for one sample
            features = feature_map.view(B, C, H * W).permute(0, 2, 1)
            sample_features = features[0]  # First sample
            
            print(f"  Sample features shape: {sample_features.shape}")
            
            # Time similarity computation
            start_time = time.time()
            features_norm = torch.nn.functional.normalize(sample_features, p=2, dim=1)
            similarity_matrix = torch.mm(features_norm, features_norm.t())
            similarity_time = time.time() - start_time
            
            print(f"  Similarity matrix shape: {similarity_matrix.shape}")
            print(f"  Similarity computation time: {similarity_time:.4f}s")
            print(f"  Estimated time for full batch: {similarity_time * B:.4f}s")
            
            # Call original forward
            return original_forward(feature_map)
        
        model.community_clustering.forward = debug_forward
    
    # Forward pass
    print(f"\n🚀 Running forward pass with batch_size={batch_size}")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(sat_images, drone_images)
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total forward pass time: {total_time:.3f}s")
    print(f"⏱️  Time per sample: {total_time/batch_size:.3f}s")
    
    # Analyze outputs
    if isinstance(outputs, dict) and 'satellite' in outputs:
        sat_outputs = outputs['satellite']
        if 'predictions' in sat_outputs:
            predictions = sat_outputs['predictions']
            print(f"\n📊 Output predictions: {len(predictions)} prediction levels")
            for i, pred in enumerate(predictions):
                if isinstance(pred, torch.Tensor):
                    print(f"  Prediction {i}: {pred.shape}")


def estimate_complexity():
    """Estimate computational complexity."""
    print(f"\n{'='*60}")
    print("📈 COMPUTATIONAL COMPLEXITY ANALYSIS")
    print(f"{'='*60}")
    
    # Typical ResNet18 output after global average pooling
    # Input: 256×256 → ResNet18 → Feature map: ~8×8 or 16×16
    
    spatial_sizes = [8, 16, 32]  # Possible spatial resolutions
    batch_sizes = [8, 16, 32, 64]
    
    print(f"{'Spatial':<10} {'Points':<10} {'Similarity':<15} {'Batch=8':<10} {'Batch=64':<10}")
    print(f"{'-'*10} {'-'*10} {'-'*15} {'-'*10} {'-'*10}")
    
    for spatial in spatial_sizes:
        points = spatial * spatial
        similarity_ops = points * points
        
        time_batch_8 = similarity_ops * 8
        time_batch_64 = similarity_ops * 64
        
        print(f"{spatial}×{spatial:<7} {points:<10} {similarity_ops:<15,} {time_batch_8:<10,} {time_batch_64:<10,}")
    
    print(f"\nNote: Numbers represent relative computational operations")
    print(f"If spatial resolution is 16×16, batch=64 should be ~8x slower than batch=8")


if __name__ == "__main__":
    debug_feature_sizes()
    estimate_complexity()
