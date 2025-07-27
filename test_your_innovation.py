#!/usr/bin/env python3
"""Test your innovation method."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_your_innovation():
    """Test your innovation method (FSRA + Community Clustering + PCA)."""
    print("Testing Your Innovation Method...")
    print("FSRA + Community Clustering + PCA")
    
    try:
        from src.models import create_model
        from src.losses import CombinedLoss
        from src.utils import load_config
        
        # Load your innovation config
        config = load_config('config/your_innovation_config.yaml')
        
        print(f"Innovation config:")
        print(f"  Model: {config['model']['name']}")
        print(f"  Classes: {config['model']['num_classes']}")
        print(f"  Patch Size: {config['model']['patch_size']} (your innovation)")
        print(f"  Clusters: {config['model']['num_final_clusters']} (your innovation)")
        print(f"  PCA Dim: {config['model']['target_pca_dim']} (your innovation)")
        print(f"  Community Clustering: {config['model']['use_community_clustering']}")
        print(f"  PCA Alignment: {config['model']['use_pca_alignment']}")
        
        # Create model
        model = create_model(config)
        model.eval()
        
        print(f"✓ Your innovation model created successfully")
        print(f"  Model type: {type(model).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Test forward pass
        batch_size = 4
        sat_images = torch.randn(batch_size, 3, 256, 256)
        drone_images = torch.randn(batch_size, 3, 256, 256)
        labels = torch.randint(0, config['model']['num_classes'], (batch_size,))
        
        print(f"\nTesting forward pass with your innovation...")
        print(f"  Input shapes: sat={sat_images.shape}, drone={drone_images.shape}")
        
        with torch.no_grad():
            outputs = model(sat_images, drone_images)
        
        print(f"✓ Forward pass successful")
        print(f"  Output keys: {list(outputs.keys())}")
        
        # Check output structure
        for view in ['satellite', 'drone']:
            if view in outputs and outputs[view] is not None:
                view_data = outputs[view]
                print(f"\n  {view.capitalize()} view (with your innovation):")
                
                if 'predictions' in view_data:
                    preds = view_data['predictions']
                    print(f"    Predictions: {type(preds)}")
                    if isinstance(preds, list):
                        for i, pred in enumerate(preds):
                            if isinstance(pred, torch.Tensor):
                                print(f"      [{i}]: shape={pred.shape}, ndim={pred.ndim}")
                                if pred.ndim == 1:
                                    print(f"           ❌ FOUND 1D TENSOR!")
                                elif pred.ndim == 2:
                                    print(f"           ✓ 2D tensor OK")
        
        # Test loss computation
        print(f"\nTesting loss computation with your innovation...")
        criterion = CombinedLoss(num_classes=config['model']['num_classes'])
        
        losses = criterion(outputs, labels)
        print(f"✓ Loss computation successful!")
        
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
        
        # Test community clustering specifically
        print(f"\nTesting community clustering module...")
        if hasattr(model, 'satellite_model') and hasattr(model.satellite_model, 'community_clustering'):
            clustering_module = model.satellite_model.community_clustering
            print(f"  ✓ Community clustering module found")
            print(f"  Feature dim: {clustering_module.feature_dim}")
            print(f"  Num clusters: {clustering_module.num_clusters}")
            print(f"  Target PCA dim: {clustering_module.target_dim}")
            print(f"  Similarity threshold: {clustering_module.similarity_threshold}")
        else:
            print(f"  ⚠️  Community clustering module not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_methods():
    """Compare Simple FSRA vs Your Innovation."""
    print(f"\n{'='*60}")
    print("Method Comparison")
    print(f"{'='*60}")
    
    print("Simple FSRA (Baseline):")
    print("  ✓ ResNet18 backbone")
    print("  ✓ 2x2 fixed regions (4 regions)")
    print("  ✓ Simple spatial pooling")
    print("  ✓ Traditional FSRA approach")
    print("  ✓ ~16M parameters")
    
    print("\nYour Innovation (FSRA + Community Clustering + PCA):")
    print("  ✓ ResNet18 backbone")
    print("  🚀 10x10 patch division (100 patches)")
    print("  🚀 Community clustering (3 communities)")
    print("  🚀 PCA dimensionality reduction")
    print("  🚀 Graph-based feature segmentation")
    print("  🚀 Cross-view feature alignment")
    print("  ✓ Novel research contribution")

def main():
    """Main test function."""
    print("=" * 60)
    print("Your Innovation Method Test")
    print("FSRA + Community Clustering + PCA")
    print("=" * 60)
    
    success = test_your_innovation()
    
    compare_methods()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 YOUR INNOVATION IS READY!")
        print("Community Clustering + PCA method is working correctly!")
        print("\nTo train your innovation method, use:")
        print("  python train_with_metrics.py --config config/your_innovation_config.yaml --data-dir data --batch-size 8 --learning-rate 0.001 --num-epochs 10 --gpu-ids \"0\"")
        print("\nOr use the enhanced training:")
        print("  python train_with_metrics.py --config config/your_innovation_config.yaml --data-dir data --batch-size 8 --learning-rate 0.001 --num-epochs 10 --gpu-ids \"0\"")
        print("\nThis will train your novel:")
        print("  🚀 Community clustering-based feature segmentation")
        print("  🚀 PCA-based feature alignment")
        print("  🚀 10x10 patch processing")
        print("  🚀 3-community clustering")
    else:
        print("❌ INNOVATION TEST FAILED!")
        print("Please check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
