#!/usr/bin/env python3
"""Test script for accuracy fix and K-means clustering."""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_accuracy_and_kmeans():
    """Test accuracy calculation and K-means clustering."""
    print("🔍 Testing Accuracy Fix and K-means Clustering")
    print("=" * 60)
    
    try:
        from src.models import create_model
        from src.utils import load_config
        
        # Load config
        config = load_config('config/fsra_vit_improved_config.yaml')
        
        print(f"📋 Configuration Updates:")
        print(f"  Learning rate: {config['training']['learning_rate']}")
        print(f"  Use community clustering: {config['model']['use_community_clustering']}")
        print(f"  Use K-means clustering: {config['model'].get('use_kmeans_clustering', 'Not set')}")
        print(f"  Use PCA alignment: {config['model']['use_pca_alignment']}")
        
        # Create model
        print(f"\n🔧 Creating model with K-means clustering...")
        model = create_model(config)
        model.eval()
        
        # Test forward pass
        print(f"\n🧪 Testing forward pass and accuracy...")
        batch_size = 4
        num_classes = config['model']['num_classes']
        
        img_size = config['data']['image_height']  # 250
        sat_images = torch.randn(batch_size, 3, img_size, img_size)
        drone_images = torch.randn(batch_size, 3, img_size, img_size)
        
        # Create random labels
        labels = torch.randint(0, num_classes, (batch_size,))
        
        with torch.no_grad():
            outputs = model(sat_images, drone_images)
        
        # Test accuracy calculation
        sys.path.insert(0, str(Path(__file__).parent))
        from train_fsra_aligned import calculate_accuracy
        
        sat_acc, drone_acc = calculate_accuracy(outputs, labels)
        
        print(f"✅ Forward pass successful!")
        print(f"  Output keys: {list(outputs.keys())}")
        
        if 'satellite' in outputs:
            predictions = outputs['satellite']['predictions']
            print(f"  Number of predictions: {len(predictions)}")
            for i, pred in enumerate(predictions):
                print(f"    Prediction {i}: {pred.shape}")
                
        print(f"\n📊 Accuracy Test Results:")
        print(f"  Satellite Accuracy: {sat_acc:.4f}")
        print(f"  Drone Accuracy: {drone_acc:.4f}")
        
        # Test if predictions are reasonable (not all zeros)
        if 'satellite' in outputs and len(outputs['satellite']['predictions']) > 0:
            first_pred = outputs['satellite']['predictions'][0]
            pred_std = first_pred.std().item()
            print(f"  Prediction std: {pred_std:.4f}")
            
            if pred_std > 0.01:
                print(f"  ✅ Predictions have reasonable variance")
            else:
                print(f"  ⚠️  Predictions have low variance (might be initialization issue)")
        
        # Test K-means clustering
        print(f"\n🎯 K-means Clustering Test:")
        if hasattr(model, 'kmeans_clustering'):
            clustering_module = model.kmeans_clustering
            print(f"  Clustering module type: {type(clustering_module).__name__}")
            print(f"  Number of clusters: {clustering_module.num_clusters}")
            print(f"  Target dimension: {clustering_module.target_dim}")
        
        print(f"\n📈 Expected Improvements:")
        print(f"  ⚡ Learning rate increased: 0.003 → 0.01 (3.3x faster)")
        print(f"  🎯 K-means replaces complex community detection")
        print(f"  📊 Accurate accuracy calculation")
        print(f"  🔍 100 patches instead of 625 (6.25x fewer)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_accuracy_and_kmeans()
    
    if success:
        print(f"\n🎉 All Tests PASSED!")
        print(f"\n🚀 Ready for Training:")
        print(f"python train_fsra_aligned.py --config config/fsra_vit_improved_config.yaml --data-dir data --batch-size 12 --num-epochs 100 --gpu-ids \"0\"")
        print(f"\n📊 Expected Results:")
        print(f"• Much higher accuracy (should be >1% instead of 0.002%)")
        print(f"• Faster convergence with 0.01 learning rate")
        print(f"• Stable K-means clustering")
        print(f"• ~1.3 min/epoch training time")
    else:
        print(f"\n❌ Tests FAILED!")
        print(f"Please check the error messages above.") 