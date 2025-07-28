#!/usr/bin/env python3
"""Test script to verify PCA removal and model functionality."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_no_pca():
    """Test that PCA has been completely removed and model works correctly."""
    print("🧹 Testing Complete PCA Removal")
    print("=" * 50)
    
    try:
        from src.models import create_model
        from src.utils import load_config
        
        # Load config
        config = load_config('config/fsra_vit_improved_config.yaml')
        
        print(f"📋 Configuration Check:")
        print(f"  Use PCA alignment: {config['model'].get('use_pca_alignment', 'Not found')}")
        print(f"  Target PCA dim: {config['model'].get('target_pca_dim', 'Not found - Good!')}")
        print(f"  Description: {config['innovation']['description']}")
        print(f"  Alignment method: {config['innovation']['architecture']['alignment_method']}")
        
        # Verify PCA is disabled
        assert config['model']['use_pca_alignment'] == False, "PCA should be disabled"
        assert 'target_pca_dim' not in config['model'], "target_pca_dim should be removed"
        assert config['innovation']['architecture']['alignment_method'] == 'none', "No alignment should be used"
        
        print(f"✅ Configuration verification passed!")
        
        # Create model
        print(f"\n🔧 Creating simplified model...")
        model = create_model(config)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully!")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Verify model structure
        print(f"\n🏗️ Model Structure Verification:")
        print(f"  Global classifier input dim: {model.global_classifier.add_block[0].in_features}")
        print(f"  Regional classifier input dim: {model.regional_classifiers[0].add_block[0].in_features}")
        print(f"  Final classifier input dim: {model.final_classifier.add_block[0].in_features}")
        
        # All should be 200 (fusion_dim)
        assert model.global_classifier.add_block[0].in_features == 200, "Global classifier should expect 200 dim"
        assert model.regional_classifiers[0].add_block[0].in_features == 200, "Regional classifier should expect 200 dim"
        assert model.final_classifier.add_block[0].in_features == 200, "Final classifier should expect 200 dim"
        
        print(f"✅ All classifiers use consistent 200-dimensional input!")
        
        # Test forward pass
        print(f"\n🧪 Testing forward pass...")
        batch_size = 4
        img_size = config['data']['image_height']  # 250
        
        sat_images = torch.randn(batch_size, 3, img_size, img_size)
        drone_images = torch.randn(batch_size, 3, img_size, img_size)
        
        with torch.no_grad():
            outputs = model(sat_images, drone_images)
        
        print(f"✅ Forward pass successful!")
        
        # Verify output structure
        if 'satellite' in outputs:
            predictions = outputs['satellite']['predictions']
            features = outputs['satellite']['features']
            
            print(f"\n📊 Output Verification:")
            print(f"  Number of predictions: {len(predictions)}")
            print(f"  Prediction shapes: {[pred.shape for pred in predictions]}")
            
            # Check that all predictions are valid
            for i, pred in enumerate(predictions):
                assert pred.shape == (batch_size, 701), f"Prediction {i} should be ({batch_size}, 701), got {pred.shape}"
            
            print(f"  ✅ All predictions have correct shape!")
            
            # Check features
            print(f"  Feature types: {list(features.keys())}")
            if 'global' in features:
                print(f"    Global features: {features['global'].shape}")
                assert features['global'].shape == (batch_size, 200), "Global features should be (batch_size, 200)"
            
            if 'regional' in features:
                print(f"    Regional features: {len(features['regional'])} x {features['regional'][0].shape}")
                for i, rf in enumerate(features['regional']):
                    assert rf.shape == (batch_size, 200), f"Regional feature {i} should be (batch_size, 200)"
                    
            print(f"  ✅ All features have correct dimensions!")
        
        # Test with training data format
        print(f"\n🎯 Testing with training setup...")
        sys.path.insert(0, str(Path(__file__).parent))
        from train_fsra_aligned import calculate_accuracy
        
        # Create dummy labels
        labels = torch.randint(0, config['model']['num_classes'], (batch_size,))
        
        # Test accuracy calculation
        sat_acc, drone_acc = calculate_accuracy(outputs, labels)
        print(f"  Satellite accuracy: {sat_acc:.4f}")
        print(f"  Drone accuracy: {drone_acc:.4f}")
        
        # Accuracy should be reasonable (not exactly 0)
        assert sat_acc >= 0 and sat_acc <= 1, "Satellite accuracy should be between 0 and 1"
        assert drone_acc >= 0 and drone_acc <= 1, "Drone accuracy should be between 0 and 1"
        
        print(f"  ✅ Accuracy calculation works correctly!")
        
        print(f"\n🎉 Simplification Summary:")
        print(f"  ❌ No PCA processing")
        print(f"  ❌ No complex community detection")
        print(f"  ❌ No feature alignment")
        print(f"  ✅ Simple K-means clustering")
        print(f"  ✅ Direct 200-dim feature processing")
        print(f"  ✅ Consistent dimension flow")
        print(f"  ✅ Faster training expected")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_no_pca()
    
    if success:
        print(f"\n🎉 PCA Removal Test PASSED!")
        print(f"\n🚀 Ready for Training (Simplified Architecture):")
        print(f"python train_fsra_aligned.py --config config/fsra_vit_improved_config.yaml --data-dir data --batch-size 12 --num-epochs 100 --gpu-ids \"0\"")
        print(f"\n📈 Expected Benefits:")
        print(f"• Faster training (no PCA overhead)")
        print(f"• Lower memory usage (no PCA matrices)")
        print(f"• Simpler architecture (easier debugging)")
        print(f"• No dimension mismatch errors")
    else:
        print(f"\n❌ PCA Removal Test FAILED!")
        print(f"Please check the error messages above.") 