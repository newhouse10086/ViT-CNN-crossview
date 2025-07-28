#!/usr/bin/env python3
"""Test script for dimension fix."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_dimension_fix():
    """Test the dimension fix for global classifier."""
    print("🔧 Testing Dimension Fix")
    print("=" * 50)
    
    try:
        from src.models import create_model
        from src.utils import load_config
        
        # Load config
        config = load_config('config/fsra_vit_improved_config.yaml')
        
        print(f"📋 Model Configuration:")
        print(f"  CNN output dim: {config['model']['cnn_output_dim']}")
        print(f"  ViT output dim: {config['model']['vit_output_dim']}")
        print(f"  Fusion dim: {config['model']['cnn_output_dim'] + config['model']['vit_output_dim']}")
        print(f"  Target PCA dim: {config['model']['target_pca_dim']}")
        
        # Create model
        print(f"\n🔧 Creating model...")
        model = create_model(config)
        model.eval()
        
        print(f"✅ Model created successfully!")
        
        # Print model structure
        print(f"\n🏗️ Model Structure:")
        print(f"  Global classifier input: {model.global_classifier.add_block[0].in_features}")
        print(f"  Global classifier bottleneck: {model.global_classifier.add_block[0].out_features}")
        print(f"  Regional classifier input: {model.regional_classifiers[0].add_block[0].in_features}")
        print(f"  Regional classifier bottleneck: {model.regional_classifiers[0].add_block[0].out_features}")
        print(f"  Final classifier input: {model.final_classifier.add_block[0].in_features}")
        print(f"  Final classifier bottleneck: {model.final_classifier.add_block[0].out_features}")
        
        # Test forward pass
        print(f"\n🧪 Testing forward pass...")
        batch_size = 4
        img_size = config['data']['image_height']  # 250
        
        sat_images = torch.randn(batch_size, 3, img_size, img_size)
        drone_images = torch.randn(batch_size, 3, img_size, img_size)
        
        with torch.no_grad():
            outputs = model(sat_images, drone_images)
        
        print(f"✅ Forward pass successful!")
        
        # Check output structure
        if 'satellite' in outputs:
            predictions = outputs['satellite']['predictions']
            features = outputs['satellite']['features']
            
            print(f"\n📊 Output Analysis:")
            print(f"  Number of predictions: {len(predictions)}")
            for i, pred in enumerate(predictions):
                print(f"    Prediction {i}: {pred.shape}")
            
            print(f"  Feature types: {list(features.keys())}")
            if 'global' in features:
                print(f"    Global features: {features['global'].shape}")
            if 'regional' in features:
                print(f"    Regional features: {len(features['regional'])} x {features['regional'][0].shape}")
            if 'final' in features:
                print(f"    Final features: {features['final'].shape}")
        
        print(f"\n🎯 Dimension Verification:")
        print(f"  ✅ No dimension mismatch errors!")
        print(f"  ✅ All matrix multiplications successful!")
        print(f"  ✅ Model ready for training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dimension_fix()
    
    if success:
        print(f"\n🎉 Dimension Fix SUCCESSFUL!")
        print(f"\n🚀 Now you can run training:")
        print(f"python train_fsra_aligned.py --config config/fsra_vit_improved_config.yaml --data-dir data --batch-size 12 --num-epochs 100 --gpu-ids \"0\"")
    else:
        print(f"\n❌ Dimension Fix FAILED!")
        print(f"Please check the error messages above.") 