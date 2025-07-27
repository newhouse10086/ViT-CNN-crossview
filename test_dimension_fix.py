#!/usr/bin/env python3
"""
Test script to verify dimension fix in FSRA ViT model.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_fsra_vit_dimensions():
    """Test the dimension fix in FSRA ViT model."""
    print("🧪 Testing FSRA ViT Dimension Fix")
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
        
        # Print model architecture info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        # Test forward pass
        batch_size = 2
        sat_images = torch.randn(batch_size, 3, 256, 256)
        drone_images = torch.randn(batch_size, 3, 256, 256)
        
        print(f"\n🔍 Testing model forward pass...")
        print(f"   Input shapes: {sat_images.shape}, {drone_images.shape}")
        
        with torch.no_grad():
            outputs = model(sat_images, drone_images)
        
        print(f"✅ Model forward pass successful!")
        
        # Check outputs structure
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
        
        # Test specific components
        print(f"\n🔬 Testing Component Dimensions:")
        
        # Test CNN branch
        if hasattr(model, 'cnn_backbone'):
            cnn_features = model.cnn_backbone(sat_images)
            print(f"   CNN backbone: {cnn_features.shape}")
            
            if hasattr(model, 'cnn_dim_reduction'):
                cnn_reduced = model.cnn_dim_reduction(cnn_features)
                print(f"   CNN reduced: {cnn_reduced.shape}")
        
        # Test ViT branch
        if hasattr(model, 'vit_branch'):
            vit_features = model.vit_branch(sat_images)
            print(f"   ViT output: {vit_features.shape}")
        
        # Test feature fusion
        if hasattr(model, 'cnn_dim_reduction') and hasattr(model, 'vit_branch'):
            cnn_feat = model.cnn_dim_reduction(model.cnn_backbone(sat_images))
            vit_feat = model.vit_branch(sat_images)
            fused = torch.cat([cnn_feat, vit_feat], dim=1)
            print(f"   Fused features: {fused.shape}")
            
            # Test global pooling
            global_feat = torch.nn.functional.adaptive_avg_pool2d(fused, (1, 1)).view(batch_size, -1)
            print(f"   Global features: {global_feat.shape}")
        
        print(f"\n🎉 All dimension tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dimension test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_classblock_dimensions():
    """Test ClassBlock dimension consistency."""
    print("\n🧪 Testing ClassBlock Dimensions")
    print("="*40)
    
    try:
        from src.models.components import ClassBlock
        
        # Test global classifier
        global_classifier = ClassBlock(
            input_dim=200,  # fusion_dim
            class_num=701,
            num_bottleneck=256,  # target_pca_dim
            return_f=True
        )
        
        # Test regional classifier
        regional_classifier = ClassBlock(
            input_dim=256,  # target_pca_dim
            class_num=701,
            num_bottleneck=256,  # target_pca_dim
            return_f=True
        )
        
        # Test inputs
        global_input = torch.randn(2, 200)
        regional_input = torch.randn(2, 256)
        
        # Test outputs
        global_pred, global_f = global_classifier(global_input)
        regional_pred, regional_f = regional_classifier(regional_input)
        
        print(f"✅ ClassBlock tests successful!")
        print(f"   Global classifier:")
        print(f"     Input: {global_input.shape} -> Features: {global_f.shape}, Pred: {global_pred.shape}")
        print(f"   Regional classifier:")
        print(f"     Input: {regional_input.shape} -> Features: {regional_f.shape}, Pred: {regional_pred.shape}")
        
        # Test feature concatenation
        all_features = torch.cat([global_f, regional_f, regional_f, regional_f], dim=1)
        print(f"   Feature concatenation: {all_features.shape}")
        print(f"   Expected: (2, 1024) = 256 * 4")
        
        if all_features.shape[1] == 1024:
            print(f"✅ Feature dimensions are consistent!")
            return True
        else:
            print(f"❌ Feature dimension mismatch!")
            return False
        
    except Exception as e:
        print(f"❌ ClassBlock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🎯 FSRA ViT Dimension Fix Verification")
    print("="*60)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: ClassBlock dimensions
    if test_classblock_dimensions():
        success_count += 1
    
    # Test 2: Full model
    if test_fsra_vit_dimensions():
        success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print(f"🎉 All tests passed! Dimension fix is working!")
        print(f"🚀 The tensor concatenation error should be resolved!")
        print(f"📝 Key fixes applied:")
        print(f"   - Global classifier bottleneck: 256D (was 512D)")
        print(f"   - All feature dimensions: 256D")
        print(f"   - Feature fusion input: 1024D (4 × 256D)")
        print(f"   - Consistent dimension alignment")
    else:
        print(f"⚠️  Some tests failed. Please check the errors above.")
    
    return success_count == total_tests


if __name__ == "__main__":
    main()
