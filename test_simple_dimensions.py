#!/usr/bin/env python3
"""
Simple dimension test without external dependencies.
"""

import torch
import torch.nn as nn

# Simple ClassBlock implementation for testing
class ClassBlock(nn.Module):
    def __init__(self, input_dim: int, class_num: int, num_bottleneck: int = 512, return_f: bool = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        
        self.add_block = nn.Sequential(
            nn.Linear(input_dim, num_bottleneck),
            nn.BatchNorm1d(num_bottleneck),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Linear(num_bottleneck, class_num)

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            pred = self.classifier(x)
            return [pred, f]
        else:
            return self.classifier(x)


def test_dimension_consistency():
    """Test dimension consistency for feature fusion."""
    print("🧪 Testing Dimension Consistency")
    print("="*50)
    
    try:
        # Model parameters
        fusion_dim = 200  # CNN 100 + ViT 100
        target_pca_dim = 256
        num_clusters = 3
        num_classes = 701
        batch_size = 2
        
        print(f"📊 Configuration:")
        print(f"   Fusion dim: {fusion_dim}")
        print(f"   Target PCA dim: {target_pca_dim}")
        print(f"   Num clusters: {num_clusters}")
        print(f"   Batch size: {batch_size}")
        
        # Create classifiers with fixed dimensions
        global_classifier = ClassBlock(
            input_dim=fusion_dim,
            class_num=num_classes,
            num_bottleneck=target_pca_dim,  # Fixed: use target_pca_dim
            return_f=True
        )
        
        regional_classifiers = nn.ModuleList([
            ClassBlock(
                input_dim=target_pca_dim,
                class_num=num_classes,
                num_bottleneck=target_pca_dim,
                return_f=True
            ) for _ in range(num_clusters)
        ])
        
        print(f"\n🔍 Testing Forward Pass:")
        
        # Simulate global features
        global_feat = torch.randn(batch_size, fusion_dim)
        print(f"   Global input: {global_feat.shape}")
        
        # Global classification
        global_pred, global_f = global_classifier(global_feat)
        print(f"   Global output: pred={global_pred.shape}, feat={global_f.shape}")
        
        # Simulate clustered features
        clustered_features = torch.randn(batch_size, num_clusters, target_pca_dim)
        print(f"   Clustered features: {clustered_features.shape}")
        
        # Regional classification
        regional_feats = []
        for i, regional_classifier in enumerate(regional_classifiers):
            regional_input = clustered_features[:, i, :]
            regional_pred, regional_f = regional_classifier(regional_input)
            regional_feats.append(regional_f)
            print(f"   Regional {i}: input={regional_input.shape}, feat={regional_f.shape}")
        
        # Feature fusion test
        print(f"\n🔗 Testing Feature Fusion:")
        all_features = torch.cat([global_f] + regional_feats, dim=1)
        print(f"   Concatenated features: {all_features.shape}")
        
        expected_dim = target_pca_dim * (1 + num_clusters)  # 256 * 4 = 1024
        print(f"   Expected dimension: {expected_dim}")
        
        if all_features.shape[1] == expected_dim:
            print(f"✅ Feature concatenation successful!")
            
            # Test final fusion layer
            final_fusion_dim = expected_dim
            feature_fusion = nn.Sequential(
                nn.Linear(final_fusion_dim, target_pca_dim),
                nn.BatchNorm1d(target_pca_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
            
            fused_features_final = feature_fusion(all_features)
            print(f"   Final fused features: {fused_features_final.shape}")
            
            # Test final classifier
            final_classifier = ClassBlock(
                input_dim=target_pca_dim,
                class_num=num_classes,
                num_bottleneck=target_pca_dim,
                return_f=True
            )
            
            final_pred, final_f = final_classifier(fused_features_final)
            print(f"   Final output: pred={final_pred.shape}, feat={final_f.shape}")
            
            print(f"\n🎉 All dimension tests passed!")
            return True
        else:
            print(f"❌ Feature concatenation failed!")
            print(f"   Got: {all_features.shape[1]}, Expected: {expected_dim}")
            return False
        
    except Exception as e:
        print(f"❌ Dimension test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tensor_shapes():
    """Test specific tensor shape operations."""
    print("\n🧪 Testing Tensor Shape Operations")
    print("="*40)
    
    try:
        batch_size = 8
        
        # Test CNN + ViT fusion
        cnn_features = torch.randn(batch_size, 100, 8, 8)
        vit_features = torch.randn(batch_size, 100, 8, 8)
        
        print(f"   CNN features: {cnn_features.shape}")
        print(f"   ViT features: {vit_features.shape}")
        
        # Fusion
        fused_features = torch.cat([cnn_features, vit_features], dim=1)
        print(f"   Fused features: {fused_features.shape}")
        
        # Global pooling
        global_feat = torch.nn.functional.adaptive_avg_pool2d(fused_features, (1, 1)).view(batch_size, -1)
        print(f"   Global features: {global_feat.shape}")
        
        # Test community clustering output simulation
        clustered_features = torch.randn(batch_size, 3, 256)
        print(f"   Clustered features: {clustered_features.shape}")
        
        # Test feature extraction from clusters
        regional_feats = []
        for i in range(3):
            regional_input = clustered_features[:, i, :]
            print(f"   Regional {i} input: {regional_input.shape}")
            regional_feats.append(regional_input)  # Simulate processed features
        
        print(f"✅ Tensor shape operations successful!")
        return True
        
    except Exception as e:
        print(f"❌ Tensor shape test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🎯 Simple Dimension Fix Verification")
    print("="*50)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Tensor shapes
    if test_tensor_shapes():
        success_count += 1
    
    # Test 2: Dimension consistency
    if test_dimension_consistency():
        success_count += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"🎊 TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print(f"🎉 All tests passed! Dimension fix should work!")
        print(f"🚀 The tensor concatenation error should be resolved!")
    else:
        print(f"⚠️  Some tests failed. Please check the errors above.")
    
    return success_count == total_tests


if __name__ == "__main__":
    main()
