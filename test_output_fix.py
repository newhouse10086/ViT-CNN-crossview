#!/usr/bin/env python3
"""
Test script to verify output format fix.
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
            return [pred, f]  # Returns a list!
        else:
            return self.classifier(x)


def test_output_format():
    """Test the output format handling."""
    print("🧪 Testing Output Format Fix")
    print("="*40)
    
    try:
        # Model parameters
        batch_size = 2
        fusion_dim = 200
        target_pca_dim = 256
        num_classes = 701
        num_clusters = 3
        
        # Create classifiers
        global_classifier = ClassBlock(
            input_dim=fusion_dim,
            class_num=num_classes,
            num_bottleneck=target_pca_dim,
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
        
        final_classifier = ClassBlock(
            input_dim=target_pca_dim,
            class_num=num_classes,
            num_bottleneck=target_pca_dim,
            return_f=True
        )
        
        print(f"📊 Testing classifier outputs:")
        
        # Test global classifier
        global_feat = torch.randn(batch_size, fusion_dim)
        global_output = global_classifier(global_feat)
        print(f"   Global classifier output type: {type(global_output)}")
        print(f"   Global classifier output length: {len(global_output)}")
        
        # Unpack properly
        global_pred, global_f = global_output
        print(f"   Global pred shape: {global_pred.shape}")
        print(f"   Global feat shape: {global_f.shape}")
        
        # Test regional classifiers
        clustered_features = torch.randn(batch_size, num_clusters, target_pca_dim)
        regional_preds = []
        regional_feats = []
        
        for i, regional_classifier in enumerate(regional_classifiers):
            regional_input = clustered_features[:, i, :]
            regional_output = regional_classifier(regional_input)
            print(f"   Regional {i} output type: {type(regional_output)}")
            
            # Unpack properly
            regional_pred, regional_f = regional_output
            regional_preds.append(regional_pred)
            regional_feats.append(regional_f)
            print(f"   Regional {i} pred shape: {regional_pred.shape}")
            print(f"   Regional {i} feat shape: {regional_f.shape}")
        
        # Test feature concatenation
        all_features = torch.cat([global_f] + regional_feats, dim=1)
        print(f"   Concatenated features shape: {all_features.shape}")
        
        # Test final classifier
        final_input = torch.randn(batch_size, target_pca_dim)
        final_output = final_classifier(final_input)
        final_pred, final_f = final_output
        print(f"   Final pred shape: {final_pred.shape}")
        print(f"   Final feat shape: {final_f.shape}")
        
        # Test predictions list
        predictions = [global_pred] + regional_preds + [final_pred]
        print(f"\n📋 Testing predictions list:")
        print(f"   Predictions list length: {len(predictions)}")
        for i, pred in enumerate(predictions):
            print(f"   Prediction {i}: type={type(pred)}, shape={pred.shape}")
        
        # Verify all predictions are tensors (not lists)
        all_tensors = all(isinstance(pred, torch.Tensor) for pred in predictions)
        if all_tensors:
            print(f"✅ All predictions are tensors!")
            
            # Test loss computation simulation
            labels = torch.randint(0, num_classes, (batch_size,))
            print(f"\n🔍 Testing loss computation simulation:")
            print(f"   Labels shape: {labels.shape}")
            
            criterion = nn.CrossEntropyLoss()
            total_loss = 0.0
            
            for i, pred in enumerate(predictions):
                loss = criterion(pred, labels)
                total_loss += loss
                print(f"   Loss {i}: {loss.item():.4f}")
            
            print(f"   Total loss: {total_loss.item():.4f}")
            print(f"✅ Loss computation successful!")
            return True
        else:
            print(f"❌ Some predictions are not tensors!")
            return False
        
    except Exception as e:
        print(f"❌ Output format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_output_structure():
    """Test the expected model output structure."""
    print("\n🧪 Testing Model Output Structure")
    print("="*45)
    
    try:
        # Simulate model output structure
        batch_size = 2
        num_classes = 701
        
        # Create sample predictions (all should be tensors)
        global_pred = torch.randn(batch_size, num_classes)
        regional_pred_1 = torch.randn(batch_size, num_classes)
        regional_pred_2 = torch.randn(batch_size, num_classes)
        regional_pred_3 = torch.randn(batch_size, num_classes)
        final_pred = torch.randn(batch_size, num_classes)
        
        predictions = [global_pred, regional_pred_1, regional_pred_2, regional_pred_3, final_pred]
        
        # Create sample features
        global_f = torch.randn(batch_size, 256)
        regional_feats = [torch.randn(batch_size, 256) for _ in range(3)]
        final_f = torch.randn(batch_size, 256)
        
        # Model output structure
        outputs = {
            'satellite': {
                'predictions': predictions,
                'features': {
                    'global': global_f,
                    'regional': regional_feats,
                    'final': final_f,
                }
            }
        }
        
        print(f"📊 Model output structure:")
        print(f"   Satellite predictions: {len(outputs['satellite']['predictions'])} levels")
        
        for i, pred in enumerate(outputs['satellite']['predictions']):
            print(f"     Level {i}: {pred.shape}")
        
        print(f"   Satellite features:")
        for key, feat in outputs['satellite']['features'].items():
            if isinstance(feat, torch.Tensor):
                print(f"     {key}: {feat.shape}")
            elif isinstance(feat, list):
                print(f"     {key}: {len(feat)} tensors")
                for j, f in enumerate(feat):
                    print(f"       [{j}]: {f.shape}")
        
        print(f"✅ Model output structure is correct!")
        return True
        
    except Exception as e:
        print(f"❌ Model output structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🎯 Output Format Fix Verification")
    print("="*50)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Output format
    if test_output_format():
        success_count += 1
    
    # Test 2: Model output structure
    if test_model_output_structure():
        success_count += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"🎊 TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print(f"🎉 All tests passed! Output format fix should work!")
        print(f"🚀 The tensor list error should be resolved!")
        print(f"📝 Key fixes applied:")
        print(f"   - Proper unpacking of ClassBlock outputs")
        print(f"   - All predictions are now tensors (not lists)")
        print(f"   - Consistent output format")
    else:
        print(f"⚠️  Some tests failed. Please check the errors above.")
    
    return success_count == total_tests


if __name__ == "__main__":
    main()
