#!/usr/bin/env python3
"""
Debug and fix model - comprehensive shape and device checking.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config_utils import load_config
from src.models import create_model


def debug_model_shapes():
    """Debug model forward pass shapes step by step."""
    print("🔍 DEBUGGING MODEL SHAPES")
    print("="*50)
    
    # Load config
    config = load_config("config/fsra_vit_improved_config.yaml")
    
    # Create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_model(config)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model created and moved to {device}")
    
    # Test inputs
    batch_size = 4
    sat_images = torch.randn(batch_size, 3, 256, 256).to(device)
    drone_images = torch.randn(batch_size, 3, 256, 256).to(device)
    labels = torch.randint(0, 701, (batch_size,)).to(device)
    
    print(f"\n📊 Input shapes:")
    print(f"  sat_images: {sat_images.shape}")
    print(f"  drone_images: {drone_images.shape}")
    print(f"  labels: {labels.shape}")
    
    # Forward pass with detailed debugging
    print(f"\n🔍 Forward pass debugging:")
    
    with torch.no_grad():
        try:
            # Step 1: CNN backbone
            x = sat_images
            B = x.shape[0]
            print(f"  Input x: {x.shape}")
            
            # CNN Branch
            cnn_features = model.cnn_backbone(x)
            print(f"  CNN backbone output: {cnn_features.shape}")
            
            cnn_features = model.cnn_dim_reduction(cnn_features)
            print(f"  CNN reduced: {cnn_features.shape}")
            
            # ViT Branch
            vit_features = model.vit_branch(x)
            print(f"  ViT output: {vit_features.shape}")
            
            # Feature Fusion
            fused_features = torch.cat([cnn_features, vit_features], dim=1)
            print(f"  Fused features: {fused_features.shape}")
            
            # Global pooling
            global_feat = torch.nn.functional.adaptive_avg_pool2d(fused_features, (1, 1)).view(B, -1)
            print(f"  Global feat: {global_feat.shape}")
            
            # Global classification
            global_output = model.global_classifier(global_feat)
            print(f"  Global output type: {type(global_output)}")
            if isinstance(global_output, (list, tuple)):
                global_pred, global_f = global_output
                print(f"  Global pred: {global_pred.shape}")
                print(f"  Global features: {global_f.shape}")
            else:
                print(f"  Global output: {global_output.shape}")
            
            # Community clustering
            clustered_features, communities = model.community_clustering(fused_features)
            print(f"  Clustered features: {clustered_features.shape}")
            
            # Regional classification
            regional_preds = []
            regional_feats = []
            
            for i, regional_classifier in enumerate(model.regional_classifiers):
                regional_input = clustered_features[:, i, :]
                print(f"  Regional {i} input: {regional_input.shape}")
                
                regional_output = regional_classifier(regional_input)
                if isinstance(regional_output, (list, tuple)):
                    regional_pred, regional_f = regional_output
                    print(f"  Regional {i} pred: {regional_pred.shape}")
                    print(f"  Regional {i} feat: {regional_f.shape}")
                    regional_preds.append(regional_pred)
                    regional_feats.append(regional_f)
                else:
                    print(f"  Regional {i} output: {regional_output.shape}")
                    regional_preds.append(regional_output)
                    regional_feats.append(regional_output)  # Fallback
            
            # Feature fusion for final
            print(f"  Global_f shape: {global_f.shape}")
            for i, rf in enumerate(regional_feats):
                print(f"  Regional feat {i}: {rf.shape}")
            
            all_features = torch.cat([global_f] + regional_feats, dim=1)
            print(f"  All features concatenated: {all_features.shape}")
            
            fused_features_final = model.feature_fusion(all_features)
            print(f"  Final fused features: {fused_features_final.shape}")
            
            # Final classification
            final_output = model.final_classifier(fused_features_final)
            if isinstance(final_output, (list, tuple)):
                final_pred, final_f = final_output
                print(f"  Final pred: {final_pred.shape}")
                print(f"  Final feat: {final_f.shape}")
            else:
                print(f"  Final output: {final_output.shape}")
            
            print(f"\n✅ Forward pass completed successfully!")
            
            # Test full forward
            outputs = model(sat_images, drone_images)
            print(f"\n📋 Full model outputs:")
            print(f"  Output keys: {list(outputs.keys())}")
            
            if 'satellite' in outputs:
                sat_outputs = outputs['satellite']
                print(f"  Satellite keys: {list(sat_outputs.keys())}")
                
                if 'predictions' in sat_outputs:
                    predictions = sat_outputs['predictions']
                    print(f"  Predictions count: {len(predictions)}")
                    for i, pred in enumerate(predictions):
                        print(f"    Pred {i}: {pred.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            return False


def debug_loss_function():
    """Debug loss function with model outputs."""
    print(f"\n🔍 DEBUGGING LOSS FUNCTION")
    print("="*50)
    
    try:
        from src.losses.fsra_style_loss import create_fsra_style_loss
        
        # Create loss function
        criterion = create_fsra_style_loss(
            num_classes=701,
            classification_weight=1.0,
            triplet_weight=1.0,
            center_weight=0.0,  # Disable center loss for now
            triplet_margin=0.3
        )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        criterion = criterion.to(device)
        
        print(f"✅ Loss function created and moved to {device}")
        
        # Create dummy outputs
        batch_size = 4
        num_classes = 701
        
        # Simulate model outputs
        outputs = {
            'satellite': {
                'predictions': [
                    torch.randn(batch_size, num_classes).to(device),  # global
                    torch.randn(batch_size, num_classes).to(device),  # regional 1
                    torch.randn(batch_size, num_classes).to(device),  # regional 2
                    torch.randn(batch_size, num_classes).to(device),  # regional 3
                    torch.randn(batch_size, num_classes).to(device),  # final
                ],
                'features': {
                    'global': torch.randn(batch_size, 256).to(device),
                    'regional': [
                        torch.randn(batch_size, 256).to(device),
                        torch.randn(batch_size, 256).to(device),
                        torch.randn(batch_size, 256).to(device),
                    ],
                    'final': torch.randn(batch_size, 256).to(device),
                }
            }
        }
        
        labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        
        print(f"📊 Test data:")
        print(f"  Batch size: {batch_size}")
        print(f"  Num classes: {num_classes}")
        print(f"  Labels: {labels.shape}")
        print(f"  Predictions: {len(outputs['satellite']['predictions'])}")
        
        # Test loss computation
        losses = criterion(outputs, labels)
        
        print(f"✅ Loss computation successful!")
        print(f"📊 Loss breakdown:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f}")
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in loss function: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main debugging function."""
    print("🎯 COMPREHENSIVE MODEL DEBUG SESSION")
    print("="*60)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Model shapes
    if debug_model_shapes():
        success_count += 1
    
    # Test 2: Loss function
    if debug_loss_function():
        success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎊 DEBUG SUMMARY")
    print(f"{'='*60}")
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print(f"🎉 All tests passed! Model is ready for training.")
        print(f"💡 Recommended training command:")
        print(f"python train_fsra_style.py \\")
        print(f"    --config config/fsra_vit_improved_config.yaml \\")
        print(f"    --data-dir data \\")
        print(f"    --batch-size 8 \\")
        print(f"    --learning-rate 0.001 \\")
        print(f"    --num-epochs 50 \\")
        print(f"    --triplet-weight 1.0 \\")
        print(f"    --center-weight 0.0 \\")
        print(f"    --gpu-ids \"0\"")
    else:
        print(f"⚠️  Some tests failed. Please check the errors above.")
    
    return success_count == total_tests


if __name__ == "__main__":
    main()
