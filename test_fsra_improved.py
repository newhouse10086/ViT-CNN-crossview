#!/usr/bin/env python3
"""Test FSRA Improved model."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_fsra_improved():
    """Test FSRA Improved model creation and forward pass."""
    print("Testing FSRA Improved model...")
    
    try:
        from src.models import create_model
        from src.losses import CombinedLoss
        from src.utils import load_config
        
        # Load FSRA Improved config
        config = load_config('config/fsra_improved_config.yaml')
        
        print(f"Model config:")
        print(f"  Name: {config['model']['name']}")
        print(f"  Classes: {config['model']['num_classes']}")
        print(f"  Clusters: {config['model']['num_final_clusters']}")
        print(f"  Feature dim: {config['model']['feature_dim']}")
        print(f"  Views: {config['data']['views']}")
        
        # Create model
        model = create_model(config)
        model.eval()
        
        print(f"✓ FSRA Improved model created successfully")
        print(f"  Model type: {type(model).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Test forward pass
        batch_size = 2
        sat_images = torch.randn(batch_size, 3, 256, 256)
        drone_images = torch.randn(batch_size, 3, 256, 256)
        labels = torch.randint(0, config['model']['num_classes'], (batch_size,))
        
        print(f"\nTesting forward pass...")
        print(f"  Input shapes: sat={sat_images.shape}, drone={drone_images.shape}")
        
        with torch.no_grad():
            outputs = model(sat_images, drone_images)
        
        print(f"✓ Forward pass successful")
        print(f"  Output keys: {list(outputs.keys())}")
        
        # Check output structure
        for view in ['satellite', 'drone']:
            if view in outputs and outputs[view] is not None:
                view_data = outputs[view]
                print(f"  {view.capitalize()} view:")
                
                if 'predictions' in view_data:
                    preds = view_data['predictions']
                    print(f"    Predictions: {type(preds)}")
                    if isinstance(preds, list):
                        for i, pred in enumerate(preds):
                            if isinstance(pred, torch.Tensor):
                                print(f"      [{i}]: {pred.shape}")
                
                if 'features' in view_data:
                    feats = view_data['features']
                    print(f"    Features: {type(feats)}")
                    if isinstance(feats, list):
                        for i, feat in enumerate(feats):
                            if isinstance(feat, torch.Tensor):
                                print(f"      [{i}]: {feat.shape}")
        
        # Check alignment
        if 'alignment' in outputs and outputs['alignment'] is not None:
            align_data = outputs['alignment']
            print(f"  Alignment:")
            for key, value in align_data.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape}")
        
        # Test loss computation
        print(f"\nTesting loss computation...")
        criterion = CombinedLoss(num_classes=config['model']['num_classes'])
        
        losses = criterion(outputs, labels)
        print(f"✓ Loss computation successful!")
        
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
        
        # Test training step
        print(f"\nTesting training step...")
        from src.optimizers import create_optimizer_with_config
        
        train_config = {
            'training': {
                'learning_rate': 0.01,
                'optimizer': 'sgd'
            }
        }
        
        optimizer, _ = create_optimizer_with_config(model, train_config)
        
        model.train()
        optimizer.zero_grad()
        
        outputs = model(sat_images, drone_images)
        losses = criterion(outputs, labels)
        total_loss = losses['total']
        
        total_loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful!")
        print(f"  Loss: {total_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_community_clustering():
    """Test community clustering module separately."""
    print("\nTesting community clustering module...")
    
    try:
        from src.models.fsra_improved import CommunityClusteringModule
        
        # Create module
        clustering = CommunityClusteringModule(feature_dim=512, num_clusters=3)
        
        # Test input
        batch_size = 2
        feature_map = torch.randn(batch_size, 512, 16, 16)
        
        print(f"  Input feature map: {feature_map.shape}")
        
        # Forward pass
        clustered_features, communities = clustering(feature_map)
        
        print(f"✓ Community clustering successful!")
        print(f"  Clustered features: {clustered_features.shape}")
        print(f"  Number of communities per batch: {[len(comm) for comm in communities]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Community clustering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("FSRA Improved Model Test")
    print("=" * 60)
    
    success = True
    
    if not test_community_clustering():
        success = False
    
    if not test_fsra_improved():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("FSRA Improved model is working correctly!")
        print("\nYou can now run training with:")
        print("  python train.py --config config/fsra_improved_config.yaml --data-dir data --batch-size 8 --learning-rate 0.001 --num-epochs 10 --gpu-ids \"0\"")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
