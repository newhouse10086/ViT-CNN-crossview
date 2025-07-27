#!/usr/bin/env python3
"""Test script to verify ViT-CNN-crossview installation."""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test all important imports."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision import failed: {e}")
        return False
    
    # Test project imports
    try:
        from src.models import create_model
        print("✓ Models module")
    except ImportError as e:
        print(f"✗ Models module import failed: {e}")
        return False
    
    try:
        from src.datasets import make_dataloader
        print("✓ Datasets module")
    except ImportError as e:
        print(f"✗ Datasets module import failed: {e}")
        return False
    
    try:
        from src.losses import CombinedLoss
        print("✓ Losses module")
    except ImportError as e:
        print(f"✗ Losses module import failed: {e}")
        return False
    
    try:
        from src.utils import setup_logger, load_config
        print("✓ Utils module")
    except ImportError as e:
        print(f"✗ Utils module import failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from src.models import create_model
        
        # Test config
        config = {
            'model': {
                'name': 'ViTCNN',
                'num_classes': 10,
                'num_final_clusters': 3,
                'use_pretrained_resnet': False,  # Disable for testing
                'use_pretrained_vit': False,
                'return_features': True
            },
            'data': {
                'views': 2
            }
        }
        
        model = create_model(config)
        print("✓ Model created successfully")
        
        # Test model info
        from src.models.model_factory import get_model_info
        info = get_model_info(model)
        print(f"  Total parameters: {info['total_parameters']:,}")
        print(f"  Model size: {info['model_size_mb']:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return False


def test_dataloader():
    """Test dataloader creation."""
    print("\nTesting dataloader creation...")
    
    try:
        from src.datasets import create_dummy_dataset, make_dataloader
        
        # Create dummy data
        dummy_dir = "test_data"
        create_dummy_dataset(dummy_dir, num_classes=5, images_per_class=2)
        print("✓ Dummy dataset created")
        
        # Test config
        config = {
            'data': {
                'data_dir': dummy_dir,
                'batch_size': 4,
                'num_workers': 0,  # Use 0 for testing
                'image_height': 256,
                'image_width': 256,
                'views': 2,
                'sample_num': 2,
                'pad': 0,
                'color_jitter': False,
                'random_erasing_prob': 0.0
            },
            'training': {
                'use_data_augmentation': False
            }
        }
        
        dataloader, class_names, dataset_sizes = make_dataloader(config, create_dummy=True)
        print("✓ Dataloader created successfully")
        print(f"  Classes: {len(class_names)}")
        print(f"  Dataset sizes: {dataset_sizes}")
        
        # Test one batch
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                print(f"  Batch shapes: sat={sat_images.shape}, drone={drone_images.shape}")
                break
        
        # Cleanup
        import shutil
        shutil.rmtree(dummy_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"✗ Dataloader test failed: {e}")
        traceback.print_exc()
        return False


def test_loss_function():
    """Test loss function."""
    print("\nTesting loss function...")
    
    try:
        import torch
        from src.losses import CombinedLoss
        
        # Create loss function
        criterion = CombinedLoss(
            num_classes=10,
            triplet_weight=0.3,
            kl_weight=0.0,
            alignment_weight=1.0,
            use_kl_loss=False
        )
        print("✓ Loss function created")
        
        # Test with dummy data
        batch_size = 4
        num_classes = 10
        feature_dim = 512
        
        # Mock outputs
        outputs = {
            'satellite': {
                'predictions': [torch.randn(batch_size, num_classes)],
                'features': [torch.randn(batch_size, feature_dim)]
            },
            'drone': {
                'predictions': [torch.randn(batch_size, num_classes)],
                'features': [torch.randn(batch_size, feature_dim)]
            },
            'alignment': {
                'satellite_aligned': torch.randn(batch_size, feature_dim),
                'drone_aligned': torch.randn(batch_size, feature_dim),
                'satellite_original': torch.randn(batch_size, feature_dim),
                'drone_original': torch.randn(batch_size, feature_dim)
            }
        }
        
        labels = torch.randint(0, num_classes, (batch_size,))
        
        losses = criterion(outputs, labels)
        print("✓ Loss computation successful")
        print(f"  Total loss: {losses['total'].item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from src.utils import load_config, save_config
        
        # Create test config
        test_config = {
            'model': {'name': 'ViTCNN', 'num_classes': 10},
            'data': {'batch_size': 16, 'views': 2},
            'training': {'num_epochs': 100, 'learning_rate': 0.001}
        }
        
        # Save and load config
        config_path = "test_config.yaml"
        save_config(test_config, config_path)
        loaded_config = load_config(config_path)
        
        print("✓ Configuration save/load successful")
        
        # Cleanup
        os.remove(config_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_training_script():
    """Test training script with dummy data."""
    print("\nTesting training script...")
    
    try:
        import subprocess
        import tempfile
        
        # Create temporary config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = """
model:
  name: "ViTCNN"
  num_classes: 5
  num_final_clusters: 2
  use_pretrained_resnet: false
  use_pretrained_vit: false

data:
  data_dir: "test_dummy_data"
  batch_size: 2
  num_workers: 0
  views: 2
  sample_num: 1

training:
  num_epochs: 2
  learning_rate: 0.01

system:
  gpu_ids: "0"
  use_gpu: false
  log_dir: "test_logs"
  checkpoint_dir: "test_checkpoints"
"""
            f.write(config_content)
            config_path = f.name
        
        # Run training script
        cmd = [
            sys.executable, "train.py",
            "--config", config_path,
            "--create-dummy-data",
            "--experiment-name", "test_run"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Training script test successful")
        else:
            print(f"✗ Training script failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        
        # Cleanup
        os.unlink(config_path)
        import shutil
        shutil.rmtree("test_dummy_data", ignore_errors=True)
        shutil.rmtree("test_logs", ignore_errors=True)
        shutil.rmtree("test_checkpoints", ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"✗ Training script test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("ViT-CNN-crossview Installation Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Creation Test", test_model_creation),
        ("Dataloader Test", test_dataloader),
        ("Loss Function Test", test_loss_function),
        ("Configuration Test", test_config_loading),
        ("Training Script Test", test_training_script),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"{'=' * 60}")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
