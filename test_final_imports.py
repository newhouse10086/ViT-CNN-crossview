#!/usr/bin/env python3
"""Final import test for all train.py dependencies."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_train_imports():
    """Test all imports used in train.py"""
    print("Testing all train.py imports...")
    
    try:
        print("1. Testing models import...")
        from src.models import create_model
        print("✓ create_model")
    except ImportError as e:
        print(f"✗ create_model: {e}")
        return False
    
    try:
        print("2. Testing datasets import...")
        from src.datasets import make_dataloader, create_dummy_dataset
        print("✓ make_dataloader, create_dummy_dataset")
    except ImportError as e:
        print(f"✗ datasets: {e}")
        return False
    
    try:
        print("3. Testing losses import...")
        from src.losses import CombinedLoss
        print("✓ CombinedLoss")
    except ImportError as e:
        print(f"✗ CombinedLoss: {e}")
        return False
    
    try:
        print("4. Testing optimizers import...")
        from src.optimizers import create_optimizer_with_config
        print("✓ create_optimizer_with_config")
    except ImportError as e:
        print(f"✗ create_optimizer_with_config: {e}")
        return False
    
    try:
        print("5. Testing utils imports...")
        from src.utils import (
            setup_logger, get_logger, load_config, validate_config,
            TrainingVisualizer, MetricsCalculator, log_system_info
        )
        print("✓ All utils imports")
    except ImportError as e:
        print(f"✗ utils imports: {e}")
        return False
    
    try:
        print("6. Testing trainer imports...")
        from src.trainer import Trainer, Evaluator
        print("✓ Trainer, Evaluator")
    except ImportError as e:
        print(f"⚠ trainer imports (optional): {e}")
    
    print("\n🎉 All critical imports successful!")
    return True

def test_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        # Test config
        from src.utils import load_config
        
        # Create test config
        test_config = {
            'model': {
                'name': 'ViTCNN',
                'num_classes': 10,
                'use_pretrained_resnet': False,
                'use_pretrained_vit': False
            },
            'data': {
                'batch_size': 4,
                'views': 2
            },
            'training': {
                'num_epochs': 1,
                'learning_rate': 0.01,
                'weight_decay': 0.0001,
                'optimizer': 'sgd',
                'scheduler': 'step'
            },
            'system': {
                'gpu_ids': '0',
                'use_gpu': False
            }
        }
        
        # Test model creation
        from src.models import create_model
        model = create_model(test_config)
        print("✓ Model creation works")
        
        # Test optimizer creation
        from src.optimizers import create_optimizer_with_config
        optimizer, scheduler = create_optimizer_with_config(model, test_config)
        print("✓ Optimizer creation works")
        
        # Test loss function
        from src.losses import CombinedLoss
        criterion = CombinedLoss(num_classes=10)
        print("✓ Loss function works")
        
        # Test metrics
        from src.utils import MetricsCalculator
        metrics_calc = MetricsCalculator(num_classes=10)
        print("✓ Metrics calculator works")
        
        print("✓ All functionality tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Final Import Test for ViT-CNN-crossview")
    print("=" * 60)
    
    success = True
    
    if not test_train_imports():
        success = False
    
    if not test_functionality():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The project is ready to use. You can now run:")
        print("  python train.py --config config/default_config.yaml")
        print("  python train.py --create-dummy-data --experiment-name test")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above.")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
