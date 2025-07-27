#!/usr/bin/env python3
"""Test script to verify all imports work after fixes."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test all important imports."""
    print("Testing imports after fixes...")
    
    try:
        print("1. Testing utils imports...")
        from src.utils import MetricsCalculator, AverageMeter, RankingMetricsCalculator
        from src.utils import setup_logger, get_logger
        from src.utils import load_config, save_config, merge_configs
        from src.utils import TrainingVisualizer
        print("✓ Utils imports successful")
    except ImportError as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    try:
        print("2. Testing optimizers imports...")
        from src.optimizers import create_optimizer, create_scheduler, create_optimizer_with_config
        print("✓ Optimizers imports successful")
    except ImportError as e:
        print(f"✗ Optimizers import failed: {e}")
        return False
    
    try:
        print("3. Testing models imports...")
        from src.models import create_model
        print("✓ Models imports successful")
    except ImportError as e:
        print(f"✗ Models import failed: {e}")
        return False
    
    try:
        print("4. Testing datasets imports...")
        from src.datasets import make_dataloader, create_dummy_dataset
        print("✓ Datasets imports successful")
    except ImportError as e:
        print(f"✗ Datasets import failed: {e}")
        return False
    
    try:
        print("5. Testing losses imports...")
        from src.losses import CombinedLoss, TripletLoss
        print("✓ Losses imports successful")
    except ImportError as e:
        print(f"✗ Losses import failed: {e}")
        return False
    
    try:
        print("6. Testing trainer imports...")
        from src.trainer import Trainer, Evaluator
        print("✓ Trainer imports successful")
    except ImportError as e:
        print(f"✗ Trainer import failed: {e}")
        return False
    
    print("\n🎉 All imports successful!")
    return True

def test_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        # Test config loading
        from src.utils import load_config
        
        # Test with default config if it exists
        if Path("config/default_config.yaml").exists():
            config = load_config("config/default_config.yaml")
            print("✓ Config loading works")
        else:
            print("⚠ Default config not found, creating test config")
            config = {
                'model': {'name': 'ViTCNN', 'num_classes': 10},
                'data': {'batch_size': 4, 'views': 2},
                'training': {'num_epochs': 1}
            }
        
        # Test model creation
        from src.models import create_model
        model = create_model(config)
        print("✓ Model creation works")
        
        # Test optimizer creation
        from src.optimizers import create_optimizer_with_config
        optimizer, scheduler = create_optimizer_with_config(model, config)
        print("✓ Optimizer creation works")
        
        # Test loss function
        from src.losses import CombinedLoss
        criterion = CombinedLoss(num_classes=config['model']['num_classes'])
        print("✓ Loss function creation works")
        
        # Test metrics calculator
        from src.utils import MetricsCalculator, RankingMetricsCalculator
        metrics_calc = MetricsCalculator(num_classes=config['model']['num_classes'])
        ranking_calc = RankingMetricsCalculator()
        print("✓ Metrics calculators work")
        
        print("✓ All functionality tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Import and Functionality Test (After Fixes)")
    print("=" * 60)
    
    success = True
    
    if not test_imports():
        success = False
    
    if not test_functionality():
        success = False
    
    if success:
        print("\n✅ All tests passed! The fixes are working correctly.")
        print("\nYou can now run:")
        print("  python train.py --config config/default_config.yaml")
        print("  python train.py --create-dummy-data --experiment-name test")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)
