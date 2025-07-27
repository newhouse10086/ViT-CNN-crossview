#!/usr/bin/env python3
"""Test enhanced training with metrics."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_metrics_logger():
    """Test the metrics logger."""
    print("Testing MetricsLogger...")
    
    try:
        from src.utils.metrics_logger import MetricsLogger
        
        # Create logger
        logger = MetricsLogger(log_dir="test_logs", experiment_name="test_experiment")
        
        # Test logging some fake metrics
        for epoch in range(1, 4):
            fake_metrics = {
                'train_loss': 10.0 - epoch,
                'train_accuracy': 0.1 * epoch,
                'train_precision': 0.1 * epoch + 0.05,
                'train_recall': 0.1 * epoch + 0.03,
                'train_f1_score': 0.1 * epoch + 0.04,
                'learning_rate': 0.001,
                'epoch_time': 120.0,
                'avg_batch_time': 0.5,
                'batches_processed': 100
            }
            
            logger.log_epoch_metrics(epoch, fake_metrics)
            print(f"✓ Logged metrics for epoch {epoch}")
        
        # Test summary report
        summary = logger.generate_summary_report()
        print("✓ Generated summary report")
        print(summary[:200] + "...")
        
        return True
        
    except Exception as e:
        print(f"❌ MetricsLogger test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_training_import():
    """Test importing the enhanced training script."""
    print("\nTesting enhanced training imports...")
    
    try:
        # Test imports
        from src.utils.metrics_logger import MetricsLogger
        from src.utils import load_config, setup_logging, set_seed, get_device_info
        from src.models import create_model
        from src.datasets import make_dataloader
        from src.losses import CombinedLoss
        from src.optimizers import create_optimizer_with_config
        
        print("✓ All imports successful")
        
        # Test config loading
        config = load_config('config/simple_fsra_config.yaml')
        print(f"✓ Config loaded: {config['model']['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced training import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Enhanced Training Test")
    print("=" * 60)
    
    success = True
    
    if not test_metrics_logger():
        success = False
    
    if not test_enhanced_training_import():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("Enhanced training with metrics is ready!")
        print("\nYou can now run enhanced training with:")
        print("  python train_with_metrics.py --config config/simple_fsra_config.yaml --data-dir data --batch-size 8 --learning-rate 0.001 --num-epochs 5 --gpu-ids \"0\"")
        print("\nThis will generate:")
        print("  📊 Detailed epoch metrics (accuracy, precision, recall, F1)")
        print("  📈 Training plots (loss, accuracy, learning rate)")
        print("  📋 CSV and JSON metrics files")
        print("  📝 Summary report")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
