#!/usr/bin/env python3
"""Test sklearn compatibility fix."""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_sklearn_imports():
    """Test sklearn imports."""
    print("Testing sklearn imports...")
    
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        print("✓ Basic sklearn metrics OK")
    except ImportError as e:
        print(f"✗ Basic sklearn metrics failed: {e}")
        return False
    
    try:
        # This should work with our fallback
        from src.utils.metrics import top_k_accuracy_score
        print("✓ top_k_accuracy_score import OK (with fallback if needed)")
    except ImportError as e:
        print(f"✗ top_k_accuracy_score import failed: {e}")
        return False
    
    return True

def test_top_k_accuracy():
    """Test top-k accuracy function."""
    print("Testing top-k accuracy function...")
    
    try:
        from src.utils.metrics import top_k_accuracy_score
        
        # Test data
        y_true = np.array([0, 1, 2, 1, 0])
        y_prob = np.array([
            [0.8, 0.1, 0.1],  # Correct: 0
            [0.2, 0.7, 0.1],  # Correct: 1
            [0.1, 0.2, 0.7],  # Correct: 2
            [0.3, 0.6, 0.1],  # Correct: 1
            [0.9, 0.05, 0.05] # Correct: 0
        ])
        
        # Test top-1 accuracy
        top1_acc = top_k_accuracy_score(y_true, y_prob, k=1)
        print(f"✓ Top-1 accuracy: {top1_acc:.3f}")
        
        # Test top-2 accuracy
        top2_acc = top_k_accuracy_score(y_true, y_prob, k=2)
        print(f"✓ Top-2 accuracy: {top2_acc:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ top-k accuracy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_calculator():
    """Test MetricsCalculator with sklearn compatibility."""
    print("Testing MetricsCalculator...")
    
    try:
        from src.utils import MetricsCalculator
        
        # Create calculator
        calc = MetricsCalculator(num_classes=3)
        
        # Test data
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 1, 0])
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.3, 0.6, 0.1],
            [0.9, 0.05, 0.05]
        ])
        
        # Update calculator
        calc.update(y_pred, y_true, y_prob)
        
        # Compute metrics
        metrics = calc.compute_metrics()
        
        print(f"✓ Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"✓ Precision: {metrics.get('precision_macro', 0):.3f}")
        print(f"✓ Recall: {metrics.get('recall_macro', 0):.3f}")
        
        if 'top_1_accuracy' in metrics:
            print(f"✓ Top-1 accuracy: {metrics['top_1_accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ MetricsCalculator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_imports():
    """Test full project imports."""
    print("Testing full project imports...")
    
    try:
        from src.utils import MetricsCalculator, load_config, setup_logger
        from src.models import create_model
        from src.losses import CombinedLoss
        from src.optimizers import create_optimizer_with_config
        print("✓ All project imports successful")
        return True
    except Exception as e:
        print(f"✗ Project imports failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Sklearn Compatibility Test")
    print("=" * 60)
    
    success = True
    
    if not test_sklearn_imports():
        success = False
    
    print()
    if not test_top_k_accuracy():
        success = False
    
    print()
    if not test_metrics_calculator():
        success = False
    
    print()
    if not test_full_imports():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL SKLEARN COMPATIBILITY TESTS PASSED!")
        print("The sklearn compatibility issues have been resolved.")
        print("You can now run:")
        print("  python train.py --config config/default_config.yaml")
        print("  python train_cpu.py --create-dummy-data")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please run: python complete_server_fix.py")
        print("Or check your sklearn version: pip install scikit-learn>=0.24.0")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
