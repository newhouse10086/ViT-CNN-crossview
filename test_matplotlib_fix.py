#!/usr/bin/env python3
"""Test matplotlib/seaborn compatibility fix."""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_matplotlib_import():
    """Test matplotlib import and style."""
    print("Testing matplotlib import and style...")
    
    try:
        import matplotlib.pyplot as plt
        print(f"✓ Matplotlib version: {plt.matplotlib.__version__}")
        
        # Test available styles
        available_styles = plt.style.available
        print(f"✓ Available styles: {len(available_styles)} styles")
        
        # Check for seaborn styles
        seaborn_styles = [s for s in available_styles if 'seaborn' in s]
        if seaborn_styles:
            print(f"✓ Seaborn styles available: {seaborn_styles}")
        else:
            print("⚠ No seaborn styles available")
        
        return True
        
    except Exception as e:
        print(f"✗ Matplotlib test failed: {e}")
        return False

def test_seaborn_import():
    """Test seaborn import."""
    print("Testing seaborn import...")
    
    try:
        import seaborn as sns
        print(f"✓ Seaborn version: {sns.__version__}")
        return True
    except ImportError:
        print("⚠ Seaborn not available")
        return True  # Not critical
    except Exception as e:
        print(f"✗ Seaborn test failed: {e}")
        return False

def test_visualization_import():
    """Test visualization module import."""
    print("Testing visualization module import...")
    
    try:
        from src.utils.visualization import TrainingVisualizer, plot_confusion_matrix
        print("✓ Visualization module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Visualization import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_confusion_matrix_plot():
    """Test confusion matrix plotting."""
    print("Testing confusion matrix plotting...")
    
    try:
        from src.utils.visualization import plot_confusion_matrix
        
        # Create test confusion matrix
        cm = np.array([[10, 2, 1], [1, 15, 2], [0, 1, 12]])
        class_names = ['Class A', 'Class B', 'Class C']
        
        # Test plotting
        fig = plot_confusion_matrix(
            cm, 
            class_names=class_names,
            title="Test Confusion Matrix",
            show=False  # Don't display
        )
        
        print("✓ Confusion matrix plotting successful")
        
        # Close figure to free memory
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"✗ Confusion matrix plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_visualizer():
    """Test TrainingVisualizer class."""
    print("Testing TrainingVisualizer...")
    
    try:
        from src.utils.visualization import TrainingVisualizer
        
        # Create visualizer
        visualizer = TrainingVisualizer(
            save_dir="test_logs",
            experiment_name="test_experiment"
        )
        
        # Test updating metrics
        test_metrics = {
            'train_loss': 0.5,
            'train_accuracy': 0.8,
            'val_loss': 0.6,
            'val_accuracy': 0.75,
            'learning_rate': 0.001
        }
        
        visualizer.update_metrics(1, test_metrics)
        print("✓ TrainingVisualizer update successful")
        
        return True
        
    except Exception as e:
        print(f"✗ TrainingVisualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_imports():
    """Test full project imports."""
    print("Testing full project imports...")
    
    try:
        from src.utils import TrainingVisualizer, MetricsCalculator
        from src.models import create_model
        from src.losses import CombinedLoss
        print("✓ All project imports successful")
        return True
    except Exception as e:
        print(f"✗ Project imports failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Matplotlib/Seaborn Compatibility Test")
    print("=" * 60)
    
    success = True
    
    if not test_matplotlib_import():
        success = False
    
    print()
    if not test_seaborn_import():
        success = False
    
    print()
    if not test_visualization_import():
        success = False
    
    print()
    if not test_confusion_matrix_plot():
        success = False
    
    print()
    if not test_training_visualizer():
        success = False
    
    print()
    if not test_full_imports():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL MATPLOTLIB/SEABORN TESTS PASSED!")
        print("The visualization compatibility issues have been resolved.")
        print("You can now run:")
        print("  python train.py --config config/default_config.yaml")
        print("  python train_cpu.py --create-dummy-data")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please run: python complete_server_fix.py")
        print("Or check your matplotlib/seaborn versions")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
