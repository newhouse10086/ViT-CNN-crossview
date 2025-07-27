#!/usr/bin/env python3
"""
Test script for enhanced memory monitoring training.
Tests the updated train_with_memory_monitor.py with comprehensive metrics.
"""

import os
import sys
import subprocess
from pathlib import Path

def test_memory_training():
    """Test the enhanced memory monitoring training script."""
    print("🧪 Testing Enhanced Memory Monitoring Training")
    print("="*70)
    
    # Check if config file exists
    config_path = "config/your_innovation_config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Please make sure the config file exists.")
        return False
    
    print(f"✅ Config file found: {config_path}")
    
    # Test command
    cmd = [
        "python", "train_with_memory_monitor.py",
        "--config", config_path,
        "--data-dir", "data",
        "--batch-size", "4",  # Small batch for testing
        "--num-epochs", "2",  # Just 2 epochs for testing
        "--gpu-ids", "0"
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    print("="*70)
    
    try:
        # Run the training script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Memory monitoring training executed successfully!")
            print("\n📊 Output preview:")
            print("-" * 50)
            # Show last 25 lines of output
            output_lines = result.stdout.split('\n')
            for line in output_lines[-25:]:
                if line.strip():
                    print(line)
            print("-" * 50)
            
            # Check for key metrics in output
            output = result.stdout
            metrics_found = []
            
            if "Accuracy:" in output:
                metrics_found.append("✅ Accuracy")
            if "Recall:" in output:
                metrics_found.append("✅ Recall")
            if "AUC:" in output:
                metrics_found.append("✅ AUC")
            if "F1-Score:" in output:
                metrics_found.append("✅ F1-Score")
            if "Top-5 Accuracy:" in output:
                metrics_found.append("✅ Top-5 Accuracy")
            if "GPU Memory Increase:" in output:
                metrics_found.append("✅ GPU Memory Monitoring")
            if "Time Degradation:" in output:
                metrics_found.append("✅ Time Degradation")
            if "MEMORY ANALYSIS:" in output:
                metrics_found.append("✅ Memory Analysis")
            if "MEMORY HEALTH ASSESSMENT:" in output:
                metrics_found.append("✅ Memory Health Assessment")
            if "PERFORMANCE ASSESSMENT:" in output:
                metrics_found.append("✅ Performance Assessment")
            
            print(f"\n📈 Metrics & Memory Features Detection:")
            for metric in metrics_found:
                print(f"  {metric}")
            
            if len(metrics_found) >= 8:
                print(f"\n🎉 Enhanced memory monitoring working correctly!")
                return True
            else:
                print(f"\n⚠️  Some features may be missing. Found {len(metrics_found)}/10 expected features.")
                return False
                
        else:
            print(f"❌ Memory training script failed with return code: {result.returncode}")
            print(f"\nError output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Memory training script timed out (5 minutes)")
        print("This might be normal for larger datasets.")
        return False
        
    except Exception as e:
        print(f"❌ Error running memory training script: {e}")
        return False


def test_memory_metrics_calculator():
    """Test the MemoryMonitoringMetrics class directly."""
    print("\n🧮 Testing MemoryMonitoringMetrics")
    print("="*50)
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Import the updated train_with_memory_monitor module
        import torch
        import numpy as np
        
        # Import from train_with_memory_monitor
        sys.path.insert(0, str(Path(__file__).parent))
        from train_with_memory_monitor import MemoryMonitoringMetrics
        
        # Create test data
        num_classes = 10
        batch_size = 32
        
        calc = MemoryMonitoringMetrics(num_classes)
        
        # Simulate some predictions and targets
        for _ in range(5):  # 5 batches
            predictions = torch.randn(batch_size, num_classes)
            targets = torch.randint(0, num_classes, (batch_size,))
            loss = torch.tensor(2.5)
            
            calc.update(predictions, targets, loss)
        
        # Compute metrics
        metrics = calc.compute_metrics()
        
        print("✅ MemoryMonitoringMetrics test successful!")
        print("📊 Sample metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Check if all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'auc', 
            'top5_accuracy', 'macro_precision', 'macro_recall', 
            'macro_f1', 'avg_loss', 'num_samples'
        ]
        
        missing_metrics = [m for m in expected_metrics if m not in metrics]
        if missing_metrics:
            print(f"⚠️  Missing metrics: {missing_metrics}")
            return False
        else:
            print("✅ All expected metrics present!")
            return True
            
    except Exception as e:
        print(f"❌ MemoryMonitoringMetrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_display_functions():
    """Test the display functions."""
    print("\n🎨 Testing Display Functions")
    print("="*40)
    
    try:
        # Import display functions
        sys.path.insert(0, str(Path(__file__).parent))
        from train_with_memory_monitor import (
            print_detailed_metrics_with_memory,
            print_epoch_summary_with_memory,
            print_memory_training_summary
        )
        
        # Create mock metrics
        mock_metrics = {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1_score': 0.85,
            'auc': 0.92,
            'top5_accuracy': 0.95,
            'macro_precision': 0.82,
            'macro_recall': 0.86,
            'macro_f1': 0.84,
            'avg_loss': 0.45,
            'num_samples': 1000,
            'epoch_time': 120.5,
            'avg_batch_time': 0.8,
            'time_degradation_percent': 5.2,
            'gpu_memory_increase': 150.0,
            'ram_memory_increase': 0.5,
            'final_gpu_memory': 2500.0,
            'final_ram_memory': 8.2,
            'successful_batches': 125,
            'total_batches': 125
        }
        
        print("Testing detailed metrics display...")
        print_detailed_metrics_with_memory(mock_metrics, 1, 10, 0.001)
        
        print("\nTesting epoch summary...")
        print_epoch_summary_with_memory(mock_metrics, 1, 10)
        
        print("\nTesting training summary...")
        mock_history = [
            {'epoch': 1, 'accuracy': 0.75, 'auc': 0.85, 'loss': 0.65, 
             'f1_score': 0.73, 'recall': 0.76, 'precision': 0.74,
             'gpu_memory_increase': 100, 'time_degradation': 2.0},
            {'epoch': 2, 'accuracy': 0.85, 'auc': 0.92, 'loss': 0.45,
             'f1_score': 0.85, 'recall': 0.87, 'precision': 0.83,
             'gpu_memory_increase': 150, 'time_degradation': 5.2}
        ]
        print_memory_training_summary(mock_history, 0.85, 0.92, 2)
        
        print("✅ Display functions test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Display functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🎯 Enhanced Memory Monitoring Training Test Suite")
    print("="*70)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: MemoryMonitoringMetrics
    if test_memory_metrics_calculator():
        success_count += 1
    
    # Test 2: Display Functions
    if test_display_functions():
        success_count += 1
    
    # Test 3: Full Memory Training (optional, requires data)
    data_dir = "data"
    if os.path.exists(data_dir):
        print(f"\n📁 Data directory found, testing full memory training...")
        if test_memory_training():
            success_count += 1
    else:
        print(f"\n📁 Data directory not found, skipping full training test")
        print(f"   (This is normal if you haven't set up the dataset yet)")
        total_tests = 2  # Only count the first two tests
    
    # Summary
    print(f"\n{'='*70}")
    print(f"🎊 TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print(f"🎉 All tests passed! Enhanced memory monitoring is ready!")
        print(f"🚀 You can now use train_with_memory_monitor.py with comprehensive metrics!")
    else:
        print(f"⚠️  Some tests failed. Please check the issues above.")
    
    return success_count == total_tests


if __name__ == "__main__":
    main()
