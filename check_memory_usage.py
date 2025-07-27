#!/usr/bin/env python3
"""
Check memory usage during training with different batch sizes.
"""

import torch
import psutil
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import load_config
from src.models import create_model


def check_memory_usage(batch_size):
    """Check memory usage for a specific batch size."""
    print(f"\n🔍 Checking memory usage for batch_size={batch_size}")
    
    # Load config
    config = load_config('config/your_innovation_config.yaml')
    config['data']['batch_size'] = batch_size
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    model.train()
    
    # Create dummy data
    sat_images = torch.randn(batch_size, 3, 256, 256).to(device)
    drone_images = torch.randn(batch_size, 3, 256, 256).to(device)
    sat_labels = torch.randint(0, 701, (batch_size,)).to(device)
    
    # Check initial memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        print(f"Initial GPU memory: {initial_memory:.1f} MB")
    
    ram_initial = psutil.virtual_memory().used / 1024**3  # GB
    print(f"Initial RAM: {ram_initial:.1f} GB")
    
    # Forward pass
    start_time = time.time()
    
    try:
        outputs = model(sat_images, drone_images)
        forward_time = time.time() - start_time
        
        # Check memory after forward
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            print(f"Peak GPU memory: {peak_memory:.1f} MB")
            print(f"Current GPU memory: {current_memory:.1f} MB")
            print(f"Memory increase: {current_memory - initial_memory:.1f} MB")
        
        ram_current = psutil.virtual_memory().used / 1024**3  # GB
        print(f"Current RAM: {ram_current:.1f} GB")
        print(f"RAM increase: {(ram_current - ram_initial)*1024:.1f} MB")
        
        print(f"Forward pass time: {forward_time:.3f}s")
        print(f"Time per sample: {forward_time/batch_size:.3f}s")
        
        return {
            'batch_size': batch_size,
            'forward_time': forward_time,
            'time_per_sample': forward_time/batch_size,
            'gpu_memory_mb': current_memory - initial_memory if torch.cuda.is_available() else 0,
            'ram_memory_mb': (ram_current - ram_initial) * 1024,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            'batch_size': batch_size,
            'success': False,
            'error': str(e)
        }
    
    finally:
        del model, sat_images, drone_images, sat_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Main function."""
    print("🎯 Memory Usage Analysis for FSRA_IMPROVED")
    
    batch_sizes = [8, 16, 32, 64]
    results = []
    
    for batch_size in batch_sizes:
        result = check_memory_usage(batch_size)
        results.append(result)
        time.sleep(2)  # Wait between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 MEMORY USAGE SUMMARY")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        print(f"{'Batch':<8} {'Time(s)':<10} {'Time/Sample':<12} {'GPU(MB)':<10} {'RAM(MB)':<10}")
        print(f"{'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
        
        for result in successful_results:
            print(f"{result['batch_size']:<8} "
                  f"{result['forward_time']:<10.3f} "
                  f"{result['time_per_sample']:<12.4f} "
                  f"{result['gpu_memory_mb']:<10.1f} "
                  f"{result['ram_memory_mb']:<10.1f}")
        
        # Check if time per sample increases significantly
        base_result = successful_results[0]  # batch_size=8
        base_time_per_sample = base_result['time_per_sample']
        
        print(f"\n🔍 Time per sample analysis (baseline: batch_size={base_result['batch_size']}):")
        for result in successful_results:
            ratio = result['time_per_sample'] / base_time_per_sample
            if ratio > 1.5:
                status = "⚠️  SLOW"
            elif ratio > 1.2:
                status = "⚡ OK"
            else:
                status = "✅ GOOD"
            
            print(f"  Batch {result['batch_size']}: {ratio:.2f}x slower {status}")
    
    else:
        print("❌ No successful tests")


if __name__ == "__main__":
    main()
