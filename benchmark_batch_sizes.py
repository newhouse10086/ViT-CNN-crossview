#!/usr/bin/env python3
"""
Benchmark different batch sizes for FSRA_IMPROVED to find optimal performance.
"""

import os
import sys
import time
import torch
import psutil
import GPUtil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import load_config
from src.models import create_model
from src.datasets import make_dataloader


def get_gpu_memory():
    """Get GPU memory usage."""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return gpu.memoryUsed, gpu.memoryTotal
    except:
        pass
    return 0, 0


def benchmark_batch_size(batch_size, config_path="config/your_innovation_config.yaml"):
    """Benchmark a specific batch size."""
    print(f"\n{'='*60}")
    print(f"🔍 Benchmarking Batch Size: {batch_size}")
    print(f"{'='*60}")
    
    # Load config
    config = load_config(config_path)
    config['data']['batch_size'] = batch_size
    
    # Adjust learning rate for batch size
    base_lr = 0.001
    config['training']['learning_rate'] = base_lr * (batch_size / 8)  # Scale LR
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    try:
        # Create model
        start_time = time.time()
        model = create_model(config)
        model = model.to(device)
        model_time = time.time() - start_time
        
        # Create dataloader
        start_time = time.time()
        dataloader, class_names, dataset_sizes = make_dataloader(config)
        dataloader_time = time.time() - start_time
        
        # Get memory usage
        gpu_used, gpu_total = get_gpu_memory()
        ram_used = psutil.virtual_memory().used / 1024**3
        
        print(f"✅ Model Creation: {model_time:.2f}s")
        print(f"✅ Dataloader Creation: {dataloader_time:.2f}s")
        print(f"📊 GPU Memory: {gpu_used:.1f}/{gpu_total:.1f} MB")
        print(f"📊 RAM Usage: {ram_used:.1f} GB")
        
        # Benchmark forward pass
        model.train()
        batch_times = []
        
        print(f"🚀 Running forward pass benchmark...")
        
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_idx >= 5:  # Only test first 5 batches
                break
                
            try:
                start_time = time.time()
                
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                    
                    sat_images = sat_images.to(device)
                    drone_images = drone_images.to(device)
                    sat_labels = sat_labels.to(device)
                    
                    # Forward pass
                    with torch.no_grad():
                        if hasattr(model, 'module'):
                            outputs = model.module(sat_images, drone_images)
                        else:
                            outputs = model(sat_images, drone_images)
                    
                    batch_time = time.time() - start_time
                    batch_times.append(batch_time)
                    
                    # Get updated memory
                    gpu_used_batch, _ = get_gpu_memory()
                    
                    print(f"  Batch {batch_idx+1}: {batch_time:.3f}s, GPU: {gpu_used_batch:.1f}MB")
                    
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                return None
        
        if batch_times:
            avg_batch_time = sum(batch_times) / len(batch_times)
            estimated_epoch_time = avg_batch_time * len(dataloader)
            
            result = {
                'batch_size': batch_size,
                'avg_batch_time': avg_batch_time,
                'estimated_epoch_time': estimated_epoch_time,
                'gpu_memory_used': gpu_used,
                'ram_used': ram_used,
                'total_batches': len(dataloader),
                'samples_per_second': batch_size / avg_batch_time,
                'success': True
            }
            
            print(f"\n📈 Results:")
            print(f"  Average Batch Time: {avg_batch_time:.3f}s")
            print(f"  Estimated Epoch Time: {estimated_epoch_time/60:.1f} minutes")
            print(f"  Samples/Second: {result['samples_per_second']:.1f}")
            print(f"  Total Batches: {len(dataloader)}")
            
            return result
        else:
            print(f"❌ No successful batches")
            return None
            
    except Exception as e:
        print(f"❌ Failed to benchmark batch size {batch_size}: {e}")
        return None
    
    finally:
        # Clean up
        if 'model' in locals():
            del model
        if 'dataloader' in locals():
            del dataloader
        torch.cuda.empty_cache()


def main():
    """Main benchmarking function."""
    print("🎯 FSRA_IMPROVED Batch Size Benchmark")
    print("Testing different batch sizes to find optimal performance")
    
    # Test different batch sizes
    batch_sizes = [4, 8, 16, 32, 64]
    results = []
    
    for batch_size in batch_sizes:
        result = benchmark_batch_size(batch_size)
        if result:
            results.append(result)
        
        # Wait a bit between tests
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    if results:
        print(f"{'Batch Size':<12} {'Epoch Time':<12} {'Samples/s':<12} {'GPU Memory':<12}")
        print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        best_throughput = max(results, key=lambda x: x['samples_per_second'])
        best_time = min(results, key=lambda x: x['estimated_epoch_time'])
        
        for result in results:
            epoch_min = result['estimated_epoch_time'] / 60
            throughput = result['samples_per_second']
            gpu_mem = result['gpu_memory_used']
            
            marker = ""
            if result == best_throughput:
                marker += " 🚀"  # Best throughput
            if result == best_time:
                marker += " ⚡"  # Best time
            
            print(f"{result['batch_size']:<12} {epoch_min:<12.1f} {throughput:<12.1f} {gpu_mem:<12.1f}{marker}")
        
        print(f"\n🎯 Recommendations:")
        print(f"  🚀 Best Throughput: Batch Size {best_throughput['batch_size']} ({best_throughput['samples_per_second']:.1f} samples/s)")
        print(f"  ⚡ Fastest Epoch: Batch Size {best_time['batch_size']} ({best_time['estimated_epoch_time']/60:.1f} min/epoch)")
        
        # Find sweet spot (good balance)
        efficiency_scores = []
        for result in results:
            # Score based on throughput and reasonable epoch time
            throughput_score = result['samples_per_second'] / best_throughput['samples_per_second']
            time_score = best_time['estimated_epoch_time'] / result['estimated_epoch_time']
            efficiency_score = (throughput_score + time_score) / 2
            efficiency_scores.append((result, efficiency_score))
        
        best_balance = max(efficiency_scores, key=lambda x: x[1])[0]
        print(f"  ⚖️  Best Balance: Batch Size {best_balance['batch_size']}")
        
    else:
        print("❌ No successful benchmarks")


if __name__ == "__main__":
    main()
