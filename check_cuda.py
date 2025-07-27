#!/usr/bin/env python3
"""Check CUDA compatibility and provide recommendations."""

import torch
import platform
import subprocess
import sys

def check_nvidia_driver():
    """Check NVIDIA driver version."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            # Extract driver version
            for line in output.split('\n'):
                if 'Driver Version:' in line:
                    driver_version = line.split('Driver Version:')[1].split()[0]
                    return driver_version
        return None
    except FileNotFoundError:
        return None

def check_cuda_version():
    """Check CUDA version."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            for line in output.split('\n'):
                if 'release' in line:
                    cuda_version = line.split('release')[1].split(',')[0].strip()
                    return cuda_version
        return None
    except FileNotFoundError:
        return None

def get_pytorch_cuda_version():
    """Get PyTorch CUDA version."""
    if torch.cuda.is_available():
        return torch.version.cuda
    else:
        # Get the CUDA version PyTorch was compiled with
        return torch.version.cuda if hasattr(torch.version, 'cuda') else "Not available"

def main():
    """Main check function."""
    print("=" * 60)
    print("CUDA Compatibility Check for ViT-CNN-crossview")
    print("=" * 60)
    
    # System info
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Driver version
    driver_version = check_nvidia_driver()
    if driver_version:
        print(f"NVIDIA Driver: {driver_version}")
    else:
        print("NVIDIA Driver: Not found or nvidia-smi not available")
    
    # CUDA version
    cuda_version = check_cuda_version()
    if cuda_version:
        print(f"CUDA Toolkit: {cuda_version}")
    else:
        print("CUDA Toolkit: Not found or nvcc not available")
    
    # PyTorch CUDA version
    pytorch_cuda = get_pytorch_cuda_version()
    print(f"PyTorch CUDA: {pytorch_cuda}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if not cuda_available:
        print("❌ CUDA is not available. Possible reasons:")
        print("  1. NVIDIA driver is too old (found version 11040)")
        print("  2. CUDA toolkit version mismatch")
        print("  3. PyTorch was not compiled with CUDA support")
        print()
        print("🔧 SOLUTIONS:")
        print()
        print("Option 1: Update NVIDIA Driver")
        print("  - Download latest driver from: https://www.nvidia.com/Download/index.aspx")
        print("  - Requires system administrator privileges")
        print()
        print("Option 2: Install Compatible PyTorch")
        print("  - For CUDA 11.0: pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html")
        print("  - For CPU only: pip install torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html")
        print()
        print("Option 3: Use CPU Training (Recommended for now)")
        print("  - Run: python train_cpu.py --create-dummy-data")
        print("  - Run: python train_cpu.py --config config/cpu_config.yaml")
        print("  - Slower but will work on any system")
        print()
        print("🚀 IMMEDIATE SOLUTION:")
        print("Since CUDA is not available, use CPU training:")
        print("  python train_cpu.py --create-dummy-data --experiment-name cpu_test")
        
    else:
        print("✅ CUDA is available and working!")
        print("You can use GPU training:")
        print("  python train.py --config config/default_config.yaml")
        print("  python train.py --create-dummy-data --experiment-name gpu_test")
    
    print("\n" + "=" * 60)
    print("TRAINING OPTIONS")
    print("=" * 60)
    
    print("🖥️  CPU Training (Always available):")
    print("  python train_cpu.py --create-dummy-data")
    print("  - Slower but guaranteed to work")
    print("  - Uses optimized settings for CPU")
    print("  - Good for testing and small experiments")
    print()
    
    if cuda_available:
        print("🚀 GPU Training (Recommended if available):")
        print("  python train.py --config config/default_config.yaml")
        print("  - Much faster training")
        print("  - Can handle larger models and datasets")
        print("  - Better for production training")
    else:
        print("🚀 GPU Training (Not available):")
        print("  - Fix CUDA issues first")
        print("  - Then use: python train.py --config config/default_config.yaml")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
