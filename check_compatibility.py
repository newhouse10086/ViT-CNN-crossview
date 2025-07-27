#!/usr/bin/env python3
"""Check environment compatibility and provide fixes."""

import sys
import importlib
import subprocess
from packaging import version

def check_python_version():
    """Check Python version."""
    python_version = sys.version_info
    print(f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    elif python_version < (3, 8):
        print("⚠️  Python 3.8+ recommended")
        return True
    else:
        print("✅ Python version OK")
        return True

def check_package(package_name, min_version=None, import_name=None):
    """Check if package is available and meets version requirements."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        if hasattr(module, '__version__'):
            pkg_version = module.__version__
        else:
            pkg_version = "unknown"
        
        print(f"{package_name}: {pkg_version}")
        
        if min_version and pkg_version != "unknown":
            if version.parse(pkg_version) < version.parse(min_version):
                print(f"❌ {package_name} {min_version}+ required, found {pkg_version}")
                return False
        
        print(f"✅ {package_name} OK")
        return True
        
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False

def check_sklearn_functions():
    """Check specific sklearn functions."""
    try:
        from sklearn.metrics import top_k_accuracy_score
        print("✅ sklearn.metrics.top_k_accuracy_score available")
        return True
    except ImportError:
        print("⚠️  sklearn.metrics.top_k_accuracy_score not available (will use fallback)")
        return True  # Not critical, we have fallback

def check_torch_compatibility():
    """Check PyTorch compatibility."""
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"   Devices: {torch.cuda.device_count()}")
        else:
            print("⚠️  CUDA not available (will use CPU)")
        
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def provide_fixes():
    """Provide installation commands for missing packages."""
    print("\n" + "=" * 60)
    print("INSTALLATION FIXES")
    print("=" * 60)
    
    print("For PyTorch 1.10.2 environment compatibility:")
    print()
    
    print("1. Update scikit-learn (for top_k_accuracy_score):")
    print("   conda install scikit-learn>=0.24.0")
    print("   # or")
    print("   pip install scikit-learn>=0.24.0")
    print()
    
    print("2. Install missing packages:")
    print("   conda install numpy pandas matplotlib seaborn")
    print("   conda install networkx")
    print("   # or")
    print("   pip install numpy pandas matplotlib seaborn networkx")
    print()
    
    print("3. For better PyTorch compatibility:")
    print("   # Switch to PyTorch 2.1.0 environment if available")
    print("   conda activate PyTorch-2.1.0")
    print()
    
    print("4. Install additional dependencies:")
    print("   pip install timm")  # For vision transformer models
    print("   pip install psutil")  # For system monitoring
    print()

def main():
    """Main compatibility check."""
    print("=" * 60)
    print("Environment Compatibility Check")
    print("=" * 60)
    
    all_good = True
    
    # Check Python
    if not check_python_version():
        all_good = False
    
    print()
    
    # Check core packages
    packages = [
        ("torch", "1.8.0"),
        ("torchvision", "0.9.0"),
        ("numpy", "1.19.0"),
        ("sklearn", "0.24.0", "sklearn"),
        ("pandas", "1.1.0"),
        ("matplotlib", "3.3.0"),
        ("seaborn", "0.11.0"),
        ("networkx", "2.5"),
    ]
    
    for package_info in packages:
        if len(package_info) == 2:
            package, min_ver = package_info
            import_name = package
        else:
            package, min_ver, import_name = package_info
        
        if not check_package(package, min_ver, import_name):
            all_good = False
    
    print()
    
    # Check PyTorch
    if not check_torch_compatibility():
        all_good = False
    
    print()
    
    # Check specific functions
    check_sklearn_functions()
    
    print("\n" + "=" * 60)
    if all_good:
        print("✅ ENVIRONMENT COMPATIBLE!")
        print("You can run the training script:")
        print("  python train.py --config config/default_config.yaml")
        print("  python train_cpu.py --create-dummy-data")
    else:
        print("❌ COMPATIBILITY ISSUES FOUND")
        provide_fixes()
    
    print("=" * 60)
    
    # Test imports
    print("\nTesting project imports...")
    try:
        sys.path.append('src')
        from src.utils import MetricsCalculator
        from src.models import create_model
        print("✅ Project imports successful")
    except Exception as e:
        print(f"❌ Project import failed: {e}")
        print("Run the compatibility fixes above and try again.")

if __name__ == "__main__":
    main()
