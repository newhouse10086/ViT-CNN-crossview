#!/usr/bin/env python3
"""Server-side fix for import issues."""

import os
import shutil
from pathlib import Path

def fix_utils_init():
    """Fix src/utils/__init__.py"""
    content = '''"""Utilities for ViT-CNN-crossview."""

from .metrics import MetricsCalculator, AverageMeter, RankingMetricsCalculator
from .visualization import TrainingVisualizer, plot_training_curves, plot_confusion_matrix, plot_roc_curves
from .logger import setup_logger, get_logger, log_system_info
from .config_utils import load_config, save_config, merge_configs, validate_config

__all__ = [
    'MetricsCalculator',
    'AverageMeter',
    'RankingMetricsCalculator',
    'TrainingVisualizer',
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'setup_logger',
    'get_logger',
    'log_system_info',
    'load_config',
    'save_config',
    'merge_configs',
    'validate_config'
]'''
    
    with open('src/utils/__init__.py', 'w') as f:
        f.write(content)
    print("✓ Fixed src/utils/__init__.py")

def fix_optimizers_init():
    """Fix src/optimizers/__init__.py"""
    content = '''"""Optimizers for ViT-CNN-crossview."""

from .optimizer_factory import create_optimizer, create_scheduler, create_optimizer_with_config
from .lr_schedulers import WarmupCosineScheduler, WarmupLinearScheduler

__all__ = [
    'create_optimizer',
    'create_scheduler',
    'create_optimizer_with_config',
    'WarmupCosineScheduler',
    'WarmupLinearScheduler'
]'''
    
    with open('src/optimizers/__init__.py', 'w') as f:
        f.write(content)
    print("✓ Fixed src/optimizers/__init__.py")

def fix_src_init():
    """Fix src/__init__.py to avoid circular imports"""
    content = '''"""ViT-CNN-crossview source package."""

# Version information
__version__ = "1.0.0"
__author__ = "newhouse10086"

# Import modules with error handling
try:
    from . import utils
except ImportError as e:
    print(f"Warning: Could not import utils: {e}")
    utils = None

try:
    from . import models
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    models = None

try:
    from . import datasets
except ImportError as e:
    print(f"Warning: Could not import datasets: {e}")
    datasets = None

try:
    from . import losses
except ImportError as e:
    print(f"Warning: Could not import losses: {e}")
    losses = None

try:
    from . import optimizers
except ImportError as e:
    print(f"Warning: Could not import optimizers: {e}")
    optimizers = None

try:
    from . import trainer
except ImportError as e:
    print(f"Warning: Could not import trainer: {e}")
    trainer = None

__all__ = [
    "models",
    "datasets", 
    "losses",
    "optimizers",
    "utils",
    "trainer"
]'''
    
    with open('src/__init__.py', 'w') as f:
        f.write(content)
    print("✓ Fixed src/__init__.py")

def test_imports():
    """Test if imports work after fixes"""
    import sys
    sys.path.insert(0, 'src')
    
    try:
        from src.utils import RankingMetricsCalculator
        print("✓ RankingMetricsCalculator import OK")
    except Exception as e:
        print(f"✗ RankingMetricsCalculator: {e}")
        return False
    
    try:
        from src.optimizers import create_optimizer_with_config
        print("✓ create_optimizer_with_config import OK")
    except Exception as e:
        print(f"✗ create_optimizer_with_config: {e}")
        return False
    
    try:
        from src.models import create_model
        from src.datasets import make_dataloader
        from src.losses import CombinedLoss
        from src.utils import load_config, MetricsCalculator
        print("✓ All main imports OK")
    except Exception as e:
        print(f"✗ Main imports: {e}")
        return False
    
    return True

def main():
    """Main fix function"""
    print("=" * 50)
    print("ViT-CNN-crossview Server Fix")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("Error: Please run this script from the ViT-CNN-crossview directory")
        return 1
    
    # Apply fixes
    print("Applying fixes...")
    fix_utils_init()
    fix_optimizers_init()
    fix_src_init()
    
    # Test imports
    print("\nTesting imports...")
    if test_imports():
        print("\n🎉 All fixes applied successfully!")
        print("You can now run:")
        print("  python train.py --config config/default_config.yaml")
        print("  python train.py --create-dummy-data --experiment-name test")
    else:
        print("\n❌ Some imports still failing")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
