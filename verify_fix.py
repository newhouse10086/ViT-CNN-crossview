#!/usr/bin/env python3
"""Verify all fixes are working."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Verifying all imports...")

# Test the specific imports that were failing
imports_to_test = [
    ("src.utils", "RankingMetricsCalculator"),
    ("src.utils", "plot_roc_curves"),
    ("src.utils", "log_system_info"),
    ("src.utils", "validate_config"),
    ("src.optimizers", "create_optimizer_with_config"),
]

all_good = True

for module, item in imports_to_test:
    try:
        exec(f"from {module} import {item}")
        print(f"✓ {item}")
    except ImportError as e:
        print(f"✗ {item}: {e}")
        all_good = False

if all_good:
    print("\n🎉 All imports working!")
else:
    print("\n❌ Some imports still failing")

print("\nTesting train.py imports...")
try:
    from src.models import create_model
    from src.datasets import make_dataloader, create_dummy_dataset
    from src.losses import CombinedLoss
    from src.optimizers import create_optimizer_with_config
    from src.utils import (
        setup_logger, get_logger, load_config, validate_config,
        TrainingVisualizer, MetricsCalculator, log_system_info
    )
    print("✓ All train.py imports working!")
except ImportError as e:
    print(f"✗ Train.py imports failed: {e}")
    all_good = False

if all_good:
    print("\n🎉 ALL FIXES SUCCESSFUL! You can now run train.py")
else:
    print("\n❌ Some issues remain")
