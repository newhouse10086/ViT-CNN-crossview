#!/usr/bin/env python3
"""Quick test to verify the fixes."""

import sys
import os
from pathlib import Path

print("Python version:", sys.version)
print("Current directory:", os.getcwd())

# Add src to path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

# Test the specific imports that were failing
print("\nTesting specific imports that were failing...")

try:
    print("Testing RankingMetricsCalculator...")
    from src.utils import RankingMetricsCalculator
    print("✓ RankingMetricsCalculator import successful")
except ImportError as e:
    print(f"✗ RankingMetricsCalculator import failed: {e}")

try:
    print("Testing create_optimizer_with_config...")
    from src.optimizers import create_optimizer_with_config
    print("✓ create_optimizer_with_config import successful")
except ImportError as e:
    print(f"✗ create_optimizer_with_config import failed: {e}")

# Test the main imports from train.py
print("\nTesting main train.py imports...")

try:
    from src.models import create_model
    print("✓ create_model")
except ImportError as e:
    print(f"✗ create_model: {e}")

try:
    from src.datasets import make_dataloader, create_dummy_dataset
    print("✓ datasets")
except ImportError as e:
    print(f"✗ datasets: {e}")

try:
    from src.losses import CombinedLoss
    print("✓ CombinedLoss")
except ImportError as e:
    print(f"✗ CombinedLoss: {e}")

try:
    from src.utils import (
        setup_logger, get_logger, load_config, validate_config,
        TrainingVisualizer, MetricsCalculator
    )
    print("✓ utils")
except ImportError as e:
    print(f"✗ utils: {e}")

print("\nDone!")
