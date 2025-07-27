#!/usr/bin/env python3
"""Simple import test."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing key imports...")

# Test the two problematic imports
try:
    from src.utils import RankingMetricsCalculator
    print("✓ RankingMetricsCalculator import OK")
except Exception as e:
    print(f"✗ RankingMetricsCalculator: {e}")

try:
    from src.optimizers import create_optimizer_with_config
    print("✓ create_optimizer_with_config import OK")
except Exception as e:
    print(f"✗ create_optimizer_with_config: {e}")

# Test train.py imports
try:
    from src.models import create_model
    from src.datasets import make_dataloader, create_dummy_dataset
    from src.losses import CombinedLoss
    from src.utils import setup_logger, load_config, TrainingVisualizer, MetricsCalculator
    print("✓ All train.py imports OK")
except Exception as e:
    print(f"✗ Train imports: {e}")

print("Done!")
