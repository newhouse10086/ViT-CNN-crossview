#!/usr/bin/env python3
"""Quick test for innovation method."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing your innovation method...")

try:
    from src.utils import load_config
    from src.models import create_model
    
    # Load config
    config = load_config('config/your_innovation_config.yaml')
    print(f"✓ Config loaded: {config['model']['name']}")
    print(f"  Innovation features:")
    print(f"    - Community Clustering: {config['model'].get('use_community_clustering', False)}")
    print(f"    - PCA Alignment: {config['model'].get('use_pca_alignment', False)}")
    print(f"    - Patch Size: {config['model'].get('patch_size', 'N/A')}")
    print(f"    - Clusters: {config['model'].get('num_final_clusters', 'N/A')}")
    
    # Create model
    model = create_model(config)
    print(f"✓ Innovation model created: {type(model).__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")
    
    print("\n🎉 Your innovation method is ready!")
    print("You can now train with:")
    print("  python train_innovation.py --config config/your_innovation_config.yaml --data-dir data --batch-size 8 --learning-rate 0.001 --num-epochs 10 --gpu-ids \"0\"")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
