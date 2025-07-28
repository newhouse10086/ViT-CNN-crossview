#!/usr/bin/env python3
"""
测试速度优化的修改是否正确工作。
"""

import sys
from pathlib import Path
import torch
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config_utils import load_config
from src.models import create_model

def test_speed_optimization():
    """测试速度优化修改。"""
    print("🧪 Testing Speed Optimization Changes...")
    print("=" * 50)
    
    # 加载配置
    config = load_config("config/fsra_vit_improved_config.yaml")
    print(f"✅ Configuration loaded")
    print(f"  - use_kmeans_clustering: {config['model'].get('use_kmeans_clustering', 'Not set')}")
    print(f"  - learning_rate: {config['training']['learning_rate']}")
    print(f"  - batch_size: {config['data']['batch_size']}")
    print(f"  - ViT embed_dim: {config['model']['vit']['embed_dim']}")
    print(f"  - ViT depth: {config['model']['vit']['depth']}")
    
    # 创建模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    
    model = create_model(config)
    model = model.to(device)
    print(f"✅ Model created: {config['model']['name']}")
    
    # 创建测试输入
    batch_size = 4  # 小批量测试
    sat_imgs = torch.randn(batch_size, 3, 250, 250).to(device)
    drone_imgs = torch.randn(batch_size, 3, 250, 250).to(device)
    
    print(f"\n🔧 Testing forward pass with batch_size={batch_size}...")
    
    # 预热
    model.eval()
    with torch.no_grad():
        _ = model(sat_imgs, drone_imgs)
    
    # 测试前向传播时间
    num_runs = 10
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_runs):
            outputs = model(sat_imgs, drone_imgs)
    
    avg_time = (time.time() - start_time) / num_runs
    print(f"✅ Forward pass successful")
    print(f"  - Average time per batch: {avg_time:.3f}s")
    print(f"  - Time per sample: {avg_time/batch_size:.3f}s")
    
    # 分析输出结构
    if 'satellite' in outputs:
        predictions = outputs['satellite']['predictions']
        features = outputs['satellite']['features']
        
        print(f"\n📊 Output Analysis:")
        print(f"  - Number of predictions: {len(predictions)}")
        print(f"  - Prediction shapes: {[pred.shape for pred in predictions]}")
        print(f"  - Available features: {list(features.keys())}")
        
        # 检查是否使用聚类
        if len(predictions) == 1:
            print(f"  ✅ Simplified mode: Only global classifier (Speed optimized)")
        else:
            print(f"  📊 Full mode: Multiple classifiers (Accuracy optimized)")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📈 Model Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # 内存使用情况
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  - GPU memory used: {memory_used:.1f} MB")
    
    print(f"\n🎯 Optimization Summary:")
    use_clustering = config['model'].get('use_kmeans_clustering', False)
    if use_clustering:
        print(f"  ⚠️  K-means clustering is ENABLED - Full accuracy mode")
        print(f"  ⚠️  Training will be slower but potentially more accurate")
    else:
        print(f"  🚀 K-means clustering is DISABLED - Speed optimized mode")
        print(f"  🚀 Training should be 2-3x faster")
    
    print(f"\n✅ Test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_speed_optimization()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 