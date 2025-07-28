#!/usr/bin/env python3
"""
FSRA-VMK调试脚本 - 逐步测试各个组件
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.fsra_vmk_improved import (
    VisionMambaEncoder,
    ConvNeXtV2FusionModule, 
    BidirectionalCrossViewAlignment,
    KANLinear,
    create_fsra_vmk_model
)

def test_kan_linear():
    """测试KAN线性层"""
    print("🧪 Testing KAN Linear...")
    try:
        kan = KANLinear(256, 128)
        x = torch.randn(4, 256)
        output = kan(x)
        print(f"✅ KAN Linear: {x.shape} -> {output.shape}")
        return True
    except Exception as e:
        print(f"❌ KAN Linear failed: {e}")
        traceback.print_exc()
        return False

def test_vision_mamba():
    """测试Vision Mamba编码器"""
    print("\n🧪 Testing Vision Mamba Encoder...")
    try:
        encoder = VisionMambaEncoder(img_size=256, embed_dim=384, depth=6)
        x = torch.randn(2, 3, 256, 256)
        features = encoder(x)
        print(f"✅ Vision Mamba input: {x.shape}")
        for key, feat in features.items():
            print(f"   {key}: {feat.shape}")
        return True, features
    except Exception as e:
        print(f"❌ Vision Mamba failed: {e}")
        traceback.print_exc()
        return False, None

def test_convnext_fusion(mamba_features):
    """测试ConvNeXt V2融合"""
    print("\n🧪 Testing ConvNeXt V2 Fusion...")
    try:
        fusion = ConvNeXtV2FusionModule(in_channels=384, fusion_channels=256)
        output = fusion(mamba_features)
        print(f"✅ ConvNeXt V2 Fusion: {output.shape}")
        return True, output
    except Exception as e:
        print(f"❌ ConvNeXt V2 Fusion failed: {e}")
        traceback.print_exc()
        return False, None

def test_cross_view_alignment(sat_fused, uav_fused):
    """测试双向跨视角对齐"""
    print("\n🧪 Testing Bidirectional Cross-View Alignment...")
    try:
        alignment = BidirectionalCrossViewAlignment(feature_dim=256)
        aligned = alignment(sat_fused, uav_fused)
        print(f"✅ Cross-View Alignment: {aligned.shape}")
        return True, aligned
    except Exception as e:
        print(f"❌ Cross-View Alignment failed: {e}")
        traceback.print_exc()
        return False, None

def test_complete_model():
    """测试完整模型"""
    print("\n🧪 Testing Complete FSRA-VMK Model...")
    try:
        model = create_fsra_vmk_model(num_classes=701, img_size=256, embed_dim=384, mamba_depth=6)
        
        sat_images = torch.randn(2, 3, 256, 256)
        uav_images = torch.randn(2, 3, 256, 256)
        
        with torch.no_grad():
            outputs = model(sat_images, uav_images)
        
        print(f"✅ Complete Model Success!")
        for key, value in outputs.items():
            if hasattr(value, 'shape'):
                print(f"   {key}: {value.shape}")
            else:
                print(f"   {key}: {type(value)} with {len(value)} items")
        return True
    except Exception as e:
        print(f"❌ Complete Model failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🔧 FSRA-VMK 组件调试")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # 逐步测试
    success_count = 0
    total_tests = 5
    
    # 1. 测试KAN
    if test_kan_linear():
        success_count += 1
    
    # 2. 测试Vision Mamba
    mamba_success, mamba_features = test_vision_mamba()
    if mamba_success:
        success_count += 1
        
        # 3. 测试ConvNeXt融合
        fusion_success, sat_fused = test_convnext_fusion(mamba_features)
        if fusion_success:
            success_count += 1
            # 模拟UAV特征
            uav_fused = sat_fused.clone()
            
            # 4. 测试跨视角对齐
            if test_cross_view_alignment(sat_fused, uav_fused):
                success_count += 1
    
    # 5. 测试完整模型
    if test_complete_model():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 测试结果: {success_count}/{total_tests} 组件通过")
    
    if success_count == total_tests:
        print("🎉 所有组件测试通过！")
        print("💡 建议检查训练数据或损失函数")
    else:
        print("⚠️ 发现问题组件，请根据上述错误信息修复")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 