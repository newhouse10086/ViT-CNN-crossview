"""
FSRA-CRN: Context-aware Region-alignment Network for Cross-View Image Matching
基于原始FSRA改进，专注于跨视角图像匹配任务，适用于University-1652数据集
参考最新预印本CEUSP (arXiv:2502.11408) 的创新思路
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class MultiScaleResidualBlock(nn.Module):
    """多尺度残差块 - MSRFE的基础组件"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # 主路径：多尺度卷积
        self.conv1x1 = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels//2, 3, stride, 1, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels//4, 5, stride, 2, bias=False)
        
        # 批归一化
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.bn2 = nn.BatchNorm2d(out_channels//2)
        self.bn3 = nn.BatchNorm2d(out_channels//4)
        
        # 残差连接
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.se = SEBlock(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度特征提取
        f1 = self.relu(self.bn1(self.conv1x1(x)))
        f2 = self.relu(self.bn2(self.conv3x3(x)))
        f3 = self.relu(self.bn3(self.conv5x5(x)))
        
        # 特征拼接
        out = torch.cat([f1, f2, f3], dim=1)
        
        # 注意力加权
        out = self.se(out)
        
        # 残差连接
        out = out + self.shortcut(x)
        
        return self.relu(out)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiScaleResidualFeatureExtractor(nn.Module):
    """多尺度残差特征提取器 (MSRFE) - 替代ViT的核心模块"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        
        # 初始卷积
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # 多尺度残差层
        self.layer1 = self._make_layer(64, 128, 3, stride=1)    # 1/4
        self.layer2 = self._make_layer(128, 256, 4, stride=2)   # 1/8  
        self.layer3 = self._make_layer(256, 512, 6, stride=2)   # 1/16
        self.layer4 = self._make_layer(512, 1024, 3, stride=2)  # 1/32
        
        # 特征金字塔网络 (FPN)
        self.fpn = FeaturePyramidNetwork([128, 256, 512, 1024], 256)
        
        # 自适应池化到固定尺寸
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((16, 16)),  # P1
            nn.AdaptiveAvgPool2d((8, 8)),    # P2  
            nn.AdaptiveAvgPool2d((4, 4)),    # P3
            nn.AdaptiveAvgPool2d((2, 2))     # P4
        ])
        
    def _make_layer(self, in_channels: int, out_channels: int, 
                   blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(MultiScaleResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, blocks):
            layers.append(MultiScaleResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) 输入图像
        Returns:
            多尺度特征字典
        """
        # Stem
        x = self.stem(x)  # (B, 64, H/4, W/4)
        
        # 多尺度特征提取
        c1 = self.layer1(x)   # (B, 128, H/4, W/4)
        c2 = self.layer2(c1)  # (B, 256, H/8, W/8)
        c3 = self.layer3(c2)  # (B, 512, H/16, W/16)
        c4 = self.layer4(c3)  # (B, 1024, H/32, W/32)
        
        # FPN融合
        fpn_features = self.fpn([c1, c2, c3, c4])
        
        # 自适应池化到固定尺寸
        pooled_features = []
        for i, pool in enumerate(self.adaptive_pools):
            pooled = pool(fpn_features[i])  # (B, 256, fixed_size, fixed_size)
            pooled_features.append(pooled)
        
        return {
            'P1': pooled_features[0],  # (B, 256, 16, 16)
            'P2': pooled_features[1],  # (B, 256, 8, 8)
            'P3': pooled_features[2],  # (B, 256, 4, 4)
            'P4': pooled_features[3],  # (B, 256, 2, 2)
            'raw_features': [c1, c2, c3, c4]
        }


class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络"""
    
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        
        # 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1)
            for in_channels in in_channels_list
        ])
        
        # 输出卷积
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            for _ in in_channels_list
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # 从高层到低层构建FPN
        results = []
        last_inner = self.lateral_convs[-1](features[-1])
        results.append(self.fpn_convs[-1](last_inner))
        
        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            top_down = F.interpolate(last_inner, size=lateral.shape[-2:], 
                                   mode='bilinear', align_corners=False)
            last_inner = lateral + top_down
            results.insert(0, self.fpn_convs[i](last_inner))
            
        return results


class RubiksCubeAttention(nn.Module):
    """魔方注意力模块 (RCA) - 借鉴CEUSP论文思路"""
    
    def __init__(self, channels: int, num_rotations: int = 6):
        super().__init__()
        self.channels = channels
        self.num_rotations = num_rotations
        
        # 3D旋转变换矩阵（模拟魔方的6种基本旋转）
        self.rotation_transforms = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, 1, 1) for _ in range(num_rotations)
        ])
        
        # 注意力权重计算
        self.attention_conv = nn.Conv2d(channels * num_rotations, num_rotations, 1)
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * num_rotations, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 位置编码
        self.pos_encoding = self._create_position_encoding(channels)
        
    def _create_position_encoding(self, channels: int) -> nn.Parameter:
        """创建3D位置编码（模拟魔方的空间结构）"""
        # 简化的3D位置编码，实际应用中可以更复杂
        pe = torch.zeros(1, channels, 1, 1)
        return nn.Parameter(pe, requires_grad=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 输入特征
        Returns:
            rotated_features: (B, C, H, W) 魔方注意力处理后的特征
        """
        batch_size, channels, height, width = x.shape
        
        # 添加位置编码
        x_pos = x + self.pos_encoding
        
        # 应用多种旋转变换（模拟魔方旋转）
        rotated_features = []
        for i, transform in enumerate(self.rotation_transforms):
            # 不同的旋转变换
            if i % 2 == 0:
                # 水平翻转
                rotated_x = torch.flip(x_pos, dims=[3])
            else:
                # 垂直翻转
                rotated_x = torch.flip(x_pos, dims=[2])
            
            # 应用卷积变换
            transformed = transform(rotated_x)
            rotated_features.append(transformed)
        
        # 拼接所有旋转特征
        concat_features = torch.cat(rotated_features, dim=1)  # (B, C*num_rotations, H, W)
        
        # 计算注意力权重
        attention_weights = torch.softmax(
            self.attention_conv(concat_features), dim=1
        )  # (B, num_rotations, H, W)
        
        # 加权融合
        weighted_features = []
        for i in range(self.num_rotations):
            weight = attention_weights[:, i:i+1, :, :]  # (B, 1, H, W)
            weighted = rotated_features[i] * weight
            weighted_features.append(weighted)
        
        # 最终融合
        final_concat = torch.cat(weighted_features, dim=1)
        output = self.fusion_conv(final_concat)
        
        # 残差连接
        return output + x


class DynamicContextFusion(nn.Module):
    """动态上下文融合模块 (DCF)"""
    
    def __init__(self, feature_channels: int = 256):
        super().__init__()
        self.feature_channels = feature_channels
        
        # 多尺度上下文捕获
        self.context_convs = nn.ModuleList([
            nn.Conv2d(feature_channels, feature_channels, kernel_size=k, 
                     padding=k//2, dilation=d)
            for k, d in [(1, 1), (3, 1), (3, 2), (3, 4)]
        ])
        
        # 动态权重生成网络
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_channels, feature_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 4, len(self.context_convs), 1),
            nn.Softmax(dim=1)
        )
        
        # 特征优化
        self.feature_refine = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 多尺度特征字典 {'P1', 'P2', 'P3', 'P4'}
        Returns:
            fused_features: (B, C, H, W) 融合后的特征
        """
        # 将所有特征上采样到相同尺寸 (取P1的尺寸作为基准)
        target_size = features['P1'].shape[-2:]
        aligned_features = []
        
        for key in ['P1', 'P2', 'P3', 'P4']:
            feat = features[key]
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, 
                                   mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # 特征求和作为基础特征
        base_feature = sum(aligned_features)  # (B, C, H, W)
        
        # 多尺度上下文捕获
        context_features = []
        for conv in self.context_convs:
            context_feat = conv(base_feature)
            context_features.append(context_feat)
        
        # 动态权重生成
        weights = self.weight_generator(base_feature)  # (B, num_contexts, 1, 1)
        
        # 加权融合上下文特征
        fused_context = sum(
            w.unsqueeze(1) * feat 
            for w, feat in zip(weights.split(1, dim=1), context_features)
        )
        
        # 特征优化
        output = self.feature_refine(fused_context)
        
        return output


class AdaptiveRegionAlignment(nn.Module):
    """自适应区域对齐模块 (ARA) - 改进的FSRA区域对齐策略"""
    
    def __init__(self, feature_dim: int, num_regions: int = 6):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_regions = num_regions
        
        # 区域重要性评估
        self.importance_net, = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 动态区域生成（基于特征响应）
        self.region_generator = nn.Sequential(
            nn.Conv2d(feature_dim, num_regions, 1),
            nn.Softmax(dim=1)  # 生成区域概率图
        )
        
        # 区域特征编码器
        self.region_encoders = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(feature_dim, feature_dim, 1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_regions)
        ])
        
        # 跨视角对齐网络
        self.cross_view_align = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 最终特征融合
        self.final_fusion = nn.Sequential(
            nn.Linear(feature_dim * num_regions, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, sat_features: torch.Tensor, 
                uav_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sat_features: (B, C, H, W) 卫星特征
            uav_features: (B, C, H, W) 无人机特征
        Returns:
            aligned_features: (B, C) 对齐后的特征
        """
        batch_size = sat_features.shape[0]
        
        # 评估特征重要性
        sat_importance = self.importance_net(sat_features)  # (B, 1, 1, 1)
        uav_importance = self.importance_net(uav_features)  # (B, 1, 1, 1)
        
        # 生成动态区域
        sat_regions = self.region_generator(sat_features)  # (B, num_regions, H, W)
        uav_regions = self.region_generator(uav_features)  # (B, num_regions, H, W)
        
        # 提取区域特征
        sat_region_features = []
        uav_region_features = []
        
        for i in range(self.num_regions):
            # 获取区域mask
            sat_mask = sat_regions[:, i:i+1, :, :]  # (B, 1, H, W)
            uav_mask = uav_regions[:, i:i+1, :, :]  # (B, 1, H, W)
            
            # 加权特征提取
            sat_weighted = sat_features * sat_mask  # (B, C, H, W)
            uav_weighted = uav_features * uav_mask  # (B, C, H, W)
            
            # 池化得到区域特征
            sat_region_feat = self.region_encoders[i](sat_weighted).flatten(2).mean(2)  # (B, C)
            uav_region_feat = self.region_encoders[i](uav_weighted).flatten(2).mean(2)  # (B, C)
            
            sat_region_features.append(sat_region_feat)
            uav_region_features.append(uav_region_feat)
        
        # 堆叠区域特征
        sat_regions_tensor = torch.stack(sat_region_features, dim=1)  # (B, num_regions, C)
        uav_regions_tensor = torch.stack(uav_region_features, dim=1)  # (B, num_regions, C)
        
        # 跨视角区域对齐
        aligned_sat, _ = self.cross_view_align(
            sat_regions_tensor, uav_regions_tensor, uav_regions_tensor
        )  # (B, num_regions, C)
        
        aligned_uav, _ = self.cross_view_align(
            uav_regions_tensor, sat_regions_tensor, sat_regions_tensor
        )  # (B, num_regions, C)
        
        # 特征融合
        fused_regions = (aligned_sat + aligned_uav) / 2  # (B, num_regions, C)
        
        # 展平并最终融合
        flattened = fused_regions.view(batch_size, -1)  # (B, num_regions * C)
        final_features = self.final_fusion(flattened)  # (B, C)
        
        return final_features


class FSRACRNModel(nn.Module):
    """FSRA-CRN主模型：Context-aware Region-alignment Network"""
    
    def __init__(self, num_classes: int = 701, feature_dim: int = 256):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # 1. 多尺度残差特征提取器 (MSRFE)
        self.feature_extractor = MultiScaleResidualFeatureExtractor()
        
        # 2. 魔方注意力模块 (RCA)
        self.rubiks_attention = RubiksCubeAttention(feature_dim)
        
        # 3. 动态上下文融合 (DCF)
        self.context_fusion = DynamicContextFusion(feature_dim)
        
        # 4. 自适应区域对齐 (ARA)
        self.region_alignment = AdaptiveRegionAlignment(feature_dim)
        
        # 分类器
        self.global_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # 区域分类器（保持与原始FSRA的兼容性）
        self.regional_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes)
            ) for _ in range(6)  # 6个区域
        ])
        
    def forward(self, sat_img: torch.Tensor, uav_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            sat_img: (B, 3, H, W) 卫星图像
            uav_img: (B, 3, H, W) 无人机图像
        Returns:
            输出字典包含全局和区域预测
        """
        # 1. 多尺度特征提取
        sat_ms_features = self.feature_extractor(sat_img)
        uav_ms_features = self.feature_extractor(uav_img)
        
        # 2. 动态上下文融合
        sat_context = self.context_fusion(sat_ms_features)
        uav_context = self.context_fusion(uav_ms_features)
        
        # 3. 魔方注意力增强
        sat_enhanced = self.rubiks_attention(sat_context)
        uav_enhanced = self.rubiks_attention(uav_context)
        
        # 4. 自适应区域对齐
        aligned_features = self.region_alignment(sat_enhanced, uav_enhanced)
        
        # 5. 分类预测
        global_pred = self.global_classifier(aligned_features)
        
        # 区域预测（为了与原始FSRA保持兼容）
        regional_preds = []
        for classifier in self.regional_classifiers:
            regional_pred = classifier(aligned_features)
            regional_preds.append(regional_pred)
        
        return {
            'global_prediction': global_pred,
            'regional_predictions': regional_preds,
            'aligned_features': aligned_features,
            'sat_enhanced': sat_enhanced,
            'uav_enhanced': uav_enhanced
        }


def create_fsra_crn_model(num_classes: int = 701, **kwargs) -> FSRACRNModel:
    """创建FSRA-CRN模型"""
    return FSRACRNModel(num_classes=num_classes, **kwargs)


# 示例使用
if __name__ == "__main__":
    # 创建模型
    model = create_fsra_crn_model(num_classes=701)
    
    # 示例输入
    batch_size = 4
    sat_images = torch.randn(batch_size, 3, 256, 256)
    uav_images = torch.randn(batch_size, 3, 256, 256)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(sat_images, uav_images)
    
    print("FSRA-CRN Model Output Shapes:")
    print(f"Global prediction: {outputs['global_prediction'].shape}")
    print(f"Regional predictions: {len(outputs['regional_predictions'])} regions") 
    print(f"Aligned features: {outputs['aligned_features'].shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}") 