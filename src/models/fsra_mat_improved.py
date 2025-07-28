"""
FSRA-MAT: Multi-Modal Adaptive Transformer for Cross-View Geo-Localization
基于原始FSRA的学术创新架构，融合2024-2025年最新研究成果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
import math


class GeometryAwarePositionalEncoding(nn.Module):
    """几何感知位置编码：处理尺度、旋转、距离变化"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # 传统位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
        
        # 几何感知编码
        self.scale_embedding = nn.Embedding(100, d_model)  # 尺度编码 [0.1, 10.0]
        self.rotation_embedding = nn.Embedding(360, d_model)  # 旋转编码 [0°, 359°]
        self.distance_embedding = nn.Embedding(1000, d_model)  # 距离编码
        
        # 自适应融合权重
        self.fusion_weights = nn.Parameter(torch.ones(4) / 4)
        
    def forward(self, positions: torch.Tensor, scales: torch.Tensor, 
                rotations: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch_size, seq_len) 位置索引
            scales: (batch_size, seq_len) 尺度信息 [0-99]
            rotations: (batch_size, seq_len) 旋转角度 [0-359]
            distances: (batch_size, seq_len) 距离信息 [0-999]
        """
        batch_size, seq_len = positions.shape
        
        # 基础位置编码
        pos_enc = self.pe[:seq_len, :].transpose(0, 1).expand(batch_size, -1, -1)
        
        # 几何感知编码
        scale_enc = self.scale_embedding(scales.long())
        rot_enc = self.rotation_embedding(rotations.long())
        dist_enc = self.distance_embedding(distances.long())
        
        # 自适应权重融合
        weights = F.softmax(self.fusion_weights, dim=0)
        geo_aware_encoding = (weights[0] * pos_enc + 
                             weights[1] * scale_enc +
                             weights[2] * rot_enc + 
                             weights[3] * dist_enc)
        
        return geo_aware_encoding


class GeometricSemanticDecoupling(nn.Module):
    """几何-语义解耦模块：分离处理几何和语义信息"""
    
    def __init__(self, input_dim: int, geo_dim: int, sem_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.geo_dim = geo_dim
        self.sem_dim = sem_dim
        
        # 特征分解投影
        self.geometric_projector = nn.Sequential(
            nn.Linear(input_dim, geo_dim),
            nn.LayerNorm(geo_dim),
            nn.ReLU(inplace=True)
        )
        
        self.semantic_projector = nn.Sequential(
            nn.Linear(input_dim, sem_dim),
            nn.LayerNorm(sem_dim),
            nn.ReLU(inplace=True)
        )
        
        # 几何特征编码器 - 专注于位置、尺度、方向
        self.geometry_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=geo_dim,
                nhead=8,
                dim_feedforward=geo_dim * 2,
                dropout=0.1,
                batch_first=True
            ) for _ in range(3)
        ])
        
        # 语义特征编码器 - 专注于内容和上下文
        self.semantic_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=sem_dim,
                nhead=12,
                dim_feedforward=sem_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(6)
        ])
        
        # 交互模块：允许几何和语义特征适度交互
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=min(geo_dim, sem_dim),
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch_size, seq_len, input_dim)
        Returns:
            geo_encoded: (batch_size, seq_len, geo_dim) 几何特征
            sem_encoded: (batch_size, seq_len, sem_dim) 语义特征
        """
        batch_size, seq_len, _ = features.shape
        
        # 特征分解
        geo_features = self.geometric_projector(features)
        sem_features = self.semantic_projector(features)
        
        # 几何特征编码：专注于空间关系
        geo_encoded = geo_features
        for geo_layer in self.geometry_encoder:
            geo_encoded = geo_layer(geo_encoded)
        
        # 语义特征编码：专注于内容理解
        sem_encoded = sem_features
        for sem_layer in self.semantic_encoder:
            sem_encoded = sem_layer(sem_encoded)
        
        # 跨模态交互（可选）
        if self.geo_dim == self.sem_dim:
            # 维度相同时进行交互
            enhanced_geo, _ = self.cross_modal_attention(
                geo_encoded, sem_encoded, sem_encoded
            )
            enhanced_sem, _ = self.cross_modal_attention(
                sem_encoded, geo_encoded, geo_encoded
            )
            geo_encoded = enhanced_geo + geo_encoded  # 残差连接
            sem_encoded = enhanced_sem + sem_encoded
        
        return geo_encoded, sem_encoded


class HierarchicalAttentionFusion(nn.Module):
    """层次化注意力融合：多尺度、多层次特征融合"""
    
    def __init__(self, geo_dim: int, sem_dim: int, num_scales: int = 4, num_heads: int = 8):
        super().__init__()
        self.num_scales = num_scales
        self.geo_dim = geo_dim
        self.sem_dim = sem_dim
        self.hidden_dim = max(geo_dim, sem_dim)
        
        # 维度对齐
        self.geo_projection = nn.Linear(geo_dim, self.hidden_dim) if geo_dim != self.hidden_dim else nn.Identity()
        self.sem_projection = nn.Linear(sem_dim, self.hidden_dim) if sem_dim != self.hidden_dim else nn.Identity()
        
        # 多尺度注意力层
        self.multi_scale_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=num_heads,
                batch_first=True
            ) for _ in range(num_scales)
        ])
        
        # 尺度自适应下采样
        self.downsampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, 
                         stride=2**i, padding=1) if i > 0 else nn.Identity(),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(inplace=True)
            ) for i in range(num_scales)
        ])
        
        # 自适应尺度融合
        self.scale_fusion = AdaptiveScaleFusion(num_scales, self.hidden_dim)
        
        # 最终输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, geo_features: torch.Tensor, sem_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geo_features: (batch_size, seq_len, geo_dim)
            sem_features: (batch_size, seq_len, sem_dim)
        Returns:
            fused_features: (batch_size, seq_len, hidden_dim)
        """
        # 维度对齐
        geo_aligned = self.geo_projection(geo_features)
        sem_aligned = self.sem_projection(sem_features)
        
        multi_scale_outputs = []
        
        for i, attention_layer in enumerate(self.multi_scale_attention):
            # 多尺度特征处理
            if i == 0:
                geo_scaled = geo_aligned
                sem_scaled = sem_aligned
            else:
                # 使用Conv1d进行下采样，需要转换维度 (B, L, C) -> (B, C, L)
                geo_scaled = geo_aligned.transpose(1, 2)
                sem_scaled = sem_aligned.transpose(1, 2)
                
                geo_scaled = self.downsampling_layers[i](geo_scaled).transpose(1, 2)
                sem_scaled = self.downsampling_layers[i](sem_scaled).transpose(1, 2)
            
            # 跨模态注意力：以几何特征为query，语义特征为key和value
            attended_features, attention_weights = attention_layer(
                geo_scaled, sem_scaled, sem_scaled
            )
            
            # 残差连接
            attended_features = attended_features + geo_scaled
            multi_scale_outputs.append(attended_features)
        
        # 自适应尺度融合
        fused_features = self.scale_fusion(multi_scale_outputs)
        
        # 最终投影
        output_features = self.output_projection(fused_features)
        
        return output_features


class AdaptiveScaleFusion(nn.Module):
    """自适应尺度融合模块"""
    
    def __init__(self, num_scales: int, feature_dim: int):
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        
        # 尺度重要性学习
        self.scale_importance = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # 尺度特异性变换
        self.scale_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_scales)
        ])
        
    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            multi_scale_features: List of (batch_size, seq_len, feature_dim)
        Returns:
            fused_features: (batch_size, seq_len, feature_dim)
        """
        # 将不同尺度特征上采样到相同尺度
        target_length = multi_scale_features[0].shape[1]
        aligned_features = []
        
        for i, features in enumerate(multi_scale_features):
            if features.shape[1] != target_length:
                # 使用插值对齐序列长度
                features = F.interpolate(
                    features.transpose(1, 2), 
                    size=target_length, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            
            # 尺度特异性变换
            transformed = self.scale_transforms[i](features)
            aligned_features.append(transformed)
        
        # 堆叠所有尺度特征
        stacked_features = torch.stack(aligned_features, dim=-1)  # (B, L, D, S)
        
        # 计算自适应权重（基于全局特征）
        global_feature = torch.mean(stacked_features, dim=(1, 3))  # (B, D)
        scale_weights = self.scale_importance(global_feature)  # (B, S)
        scale_weights = scale_weights.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        
        # 加权融合
        fused_features = torch.sum(stacked_features * scale_weights, dim=-1)
        
        return fused_features


class DynamicRegionAlignment(nn.Module):
    """动态区域对齐：自适应生成和对齐区域"""
    
    def __init__(self, feature_dim: int, max_regions: int = 8, min_regions: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_regions = max_regions
        self.min_regions = min_regions
        
        # 区域重要性评估
        self.region_scorer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 区域生成器
        self.region_generator = AdaptiveRegionGenerator(feature_dim, max_regions)
        
        # 跨视角区域对齐器
        self.region_aligner = CrossViewRegionAligner(feature_dim)
        
        # 注意力池化
        self.attention_pooling = AttentionPooling(feature_dim)
        
        # 区域特征融合
        self.region_fusion = nn.Sequential(
            nn.Linear(feature_dim * max_regions, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, sat_features: torch.Tensor, uav_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sat_features: (batch_size, seq_len, feature_dim) 卫星特征
            uav_features: (batch_size, seq_len, feature_dim) 无人机特征
        Returns:
            aligned_features: (batch_size, feature_dim) 对齐后的特征
        """
        batch_size = sat_features.shape[0]
        
        # 自适应区域生成
        sat_regions = self.region_generator(sat_features)
        uav_regions = self.region_generator(uav_features)
        
        # 跨视角区域对齐
        aligned_pairs = self.region_aligner(sat_regions, uav_regions)
        
        # 注意力加权池化每个区域对
        pooled_regions = []
        for sat_region, uav_region in aligned_pairs:
            pooled_feature = self.attention_pooling(sat_region, uav_region)
            pooled_regions.append(pooled_feature)
        
        # 填充到最大区域数
        while len(pooled_regions) < self.max_regions:
            pooled_regions.append(torch.zeros_like(pooled_regions[0]))
        
        # 拼接并融合区域特征
        concatenated_regions = torch.cat(pooled_regions[:self.max_regions], dim=-1)
        fused_features = self.region_fusion(concatenated_regions)
        
        return fused_features


class AdaptiveRegionGenerator(nn.Module):
    """自适应区域生成器"""
    
    def __init__(self, feature_dim: int, max_regions: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_regions = max_regions
        
        # 特征重要性评估
        self.importance_estimator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # 区域聚类网络
        self.clustering_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, max_regions),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            features: (batch_size, seq_len, feature_dim)
        Returns:
            regions: List of region features
        """
        batch_size, seq_len, _ = features.shape
        
        # 计算每个位置的重要性
        importance_scores = self.importance_estimator(features).squeeze(-1)  # (B, L)
        
        # 生成软聚类分配
        cluster_assignments = self.clustering_network(features)  # (B, L, max_regions)
        
        # 根据聚类分配生成区域
        regions = []
        for i in range(self.max_regions):
            # 获取第i个聚类的权重
            cluster_weights = cluster_assignments[:, :, i:i+1]  # (B, L, 1)
            
            # 加权重要性
            weighted_importance = importance_scores.unsqueeze(-1) * cluster_weights  # (B, L, 1)
            
            # 生成区域特征（加权平均）
            region_feature = torch.sum(features * weighted_importance, dim=1)  # (B, feature_dim)
            
            # 归一化
            weight_sum = torch.sum(weighted_importance, dim=1, keepdim=True) + 1e-8
            region_feature = region_feature / weight_sum.squeeze(-1)
            
            regions.append(region_feature)
        
        return regions


class CrossViewRegionAligner(nn.Module):
    """跨视角区域对齐器"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 相似度计算网络
        self.similarity_network = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sat_regions: List[torch.Tensor], 
                uav_regions: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            sat_regions: List of satellite region features
            uav_regions: List of UAV region features
        Returns:
            aligned_pairs: List of (sat_region, uav_region) pairs
        """
        aligned_pairs = []
        
        # 计算所有区域对之间的相似度
        similarity_matrix = torch.zeros(len(sat_regions), len(uav_regions))
        
        for i, sat_region in enumerate(sat_regions):
            for j, uav_region in enumerate(uav_regions):
                # 拼接特征并计算相似度
                combined_features = torch.cat([sat_region, uav_region], dim=-1)
                similarity = self.similarity_network(combined_features).mean()
                similarity_matrix[i, j] = similarity
        
        # 使用匈牙利算法或贪心算法进行最优匹配
        # 这里使用简化的贪心算法
        used_uav = set()
        for i in range(len(sat_regions)):
            # 找到与当前卫星区域最相似的无人机区域
            available_scores = similarity_matrix[i].clone()
            for used_j in used_uav:
                available_scores[used_j] = -1
            
            best_j = torch.argmax(available_scores).item()
            used_uav.add(best_j)
            
            aligned_pairs.append((sat_regions[i], uav_regions[best_j]))
        
        return aligned_pairs


class AttentionPooling(nn.Module):
    """注意力池化模块"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, sat_region: torch.Tensor, uav_region: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sat_region: (batch_size, feature_dim)
            uav_region: (batch_size, feature_dim)
        Returns:
            pooled_feature: (batch_size, feature_dim)
        """
        # 扩展维度以适应注意力机制
        sat_expanded = sat_region.unsqueeze(1)  # (batch_size, 1, feature_dim)
        uav_expanded = uav_region.unsqueeze(1)  # (batch_size, 1, feature_dim)
        
        # 双向注意力池化
        sat_attended, _ = self.attention(sat_expanded, uav_expanded, uav_expanded)
        uav_attended, _ = self.attention(uav_expanded, sat_expanded, sat_expanded)
        
        # 融合并归一化
        pooled_feature = self.norm(sat_attended.squeeze(1) + uav_attended.squeeze(1))
        
        return pooled_feature


class VisionLanguageCrossAttention(nn.Module):
    """视觉-语言跨模态注意力模块"""
    
    def __init__(self, vision_dim: int, text_dim: int = 384, hidden_dim: int = 768):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # 文本编码器（使用预训练模型）
        self.text_encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 维度投影
        self.vision_projector = nn.Linear(vision_dim, hidden_dim)
        self.text_projector = nn.Linear(text_dim, hidden_dim)
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=12,
            batch_first=True
        )
        
        # 特征融合
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vision_dim)
        )
        
    def forward(self, visual_features: torch.Tensor, 
                location_descriptions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_features: (batch_size, seq_len, vision_dim)
            location_descriptions: List of text descriptions
        Returns:
            enhanced_features: (batch_size, seq_len, vision_dim)
            attention_weights: (batch_size, seq_len, text_len)
        """
        batch_size, seq_len, _ = visual_features.shape
        
        # 文本编码
        encoded_texts = self.tokenizer(
            location_descriptions,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        ).to(visual_features.device)
        
        with torch.no_grad():
            text_outputs = self.text_encoder(**encoded_texts)
        
        text_features = text_outputs.last_hidden_state  # (batch_size, text_len, text_dim)
        
        # 维度投影
        vision_projected = self.vision_projector(visual_features)
        text_projected = self.text_projector(text_features)
        
        # 跨模态注意力：视觉特征作为query，文本特征作为key和value
        enhanced_vision, attention_weights = self.cross_attention(
            vision_projected, text_projected, text_projected
        )
        
        # 特征融合
        concatenated = torch.cat([enhanced_vision, vision_projected], dim=-1)
        fused_features = self.fusion_network(concatenated)
        
        # 残差连接
        enhanced_features = fused_features + visual_features
        
        return enhanced_features, attention_weights


class FSRAMATModel(nn.Module):
    """FSRA-MAT主模型：多模态自适应Transformer"""
    
    def __init__(self, 
                 num_classes: int = 701,
                 input_dim: int = 768,
                 geo_dim: int = 256,
                 sem_dim: int = 512,
                 max_regions: int = 6):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # 1. 几何感知位置编码
        self.geometry_aware_pe = GeometryAwarePositionalEncoding(input_dim)
        
        # 2. 几何-语义解耦
        self.geo_sem_decoupling = GeometricSemanticDecoupling(input_dim, geo_dim, sem_dim)
        
        # 3. 层次化注意力融合
        self.hierarchical_fusion = HierarchicalAttentionFusion(geo_dim, sem_dim)
        
        # 4. 动态区域对齐
        self.dynamic_alignment = DynamicRegionAlignment(max(geo_dim, sem_dim), max_regions)
        
        # 5. 视觉-语言跨模态注意力
        self.vision_language_attention = VisionLanguageCrossAttention(
            vision_dim=max(geo_dim, sem_dim)
        )
        
        # 最终分类器
        self.global_classifier = nn.Sequential(
            nn.Linear(max(geo_dim, sem_dim), 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # 区域分类器
        self.regional_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max(geo_dim, sem_dim), 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes)
            ) for _ in range(max_regions)
        ])
        
    def forward(self, 
                sat_img: torch.Tensor, 
                uav_img: torch.Tensor,
                location_descriptions: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            sat_img: (batch_size, 3, H, W) 卫星图像
            uav_img: (batch_size, 3, H, W) 无人机图像
            location_descriptions: 地理位置描述文本
        Returns:
            outputs: 包含预测结果和特征的字典
        """
        batch_size = sat_img.shape[0]
        
        # 假设已经通过backbone提取了特征（这里简化处理）
        # 实际应用中需要使用ViT或CNN backbone
        sat_features = self._extract_features(sat_img)  # (B, L, input_dim)
        uav_features = self._extract_features(uav_img)  # (B, L, input_dim)
        
        # 生成几何信息（简化，实际应用中需要从图像元数据获取）
        seq_len = sat_features.shape[1]
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        scales = torch.randint(0, 100, (batch_size, seq_len))  # 示例
        rotations = torch.randint(0, 360, (batch_size, seq_len))  # 示例
        distances = torch.randint(0, 1000, (batch_size, seq_len))  # 示例
        
        # 移动到正确设备
        device = sat_features.device
        positions = positions.to(device)
        scales = scales.to(device)
        rotations = rotations.to(device)
        distances = distances.to(device)
        
        # 1. 几何感知位置编码
        geo_pe = self.geometry_aware_pe(positions, scales, rotations, distances)
        sat_features_pe = sat_features + geo_pe
        uav_features_pe = uav_features + geo_pe
        
        # 2. 几何-语义解耦
        sat_geo, sat_sem = self.geo_sem_decoupling(sat_features_pe)
        uav_geo, uav_sem = self.geo_sem_decoupling(uav_features_pe)
        
        # 3. 层次化注意力融合
        sat_fused = self.hierarchical_fusion(sat_geo, sat_sem)
        uav_fused = self.hierarchical_fusion(uav_geo, uav_sem)
        
        # 4. 动态区域对齐
        aligned_features = self.dynamic_alignment(sat_fused, uav_fused)
        
        # 5. 视觉-语言跨模态注意力（如果提供文本）
        if location_descriptions is not None:
            # 将对齐特征扩展为序列格式
            expanded_features = aligned_features.unsqueeze(1)  # (B, 1, dim)
            enhanced_features, attention_weights = self.vision_language_attention(
                expanded_features, location_descriptions
            )
            final_features = enhanced_features.squeeze(1)  # (B, dim)
        else:
            final_features = aligned_features
            attention_weights = None
        
        # 6. 分类预测
        global_pred = self.global_classifier(final_features)
        
        # 区域预测（使用动态生成的区域特征）
        regional_preds = []
        for i, classifier in enumerate(self.regional_classifiers):
            regional_pred = classifier(final_features)  # 简化处理
            regional_preds.append(regional_pred)
        
        return {
            'global_prediction': global_pred,
            'regional_predictions': regional_preds,
            'final_features': final_features,
            'attention_weights': attention_weights,
            'aligned_features': aligned_features
        }
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        简化的特征提取，实际应用中应该使用预训练的backbone
        """
        batch_size, channels, height, width = images.shape
        
        # 简单的特征提取（实际中应该使用ViT或ResNet）
        features = F.adaptive_avg_pool2d(images, (16, 16))  # (B, C, 16, 16)
        features = features.flatten(2).transpose(1, 2)  # (B, 256, C)
        
        # 投影到指定维度
        if features.shape[-1] != self.input_dim:
            projection = nn.Linear(features.shape[-1], self.input_dim).to(images.device)
            features = projection(features)
        
        return features


def create_fsra_mat_model(num_classes: int = 701, **kwargs) -> FSRAMATModel:
    """创建FSRA-MAT模型"""
    return FSRAMATModel(num_classes=num_classes, **kwargs)


# 示例使用
if __name__ == "__main__":
    # 创建模型
    model = create_fsra_mat_model(num_classes=701)
    
    # 示例输入
    batch_size = 4
    sat_images = torch.randn(batch_size, 3, 256, 256)
    uav_images = torch.randn(batch_size, 3, 256, 256)
    descriptions = [
        "University campus with modern buildings and green spaces",
        "Urban area with residential buildings and parking lots", 
        "Commercial district with tall office buildings",
        "Academic complex with library and student centers"
    ]
    
    # 前向传播
    with torch.no_grad():
        outputs = model(sat_images, uav_images, descriptions)
    
    print("FSRA-MAT Model Output Shapes:")
    print(f"Global prediction: {outputs['global_prediction'].shape}")
    print(f"Regional predictions: {len(outputs['regional_predictions'])} regions")
    print(f"Final features: {outputs['final_features'].shape}")
    if outputs['attention_weights'] is not None:
        print(f"Attention weights: {outputs['attention_weights'].shape}") 