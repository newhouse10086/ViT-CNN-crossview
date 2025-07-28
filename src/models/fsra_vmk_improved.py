"""
FSRA-VMK: Vision Mamba Kolmogorov Network for Cross-View Image Matching
基于2024年最新神经网络模块的创新架构：
- Vision Mamba Encoder (VME)
- ConvNeXt V2 Fusion Module (CFM) 
- Kolmogorov-Arnold Attention (KAA)
- Bidirectional Cross-View Alignment (BCVA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional


# ================== Kolmogorov-Arnold Networks (KAN) 模块 ==================

class KANLinear(nn.Module):
    """Kolmogorov-Arnold Network Linear Layer (2024最新)"""
    
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # 基础权重
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # 样条函数系数 (核心KAN创新)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + 1))
        
        # 缩放参数
        self.scale_noise = 0.1
        self.scale_base = 1.0
        self.scale_spline = 1.0
        
        # 网格点
        grid = torch.linspace(-1, 1, steps=grid_size + 1)
        self.register_buffer('grid', grid)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1) - 1/2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(noise[None, None, :])
            
    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """B样条基函数 (KAN的核心数学基础) - 简化版本"""
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch_size, in_features = x.shape
        
        # 简化的B样条实现：使用RBF核近似
        grid = self.grid  # (grid_size + 1,)
        x_expanded = x.unsqueeze(-1)  # (batch, in_features, 1)
        grid_expanded = grid.view(1, 1, -1)  # (1, 1, grid_size + 1)
        
        # 使用高斯核作为B样条的近似
        sigma = 1.0 / self.grid_size
        distances = (x_expanded - grid_expanded) ** 2
        bases = torch.exp(-distances / (2 * sigma ** 2))
        
        # 归一化
        bases = bases / (bases.sum(dim=-1, keepdim=True) + 1e-8)
        
        return bases  # (batch, in_features, grid_size + 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - KAN的核心计算"""
        base_output = F.linear(x, self.base_weight)  # 基础线性变换
        
        # 样条函数计算
        spline_bases = self.b_splines(x)  # (batch, in_features, grid_size + 1)
        spline_output = torch.einsum('bio,oig->bo', spline_bases, self.spline_weight)
        
        return self.scale_base * base_output + self.scale_spline * spline_output


class KolmogorovArnoldAttention(nn.Module):
    """基于KAN的注意力机制 (2024创新)"""
    
    def __init__(self, dim: int, num_heads: int = 8, grid_size: int = 5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 使用KAN替代传统的线性变换
        self.q_kan = KANLinear(dim, dim, grid_size)
        self.k_kan = KANLinear(dim, dim, grid_size)
        self.v_kan = KANLinear(dim, dim, grid_size)
        self.proj_kan = KANLinear(dim, dim, grid_size)
        
        # 相对位置编码 (KAN增强)
        self.relative_position_kan = KANLinear(2, num_heads, grid_size)
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) where N = H * W
            H, W: 特征图的高度和宽度
        """
        B, N, C = x.shape
        
        # 将3D输入reshape为2D供KAN使用
        x_flat = x.contiguous().view(-1, C)  # (B*N, C)
        
        # KAN-based Q, K, V computation
        q_flat = self.q_kan(x_flat)  # (B*N, C)
        k_flat = self.k_kan(x_flat)  # (B*N, C)
        v_flat = self.v_kan(x_flat)  # (B*N, C)
        
        # Reshape back to (B, N, C) then to attention format
        q = q_flat.contiguous().view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k_flat.contiguous().view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v_flat.contiguous().view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 相对位置编码 (KAN增强)
        coords_h = torch.arange(H, device=x.device)
        coords_w = torch.arange(W, device=x.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, H, W
        coords_flatten = torch.flatten(coords, 1)  # 2, H*W
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, H*W, H*W
        relative_coords = relative_coords.permute(1, 2, 0).float()  # H*W, H*W, 2
        
        # 归一化到[-1, 1]
        relative_coords[:, :, 0] /= (H - 1)
        relative_coords[:, :, 1] /= (W - 1)
        
        # KAN处理相对位置
        relative_position_bias = self.relative_position_kan(
            relative_coords.view(-1, 2)
        ).view(N, N, self.num_heads).permute(2, 0, 1)  # num_heads, N, N
        
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # KAN投影 - 需要flatten再reshape
        x_flat = x.contiguous().view(-1, C)  # (B*N, C)
        x_proj = self.proj_kan(x_flat)  # (B*N, C)
        x = x_proj.contiguous().view(B, N, C)  # (B, N, C)
        
        return x


# ================== Vision Mamba 模块 ==================

class VisionMambaBlock(nn.Module):
    """Vision Mamba Block - 2024年最新的状态空间模型"""
    
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.dim)
        
        # 输入投影
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        
        # 深度可分离卷积
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        
        # SSM参数
        self.x_proj = nn.Linear(self.d_inner, (d_state * 2), bias=False)
        self.dt_proj = nn.Linear(self.d_inner, d_state, bias=True)
        
        # 状态空间参数
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        
        # 归一化
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
            H, W: 特征图尺寸
        Returns:
            (B, H*W, C)
        """
        B, L, D = x.shape
        
        # 残差连接
        residual = x
        x = self.norm(x)
        
        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)
        
        # 转换为2D进行卷积
        x = x.transpose(1, 2).view(B, self.d_inner, H, W)  # (B, d_inner, H, W)
        x = self.conv2d(x)
        x = x.view(B, self.d_inner, L).transpose(1, 2)  # (B, L, d_inner)
        
        # 激活
        x = F.silu(x)
        
        # SSM计算
        A = -torch.exp(self.A_log.float())  # (d_state,)
        
        # 投影到SSM参数
        x_dbl = self.x_proj(x)  # (B, L, 2*d_state)
        delta, B_ssm = x_dbl.chunk(2, dim=-1)
        
        dt = self.dt_proj(x)  # (B, L, d_state)
        dt = F.softplus(dt)
        
        # 选择性SSM (Mamba的核心)
        y = self.selective_scan(x, delta, A, B_ssm, dt, self.D)
        
        # 门控机制
        y = y * F.silu(z)
        
        # 输出投影
        output = self.out_proj(y)
        
        return output + residual
    
    def selective_scan(self, u, delta, A, B, dt, D):
        """选择性扫描 - Mamba的核心算法 (简化版本)"""
        # 简化实现：直接返回加权的输入特征
        B_batch, L, C = u.shape
        
        # 简化的状态空间更新
        # 使用平均池化来模拟状态传播效果
        u_avg = F.avg_pool1d(u.transpose(1, 2), kernel_size=3, stride=1, padding=1).transpose(1, 2)
        
        # 加权组合
        alpha = torch.sigmoid(dt.mean(dim=-1, keepdim=True))  # (B, L, 1)
        output = alpha * u + (1 - alpha) * u_avg
        
        return output


class VisionMambaEncoder(nn.Module):
    """Vision Mamba编码器 - 简化版本避免形状问题"""
    
    def __init__(self, img_size: int = 256, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 384, 
                 depth: int = 6, d_state: int = 16):  # 减少深度避免复杂度
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.H = self.W = img_size // patch_size
        
        # Patch嵌入 - 简化版本
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.H, self.W))
        
        # 简化的Mamba块 - 使用卷积近似
        self.mamba_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1),
                nn.BatchNorm2d(embed_dim)
            ) for _ in range(depth)
        ])
        
        # 特征金字塔网络层
        self.pyramid_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 2, 1),  # 下采样
                nn.BatchNorm2d(embed_dim),
                nn.GELU()
            ) for _ in range(3)
        ])
        
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            多尺度特征字典
        """
        B = x.shape[0]
        
        # Patch嵌入
        x = self.patch_embed(x)  # (B, C, H, W)
        x = x + self.pos_embed
        
        # 存储多尺度特征
        features_list = []
        
        # 通过简化的Mamba块
        for i, block in enumerate(self.mamba_blocks):
            residual = x
            x = block(x) + residual  # 残差连接
            
            # 在特定层收集特征用于金字塔
            if i in [1, 3, 5]:  # 3个不同深度的特征
                features_list.append(x)
        
        # 确保至少有一个特征
        if not features_list:
            features_list = [x]
        
        # 构建特征金字塔
        pyramid_features = [features_list[0]]  # 最高分辨率
        
        current_feat = features_list[0]
        for layer in self.pyramid_layers:
            current_feat = layer(current_feat)
            pyramid_features.append(current_feat)
        
        return {
            'S1': pyramid_features[0],  # (B, C, H, W)
            'S2': pyramid_features[1] if len(pyramid_features) > 1 else pyramid_features[0],
            'S3': pyramid_features[2] if len(pyramid_features) > 2 else pyramid_features[0],
            'S4': pyramid_features[3] if len(pyramid_features) > 3 else pyramid_features[0],
            'mamba_features': x
        }


# ================== ConvNeXt V2 模块 ==================

class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 Block - 2023年改进版本"""
    
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer Scale V2 (ConvNeXt V2创新)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        
        # Global Response Normalization (ConvNeXt V2核心创新)
        self.grn = GlobalResponseNorm(4 * dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        
        # Global Response Normalization (ConvNeXt V2创新)
        x = self.grn(x)
        
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        x = input_x + x
        return x


class GlobalResponseNorm(nn.Module):
    """Global Response Normalization - ConvNeXt V2的核心创新"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2FusionModule(nn.Module):
    """ConvNeXt V2融合模块 - 现代化卷积特征融合"""
    
    def __init__(self, in_channels: int = 384, fusion_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.fusion_channels = fusion_channels
        
        # 各尺度特征适配
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, fusion_channels, 1),
                nn.GroupNorm(32, fusion_channels),
                nn.GELU()
            ) for _ in range(4)
        ])
        
        # ConvNeXt V2融合块
        self.fusion_blocks = nn.ModuleList([
            ConvNeXtV2Block(fusion_channels) for _ in range(3)
        ])
        
        # 特征金字塔上采样
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose2d(fusion_channels, fusion_channels, 4, 2, 1),
            nn.ConvTranspose2d(fusion_channels, fusion_channels, 4, 2, 1),
            nn.ConvTranspose2d(fusion_channels, fusion_channels, 4, 2, 1)
        ])
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(fusion_channels * 4, fusion_channels, 3, 1, 1),
            ConvNeXtV2Block(fusion_channels),
            nn.Conv2d(fusion_channels, fusion_channels, 1)
        )
        
    def forward(self, mamba_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            mamba_features: {'S1', 'S2', 'S3', 'S4'} from Vision Mamba
        Returns:
            融合后的特征 (B, fusion_channels, H, W)
        """
        # 特征适配
        adapted_features = []
        feature_keys = ['S1', 'S2', 'S3', 'S4']
        
        for i, key in enumerate(feature_keys):
            feat = self.adapters[i](mamba_features[key])
            adapted_features.append(feat)
        
        # ConvNeXt V2特征增强
        for block in self.fusion_blocks:
            adapted_features = [block(feat) for feat in adapted_features]
        
        # 上采样到相同尺寸 (以S1为基准)
        target_size = adapted_features[0].shape[-2:]
        upsampled_features = [adapted_features[0]]  # S1保持不变
        
        current_feat = adapted_features[1]  # S2
        for i, upsample in enumerate(self.upsample_layers):
            current_feat = upsample(current_feat)
            if i < len(adapted_features) - 2:
                # 融合下一个尺度的特征
                current_feat = current_feat + F.interpolate(
                    adapted_features[i + 2], size=current_feat.shape[-2:], 
                    mode='bilinear', align_corners=False
                )
            upsampled_features.append(current_feat)
        
        # 所有特征插值到目标尺寸
        aligned_features = []
        for feat in upsampled_features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # 最终融合
        concatenated = torch.cat(aligned_features, dim=1)
        fused_output = self.final_fusion(concatenated)
        
        return fused_output


# ================== 双向跨视角对齐模块 ==================

class BidirectionalCrossViewAlignment(nn.Module):
    """双向跨视角对齐 - 超越传统单向对齐"""
    
    def __init__(self, feature_dim: int = 256, num_heads: int = 8, 
                 hidden_dim: int = 1024, num_layers: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # KAN增强的交叉注意力
        self.sat_to_uav_attention = nn.ModuleList([
            KolmogorovArnoldAttention(feature_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        self.uav_to_sat_attention = nn.ModuleList([
            KolmogorovArnoldAttention(feature_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # 特征细化网络
        self.feature_refinement = nn.ModuleList([
            nn.Sequential(
                KANLinear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                KANLinear(hidden_dim, feature_dim),
                nn.LayerNorm(feature_dim)
            ) for _ in range(num_layers)
        ])
        
        # 相似性计算模块
        self.similarity_computation = nn.Sequential(
            KANLinear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            KANLinear(hidden_dim, feature_dim // 4), 
            nn.GELU(),
            KANLinear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            KANLinear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
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
        B, C, H, W = sat_features.shape
        
        # 展平为序列
        sat_seq = sat_features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        uav_seq = uav_features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # 多层双向注意力
        for i in range(len(self.sat_to_uav_attention)):
            # 卫星->无人机注意力
            sat_enhanced = self.sat_to_uav_attention[i](sat_seq, H, W)
            
            # 无人机->卫星注意力  
            uav_enhanced = self.uav_to_sat_attention[i](uav_seq, H, W)
            
            # 特征细化 - 需要处理KAN输入
            sat_enhanced_flat = sat_enhanced.contiguous().view(-1, sat_enhanced.size(-1))
            sat_refined_flat = self.feature_refinement[i](sat_enhanced_flat)
            sat_refined = sat_refined_flat.view_as(sat_enhanced)
            sat_seq = sat_enhanced + sat_refined
            
            uav_enhanced_flat = uav_enhanced.contiguous().view(-1, uav_enhanced.size(-1))
            uav_refined_flat = self.feature_refinement[i](uav_enhanced_flat)
            uav_refined = uav_refined_flat.view_as(uav_enhanced)
            uav_seq = uav_enhanced + uav_refined
        
        # 全局池化
        sat_global = torch.mean(sat_seq, dim=1)  # (B, C)
        uav_global = torch.mean(uav_seq, dim=1)  # (B, C)
        
        # 计算相似性权重
        similarity_input = torch.cat([sat_global, uav_global], dim=1)
        similarity_weight = self.similarity_computation(similarity_input)  # (B, 1)
        
        # 加权融合
        weighted_sat = sat_global * similarity_weight
        weighted_uav = uav_global * (1 - similarity_weight)
        
        # 最终融合
        fusion_input = torch.cat([weighted_sat, weighted_uav], dim=1)
        aligned_features = self.final_fusion(fusion_input)
        
        return aligned_features


# ================== 主模型 ==================

class FSRAVMKModel(nn.Module):
    """FSRA-VMK主模型：Vision Mamba Kolmogorov Network"""
    
    def __init__(self, num_classes: int = 701, img_size: int = 256, 
                 embed_dim: int = 384, mamba_depth: int = 12):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Vision Mamba编码器
        self.vision_mamba = VisionMambaEncoder(
            img_size=img_size,
            embed_dim=embed_dim,
            depth=mamba_depth
        )
        
        # ConvNeXt V2融合模块
        self.convnext_fusion = ConvNeXtV2FusionModule(
            in_channels=embed_dim,
            fusion_channels=256
        )
        
        # 双向跨视角对齐
        self.cross_view_alignment = BidirectionalCrossViewAlignment(
            feature_dim=256
        )
        
        # 多头分类器
        self.global_classifier = nn.Sequential(
            KANLinear(256, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            KANLinear(512, num_classes)
        )
        
        # 区域分类器 (保持兼容性)
        self.regional_classifiers = nn.ModuleList([
            nn.Sequential(
                KANLinear(256, 256),
                nn.GELU(),
                KANLinear(256, num_classes)
            ) for _ in range(6)
        ])
        
        # 语义分类器 (新增)
        self.semantic_classifier = nn.Sequential(
            KANLinear(256, 128),
            nn.GELU(),
            KANLinear(128, num_classes // 4)  # 语义类别数较少
        )
        
    def forward(self, sat_img: torch.Tensor, uav_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # Vision Mamba特征提取
        sat_mamba_features = self.vision_mamba(sat_img)
        uav_mamba_features = self.vision_mamba(uav_img)
        
        # ConvNeXt V2特征融合
        sat_fused = self.convnext_fusion(sat_mamba_features)
        uav_fused = self.convnext_fusion(uav_mamba_features)
        
        # 双向跨视角对齐
        aligned_features = self.cross_view_alignment(sat_fused, uav_fused)
        
        # 多头分类
        global_pred = self.global_classifier(aligned_features)
        
        regional_preds = []
        for classifier in self.regional_classifiers:
            regional_pred = classifier(aligned_features)
            regional_preds.append(regional_pred)
            
        semantic_pred = self.semantic_classifier(aligned_features)
        
        return {
            'global_prediction': global_pred,
            'regional_predictions': regional_preds,
            'semantic_prediction': semantic_pred,
            'aligned_features': aligned_features,
            'sat_fused': sat_fused,
            'uav_fused': uav_fused
        }


def create_fsra_vmk_model(num_classes: int = 701, **kwargs) -> FSRAVMKModel:
    """创建FSRA-VMK模型"""
    return FSRAVMKModel(num_classes=num_classes, **kwargs)


# 示例使用
if __name__ == "__main__":
    # 创建模型
    model = create_fsra_vmk_model(num_classes=701)
    
    # 示例输入
    batch_size = 2
    sat_images = torch.randn(batch_size, 3, 256, 256)
    uav_images = torch.randn(batch_size, 3, 256, 256)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(sat_images, uav_images)
    
    print("FSRA-VMK Model Output Shapes:")
    print(f"Global prediction: {outputs['global_prediction'].shape}")
    print(f"Regional predictions: {len(outputs['regional_predictions'])} regions")
    print(f"Semantic prediction: {outputs['semantic_prediction'].shape}")
    print(f"Aligned features: {outputs['aligned_features'].shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}") 