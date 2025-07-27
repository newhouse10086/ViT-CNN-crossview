"""Cross-attention model for ViT-CNN-crossview."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrossAttentionModel(nn.Module):
    """Cross-attention model for feature alignment between views."""
    
    def __init__(self, d_model: int = 512, block_size: int = 4, num_heads: int = 8):
        """
        Initialize cross-attention model.
        
        Args:
            d_model: Model dimension
            block_size: Number of blocks/regions
            num_heads: Number of attention heads
        """
        super(CrossAttentionModel, self).__init__()
        
        self.d_model = d_model
        self.block_size = block_size
        self.num_heads = num_heads
        
        # Linear transformations for queries, keys, and values
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        
        # Class-wise softmax for attention
        self.class_softmax = nn.Softmax(dim=-1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, support_set: torch.Tensor, queries: torch.Tensor, 
                mode: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-attention.
        
        Args:
            support_set: Support set features (B, N, D)
            queries: Query features (B, 1, D)
            mode: Mode for different operations (0: training, 1: testing)
            
        Returns:
            Tuple of (query_prototype, class_prototype)
        """
        # Apply linear transformations
        support_set_ks = self.k_linear(support_set)
        queries_qs = self.q_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)
        
        # Compute attention weights
        # queries_qs: (B, 1, D), support_set_ks: (B, N, D)
        attention_scores = torch.bmm(queries_qs, support_set_ks.transpose(1, 2))
        attention_scores = attention_scores / (self.d_model ** 0.5)
        
        # Apply softmax to get attention weights
        affinity_matrix = self.class_softmax(attention_scores)
        
        # Compute class prototype using attention weights
        class_prototype = torch.bmm(affinity_matrix, support_set_vs)
        
        # Query prototype
        query_prototype = queries_vs
        
        # Apply multi-head attention for refinement
        query_refined, _ = self.multihead_attn(
            query_prototype, support_set, support_set
        )
        
        # Residual connection and normalization
        query_prototype = self.norm1(query_prototype + query_refined)
        
        # Feed-forward network
        query_ffn = self.ffn(query_prototype)
        query_prototype = self.norm2(query_prototype + query_ffn)
        
        # Apply output projection
        query_prototype = self.output_proj(query_prototype)
        class_prototype = self.output_proj(class_prototype)
        
        if mode == 0:  # Training mode
            return query_prototype, class_prototype
        elif mode == 1:  # Testing mode
            return query_prototype, support_set_vs
        else:
            raise ValueError(f"Unsupported mode: {mode}")


class FeatureAlignmentModule(nn.Module):
    """Feature alignment module using cross-attention."""
    
    def __init__(self, feature_dim: int = 512, num_regions: int = 4):
        """
        Initialize feature alignment module.
        
        Args:
            feature_dim: Feature dimension
            num_regions: Number of regions/blocks
        """
        super(FeatureAlignmentModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_regions = num_regions
        
        # Cross-attention for each region
        self.cross_attentions = nn.ModuleList([
            CrossAttentionModel(d_model=feature_dim, block_size=1, num_heads=8)
            for _ in range(num_regions)
        ])
        
        # Global cross-attention
        self.global_cross_attention = CrossAttentionModel(
            d_model=feature_dim, block_size=num_regions, num_heads=8
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim * (num_regions + 1), feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, satellite_features: torch.Tensor, 
                drone_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for feature alignment.
        
        Args:
            satellite_features: Satellite features (B, N+1, D) - global + regional
            drone_features: Drone features (B, N+1, D) - global + regional
            
        Returns:
            Tuple of aligned features (satellite, drone)
        """
        B, N_plus_1, D = satellite_features.shape
        N = N_plus_1 - 1  # Number of regions
        
        # Split global and regional features
        sat_global = satellite_features[:, 0:1, :]  # (B, 1, D)
        sat_regional = satellite_features[:, 1:, :]  # (B, N, D)
        
        drone_global = drone_features[:, 0:1, :]  # (B, 1, D)
        drone_regional = drone_features[:, 1:, :]  # (B, N, D)
        
        # Regional cross-attention
        sat_aligned_regional = []
        drone_aligned_regional = []
        
        for i in range(min(N, self.num_regions)):
            # Satellite region attends to drone regions
            sat_region_query = sat_regional[:, i:i+1, :]
            sat_aligned_region, _ = self.cross_attentions[i](
                drone_regional, sat_region_query
            )
            sat_aligned_regional.append(sat_aligned_region)
            
            # Drone region attends to satellite regions
            drone_region_query = drone_regional[:, i:i+1, :]
            drone_aligned_region, _ = self.cross_attentions[i](
                sat_regional, drone_region_query
            )
            drone_aligned_regional.append(drone_aligned_region)
        
        # Global cross-attention
        sat_global_aligned, _ = self.global_cross_attention(
            drone_features, sat_global
        )
        drone_global_aligned, _ = self.global_cross_attention(
            satellite_features, drone_global
        )
        
        # Concatenate aligned features
        if sat_aligned_regional:
            sat_aligned_regional = torch.cat(sat_aligned_regional, dim=1)
            drone_aligned_regional = torch.cat(drone_aligned_regional, dim=1)
            
            sat_all_aligned = torch.cat([sat_global_aligned, sat_aligned_regional], dim=1)
            drone_all_aligned = torch.cat([drone_global_aligned, drone_aligned_regional], dim=1)
        else:
            sat_all_aligned = sat_global_aligned
            drone_all_aligned = drone_global_aligned
        
        # Feature fusion
        sat_fused = self.feature_fusion(sat_all_aligned.flatten(1))
        drone_fused = self.feature_fusion(drone_all_aligned.flatten(1))
        
        return sat_fused, drone_fused


class AdaptiveCrossAttention(nn.Module):
    """Adaptive cross-attention with learnable attention patterns."""
    
    def __init__(self, feature_dim: int = 512, num_patterns: int = 4):
        """
        Initialize adaptive cross-attention.
        
        Args:
            feature_dim: Feature dimension
            num_patterns: Number of attention patterns
        """
        super(AdaptiveCrossAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_patterns = num_patterns
        
        # Learnable attention patterns
        self.attention_patterns = nn.Parameter(
            torch.randn(num_patterns, feature_dim, feature_dim)
        )
        
        # Pattern selection network
        self.pattern_selector = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_patterns),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, query_features: torch.Tensor, 
                support_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for adaptive cross-attention.
        
        Args:
            query_features: Query features (B, D)
            support_features: Support features (B, D)
            
        Returns:
            Aligned query features (B, D)
        """
        B, D = query_features.shape
        
        # Concatenate features for pattern selection
        combined_features = torch.cat([query_features, support_features], dim=-1)
        
        # Select attention pattern
        pattern_weights = self.pattern_selector(combined_features)  # (B, num_patterns)
        
        # Compute weighted attention patterns
        weighted_patterns = torch.einsum('bp,pij->bij', pattern_weights, self.attention_patterns)
        
        # Apply attention
        aligned_features = torch.bmm(
            query_features.unsqueeze(1), weighted_patterns
        ).squeeze(1)
        
        # Output projection
        output = self.output_proj(aligned_features)
        
        return output
