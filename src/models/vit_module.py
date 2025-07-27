"""
Vision Transformer Module for FSRA_IMPROVED
Implements patch-based ViT for 10x10 patch processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class PatchEmbedding(nn.Module):
    """
    Patch embedding module for ViT.
    Divides input image into patches and embeds them.
    """
    
    def __init__(self, img_size: int = 256, patch_size: int = 10, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2  # 10x10 = 100 patches
        self.patch_dim = in_channels * patch_size * patch_size  # 3 * 10 * 10 = 300
        
        # Linear projection of flattened patches
        self.projection = nn.Linear(self.patch_dim, embed_dim)
        
        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Class token (optional, for compatibility)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of patch embedding.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Embedded patches of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Ensure input size is correct
        assert H == self.img_size and W == self.img_size, f"Input size {H}x{W} doesn't match expected {self.img_size}x{self.img_size}"
        
        # Extract patches using unfold
        # unfold(dimension, size, step)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Shape: (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
        
        # Reshape to (B, num_patches, patch_dim)
        num_patches_h = H // self.patch_size  # 25
        num_patches_w = W // self.patch_size  # 25
        patches = patches.contiguous().view(B, C, num_patches_h * num_patches_w, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous()  # (B, num_patches, C, patch_size^2)
        patches = patches.view(B, self.num_patches, self.patch_dim)  # (B, 100, 300)
        
        # Linear projection
        embeddings = self.projection(patches)  # (B, 100, embed_dim)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embeddings
        
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for ViT.
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x: Input tensor of shape (B, N, embed_dim)
            
        Returns:
            Output tensor of shape (B, N, embed_dim)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, embed_dim)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP.
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape (B, N, embed_dim)
            
        Returns:
            Output tensor of shape (B, N, embed_dim)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for patch-based image processing.
    Designed for 10x10 patch division of 256x256 images.
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 10,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 6,  # Reduced depth for efficiency
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_dim: int = 100  # Target output dimension for CNN fusion
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches  # 100 patches
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection to match CNN features
        self.output_proj = nn.Linear(embed_dim, output_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Vision Transformer.
        
        Args:
            x: Input tensor of shape (B, 3, 256, 256)
            
        Returns:
            Output tensor of shape (B, output_dim, 8, 8) to match CNN scale
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, 100, embed_dim)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)  # (B, 100, embed_dim)
        
        # Output projection
        x = self.output_proj(x)  # (B, 100, output_dim)
        
        # Reshape to spatial format to match CNN: (B, 100, output_dim) -> (B, output_dim, 10, 10)
        B = x.shape[0]
        x = x.permute(0, 2, 1).contiguous()  # (B, output_dim, 100)
        x = x.view(B, self.output_dim, 10, 10)  # (B, output_dim, 10, 10)
        
        # Resize to 8x8 to match CNN spatial scale using adaptive pooling
        x = F.adaptive_avg_pool2d(x, (8, 8))  # (B, output_dim, 8, 8)
        
        return x


def create_vit_module(config: dict) -> VisionTransformer:
    """
    Create ViT module based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VisionTransformer instance
    """
    vit_config = config.get('vit', {})
    
    return VisionTransformer(
        img_size=config.get('image_size', 256),
        patch_size=config.get('patch_size', 10),
        in_channels=3,
        embed_dim=vit_config.get('embed_dim', 768),
        depth=vit_config.get('depth', 6),
        num_heads=vit_config.get('num_heads', 12),
        mlp_ratio=vit_config.get('mlp_ratio', 4.0),
        dropout=vit_config.get('dropout', 0.1),
        output_dim=config.get('vit_output_dim', 100)
    )
