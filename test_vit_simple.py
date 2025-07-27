#!/usr/bin/env python3
"""
Simple test for ViT module without dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Copy the ViT classes directly to avoid import issues
class PatchEmbedding(nn.Module):
    """Patch embedding module for ViT."""
    
    def __init__(self, img_size: int = 256, patch_size: int = 10, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2  # 256//10 = 25, so 25x25 = 625 patches
        self.patch_dim = in_channels * patch_size * patch_size  # 3 * 10 * 10 = 300
        
        # Linear projection of flattened patches
        self.projection = nn.Linear(self.patch_dim, embed_dim)
        
        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of patch embedding."""
        B, C, H, W = x.shape
        
        # Ensure input size is correct
        assert H == self.img_size and W == self.img_size, f"Input size {H}x{W} doesn't match expected {self.img_size}x{self.img_size}"
        
        # Extract patches using unfold
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        
        # Reshape to (B, num_patches, patch_dim)
        num_patches_h = H // self.patch_size  # 256//10 = 25
        num_patches_w = W // self.patch_size  # 256//10 = 25
        patches = patches.contiguous().view(B, C, num_patches_h * num_patches_w, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous()  # (B, num_patches, C, patch_size^2)
        patches = patches.view(B, self.num_patches, self.patch_dim)  # (B, 625, 300)
        
        # Linear projection
        embeddings = self.projection(patches)  # (B, 625, embed_dim)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embeddings
        
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism for ViT."""
    
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
        """Forward pass of multi-head self-attention."""
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
    """Transformer block with self-attention and MLP."""
    
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
        """Forward pass of transformer block."""
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for patch-based image processing."""
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 10,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_dim: int = 100
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches  # 625 patches
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection to match CNN features
        self.output_proj = nn.Linear(embed_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Vision Transformer."""
        # Patch embedding
        x = self.patch_embed(x)  # (B, 625, embed_dim)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)  # (B, 625, embed_dim)
        
        # Output projection
        x = self.output_proj(x)  # (B, 625, output_dim)
        
        # Reshape to spatial format: (B, num_patches, output_dim) -> (B, output_dim, H_patches, W_patches)
        B = x.shape[0]
        x = x.permute(0, 2, 1).contiguous()  # (B, output_dim, num_patches)
        
        # Calculate spatial dimensions
        patches_per_side = int(self.num_patches ** 0.5)  # 25 for 625 patches
        x = x.view(B, self.output_dim, patches_per_side, patches_per_side)  # (B, output_dim, 25, 25)
        
        # Resize to 8x8 to match CNN spatial scale using adaptive pooling
        x = F.adaptive_avg_pool2d(x, (8, 8))  # (B, output_dim, 8, 8)
        
        return x


def test_vit_module():
    """Test the fixed ViT module."""
    print("🧪 Testing Fixed ViT Module")
    print("="*40)
    
    try:
        # Create ViT with correct parameters
        vit = VisionTransformer(
            img_size=256,
            patch_size=10,
            embed_dim=768,
            depth=6,
            output_dim=100
        )
        
        print(f"✅ ViT created successfully!")
        print(f"   Patch size: {vit.patch_embed.patch_size}")
        print(f"   Num patches: {vit.patch_embed.num_patches}")
        print(f"   Expected: 256//10 = 25, so 25×25 = 625 patches")
        
        # Test forward pass
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        
        print(f"\n🔍 Testing forward pass...")
        print(f"   Input shape: {input_tensor.shape}")
        
        with torch.no_grad():
            output = vit(input_tensor)
        
        print(f"   Output shape: {output.shape}")
        print(f"   Expected: ({batch_size}, 100, 8, 8)")
        
        if output.shape == (batch_size, 100, 8, 8):
            print(f"✅ ViT forward pass successful!")
            return True
        else:
            print(f"❌ Output shape mismatch!")
            return False
            
    except Exception as e:
        print(f"❌ ViT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🎯 ViT Module Fix Verification (Standalone)")
    print("="*50)
    
    if test_vit_module():
        print(f"\n🎉 ViT fix is working!")
        print(f"🚀 The shape error should be resolved!")
        print(f"📝 Key fixes applied:")
        print(f"   - Corrected num_patches: 625 (25×25) instead of 100")
        print(f"   - Fixed spatial reshaping: 25×25 instead of 10×10")
        print(f"   - Proper adaptive pooling to 8×8")
    else:
        print(f"\n⚠️  ViT test failed. Please check the error above.")


if __name__ == "__main__":
    main()
