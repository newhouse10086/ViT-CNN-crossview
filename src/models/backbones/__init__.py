"""Backbone networks for ViT-CNN-crossview."""

from .vit_pytorch import vit_small_patch16_224
from .resnet import resnet18_backbone

__all__ = [
    'vit_small_patch16_224',
    'resnet18_backbone'
]
