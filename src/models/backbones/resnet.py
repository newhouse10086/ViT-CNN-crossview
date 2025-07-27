"""ResNet backbone for ViT-CNN-crossview."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNet18Backbone(nn.Module):
    """ResNet18 backbone for feature extraction."""
    
    def __init__(self, pretrained: bool = True, output_stride: int = 16):
        """
        Initialize ResNet18 backbone.
        
        Args:
            pretrained: Whether to use pretrained weights
            output_stride: Output stride for feature maps
        """
        super(ResNet18Backbone, self).__init__()
        
        # Load ResNet18 with version compatibility
        try:
            # Try new torchvision API (v0.13+)
            if pretrained:
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet18(weights=None)
        except (AttributeError, TypeError):
            # Fallback to old torchvision API (< v0.13)
            resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Modify stride if needed
        if output_stride == 8:
            self.layer3[0].conv1.stride = (1, 1)
            self.layer3[0].downsample[0].stride = (1, 1)
            self.layer4[0].conv1.stride = (1, 1)
            self.layer4[0].downsample[0].stride = (1, 1)
        elif output_stride == 16:
            self.layer4[0].conv1.stride = (1, 1)
            self.layer4[0].downsample[0].stride = (1, 1)
        
        # Feature dimensions
        self.feature_dim = 512
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Feature tensor of shape (B, 512, H/16, W/16)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


def resnet18_backbone(pretrained: bool = True, output_stride: int = 16) -> ResNet18Backbone:
    """
    Create ResNet18 backbone.
    
    Args:
        pretrained: Whether to use pretrained weights
        output_stride: Output stride for feature maps
        
    Returns:
        ResNet18 backbone model
    """
    return ResNet18Backbone(pretrained=pretrained, output_stride=output_stride)
