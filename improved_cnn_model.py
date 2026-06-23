"""Improved CNN model with transfer learning"""

import torch
import torch.nn as nn
from torchvision import models


class ImprovedGenreClassifier(nn.Module):
    """
    EfficientNet-B0 with transfer learning for music genre classification.
    
    Strategy:
    - Adapt first conv layer for 1-channel input
    - Start with frozen backbone (phase 1)
    - Unfreeze for fine-tuning (phase 2)
    - Custom classifier head with dropout
    """
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of genre classes to predict
        """
        super().__init__()
        
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)
        
        # Adapt first conv to accept 1-channel input
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        
        # Initialize with mean of 3 channel weights
        with torch.no_grad():
            new_conv.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.features[0][0] = new_conv
        
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Custom classifier head
        in_features = backbone.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters (phase 1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters (phase 2 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = True