from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


@dataclass
class ModelConfig:
    num_classes: int = 4  # Attentive, Distracted, Confused, Disengaged
    pretrained: bool = True
    dropout_p: float = 0.2
    arch: str = "resnet18"  # or "resnet50"


class EngagementNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        arch = (config.arch or "resnet18").lower()
        if arch == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if config.pretrained else None
            self.backbone = models.resnet50(weights=weights)
        else:
            weights = models.ResNet18_Weights.DEFAULT if config.pretrained else None
            self.backbone = models.resnet18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(config.dropout_p),
            nn.Linear(in_feats, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_model(num_classes: int = 4, pretrained: bool = True, dropout_p: float = 0.2, arch: str = "resnet18") -> nn.Module:
    return EngagementNet(ModelConfig(num_classes=num_classes, pretrained=pretrained, dropout_p=dropout_p, arch=arch))
