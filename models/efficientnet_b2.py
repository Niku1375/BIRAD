import torch
import torch.nn as nn
import timm

class EfficientNetB2Paired(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b2", pretrained=True, num_classes=0)
        feat_dim      = self.backbone.num_features
        self.head     = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, cc, mlo):
        f_cc  = self.backbone(cc)
        f_mlo = self.backbone(mlo)
        return self.head(torch.cat([f_cc, f_mlo], dim=1))

def get_model(num_classes=5):
    return EfficientNetB2Paired(num_classes)