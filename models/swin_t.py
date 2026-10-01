import torch
import torch.nn as nn
import timm

class SwinTPaired(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224",
                                           pretrained=True, num_classes=0)
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
    return SwinTPaired(num_classes)