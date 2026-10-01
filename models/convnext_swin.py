import torch
import torch.nn as nn
import timm

class ConvNeXtSwinPaired(nn.Module):
    """
    Proposed hybrid:
    ConvNeXt-Tiny (local features) → Swin Transformer encoder (global context)
    Applied to CC and MLO independently → concat → classify
    """
    def __init__(self, num_classes=5):
        super().__init__()
        # ConvNeXt-Tiny as local feature extractor (remove head)
        convnext      = timm.create_model("convnext_tiny", pretrained=True, num_classes=0,
                                           global_pool="")
        # feature maps from last stage: [B, 768, 7, 7] at 224x224
        self.convnext = convnext

        # Swin encoder — takes sequence of tokens
        # We use swin_tiny and replace patch embed with identity-compatible input
        self.swin     = timm.create_model("swin_tiny_patch4_window7_224",
                                           pretrained=True, num_classes=0)

        # project ConvNeXt output to Swin input space
        # ConvNeXt last stage: 768 channels, 7x7 spatial
        # We flatten to [B, 49, 768] and project to [B, 49, 96] (Swin embed dim)
        self.proj     = nn.Linear(768, 96)

        swin_feat_dim = self.swin.num_features  # 768
        self.head     = nn.Sequential(
            nn.Linear(swin_feat_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def encode(self, x):
        # ConvNeXt feature maps
        feat = self.convnext(x)           # [B, 768, 7, 7]
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1)  # [B, 7, 7, 768]
        feat = feat.reshape(B, H*W, C)   # [B, 49, 768]
        feat = self.proj(feat)            # [B, 49, 96]

        # Pass through Swin layers (skip patch embed — inject directly)
        # Reshape back to [B, H, W, C] for Swin layers
        feat = feat.reshape(B, H, W, 96)
        for layer in self.swin.layers:
            feat = layer(feat)            # Swin stages

        # Global average pool
        feat = feat.mean(dim=[1, 2])     # [B, 768]
        return feat

    def forward(self, cc, mlo):
        f_cc  = self.encode(cc)
        f_mlo = self.encode(mlo)
        return self.head(torch.cat([f_cc, f_mlo], dim=1))

def get_model(num_classes=5):
    return ConvNeXtSwinPaired(num_classes)