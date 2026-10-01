# Hybrid ConvNeXt-Swin Transformer for BI-RADS Classification

A deep learning pipeline for 5-class BI-RADS classification of mammograms using the VinDr-Mammo dataset. The proposed model pairs a ConvNeXt-Tiny backbone with a Swin Transformer encoder for breast-level classification using paired CC and MLO views.

## Pipeline Overview


Raw DICOM/PNG → Preprocessing → Paired Dataset → Train → Evaluate


## Files

| File | Purpose |
|------|---------|
| `eda.py` | Exploratory data analysis — label distribution, split verification, image statistics |
| `preprocessing.py` | CLAHE enhancement, breast crop, resize to 224×224, builds paired CC+MLO CSV |
| `dataset.py` | PyTorch Dataset for paired CC+MLO breast-level loading, train/val split, MixUp |
| `loss.py` | Named loss functions: `ce`, `ce_weighted`, `focal`, `focal_ls` |
| `train.py` | Training pipeline — pass `--model` and `--loss` as arguments |
| `evaluate.py` | Test set evaluation — F1, AUROC, confusion matrix, BI-RADS 3v4 analysis |
| `run_all.sh` | Runs all model+loss experiment combinations sequentially |

## Models

| File | Architecture |
|------|-------------|
| `models/resnet50.py` | ResNet-50 baseline with paired CC+MLO feature fusion |
| `models/efficientnet_b2.py` | EfficientNet-B2 baseline |
| `models/efficientnet_b4.py` | EfficientNet-B4 baseline |
| `models/swin_t.py` | Swin Transformer-Tiny baseline |
| `models/convnext_t.py` | ConvNeXt-Tiny baseline |
| `models/convnext_swin.py` | **Proposed hybrid** — ConvNeXt-Tiny + Swin Transformer encoder |

## Training

```bash
python train.py --model convnext_swin --loss focal_ls
python train.py --model resnet50 --loss ce_weighted
```

## Dataset

VinDr-Mammo — 5,000 studies, 20,000 mammograms, 5-class BI-RADS labels.
Available at: https://physionet.org/content/vindr-mammo/
