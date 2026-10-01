#!/bin/bash
# run_all.sh
# Runs all model+loss combinations sequentially.
# Edit the EXPERIMENTS array to add/remove configs.
# Usage: bash run_all.sh

source /workspace/BIRAD/bir/bin/activate
cd /workspace/BIRAD

# Format: "model loss"
EXPERIMENTS=(
    # Phase 1 — ablate loss function on ConvNeXt-T
    "convnext_t ce"
    "convnext_t ce_weighted"
    "convnext_t focal"
    "convnext_t focal_ls"

    # Phase 2 — best loss on all models (change focal_ls if needed)
    "resnet50        focal_ls"
    "efficientnet_b2 focal_ls"
    "efficientnet_b4 focal_ls"
    "swin_t          focal_ls"

    # Phase 3 — proposed hybrid
    "convnext_swin focal_ls"
)

TOTAL=${#EXPERIMENTS[@]}
COUNT=1

for EXP in "${EXPERIMENTS[@]}"; do
    MODEL=$(echo $EXP | awk '{print $1}')
    LOSS=$(echo $EXP  | awk '{print $2}')

    echo ""
    echo "========================================================"
    echo "[$COUNT/$TOTAL] Model: $MODEL | Loss: $LOSS"
    echo "========================================================"

    python train.py --model $MODEL --loss $LOSS

    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed: $MODEL $LOSS — skipping"
    fi

    COUNT=$((COUNT + 1))
done

echo ""
echo "========================================================"
echo "ALL EXPERIMENTS DONE"
echo "========================================================"