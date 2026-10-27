#!/bin/bash
python run.py \
    --ckpt ./runs/20261026_205442_635036/models/best_model_wts.pt \
    --cnn_backbone resnet50 \
    --n_classes 50 \
    --fr_per_vid 32 \
    --batch_size 16 \
    --mode eval
