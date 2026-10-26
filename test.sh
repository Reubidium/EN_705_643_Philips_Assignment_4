#!/bin/bash
python run.py \
    --ckpt ./runs/20261025_191345_023974/models/best_model_wts.pt \
    --cnn_backbone resnet50 \
    --n_classes 50 \
    --batch_size 16 \
    --mode eval
