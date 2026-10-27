#!/bin/bash
python run.py \
    --ckpt "$1" \
    --cnn_backbone convnext_tiny \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 16 \
    --mode eval
