#!/bin/bash
python run.py \
    --frame_dir UCF50_Captured_Frames \
    --train_size 0.60 \
    --test_size 0.20 \
    --model_type lrcn \
    --cnn_backbone convnext_tiny \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 6 \
    --mode 'train' \
    --n_epochs 100 \
    --num_workers 4 \
    --dropout 0.3 \
    --random_temporal_sampling
