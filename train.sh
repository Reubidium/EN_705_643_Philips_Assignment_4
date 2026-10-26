#!/bin/bash
python run.py \
    --frame_dir UCF50_Captured_Frames \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --cnn_backbone resnet50 \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 8 \
    --mode 'train' \
    --n_epochs 100 \
    --num_workers 4 \
    --fine_tune_last_block