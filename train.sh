#!/bin/bash
python run.py \
    --frame_dir UCF50_Captured_Frames \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 16 \
    --mode 'train' \
    --n_epochs 30 \
    --num_workers 4