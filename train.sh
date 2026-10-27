#!/bin/bash
python run.py \
    --frame_dir UCF50_Captured_Frames \
    --train_size 0.60 \
    --test_size 0.20 \
    --model_type lrcn \
    --cnn_backbone resnet50 \
    --n_classes 50 \
    --fr_per_vid 32 \
    --batch_size 32 \
    --mode 'train' \
    --n_epochs 100 \
    --num_workers 6 \
    --dropout 0.3 \
    --fine_tune_last_block