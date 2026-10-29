#!/bin/bash
python run.py \
    --ckpt ./runs/YOUR_ATTENTION_RUN/models/best_model_wts.pt \
    --cnn_backbone convnext_tiny \
    --rnn_hidden_size 100 \
    --rnn_n_layers 1 \
    --bidirectional \
    --rnn_layer_norm \
    --temporal_attention \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 16 \
    --mode eval
